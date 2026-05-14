import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MAX_SEQ_LEN, contains_hangul


SERVICE_NUM_KV = int(os.getenv("MERGEPRAG_SERVICE_NUM_KV", "8"))
SERVICE_HIDDEN_DIM = int(os.getenv("MERGEPRAG_SERVICE_HIDDEN_DIM", "1024"))
SERVICE_ALPHA = float(os.getenv("MERGEPRAG_SERVICE_ALPHA", "0.3"))
SERVICE_USE_CONTEXTUAL = os.getenv("MERGEPRAG_SERVICE_USE_CONTEXTUAL", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SERVICE_RMS_CLAMP = float(os.getenv("MERGEPRAG_SERVICE_RMS_CLAMP", "0.5"))
SERVICE_SKIP_SCALE = float(os.getenv("MERGEPRAG_SERVICE_SKIP_SCALE", "0.0"))
SERVICE_POOLING_MODE = os.getenv("MERGEPRAG_SERVICE_POOLING_MODE", "slot").strip().lower()
SERVICE_MAX_MEMORY_TOKENS = int(os.getenv("MERGEPRAG_SERVICE_MAX_MEMORY_TOKENS", "128"))
SERVICE_QUESTION_CONDITIONED = os.getenv("MERGEPRAG_SERVICE_QUESTION_CONDITIONED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEBUG_HOOK = os.getenv("MERGEPRAG_DEBUG_HOOK", "0").strip().lower() in {"1", "true", "yes", "on"}
SERVICE_SYSTEM_PROMPT_EN = os.getenv(
    "MERGEPRAG_SERVICE_SYSTEM_PROMPT",
    (
        "You are a strict lecture-memory QA assistant. "
        "Use only the injected lecture memory or the provided lecture content. "
        "If the answer is not explicitly supported, answer exactly: Unknown. "
        "Do not guess. Do not explain your reasoning. "
        "Return only the final answer, preferably a short phrase."
    ),
)
SERVICE_SYSTEM_PROMPT_KO = os.getenv(
    "MERGEPRAG_SERVICE_SYSTEM_PROMPT_KO",
    (
        "당신은 엄격한 수업 메모리 질의응답 조교입니다. "
        "주입된 강의 메모리 또는 제공된 강의 내용만 근거로 답하세요. "
        "정답 근거가 명시적으로 없으면 정확히 '모름'이라고만 답하세요. "
        "추측하지 마세요. 추론 과정을 쓰지 마세요. "
        "최종 답만 짧은 구로 출력하세요."
    ),
)


def select_service_system_prompt(question: str) -> str:
    return SERVICE_SYSTEM_PROMPT_KO if contains_hangul(question) else SERVICE_SYSTEM_PROMPT_EN


class ServiceMemoryHyperNetwork(nn.Module):
    """Service-oriented HyperKV generator.

    The local paper code compresses a passage to one attentive pooled vector.
    Our service needs to preserve the entities and relations inside a short
    lecture sentence, so this module keeps multiple memory slots. Each slot has
    its own learned query over passage token features and then produces one K/V
    vector.
    """

    def __init__(
        self,
        d_model: int,
        feature_dim: int,
        num_kv: int = SERVICE_NUM_KV,
        hidden_dim: int = SERVICE_HIDDEN_DIM,
        skip_scale: float = SERVICE_SKIP_SCALE,
        rms_clamp: float = SERVICE_RMS_CLAMP,
        pooling_mode: str = SERVICE_POOLING_MODE,
        max_memory_tokens: int = SERVICE_MAX_MEMORY_TOKENS,
    ):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.num_kv = num_kv
        self.skip_scale = skip_scale
        self.rms_clamp = rms_clamp
        self.pooling_mode = pooling_mode
        self.max_memory_tokens = max_memory_tokens

        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.slot_queries = nn.Parameter(torch.randn(num_kv, d_model) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.linear_K = nn.Linear(hidden_dim, d_model)
        self.linear_V = nn.Linear(hidden_dim, d_model)
        self.skip_K = nn.Linear(d_model, d_model)
        self.skip_V = nn.Linear(d_model, d_model)

    def _rms_clamp(self, x: torch.Tensor) -> torch.Tensor:
        if self.rms_clamp <= 0:
            return x
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        scale = torch.clamp(self.rms_clamp / rms, max=1.0)
        return x * scale

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor):
        x = self.input_proj(self.input_norm(features))
        if self.pooling_mode == "token":
            if self.max_memory_tokens > 0 and x.size(1) > self.max_memory_tokens:
                x = x[:, : self.max_memory_tokens, :]
                attention_mask = attention_mask[:, : self.max_memory_tokens]
            pooled = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
            hidden = self.mlp(pooled)
            att_weights = None
        elif self.pooling_mode == "slot":
            scores = torch.einsum("btd,sd->bts", x, self.slot_queries) / math.sqrt(self.d_model)
            scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, torch.finfo(scores.dtype).min)
            att_weights = torch.softmax(scores, dim=1)
            pooled = torch.einsum("btd,bts->bsd", x, att_weights)
            hidden = self.mlp(pooled)
        else:
            raise ValueError(f"Unsupported service memory pooling mode: {self.pooling_mode}")
        K = self.linear_K(hidden)
        V = self.linear_V(hidden)
        if self.skip_scale != 0:
            K = K + self.skip_scale * self.skip_K(pooled)
            V = V + self.skip_scale * self.skip_V(pooled)
        return {
            "encoded": x,
            "pooled": pooled,
            "hidden": hidden,
            "K": self._rms_clamp(K),
            "V": self._rms_clamp(V),
            "att_weights": att_weights,
        }


def build_memory_text(passage: str, question: str | None = None) -> str:
    if question and SERVICE_QUESTION_CONDITIONED:
        if contains_hangul(f"{question}\n{passage}"):
            return f"질문: {question}\n강의 내용: {passage}"
        return f"Question: {question}\nLecture content: {passage}"
    return passage


def tokenize_passage(tokenizer, passage: str, device):
    encoded = tokenizer(
        passage,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


@torch.no_grad()
def passage_features(model, tokenizer, passage: str, device, use_contextual: bool = SERVICE_USE_CONTEXTUAL, question: str | None = None):
    encoded = tokenize_passage(tokenizer, build_memory_text(passage, question), device)
    raw = model.model.embed_tokens(encoded["input_ids"]).to(dtype=torch.float32)
    if not use_contextual:
        features = raw
    else:
        outputs = model.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        contextual = outputs.last_hidden_state.to(dtype=torch.float32)
        features = torch.cat([raw, contextual], dim=-1)
    return features, encoded["attention_mask"], encoded["input_ids"]


def encode_memory(
    model,
    hypernet,
    tokenizer,
    passage: str,
    device,
    use_contextual: bool = SERVICE_USE_CONTEXTUAL,
    question: str | None = None,
):
    features, attention_mask, input_ids = passage_features(
        model,
        tokenizer,
        passage,
        device,
        use_contextual=use_contextual,
        question=question,
    )
    memory = hypernet(features, attention_mask)
    memory["input_ids"] = input_ids
    memory["attention_mask"] = attention_mask
    return memory


def cross_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_heads: int) -> torch.Tensor:
    squeezed = False
    if Q.dim() == 2:
        Q = Q.unsqueeze(0)
        squeezed = True
    if K.dim() == 2:
        K = K.unsqueeze(0)
    if V.dim() == 2:
        V = V.unsqueeze(0)

    batch, query_len, d_model = Q.shape
    key_len = K.shape[1]
    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model={d_model} must be divisible by num_heads={num_heads}. "
            "Check model.config.num_attention_heads."
        )
    d_k = d_model // num_heads
    Qh = Q.view(batch, query_len, num_heads, d_k).transpose(1, 2)
    Kh = K.view(batch, key_len, num_heads, d_k).transpose(1, 2)
    Vh = V.view(batch, key_len, num_heads, d_k).transpose(1, 2)
    att = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(d_k)
    out = att.softmax(dim=-1) @ Vh
    out = out.transpose(1, 2).contiguous().view(batch, query_len, d_model)
    return out.squeeze(0) if squeezed else out


def model_num_heads(model) -> int:
    return int(getattr(model.config, "num_attention_heads"))


def make_memory_hook(K: torch.Tensor, V: torch.Tensor, num_heads: int, alpha: float = SERVICE_ALPHA):
    call_count = [0]

    def hook_fn(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        K_local = K.to(device=hidden.device, dtype=hidden.dtype)
        V_local = V.to(device=hidden.device, dtype=hidden.dtype)
        delta = cross_attention(hidden, K_local, V_local, num_heads=num_heads)
        scaled_delta = alpha * delta
        new_hidden = hidden + scaled_delta
        if DEBUG_HOOK and call_count[0] < 3:
            with torch.no_grad():
                hidden_norm = hidden.norm(dim=-1).mean().item()
                delta_norm = scaled_delta.norm(dim=-1).mean().item()
                ratio = delta_norm / max(hidden_norm, 1e-8)
                print(
                    f"[HOOK call={call_count[0]}] hidden_norm={hidden_norm:.4f} "
                    f"delta_norm={delta_norm:.4f} ratio={ratio:.4f}"
                )
            call_count[0] += 1
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    return hook_fn


def build_user_prompt(question: str) -> str:
    if contains_hangul(question):
        return f"질문: {question}\n정답만 짧게 답하세요."
    return f"Question: {question}\nAnswer with only the short final answer."


def build_direct_user_prompt(question: str, passage: str) -> str:
    if contains_hangul(f"{question}\n{passage}"):
        return f"강의 내용:\n{passage}\n\n질문: {question}\n정답만 짧게 답하세요."
    return f"Lecture content:\n{passage}\n\nQuestion: {question}\nAnswer with only the short final answer."


def build_chat_prompt(tokenizer, question: str, answer: str = "") -> str:
    messages = [
        {"role": "system", "content": select_service_system_prompt(question)},
        {"role": "user", "content": build_user_prompt(question)},
    ]
    if answer:
        messages.append({"role": "assistant", "content": answer})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not bool(answer),
        enable_thinking=False,
    )


def build_direct_chat_prompt(tokenizer, question: str, passage: str, answer: str = "") -> str:
    messages = [
        {"role": "system", "content": select_service_system_prompt(question)},
        {"role": "user", "content": build_direct_user_prompt(question, passage)},
    ]
    if answer:
        messages.append({"role": "assistant", "content": answer})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not bool(answer),
        enable_thinking=False,
    )


def tokenize_qa(tokenizer, question: str, answer: str, device):
    prompt = build_chat_prompt(tokenizer, question)
    answer_text = f"{answer}{tokenizer.eos_token}"
    tok_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    tok_answer = tokenizer(
        answer_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    input_ids = torch.cat((tok_prompt["input_ids"], tok_answer["input_ids"]), dim=-1).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels = torch.cat(
        (
            torch.full((1, tok_prompt["input_ids"].shape[1]), -100, dtype=torch.long),
            tok_answer["input_ids"],
        ),
        dim=-1,
    ).to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tokenize_direct_qa(tokenizer, question: str, passage: str, answer: str, device):
    prompt = build_direct_chat_prompt(tokenizer, question, passage)
    answer_text = f"{answer}{tokenizer.eos_token}"
    tok_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    tok_answer = tokenizer(
        answer_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    input_ids = torch.cat((tok_prompt["input_ids"], tok_answer["input_ids"]), dim=-1).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels = torch.cat(
        (
            torch.full((1, tok_prompt["input_ids"].shape[1]), -100, dtype=torch.long),
            tok_answer["input_ids"],
        ),
        dim=-1,
    ).to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def compute_answer_loss(logits: torch.Tensor, labels: torch.Tensor):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != -100
    if valid.sum() == 0:
        return None
    return F.cross_entropy(shift_logits[valid], shift_labels[valid])


def forward_with_memory(model, target_layer, K: torch.Tensor, V: torch.Tensor, tok, alpha: float = SERVICE_ALPHA):
    hook = target_layer.register_forward_hook(
        make_memory_hook(K, V, num_heads=model_num_heads(model), alpha=alpha)
    )
    try:
        logits = model(**tok)["logits"]
    finally:
        hook.remove()
    return logits


def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dim() >= 3 and b.dim() >= 3 and a.size(1) != b.size(1):
        keep = min(a.size(1), b.size(1))
        a = a[:, :keep]
        b = b[:, :keep]
    return F.cosine_similarity(a.flatten(1), b.flatten(1)).mean()


def slot_diversity_loss(K: torch.Tensor, V: torch.Tensor, target: float = 0.3):
    if K.size(1) <= 1:
        return K.new_tensor(0.0)
    # Diversity regularization is meant for a small number of learned summary
    # slots. In token-memory mode, each token is already a separate memory item;
    # forcing all token memories to be nearly orthogonal destroys useful local
    # lexical/relational structure.
    if K.size(1) > SERVICE_NUM_KV:
        return K.new_tensor(0.0)

    def penalty(x):
        x = F.normalize(x, dim=-1)
        sim = torch.matmul(x, x.transpose(1, 2)).abs()
        mask = ~torch.eye(sim.size(-1), dtype=torch.bool, device=sim.device).unsqueeze(0)
        offdiag = sim.masked_select(mask)
        return F.relu(offdiag - target).mean()

    return penalty(K) + penalty(V)
