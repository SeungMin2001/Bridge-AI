import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MAX_SEQ_LEN, contains_hangul


SERVICE_NUM_KV = int(os.getenv("MERGEPRAG_SERVICE_NUM_KV", "8"))
SERVICE_HIDDEN_DIM = int(os.getenv("MERGEPRAG_SERVICE_HIDDEN_DIM", "1024"))
SERVICE_ALPHA = float(os.getenv("MERGEPRAG_SERVICE_ALPHA", "0.3"))
SERVICE_USE_CONTEXTUAL = os.getenv("MERGEPRAG_SERVICE_USE_CONTEXTUAL", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SERVICE_RMS_CLAMP = float(os.getenv("MERGEPRAG_SERVICE_RMS_CLAMP", "0.5"))
SERVICE_SKIP_SCALE = float(os.getenv("MERGEPRAG_SERVICE_SKIP_SCALE", "0.5"))
SERVICE_SYSTEM_PROMPT_EN = os.getenv(
    "MERGEPRAG_SERVICE_SYSTEM_PROMPT",
    (
        "You are a helpful lecture assistant. "
        "Answer using the injected lecture memory. "
        "Keep the answer concise and use the same language as the question."
    ),
)
SERVICE_SYSTEM_PROMPT_KO = os.getenv(
    "MERGEPRAG_SERVICE_SYSTEM_PROMPT_KO",
    (
        "당신은 수업 내용을 기억해 답하는 조교입니다. "
        "주입된 강의 메모리를 사용해 답하세요. "
        "질문과 같은 언어로 간결하게 답하세요."
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
    ):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.num_kv = num_kv
        self.skip_scale = skip_scale
        self.rms_clamp = rms_clamp

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
        scores = torch.einsum("btd,sd->bts", x, self.slot_queries) / math.sqrt(self.d_model)
        scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, torch.finfo(scores.dtype).min)
        att_weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("btd,bts->bsd", x, att_weights)
        hidden = self.mlp(pooled)
        K = self.linear_K(hidden) + self.skip_scale * self.skip_K(pooled)
        V = self.linear_V(hidden) + self.skip_scale * self.skip_V(pooled)
        return {
            "encoded": x,
            "pooled": pooled,
            "hidden": hidden,
            "K": self._rms_clamp(K),
            "V": self._rms_clamp(V),
            "att_weights": att_weights,
        }


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
def passage_features(model, tokenizer, passage: str, device, use_contextual: bool = SERVICE_USE_CONTEXTUAL):
    encoded = tokenize_passage(tokenizer, passage, device)
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


def encode_memory(model, hypernet, tokenizer, passage: str, device, use_contextual: bool = SERVICE_USE_CONTEXTUAL):
    features, attention_mask, input_ids = passage_features(
        model,
        tokenizer,
        passage,
        device,
        use_contextual=use_contextual,
    )
    memory = hypernet(features, attention_mask)
    memory["input_ids"] = input_ids
    memory["attention_mask"] = attention_mask
    return memory


def cross_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_heads: int = 8) -> torch.Tensor:
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
        raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
    d_k = d_model // num_heads
    Qh = Q.view(batch, query_len, num_heads, d_k).transpose(1, 2)
    Kh = K.view(batch, key_len, num_heads, d_k).transpose(1, 2)
    Vh = V.view(batch, key_len, num_heads, d_k).transpose(1, 2)
    att = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(d_k)
    out = att.softmax(dim=-1) @ Vh
    out = out.transpose(1, 2).contiguous().view(batch, query_len, d_model)
    return out.squeeze(0) if squeezed else out


def make_memory_hook(K: torch.Tensor, V: torch.Tensor, alpha: float = SERVICE_ALPHA):
    def hook_fn(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        K_local = K.to(device=hidden.device, dtype=hidden.dtype)
        V_local = V.to(device=hidden.device, dtype=hidden.dtype)
        new_hidden = hidden + alpha * cross_attention(hidden, K_local, V_local)
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    return hook_fn


def build_user_prompt(question: str) -> str:
    return question


def build_direct_user_prompt(question: str, passage: str) -> str:
    if contains_hangul(f"{question}\n{passage}"):
        return f"강의 내용:\n{passage}\n\n질문: {question}"
    return f"Lecture content:\n{passage}\n\nQuestion: {question}"


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
    hook = target_layer.register_forward_hook(make_memory_hook(K, V, alpha=alpha))
    try:
        logits = model(**tok)["logits"]
    finally:
        hook.remove()
    return logits


def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.flatten(1), b.flatten(1)).mean()


def slot_diversity_loss(K: torch.Tensor, V: torch.Tensor, target: float = 0.3):
    if K.size(1) <= 1:
        return K.new_tensor(0.0)

    def penalty(x):
        x = F.normalize(x, dim=-1)
        sim = torch.matmul(x, x.transpose(1, 2)).abs()
        mask = ~torch.eye(sim.size(-1), dtype=torch.bool, device=sim.device).unsqueeze(0)
        offdiag = sim.masked_select(mask)
        return F.relu(offdiag - target).mean()

    return penalty(K) + penalty(V)
