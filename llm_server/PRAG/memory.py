"""MergePRAG-style HyperKV memory and injection utilities."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig

from .config import (
    ALPHA,
    HIDDEN_DIM,
    MAX_MEMORY_TOKENS,
    MAX_SEQ_LEN,
    MODEL_NAME,
    NUM_KV,
    QUESTION_CONDITIONED_MEMORY,
    USE_CONTEXTUAL_MEMORY,
    contains_hangul,
)
from .prompts import system_prompt, user_prompt


def deterministic_generation_config(tokenizer, max_new_tokens: int) -> GenerationConfig:
    """Greedy generation config without stale sampling flags from model config.

    Some chat models ship `temperature`, `top_p`, or `top_k` in their default
    generation config. We run diagnostics with `do_sample=False` for
    reproducibility, so those sampling-only fields should be cleared to avoid
    noisy Transformers warnings during validation probes.
    """
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )


class HyperKVGenerator(nn.Module):
    """MergePRAG-style HyperKV generator adapted for service QA.

    The paper path is preserved: passage features -> attentive pooling -> MLP
    -> linear K/V. For the service setting, the input features can include the
    base model's contextual hidden states, and the memory text can include the
    user question. This lets K/V encode "the answer to this question from this
    retrieved passage" instead of a question-agnostic bag of token embeddings.
    """

    def __init__(
        self,
        d_model: int,
        num_kv: int = NUM_KV,
        hidden_dim: int = HIDDEN_DIM,
        feature_dim: int | None = None,
        question_fusion: str = "none",
        legacy: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim or d_model
        self.num_kv = num_kv
        self.question_fusion = question_fusion or "none"
        self.legacy = legacy
        if self.question_fusion not in {"none", "text_concat", "feature_concat", "kv_adapter"}:
            raise ValueError(f"Unsupported question_fusion={self.question_fusion!r}")
        if legacy and self.question_fusion in {"feature_concat", "kv_adapter"}:
            raise ValueError(f"{self.question_fusion} question fusion is not supported with legacy HyperKV.")
        if legacy:
            self.att_pool = nn.Linear(d_model, 1)
        else:
            if self.question_fusion == "feature_concat":
                self.question_feature_fusion = nn.Sequential(
                    nn.LayerNorm(self.feature_dim * 2),
                    nn.Linear(self.feature_dim * 2, self.feature_dim),
                    nn.GELU(),
                    nn.LayerNorm(self.feature_dim),
                )
            if self.question_fusion == "kv_adapter":
                self.k_adapter = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
                self.v_adapter = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
            self.input_norm = nn.LayerNorm(self.feature_dim)
            self.input_proj = nn.Sequential(
                nn.Linear(self.feature_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.att_pool = nn.Linear(d_model, num_kv)
        activation = nn.ReLU if legacy else nn.GELU
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            activation(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
        )
        if legacy:
            self.linear_K = nn.Linear(hidden_dim, num_kv * d_model)
            self.linear_V = nn.Linear(hidden_dim, num_kv * d_model)
        else:
            self.linear_K = nn.Linear(hidden_dim, d_model)
            self.linear_V = nn.Linear(hidden_dim, d_model)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        question_features: torch.Tensor | None = None,
    ):
        if MAX_MEMORY_TOKENS > 0 and features.size(1) > MAX_MEMORY_TOKENS:
            features = features[:, :MAX_MEMORY_TOKENS]
            attention_mask = attention_mask[:, :MAX_MEMORY_TOKENS] if attention_mask is not None else None
        if question_features is not None:
            if self.question_fusion != "feature_concat" or not hasattr(self, "question_feature_fusion"):
                raise ValueError("question_features require question_fusion='feature_concat'.")
            if question_features.dim() == 2:
                question_features = question_features.unsqueeze(1)
            if question_features.size(1) == 1:
                question_features = question_features.expand(-1, features.size(1), -1)
            if question_features.shape[:2] != features.shape[:2]:
                raise ValueError(
                    f"Question features must align with memory tokens: {question_features.shape} vs {features.shape}"
                )
            features = self.question_feature_fusion(torch.cat([features, question_features.to(features.dtype)], dim=-1))
        if self.legacy:
            encoded = features
            scores = self.att_pool(encoded).squeeze(-1)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1)
            hidden = self.mlp(pooled)
            batch = hidden.size(0)
            K = self.linear_K(hidden).view(batch, self.num_kv, self.d_model)
            V = self.linear_V(hidden).view(batch, self.num_kv, self.d_model)
            return {"encoded": encoded, "pooled": pooled, "hidden": hidden, "K": K, "V": V, "att_weights": weights}

        encoded = self.input_proj(self.input_norm(features))
        scores = self.att_pool(encoded)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, torch.finfo(scores.dtype).min)
            
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("btd,bts->bsd", encoded, weights)
        hidden = self.mlp(pooled)
        
        K = self.linear_K(hidden)
        V = self.linear_V(hidden)
        result = {"encoded": encoded, "pooled": pooled, "hidden": hidden, "K": K, "V": V, "att_weights": weights}
        if self.question_fusion == "kv_adapter":
            k_delta = self.k_adapter(K)
            v_delta = self.v_adapter(V)
            result.update(
                {
                    "K_pre_adapter": K,
                    "V_pre_adapter": V,
                    "K_adapter_delta": k_delta,
                    "V_adapter_delta": v_delta,
                    "K": K + k_delta,
                    "V": V + v_delta,
                }
            )
        return result


def build_memory_text(passage: str, question: str | None = None, question_conditioned: bool = QUESTION_CONDITIONED_MEMORY) -> str:
    if question and question_conditioned:
        if contains_hangul(f"{question}\n{passage}"):
            return f"질문: {question}\n관련 수업/회의 내용: {passage}"
        return f"Question: {question}\nRelevant lecture/meeting content: {passage}"
    return passage


def should_text_condition(question_conditioned: bool, question_fusion: str) -> bool:
    return bool(question_conditioned and question_fusion in {"text_concat", "kv_adapter"})


def tokenize_passage(tokenizer, passage: str, device):
    encoded = tokenizer(
        passage,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    return {k: v.to(device) for k, v in encoded.items()}


@torch.no_grad()
def passage_features(
    model,
    tokenizer,
    passage: str,
    device,
    *,
    question: str | None = None,
    use_contextual: bool | None = None,
    question_conditioned: bool = QUESTION_CONDITIONED_MEMORY,
    question_fusion: str = "text_concat",
):
    memory_text = build_memory_text(
        passage,
        question,
        question_conditioned=should_text_condition(question_conditioned, question_fusion),
    )
    encoded = tokenize_passage(tokenizer, memory_text, device)
    raw = model.model.embed_tokens(encoded["input_ids"]).to(dtype=torch.float32)
    if use_contextual is None:
        use_contextual = USE_CONTEXTUAL_MEMORY
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
    return features, encoded.get("attention_mask"), encoded["input_ids"]


def masked_mean(features: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    if attention_mask is None:
        return features.mean(dim=1)
    mask = attention_mask.to(device=features.device, dtype=features.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (features * mask).sum(dim=1) / denom


def encode_memory(
    model,
    tokenizer,
    hypernet: HyperKVGenerator,
    passage: str,
    device,
    *,
    question: str | None = None,
    use_contextual: bool | None = None,
    question_conditioned: bool = QUESTION_CONDITIONED_MEMORY,
):
    if use_contextual is None:
        use_contextual = hypernet.feature_dim != hypernet.d_model
    question_fusion = getattr(hypernet, "question_fusion", "none")
    features, attention_mask, input_ids = passage_features(
        model,
        tokenizer,
        passage,
        device,
        question=question,
        use_contextual=use_contextual,
        question_conditioned=question_conditioned,
        question_fusion=question_fusion,
    )
    question_features = None
    if question and question_conditioned and question_fusion == "feature_concat":
        q_features, q_mask, _ = passage_features(
            model,
            tokenizer,
            question,
            device,
            question=None,
            use_contextual=use_contextual,
            question_conditioned=False,
            question_fusion="none",
        )
        question_features = masked_mean(q_features, q_mask)
    memory = hypernet(features, attention_mask, question_features=question_features)
    memory["input_ids"] = input_ids
    memory["attention_mask"] = attention_mask
    return memory


def orthogonal_merge_slots(base: torch.Tensor | None, update: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Merge two slot matrices by adding only the update's orthogonal component.

    This mirrors the local MergePRAG author-code path:
    `B_orth = B - Q_A(Q_A^T B)` where `Q_A` comes from QR over existing slots.
    Inputs can be `[num_kv, d_model]` or `[batch, num_kv, d_model]`; the output
    keeps the same shape as `update`/`base`, so the downstream hook does not
    need to change when multiple passages are fused into one K/V memory.
    """
    if base is None:
        return update
    if base.shape != update.shape:
        raise ValueError(f"Cannot orthogonally merge tensors with different shapes: {base.shape} vs {update.shape}")
    squeezed = False
    if base.dim() == 2:
        base = base.unsqueeze(0)
        update = update.unsqueeze(0)
        squeezed = True
    if base.dim() != 3:
        raise ValueError(f"Expected [num_kv, d_model] or [batch, num_kv, d_model], got {base.shape}")

    merged = []
    for existing, incoming in zip(base, update):
        existing_cols = existing.transpose(0, 1).to(dtype=torch.float32)
        incoming_cols = incoming.transpose(0, 1).to(dtype=torch.float32)
        q_existing, _ = torch.linalg.qr(existing_cols, mode="reduced")
        projection = q_existing @ (q_existing.transpose(0, 1) @ incoming_cols)
        orthogonal_component = incoming_cols - projection
        fused = existing_cols + orthogonal_component
        merged.append(fused.transpose(0, 1).to(dtype=update.dtype))
    out = torch.stack(merged, dim=0)
    return out.squeeze(0) if squeezed else out


def merge_memory_dicts(memories: list[dict], mode: str = "orthogonal") -> dict:
    if not memories:
        raise ValueError("Need at least one memory to merge.")
    if len(memories) == 1:
        return dict(memories[0])

    if mode != "orthogonal":
        raise ValueError(f"Unsupported memory merge mode: {mode}")
    merged_k = None
    merged_v = None
    for memory in memories:
        merged_k = orthogonal_merge_slots(merged_k, memory["K"])
        merged_v = orthogonal_merge_slots(merged_v, memory["V"])
    merged = dict(memories[0])
    merged["K"] = merged_k
    merged["V"] = merged_v
    merged["merged_count"] = len(memories)
    merged["merge_mode"] = mode
    return merged


def encode_merged_memory(
    model,
    tokenizer,
    hypernet: HyperKVGenerator,
    passages: list[str],
    device,
    *,
    question: str | None = None,
    use_contextual: bool | None = None,
    question_conditioned: bool = QUESTION_CONDITIONED_MEMORY,
    merge_mode: str = "orthogonal",
):
    clean_passages = [p for p in passages if str(p or "").strip()]
    if not clean_passages:
        raise ValueError("Need at least one non-empty passage for memory encoding.")
    memories = [
        encode_memory(
            model,
            tokenizer,
            hypernet,
            passage,
            device,
            question=question,
            use_contextual=use_contextual,
            question_conditioned=question_conditioned,
        )
        for passage in clean_passages
    ]
    merged = merge_memory_dicts(memories, mode=merge_mode)
    merged["passages"] = clean_passages
    return merged


def cross_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_heads: int) -> torch.Tensor:
    if Q.dim() == 2:
        Q = Q.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if K.dim() == 2:
        K = K.unsqueeze(0)
    if V.dim() == 2:
        V = V.unsqueeze(0)
    batch, query_len, d_model = Q.shape
    key_len = K.shape[1]
    if d_model % num_heads != 0:
        raise ValueError(f"d_model={d_model} is not divisible by num_heads={num_heads}")
    head_dim = d_model // num_heads
    Qh = Q.view(batch, query_len, num_heads, head_dim).transpose(1, 2)
    Kh = K.view(batch, key_len, num_heads, head_dim).transpose(1, 2)
    Vh = V.view(batch, key_len, num_heads, head_dim).transpose(1, 2)
    att = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(head_dim)
    out = att.softmax(dim=-1) @ Vh
    out = out.transpose(1, 2).contiguous().view(batch, query_len, d_model)
    return out.squeeze(0) if squeeze else out


def model_num_heads(model) -> int:
    return int(getattr(model.config, "num_attention_heads"))


def additive_memory(hidden: torch.Tensor, V: torch.Tensor, alpha: float, *, last_token_only: bool = False) -> torch.Tensor:
    """Inject a direct memory bias without requiring the model to attend to K/V.

    This is intended for diagnostics/ablations. The trained path remains the
    MergePRAG-style cross-attention hook, but additive injection helps determine
    whether failures come from weak attention-to-memory reads or from the memory
    vector itself not carrying the answer.
    """
    if V.dim() == 2:
        V = V.unsqueeze(0)
    memory = V.mean(dim=1).to(device=hidden.device, dtype=hidden.dtype).unsqueeze(1)
    if not last_token_only:
        return hidden + alpha * memory
    new_hidden = hidden.clone()
    new_hidden[:, -1:, :] = new_hidden[:, -1:, :] + alpha * memory
    return new_hidden


def make_memory_hook(
    K: torch.Tensor,
    V: torch.Tensor,
    num_heads: int,
    alpha: float = ALPHA,
    injection_mode: str = "attention",
):
    def hook_fn(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        K_local = K.to(device=hidden.device, dtype=hidden.dtype)
        V_local = V.to(device=hidden.device, dtype=hidden.dtype)
        if injection_mode == "attention":
            new_hidden = hidden + alpha * cross_attention(hidden, K_local, V_local, num_heads)
        elif injection_mode == "add_all":
            new_hidden = additive_memory(hidden, V_local, alpha, last_token_only=False)
        elif injection_mode == "add_last":
            new_hidden = additive_memory(hidden, V_local, alpha, last_token_only=True)
        elif injection_mode == "hybrid":
            attended = hidden + alpha * cross_attention(hidden, K_local, V_local, num_heads)
            new_hidden = additive_memory(attended, V_local, alpha, last_token_only=False)
        else:
            raise ValueError(f"Unsupported injection_mode: {injection_mode}")
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    return hook_fn


def uses_chat_prompt(tokenizer) -> bool:
    """Use chat templates only for instruction/chat checkpoints.

    The MergePRAG author path trains with a plain ``Question/Answer`` prompt.
    Base checkpoints such as ``Qwen/Qwen2.5-7B`` may not define a compatible
    chat template, so falling back to the paper-style prompt avoids silently
    training/evaluating with an instruction-only format.
    """
    model_name = str(getattr(tokenizer, "name_or_path", "") or MODEL_NAME).lower()
    return bool(getattr(tokenizer, "chat_template", None)) and any(
        marker in model_name for marker in ("instruct", "chat")
    )


def build_chat_prompt(tokenizer, question: str, answer: str = "") -> str:
    if not uses_chat_prompt(tokenizer):
        if contains_hangul(question):
            prompt = f"질문: {question}\n답변:"
        else:
            prompt = f"Question: {question}\nAnswer:"
        return f"{prompt} {answer}" if answer else prompt

    messages = [
        {"role": "system", "content": system_prompt(question)},
        {"role": "user", "content": user_prompt(question)},
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
    tok_answer = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False)
    input_ids = torch.cat([tok_prompt["input_ids"], tok_answer["input_ids"]], dim=-1).to(device)
    labels = torch.cat(
        [
            torch.full((1, tok_prompt["input_ids"].shape[1]), -100, dtype=torch.long),
            tok_answer["input_ids"],
        ],
        dim=-1,
    ).to(device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}


def compute_answer_loss(logits: torch.Tensor, labels: torch.Tensor):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != -100
    if valid.sum() == 0:
        return None
    return F.cross_entropy(shift_logits[valid], shift_labels[valid])


def forward_with_memory(
    model,
    target_layer,
    K: torch.Tensor,
    V: torch.Tensor,
    tok,
    alpha: float = ALPHA,
    injection_mode: str = "attention",
):
    hook = target_layer.register_forward_hook(
        make_memory_hook(K, V, model_num_heads(model), alpha=alpha, injection_mode=injection_mode)
    )
    try:
        logits = model(**tok)["logits"]
    finally:
        hook.remove()
    return logits
