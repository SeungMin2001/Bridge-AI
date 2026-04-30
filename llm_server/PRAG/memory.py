"""MergePRAG-style HyperKV memory and injection utilities."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ALPHA, HIDDEN_DIM, MAX_MEMORY_TOKENS, MAX_SEQ_LEN, NUM_KV
from .prompts import system_prompt, user_prompt


class HyperKVGenerator(nn.Module):
    """Paper-style HyperKVGeneratorFixed with dynamic model dimensions.

    This follows the local MergePRAG author code closely:
      token embeddings -> attentive pooling -> MLP -> linear K/V
    The difference is that num_kv can be >1 for service memory capacity.
    """

    def __init__(self, d_model: int, num_kv: int = NUM_KV, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.d_model = d_model
        self.num_kv = num_kv
        self.att_pool = nn.Linear(d_model, 1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.linear_K = nn.Linear(hidden_dim, num_kv * d_model)
        self.linear_V = nn.Linear(hidden_dim, num_kv * d_model)

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if MAX_MEMORY_TOKENS > 0 and embeddings.size(1) > MAX_MEMORY_TOKENS:
            embeddings = embeddings[:, :MAX_MEMORY_TOKENS]
            attention_mask = attention_mask[:, :MAX_MEMORY_TOKENS] if attention_mask is not None else None
        scores = self.att_pool(embeddings).squeeze(-1)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
            
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (embeddings * weights).sum(dim=1)
        hidden = self.mlp(pooled)
        batch = hidden.size(0)
        
        K = self.linear_K(hidden).view(batch, self.num_kv, self.d_model)
        V = self.linear_V(hidden).view(batch, self.num_kv, self.d_model)
        return {"pooled": pooled, "hidden": hidden, "K": K, "V": V, "att_weights": weights}


def tokenize_passage(tokenizer, passage: str, device):
    encoded = tokenizer(
        passage,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    return {k: v.to(device) for k, v in encoded.items()}


def encode_memory(model, tokenizer, hypernet: HyperKVGenerator, passage: str, device):
    encoded = tokenize_passage(tokenizer, passage, device)
    embeddings = model.model.embed_tokens(encoded["input_ids"]).to(dtype=torch.float32)
    memory = hypernet(embeddings, encoded.get("attention_mask"))
    memory["input_ids"] = encoded["input_ids"]
    memory["attention_mask"] = encoded.get("attention_mask")
    return memory


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


def make_memory_hook(K: torch.Tensor, V: torch.Tensor, num_heads: int, alpha: float = ALPHA):
    def hook_fn(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        K_local = K.to(device=hidden.device, dtype=hidden.dtype)
        V_local = V.to(device=hidden.device, dtype=hidden.dtype)
        new_hidden = hidden + alpha * cross_attention(hidden, K_local, V_local, num_heads)
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    return hook_fn


def build_chat_prompt(tokenizer, question: str, answer: str = "") -> str:
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


def forward_with_memory(model, target_layer, K: torch.Tensor, V: torch.Tensor, tok, alpha: float = ALPHA):
    hook = target_layer.register_forward_hook(make_memory_hook(K, V, model_num_heads(model), alpha=alpha))
    try:
        logits = model(**tok)["logits"]
    finally:
        hook.remove()
    return logits

