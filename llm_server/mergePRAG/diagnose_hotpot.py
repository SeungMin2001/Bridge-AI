"""
Diagnose whether the trained HyperNetwork distinguishes real processed Hotpot passages.

Usage:
  python -m llm_server.mergePRAG.diagnose_hotpot
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..run_model import run_model
from .config import (
    ALPHA,
    TRAIN_DATA_PATH,
    NUM_KV,
    load_critical_layer,
    load_hypernet_state_dict,
)
from .embedding import encode_passage_states, tokenize_conditioned_memory
from .hypernetwork import HyperNetwork
from .cross_attention import cross_attention


NUM_SAMPLES_TO_SHOW = 3
MAX_ROWS = 200
CRITICAL_LAYER = load_critical_layer()


def iter_jsonl(path: str):
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def masked_mean(hidden, mask):
    weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denom


def encode_passage(model, tokenizer, hypernet, question: str, passage: str, device):
    with torch.no_grad():
        encoded = tokenize_conditioned_memory(
            tokenizer,
            question,
            passage,
            device,
            max_length=512,
        )
        ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        question_mask = encoded["question_mask"]
        passage_mask = encoded["passage_mask"]
        focus_weight = encoded.get("focus_weight")
        emb = encode_passage_states(
            model,
            ids,
            attention_mask=attention_mask,
            use_contextual=True,
        )
        query = masked_mean(emb, question_mask)
        pooled = hypernet.pooling(emb, mask=attention_mask, query=query, focus_mask=passage_mask)
        hidden = hypernet.mlp(pooled)
        k_raw, v_raw = hypernet.lp(hidden)
        k, v = hypernet(emb, attention_mask=attention_mask, query=query, focus_mask=passage_mask)
    return pooled, hidden, k_raw, v_raw, k, v


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.view(1, -1), b.view(1, -1)).item()


def make_hook(dk, dv, alpha=ALPHA):
    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        k = dk.to(device=hidden.device, dtype=hidden.dtype)
        v = dv.to(device=hidden.device, dtype=hidden.dtype)
        delta = cross_attention(hidden, k, v)
        result = hidden + alpha * delta
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result
    return hook_fn


def decode_answer(model, tokenizer, question: str, dk, dv, layer_idx: int | None):
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Answer in English with one short sentence."},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    hook = None
    if layer_idx is not None:
        layer = model.model.layers[layer_idx]
        hook = layer.register_forward_hook(make_hook(dk, dv))
    try:
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    finally:
        if hook is not None:
            hook.remove()
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    print("[diag] loading model...")
    model, tokenizer = run_model()
    device = next(model.parameters()).device

    hypernet = HyperNetwork(model.config.hidden_size, k=NUM_KV).to(device).float()
    state_dict, load_info = load_hypernet_state_dict(map_location=device)
    hypernet.load_state_dict(state_dict)
    hypernet.eval()
    print(f"[diag] hypernet source: {load_info['source']} ({load_info['kind']}, step={load_info['step']})")
    print(f"[diag] critical layer: {CRITICAL_LAYER}")
    print(f"[diag] dataset: {TRAIN_DATA_PATH}")

    rows = []
    for idx, item in enumerate(iter_jsonl(TRAIN_DATA_PATH)):
        rows.append(item)
        if idx + 1 >= MAX_ROWS:
            break
    if len(rows) < 2:
        raise RuntimeError("Need at least 2 processed rows for diagnosis.")

    samples = rows[:NUM_SAMPLES_TO_SHOW]
    for i, sample in enumerate(samples):
        print(f"\n[sample {i}]")
        print(f"question: {sample['question']}")
        print(f"answer: {sample['answer']}")
        print(f"facts_count: {len(sample.get('facts', []))}")
        print(f"passage: {' '.join(sample.get('facts', []))[:300]}")

    # Compare first two real training samples
    s1, s2 = rows[0], rows[1]
    p1 = " ".join(s1.get("facts", []))
    p2 = " ".join(s2.get("facts", []))

    pooled1, h1, k1_raw, v1_raw, k1, v1 = encode_passage(model, tokenizer, hypernet, s1["question"], p1, device)
    pooled2, h2, k2_raw, v2_raw, k2, v2 = encode_passage(model, tokenizer, hypernet, s2["question"], p2, device)

    print("\n[real sample comparison]")
    print(f"q1: {s1['question']}")
    print(f"a1: {s1['answer']}")
    print(f"q2: {s2['question']}")
    print(f"a2: {s2['answer']}")
    print(f"pooled cosine: {cosine(pooled1, pooled2):.4f}")
    print(f"h cosine: {cosine(h1, h2):.4f}")
    print(f"K raw cosine: {cosine(k1_raw, k2_raw):.4f}")
    print(f"V raw cosine: {cosine(v1_raw, v2_raw):.4f}")
    print(f"K cosine: {cosine(k1, k2):.4f}")
    print(f"V cosine: {cosine(v1, v2):.4f}")

    # Same question, different memories
    print("\n[answer flip on same question]")
    base_q = s1["question"]
    no_hook = decode_answer(model, tokenizer, base_q, k1, v1, layer_idx=None)
    ans_1 = decode_answer(model, tokenizer, base_q, k1, v1, layer_idx=CRITICAL_LAYER)
    ans_2 = decode_answer(model, tokenizer, base_q, k2, v2, layer_idx=CRITICAL_LAYER)
    print(f"question: {base_q}")
    print(f"no hook: {no_hook}")
    print(f"hook with sample1 memory: {ans_1}")
    print(f"hook with sample2 memory: {ans_2}")


if __name__ == "__main__":
    main()
