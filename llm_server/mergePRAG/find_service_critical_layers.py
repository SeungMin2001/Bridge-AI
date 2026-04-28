"""
Service-memory critical layer finder.

This scanner matches train_service_memory.py rather than the older paper-like
train.py path:
  passage/question -> ServiceMemoryHyperNetwork -> K/V -> service chat prompt.

Run after changing MERGEPRAG_MODEL_NAME:
    python -m llm_server.mergePRAG.find_service_critical_layers
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_NAME, TRAIN_DATA_PATH, VALID_DATA_PATH
from .service_memory import (
    SERVICE_ALPHA,
    SERVICE_HIDDEN_DIM,
    SERVICE_MAX_MEMORY_TOKENS,
    SERVICE_NUM_KV,
    SERVICE_POOLING_MODE,
    SERVICE_QUESTION_CONDITIONED,
    SERVICE_RMS_CLAMP,
    SERVICE_SKIP_SCALE,
    SERVICE_USE_CONTEXTUAL,
    ServiceMemoryHyperNetwork,
    compute_answer_loss,
    cosine_flat,
    encode_memory,
    forward_with_memory,
    tokenize_qa,
)
from .train2 import MergePRAGDataset, extract_first_hard_negative


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = Path(os.getenv("MERGEPRAG_SERVICE_CRITICAL_LAYERS_PATH", str(BASE_DIR / "critical_layers.json")))
SCAN_MAX_TRAIN_SAMPLES = int(os.getenv("MERGEPRAG_SERVICE_SCAN_TRAIN_SAMPLES", "96"))
SCAN_MAX_VAL_SAMPLES = int(os.getenv("MERGEPRAG_SERVICE_SCAN_VAL_SAMPLES", "48"))
SCAN_STEPS = int(os.getenv("MERGEPRAG_SERVICE_SCAN_STEPS", "60"))
TOP_N = int(os.getenv("MERGEPRAG_SERVICE_SCAN_TOP_N", "5"))
LR = float(os.getenv("MERGEPRAG_SERVICE_SCAN_LR", "8e-5"))
LR_MIN = float(os.getenv("MERGEPRAG_SERVICE_SCAN_LR_MIN", "1e-6"))
RANK_MARGIN = float(os.getenv("MERGEPRAG_SERVICE_SCAN_RANK_MARGIN", "0.5"))


def load_model():
    print(f"[service-layer-scan] loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def make_hypernet(model, device):
    d_model = model.config.hidden_size
    feature_dim = d_model * 2 if SERVICE_USE_CONTEXTUAL else d_model
    return ServiceMemoryHyperNetwork(
        d_model=d_model,
        feature_dim=feature_dim,
        num_kv=SERVICE_NUM_KV,
        hidden_dim=SERVICE_HIDDEN_DIM,
        skip_scale=SERVICE_SKIP_SCALE,
        rms_clamp=SERVICE_RMS_CLAMP,
        pooling_mode=SERVICE_POOLING_MODE,
        max_memory_tokens=SERVICE_MAX_MEMORY_TOKENS,
    ).to(device).float()


def service_pair_objective(model, tokenizer, hypernet, target_layer, sample, device):
    negative = extract_first_hard_negative(sample)
    if not negative or not negative.get("passage") or not negative.get("answer"):
        return None

    question = sample["question"]
    gold = sample["answer"]
    neg_answer = negative["answer"]
    main_mem = encode_memory(model, hypernet, tokenizer, sample["passage"], device, question=question)
    neg_mem = encode_memory(model, hypernet, tokenizer, negative["passage"], device, question=question)
    gold_tok = tokenize_qa(tokenizer, question, gold, device)
    neg_tok = tokenize_qa(tokenizer, question, neg_answer, device)

    main_gold = compute_answer_loss(
        forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], gold_tok, alpha=SERVICE_ALPHA),
        gold_tok["labels"],
    )
    main_neg = compute_answer_loss(
        forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], neg_tok, alpha=SERVICE_ALPHA),
        neg_tok["labels"],
    )
    neg_gold = compute_answer_loss(
        forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], gold_tok, alpha=SERVICE_ALPHA),
        gold_tok["labels"],
    )
    neg_neg = compute_answer_loss(
        forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], neg_tok, alpha=SERVICE_ALPHA),
        neg_tok["labels"],
    )
    if any(loss is None for loss in (main_gold, main_neg, neg_gold, neg_neg)):
        return None

    rank = F.relu(RANK_MARGIN + main_gold - main_neg) + F.relu(RANK_MARGIN + neg_neg - neg_gold)
    objective = main_gold + neg_neg + rank
    return {
        "objective": objective,
        "main_ok": main_gold.item() < main_neg.item(),
        "neg_ok": neg_neg.item() < neg_gold.item(),
        "k_cos": cosine_flat(main_mem["K"], neg_mem["K"]),
        "v_cos": cosine_flat(main_mem["V"], neg_mem["V"]),
    }


@torch.no_grad()
def evaluate_layer(model, tokenizer, hypernet, layer_idx, dataset, device):
    target_layer = model.model.layers[layer_idx]
    hypernet.eval()
    total_loss = 0.0
    main_ok = 0
    neg_ok = 0
    flip_ok = 0
    k_cos = 0.0
    v_cos = 0.0
    count = 0
    for idx, sample in enumerate(dataset):
        if idx >= SCAN_MAX_VAL_SAMPLES:
            break
        out = service_pair_objective(model, tokenizer, hypernet, target_layer, sample, device)
        if out is None:
            continue
        total_loss += out["objective"].item()
        main_ok += int(out["main_ok"])
        neg_ok += int(out["neg_ok"])
        flip_ok += int(out["main_ok"] and out["neg_ok"])
        k_cos += out["k_cos"].item()
        v_cos += out["v_cos"].item()
        count += 1
        if torch.cuda.is_available() and idx % 24 == 0:
            torch.cuda.empty_cache()
    hypernet.train()
    denom = max(count, 1)
    return {
        "avg_loss": total_loss / denom,
        "main_ok": main_ok / denom,
        "neg_ok": neg_ok / denom,
        "flip_ok": flip_ok / denom,
        "k_cos": k_cos / denom,
        "v_cos": v_cos / denom,
        "count": count,
    }


def train_layer(model, tokenizer, layer_idx, train_dataset, val_dataset, device):
    target_layer = model.model.layers[layer_idx]
    hypernet = make_hypernet(model, device)
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=LR, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, min(len(train_dataset), SCAN_STEPS)),
        eta_min=LR_MIN,
    )

    hypernet.train()
    step = 0
    for sample in train_dataset:
        if step >= SCAN_STEPS:
            break
        out = service_pair_objective(model, tokenizer, hypernet, target_layer, sample, device)
        if out is None:
            continue
        optimizer.zero_grad()
        out["objective"].backward()
        torch.nn.utils.clip_grad_norm_(hypernet.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

    metrics = evaluate_layer(model, tokenizer, hypernet, layer_idx, val_dataset, device)
    del hypernet, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def find_service_critical_layers(model, tokenizer, top_n: int = TOP_N):
    device = next(model.parameters()).device
    train_dataset = MergePRAGDataset(TRAIN_DATA_PATH, max_samples=SCAN_MAX_TRAIN_SAMPLES)
    val_dataset = MergePRAGDataset(VALID_DATA_PATH, max_samples=SCAN_MAX_VAL_SAMPLES)
    if not train_dataset or not val_dataset:
        raise RuntimeError("Service layer scan requires non-empty train/valid datasets.")

    num_layers = len(model.model.layers)
    layer_results = []
    print(
        f"[service-layer-scan] layers={num_layers}, train={len(train_dataset)}, "
        f"valid={len(val_dataset)}, steps={SCAN_STEPS}"
    )
    print(
        f"[service-layer-scan] service config | num_kv={SERVICE_NUM_KV}, alpha={SERVICE_ALPHA}, "
        f"contextual={SERVICE_USE_CONTEXTUAL}, pooling={SERVICE_POOLING_MODE}, "
        f"question_conditioned={SERVICE_QUESTION_CONDITIONED}"
    )

    for layer_idx in range(num_layers):
        print(f"[service-layer-scan] layer {layer_idx}: short train/eval...")
        metrics = train_layer(model, tokenizer, layer_idx, train_dataset, val_dataset, device)
        result = {"layer": layer_idx, **{k: round(v, 4) for k, v in metrics.items() if k != "count"}, "count": metrics["count"]}
        layer_results.append(result)
        print(
            f"  -> loss={metrics['avg_loss']:.4f} flip={metrics['flip_ok']:.3f} "
            f"Kcos={metrics['k_cos']:.4f} Vcos={metrics['v_cos']:.4f}"
        )

    layer_results.sort(key=lambda item: (item["avg_loss"], -item["flip_ok"]))
    critical = [item["layer"] for item in layer_results[:top_n]]
    output = {
        "scanner": "service_memory",
        "model": MODEL_NAME,
        "num_layers": num_layers,
        "d_model": model.config.hidden_size,
        "critical_layers": critical,
        "all_layers": layer_results,
        "scan_steps": SCAN_STEPS,
        "scan_train_samples": SCAN_MAX_TRAIN_SAMPLES,
        "scan_val_samples": SCAN_MAX_VAL_SAMPLES,
        "service_config": {
            "num_kv": SERVICE_NUM_KV,
            "alpha": SERVICE_ALPHA,
            "use_contextual": SERVICE_USE_CONTEXTUAL,
            "skip_scale": SERVICE_SKIP_SCALE,
            "pooling_mode": SERVICE_POOLING_MODE,
            "question_conditioned": SERVICE_QUESTION_CONDITIONED,
            "rms_clamp": SERVICE_RMS_CLAMP,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[service-layer-scan] top-{top_n}: {critical}")
    print(f"[service-layer-scan] saved: {OUTPUT_PATH}")
    return critical


if __name__ == "__main__":
    model, tokenizer = load_model()
    find_service_critical_layers(model, tokenizer)
