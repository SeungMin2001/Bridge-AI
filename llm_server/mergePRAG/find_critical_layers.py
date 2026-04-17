"""
Critical Layer Finder

- 각 레이어마다 HyperNetwork를 짧게 학습한 뒤 validation loss를 비교한다.
- 랜덤 K,V를 꽂아보는 방식보다 실제 서비스용 QA-passage 설정에 가깝다.

사용법:
  python -m llm_server.mergePRAG.find_critical_layers
"""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    ALPHA,
    MODEL_NAME,
    NUM_KV,
    TRAIN_DATA_PATH,
    VALID_DATA_PATH,
)
from .hypernetwork import HyperNetwork
from .train import (
    MergePRAGDataset,
    NEGATIVE_LOSS_WEIGHT,
    NEGATIVE_MARGIN,
    REPULSION_LOSS_WEIGHT,
    compute_repulsion_loss,
    compute_loss,
    encode_memory,
    forward_with_memory,
    get_negative_sample,
    tokenize_qa,
)

OUTPUT_PATH = "llm_server/mergePRAG/critical_layers.json"
SCAN_MAX_TRAIN_SAMPLES = 96
SCAN_MAX_VAL_SAMPLES = 24
SCAN_STEPS = 60
LR = 1e-4
LR_MIN = 1e-6


def load_model():
    print(f"[Critical Layer Finder] 모델 로딩: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def evaluate_layer(model, tokenizer, hypernet, layer_idx, dataset, device):
    target_layer = model.model.layers[layer_idx]
    hypernet.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            if idx >= SCAN_MAX_VAL_SAMPLES:
                break

            tok = tokenize_qa(
                tokenizer,
                sample["question"],
                sample["answer"],
                device,
                task=sample.get("task", "final_qa"),
            )
            _, _, _, _, delta_K, delta_V = encode_memory(
                model, hypernet, tokenizer, sample["passage"], device
            )

            logits = forward_with_memory(model, target_layer, delta_K, delta_V, tok)

            loss = compute_loss(logits, tok["labels"])
            if loss is not None:
                total_loss += loss.item()
                count += 1

    hypernet.train()
    if count == 0:
        return float("inf")
    return total_loss / count


def train_layer(model, tokenizer, layer_idx, train_dataset, val_dataset, device):
    target_layer = model.model.layers[layer_idx]
    hypernet = HyperNetwork(model.config.hidden_size, k=NUM_KV).to(device).float()
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=min(len(train_dataset), SCAN_STEPS),
        eta_min=LR_MIN,
    )

    step = 0
    train_size = len(train_dataset)
    for idx, sample in enumerate(train_dataset):
        if step >= SCAN_STEPS:
            break

        _, _, _, hidden_pos, delta_K, delta_V = encode_memory(
            model, hypernet, tokenizer, sample["passage"], device
        )
        tok = tokenize_qa(
            tokenizer,
            sample["question"],
            sample["answer"],
            device,
            task=sample.get("task", "final_qa"),
        )

        logits = forward_with_memory(model, target_layer, delta_K, delta_V, tok)
        task_loss = compute_loss(logits, tok["labels"])
        if task_loss is None:
            continue

        negative_sample = get_negative_sample(train_dataset, idx)
        _, _, _, hidden_neg, neg_K, neg_V = encode_memory(
            model, hypernet, tokenizer, negative_sample["passage"], device
        )
        neg_logits = forward_with_memory(model, target_layer, neg_K, neg_V, tok)
        neg_task_loss = compute_loss(neg_logits, tok["labels"])
        if neg_task_loss is None:
            neg_task_loss = task_loss.detach()

        grounding_loss = torch.relu(NEGATIVE_MARGIN + task_loss - neg_task_loss)
        repulsion_loss = compute_repulsion_loss(
            hidden_pos,
            hidden_neg,
            delta_K,
            neg_K,
            delta_V,
            neg_V,
        )
        loss = (
            task_loss
            + NEGATIVE_LOSS_WEIGHT * grounding_loss
            + REPULSION_LOSS_WEIGHT * repulsion_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1

    val_loss = evaluate_layer(model, tokenizer, hypernet, layer_idx, val_dataset, device)
    del hypernet, optimizer, scheduler
    torch.cuda.empty_cache()
    return val_loss


def find_critical_layers(model, tokenizer, top_n=5):
    device = next(model.parameters()).device
    train_dataset = MergePRAGDataset(TRAIN_DATA_PATH, max_samples=SCAN_MAX_TRAIN_SAMPLES)
    val_dataset = MergePRAGDataset(VALID_DATA_PATH, max_samples=SCAN_MAX_VAL_SAMPLES)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Critical layer scan requires non-empty train/valid datasets.")

    num_layers = len(model.model.layers)
    layer_results = []
    print(f"[Critical Layer Finder] 총 {num_layers}개 레이어 스캔 시작")
    print(f"[Critical Layer Finder] train={len(train_dataset)}, valid={len(val_dataset)}, k={NUM_KV}, alpha={ALPHA}")

    for layer_idx in range(num_layers):
        print(f"[Layer {layer_idx}] 짧은 학습 후 val loss 측정 중...")
        val_loss = train_layer(model, tokenizer, layer_idx, train_dataset, val_dataset, device)
        layer_results.append({
            "layer": layer_idx,
            "avg_loss": round(val_loss, 4),
        })
        print(f"  -> val_loss={val_loss:.4f}")

    layer_results.sort(key=lambda x: x["avg_loss"])
    critical = [r["layer"] for r in layer_results[:top_n]]

    result = {
        "model": MODEL_NAME,
        "num_layers": num_layers,
        "d_model": model.config.hidden_size,
        "k": NUM_KV,
        "alpha": ALPHA,
        "critical_layers": critical,
        "all_layers": layer_results,
        "scan_steps": SCAN_STEPS,
        "scan_train_samples": SCAN_MAX_TRAIN_SAMPLES,
        "scan_val_samples": SCAN_MAX_VAL_SAMPLES,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[Critical Layer Finder] top-{top_n}: {critical}")
    print(f"[Critical Layer Finder] 결과 저장: {OUTPUT_PATH}")
    return critical


if __name__ == "__main__":
    model, tokenizer = load_model()
    find_critical_layers(model, tokenizer, top_n=5)
