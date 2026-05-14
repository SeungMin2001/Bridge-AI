"""
Minimal repair trainer for train2's paper-style HyperKV architecture.

The architecture is still the local paper-code HyperKV path:
passage token embeddings -> attentive pooling -> MLP -> linear K/V -> layer hook.

The repair is only in the objective: ServiceHardPair rows contain an explicit
counterfactual passage/answer, so CE-only is not enough. This trainer optimizes
both directions and adds a small pairwise ranking margin:

  main memory:     gold answer loss < negative answer loss
  negative memory: negative answer loss < gold answer loss

Run:
    python -m llm_server.mergePRAG.train2_repair
"""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_NAME, TRAIN_DATA_PATH, VALID_DATA_PATH, load_critical_layer
from .train2 import (
    HIDDEN_DIM,
    LR,
    LR_MIN,
    MAX_SAMPLES,
    MAX_VAL_SAMPLES,
    NUM_KV,
    WEIGHT_DECAY,
    MergePRAGDataset,
    PaperHyperNetwork,
    compute_loss,
    encode_memory,
    extract_first_hard_negative,
    forward_with_memory,
    tokenize_qa,
)


BASE_DIR = Path(__file__).resolve().parent
CRITICAL_LAYER = load_critical_layer()
EPOCHS = int(os.getenv("MERGEPRAG_TRAIN2_REPAIR_EPOCHS", "1"))
LOG_EVERY = int(os.getenv("MERGEPRAG_TRAIN2_REPAIR_LOG_EVERY", "25"))
EVAL_EVERY = int(os.getenv("MERGEPRAG_TRAIN2_REPAIR_EVAL_EVERY", "250"))
SAVE_EVERY = int(os.getenv("MERGEPRAG_TRAIN2_REPAIR_SAVE_EVERY", "250"))
EVAL_MAX_SAMPLES = int(os.getenv("MERGEPRAG_TRAIN2_REPAIR_EVAL_MAX_SAMPLES", "240"))
RANK_MARGIN = float(os.getenv("MERGEPRAG_TRAIN2_REPAIR_RANK_MARGIN", "0.5"))
RANK_WEIGHT = float(os.getenv("MERGEPRAG_TRAIN2_REPAIR_RANK_WEIGHT", "1.0"))
DUAL_CE_WEIGHT = float(os.getenv("MERGEPRAG_TRAIN2_REPAIR_DUAL_CE_WEIGHT", "1.0"))
GRAD_CLIP_NORM = float(os.getenv("MERGEPRAG_TRAIN2_REPAIR_GRAD_CLIP_NORM", "1.0"))

WEIGHTS_PATH = os.getenv(
    "MERGEPRAG_TRAIN2_REPAIR_WEIGHTS_PATH",
    str(BASE_DIR / "hypernet_train2_repair_weights.pt"),
)
CHECKPOINT_PATH = os.getenv(
    "MERGEPRAG_TRAIN2_REPAIR_CHECKPOINT_PATH",
    str(BASE_DIR / "hypernet_train2_repair_checkpoint.pt"),
)
LOG_PATH = os.getenv(
    "MERGEPRAG_TRAIN2_REPAIR_LOG_PATH",
    str(BASE_DIR / "train2_repair_log.json"),
)


def hard_pair_losses(model, tokenizer, hypernet, target_layer, sample, device):
    negative = extract_first_hard_negative(sample)
    if not negative or not negative.get("answer") or not negative.get("passage"):
        return None

    question = sample["question"]
    gold = sample["answer"]
    neg_answer = negative["answer"]

    _, _, _, main_K, main_V = encode_memory(model, hypernet, tokenizer, sample["passage"], device)
    _, _, _, neg_K, neg_V = encode_memory(model, hypernet, tokenizer, negative["passage"], device)
    gold_tok = tokenize_qa(tokenizer, question, gold, device)
    neg_tok = tokenize_qa(tokenizer, question, neg_answer, device)

    main_gold_logits = forward_with_memory(model, target_layer, main_K, main_V, gold_tok)
    main_neg_logits = forward_with_memory(model, target_layer, main_K, main_V, neg_tok)
    neg_gold_logits = forward_with_memory(model, target_layer, neg_K, neg_V, gold_tok)
    neg_neg_logits = forward_with_memory(model, target_layer, neg_K, neg_V, neg_tok)

    main_gold = compute_loss(main_gold_logits, gold_tok["labels"])
    main_neg = compute_loss(main_neg_logits, neg_tok["labels"])
    neg_gold = compute_loss(neg_gold_logits, gold_tok["labels"])
    neg_neg = compute_loss(neg_neg_logits, neg_tok["labels"])
    if any(loss is None for loss in (main_gold, main_neg, neg_gold, neg_neg)):
        return None

    rank_loss = F.relu(RANK_MARGIN + main_gold - main_neg) + F.relu(RANK_MARGIN + neg_neg - neg_gold)
    objective = main_gold + DUAL_CE_WEIGHT * neg_neg + RANK_WEIGHT * rank_loss

    return {
        "objective": objective,
        "main_gold": main_gold,
        "main_neg": main_neg,
        "neg_gold": neg_gold,
        "neg_neg": neg_neg,
        "rank": rank_loss,
    }


@torch.no_grad()
def evaluate(model, tokenizer, hypernet, target_layer, dataset, device):
    hypernet.eval()
    total_objective = 0.0
    total_gain = 0.0
    main_ok = 0
    neg_ok = 0
    flip_ok = 0
    count = 0

    for idx, sample in enumerate(dataset):
        if idx >= EVAL_MAX_SAMPLES:
            break
        losses = hard_pair_losses(model, tokenizer, hypernet, target_layer, sample, device)
        if losses is None:
            continue
        main_gold = losses["main_gold"].item()
        main_neg = losses["main_neg"].item()
        neg_gold = losses["neg_gold"].item()
        neg_neg = losses["neg_neg"].item()
        main_pref = main_gold < main_neg
        neg_pref = neg_neg < neg_gold
        total_objective += losses["objective"].item()
        total_gain += main_neg - main_gold
        main_ok += int(main_pref)
        neg_ok += int(neg_pref)
        flip_ok += int(main_pref and neg_pref)
        count += 1
        if torch.cuda.is_available() and idx % 50 == 0:
            torch.cuda.empty_cache()

    hypernet.train()
    denom = max(count, 1)
    return {
        "objective": total_objective / denom,
        "main_ok": main_ok / denom,
        "neg_ok": neg_ok / denom,
        "flip_ok": flip_ok / denom,
        "avg_margin": total_gain / denom,
        "count": count,
    }


def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def train():
    print(f"[train2_repair] loading model: {MODEL_NAME}")
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

    device = next(model.parameters()).device
    d_model = model.config.hidden_size
    target_layer = model.model.layers[CRITICAL_LAYER]
    hypernet = PaperHyperNetwork(d_model, num_kv=NUM_KV, hidden_dim=HIDDEN_DIM).to(device).float()
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_dataset = MergePRAGDataset(TRAIN_DATA_PATH, max_samples=MAX_SAMPLES)
    val_dataset = MergePRAGDataset(VALID_DATA_PATH, max_samples=MAX_VAL_SAMPLES)
    total_steps = max(1, len(train_dataset) * EPOCHS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LR_MIN)

    print(
        "[train2_repair] config | "
        f"layer={CRITICAL_LAYER}, num_kv={NUM_KV}, lr={LR}, "
        f"rank_weight={RANK_WEIGHT}, rank_margin={RANK_MARGIN}, dual_ce={DUAL_CE_WEIGHT}, "
        f"train={len(train_dataset)}, valid={len(val_dataset)}"
    )
    print(f"[train2_repair] outputs | weights={WEIGHTS_PATH}, checkpoint={CHECKPOINT_PATH}")

    log_data = {
        "config": {
            "model": MODEL_NAME,
            "critical_layer": CRITICAL_LAYER,
            "num_kv": NUM_KV,
            "hidden_dim": HIDDEN_DIM,
            "objective": "dual_ce_plus_bidirectional_answer_rank",
            "rank_margin": RANK_MARGIN,
            "rank_weight": RANK_WEIGHT,
            "dual_ce_weight": DUAL_CE_WEIGHT,
            "train_data_path": TRAIN_DATA_PATH,
            "valid_data_path": VALID_DATA_PATH,
        },
        "step_losses": [],
        "val_evals": [],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    best_val = float("inf")
    global_step = 0
    start_time = time.time()
    hypernet.train()

    for epoch in range(EPOCHS):
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        running = []
        for sample_idx in indices:
            sample = train_dataset[sample_idx]
            losses = hard_pair_losses(model, tokenizer, hypernet, target_layer, sample, device)
            if losses is None:
                continue

            optimizer.zero_grad()
            losses["objective"].backward()
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(hypernet.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            global_step += 1

            main_gold = losses["main_gold"].item()
            main_neg = losses["main_neg"].item()
            neg_gold = losses["neg_gold"].item()
            neg_neg = losses["neg_neg"].item()
            objective = losses["objective"].item()
            rank = losses["rank"].item()
            main_pref = main_gold < main_neg
            neg_pref = neg_neg < neg_gold
            running.append(objective)

            log_data["step_losses"].append({
                "step": global_step,
                "objective": round(objective, 4),
                "rank": round(rank, 4),
                "main_gold": round(main_gold, 4),
                "main_neg": round(main_neg, 4),
                "neg_gold": round(neg_gold, 4),
                "neg_neg": round(neg_neg, 4),
                "flip_ok": bool(main_pref and neg_pref),
            })

            if global_step % LOG_EVERY == 0:
                elapsed = (time.time() - start_time) / 60
                print(
                    f"  Step {global_step}/{total_steps} | "
                    f"obj={objective:.4f} | avg={sum(running) / len(running):.4f} | "
                    f"rank={rank:.4f} | main={main_gold:.3f}/{main_neg:.3f}:{main_pref} | "
                    f"neg={neg_gold:.3f}/{neg_neg:.3f}:{neg_pref} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}min"
                )
                running = []

            if global_step % SAVE_EVERY == 0:
                torch.save({
                    "step": global_step,
                    "hypernet": hypernet.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val": best_val,
                    "config": log_data["config"],
                }, CHECKPOINT_PATH)
                save_json(LOG_PATH, log_data)
                print(f"  [train2_repair:checkpoint] saved step {global_step}")

            if global_step % EVAL_EVERY == 0:
                metrics = evaluate(model, tokenizer, hypernet, target_layer, val_dataset, device)
                print(
                    f"  -- [train2_repair:val @ {global_step}] "
                    f"obj={metrics['objective']:.4f} | main={metrics['main_ok']:.3f} | "
                    f"neg={metrics['neg_ok']:.3f} | flip={metrics['flip_ok']:.3f} | "
                    f"margin={metrics['avg_margin']:.4f}"
                )
                log_data["val_evals"].append({"step": global_step, **metrics})
                if metrics["objective"] < best_val:
                    best_val = metrics["objective"]
                    torch.save({
                        "step": global_step,
                        "hypernet": hypernet.state_dict(),
                        "config": log_data["config"],
                        "val_metrics": metrics,
                    }, WEIGHTS_PATH)
                    print("     [train2_repair:best] saved weights")

            if torch.cuda.is_available() and global_step % 50 == 0:
                torch.cuda.empty_cache()

        metrics = evaluate(model, tokenizer, hypernet, target_layer, val_dataset, device)
        print(
            f"[train2_repair:epoch {epoch + 1}/{EPOCHS}] "
            f"obj={metrics['objective']:.4f} | main={metrics['main_ok']:.3f} | "
            f"neg={metrics['neg_ok']:.3f} | flip={metrics['flip_ok']:.3f}"
        )
        log_data["val_evals"].append({"step": global_step, **metrics})
        if metrics["objective"] < best_val:
            best_val = metrics["objective"]
            torch.save({
                "step": global_step,
                "hypernet": hypernet.state_dict(),
                "config": log_data["config"],
                "val_metrics": metrics,
            }, WEIGHTS_PATH)
            print("     [train2_repair:best] saved weights")

    log_data["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data["total_time_min"] = round((time.time() - start_time) / 60, 1)
    log_data["best_val"] = round(best_val, 4)
    log_data["total_steps"] = global_step
    save_json(LOG_PATH, log_data)
    print(f"[train2_repair] done | best_val={best_val:.4f} | weights={WEIGHTS_PATH}")


if __name__ == "__main__":
    train()
