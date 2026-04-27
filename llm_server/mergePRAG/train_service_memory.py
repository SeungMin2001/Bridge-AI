"""
Train the service-oriented lecture memory HyperNetwork.

Goal:
  A lecture passage is encoded once into K/V memory. Later, the user asks a
  question and the base model answers from the injected memory without seeing
  the passage text in the prompt.

This is not a paper-reproduction baseline. It keeps the paper's high-level
HyperKV idea but uses a service-suitable encoder:
  raw/contextual token features -> multi-slot pooling -> K/V -> layer hook.
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
from .service_memory import (
    SERVICE_ALPHA,
    SERVICE_HIDDEN_DIM,
    SERVICE_NUM_KV,
    SERVICE_RMS_CLAMP,
    SERVICE_SKIP_SCALE,
    SERVICE_USE_CONTEXTUAL,
    ServiceMemoryHyperNetwork,
    compute_answer_loss,
    cosine_flat,
    encode_memory,
    forward_with_memory,
    slot_diversity_loss,
    tokenize_qa,
)
from .train2 import MergePRAGDataset, extract_first_hard_negative


BASE_DIR = Path(__file__).resolve().parent
CRITICAL_LAYER = load_critical_layer()
LR = float(os.getenv("MERGEPRAG_SERVICE_LR", "8e-5"))
LR_MIN = float(os.getenv("MERGEPRAG_SERVICE_LR_MIN", "1e-6"))
WEIGHT_DECAY = float(os.getenv("MERGEPRAG_SERVICE_WEIGHT_DECAY", "0.0"))
EPOCHS = int(os.getenv("MERGEPRAG_SERVICE_EPOCHS", "1"))
MAX_SAMPLES = os.getenv("MERGEPRAG_SERVICE_MAX_SAMPLES")
MAX_VAL_SAMPLES = os.getenv("MERGEPRAG_SERVICE_MAX_VAL_SAMPLES")
MAX_SAMPLES = int(MAX_SAMPLES) if MAX_SAMPLES else None
MAX_VAL_SAMPLES = int(MAX_VAL_SAMPLES) if MAX_VAL_SAMPLES else None
LOG_EVERY = int(os.getenv("MERGEPRAG_SERVICE_LOG_EVERY", "25"))
EVAL_EVERY = int(os.getenv("MERGEPRAG_SERVICE_EVAL_EVERY", "250"))
SAVE_EVERY = int(os.getenv("MERGEPRAG_SERVICE_SAVE_EVERY", "250"))
EVAL_MAX_SAMPLES = int(os.getenv("MERGEPRAG_SERVICE_EVAL_MAX_SAMPLES", "240"))
RANK_MARGIN = float(os.getenv("MERGEPRAG_SERVICE_RANK_MARGIN", "0.5"))
RANK_WEIGHT = float(os.getenv("MERGEPRAG_SERVICE_RANK_WEIGHT", "1.0"))
DUAL_CE_WEIGHT = float(os.getenv("MERGEPRAG_SERVICE_DUAL_CE_WEIGHT", "1.0"))
SEPARATION_WEIGHT = float(os.getenv("MERGEPRAG_SERVICE_SEPARATION_WEIGHT", "0.25"))
SEPARATION_TARGET = float(os.getenv("MERGEPRAG_SERVICE_SEPARATION_TARGET", "0.90"))
DIVERSITY_WEIGHT = float(os.getenv("MERGEPRAG_SERVICE_DIVERSITY_WEIGHT", "0.05"))
GRAD_CLIP_NORM = float(os.getenv("MERGEPRAG_SERVICE_GRAD_CLIP_NORM", "1.0"))
RESUME_FROM_CHECKPOINT = os.getenv("MERGEPRAG_SERVICE_RESUME", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

WEIGHTS_PATH = os.getenv(
    "MERGEPRAG_SERVICE_WEIGHTS_PATH",
    str(BASE_DIR / "service_memory_weights.pt"),
)
CHECKPOINT_PATH = os.getenv(
    "MERGEPRAG_SERVICE_CHECKPOINT_PATH",
    str(BASE_DIR / "service_memory_checkpoint.pt"),
)
LOG_PATH = os.getenv(
    "MERGEPRAG_SERVICE_LOG_PATH",
    str(BASE_DIR / "service_memory_log.json"),
)


def hard_pair_objective(model, tokenizer, hypernet, target_layer, sample, device):
    negative = extract_first_hard_negative(sample)
    if not negative or not negative.get("answer") or not negative.get("passage"):
        return None

    question = sample["question"]
    gold = sample["answer"]
    neg_answer = negative["answer"]

    main_mem = encode_memory(model, hypernet, tokenizer, sample["passage"], device)
    neg_mem = encode_memory(model, hypernet, tokenizer, negative["passage"], device)
    gold_tok = tokenize_qa(tokenizer, question, gold, device)
    neg_tok = tokenize_qa(tokenizer, question, neg_answer, device)

    main_gold_logits = forward_with_memory(
        model,
        target_layer,
        main_mem["K"],
        main_mem["V"],
        gold_tok,
    )
    main_neg_logits = forward_with_memory(
        model,
        target_layer,
        main_mem["K"],
        main_mem["V"],
        neg_tok,
    )
    neg_gold_logits = forward_with_memory(
        model,
        target_layer,
        neg_mem["K"],
        neg_mem["V"],
        gold_tok,
    )
    neg_neg_logits = forward_with_memory(
        model,
        target_layer,
        neg_mem["K"],
        neg_mem["V"],
        neg_tok,
    )

    main_gold = compute_answer_loss(main_gold_logits, gold_tok["labels"])
    main_neg = compute_answer_loss(main_neg_logits, neg_tok["labels"])
    neg_gold = compute_answer_loss(neg_gold_logits, gold_tok["labels"])
    neg_neg = compute_answer_loss(neg_neg_logits, neg_tok["labels"])
    if any(loss is None for loss in (main_gold, main_neg, neg_gold, neg_neg)):
        return None

    rank = F.relu(RANK_MARGIN + main_gold - main_neg) + F.relu(RANK_MARGIN + neg_neg - neg_gold)
    k_cos = cosine_flat(main_mem["K"], neg_mem["K"])
    v_cos = cosine_flat(main_mem["V"], neg_mem["V"])
    sep = F.relu(k_cos - SEPARATION_TARGET) + F.relu(v_cos - SEPARATION_TARGET)
    div = slot_diversity_loss(main_mem["K"], main_mem["V"]) + slot_diversity_loss(neg_mem["K"], neg_mem["V"])
    objective = (
        main_gold
        + DUAL_CE_WEIGHT * neg_neg
        + RANK_WEIGHT * rank
        + SEPARATION_WEIGHT * sep
        + DIVERSITY_WEIGHT * div
    )
    return {
        "objective": objective,
        "main_gold": main_gold,
        "main_neg": main_neg,
        "neg_gold": neg_gold,
        "neg_neg": neg_neg,
        "rank": rank,
        "sep": sep,
        "div": div,
        "k_cos": k_cos,
        "v_cos": v_cos,
    }


@torch.no_grad()
def evaluate(model, tokenizer, hypernet, target_layer, dataset, device):
    hypernet.eval()
    total_obj = 0.0
    main_ok = 0
    neg_ok = 0
    flip_ok = 0
    k_cos_sum = 0.0
    v_cos_sum = 0.0
    count = 0
    for idx, sample in enumerate(dataset):
        if idx >= EVAL_MAX_SAMPLES:
            break
        out = hard_pair_objective(model, tokenizer, hypernet, target_layer, sample, device)
        if out is None:
            continue
        main_pref = out["main_gold"].item() < out["main_neg"].item()
        neg_pref = out["neg_neg"].item() < out["neg_gold"].item()
        total_obj += out["objective"].item()
        main_ok += int(main_pref)
        neg_ok += int(neg_pref)
        flip_ok += int(main_pref and neg_pref)
        k_cos_sum += out["k_cos"].item()
        v_cos_sum += out["v_cos"].item()
        count += 1
        if torch.cuda.is_available() and idx % 50 == 0:
            torch.cuda.empty_cache()
    hypernet.train()
    denom = max(count, 1)
    return {
        "objective": total_obj / denom,
        "main_ok": main_ok / denom,
        "neg_ok": neg_ok / denom,
        "flip_ok": flip_ok / denom,
        "k_cos": k_cos_sum / denom,
        "v_cos": v_cos_sum / denom,
        "count": count,
    }


def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_existing_log(path: str, fallback: dict) -> dict:
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("step_losses", [])
        data.setdefault("val_evals", [])
        data["resumed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return data
    except Exception as exc:
        print(f"[service_memory:resume] could not load log {path}: {exc}")
        return fallback


def config_matches(saved: dict, current: dict) -> bool:
    keys = (
        "model",
        "critical_layer",
        "num_kv",
        "hidden_dim",
        "feature_dim",
        "use_contextual",
        "rms_clamp",
        "skip_scale",
        "objective",
    )
    return all(saved.get(key) == current.get(key) for key in keys)


def train():
    print(f"[service_memory] loading model: {MODEL_NAME}")
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
    feature_dim = d_model * 2 if SERVICE_USE_CONTEXTUAL else d_model
    target_layer = model.model.layers[CRITICAL_LAYER]
    hypernet = ServiceMemoryHyperNetwork(
        d_model=d_model,
        feature_dim=feature_dim,
        num_kv=SERVICE_NUM_KV,
        hidden_dim=SERVICE_HIDDEN_DIM,
        skip_scale=SERVICE_SKIP_SCALE,
        rms_clamp=SERVICE_RMS_CLAMP,
    ).to(device).float()
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_dataset = MergePRAGDataset(TRAIN_DATA_PATH, max_samples=MAX_SAMPLES)
    val_dataset = MergePRAGDataset(VALID_DATA_PATH, max_samples=MAX_VAL_SAMPLES)
    total_steps = max(1, len(train_dataset) * EPOCHS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LR_MIN)

    run_config = {
        "model": MODEL_NAME,
        "critical_layer": CRITICAL_LAYER,
        "num_kv": SERVICE_NUM_KV,
        "hidden_dim": SERVICE_HIDDEN_DIM,
        "feature_dim": feature_dim,
        "use_contextual": SERVICE_USE_CONTEXTUAL,
        "alpha": SERVICE_ALPHA,
        "rms_clamp": SERVICE_RMS_CLAMP,
        "skip_scale": SERVICE_SKIP_SCALE,
        "objective": "dual_ce_rank_memory_separation_slot_diversity",
        "rank_margin": RANK_MARGIN,
        "rank_weight": RANK_WEIGHT,
        "dual_ce_weight": DUAL_CE_WEIGHT,
        "separation_weight": SEPARATION_WEIGHT,
        "separation_target": SEPARATION_TARGET,
        "diversity_weight": DIVERSITY_WEIGHT,
        "train_data_path": TRAIN_DATA_PATH,
        "valid_data_path": VALID_DATA_PATH,
    }
    print(f"[service_memory] config: {run_config}")
    print(f"[service_memory] outputs | weights={WEIGHTS_PATH}, checkpoint={CHECKPOINT_PATH}")

    log_data = {
        "config": run_config,
        "step_losses": [],
        "val_evals": [],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    best_val = float("inf")
    global_step = 0
    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        saved_config = checkpoint.get("config") or {}
        if config_matches(saved_config, run_config):
            hypernet.load_state_dict(checkpoint["hypernet"])
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            best_val = float(checkpoint.get("best_val", best_val))
            global_step = int(checkpoint.get("step", 0))
            log_data = load_existing_log(LOG_PATH, log_data)
            log_data["config"] = run_config
            print(
                f"[service_memory:resume] loaded checkpoint step={global_step} "
                f"best_val={best_val:.4f} from {CHECKPOINT_PATH}"
            )
        else:
            print("[service_memory:resume] checkpoint config mismatch; starting fresh.")
            print(f"  saved={saved_config}")
            print(f"  current={run_config}")
    elif RESUME_FROM_CHECKPOINT:
        print(f"[service_memory:resume] no checkpoint found at {CHECKPOINT_PATH}; starting fresh.")
    else:
        print("[service_memory:resume] disabled by MERGEPRAG_SERVICE_RESUME=0; starting fresh.")

    start_time = time.time()
    hypernet.train()

    for epoch in range(EPOCHS):
        if global_step >= total_steps:
            break
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        running = []
        for sample_idx in indices:
            if global_step >= total_steps:
                break
            out = hard_pair_objective(model, tokenizer, hypernet, target_layer, train_dataset[sample_idx], device)
            if out is None:
                continue

            optimizer.zero_grad()
            out["objective"].backward()
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(hypernet.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            global_step += 1

            obj = out["objective"].item()
            rank = out["rank"].item()
            sep = out["sep"].item()
            div = out["div"].item()
            main_gold = out["main_gold"].item()
            main_neg = out["main_neg"].item()
            neg_gold = out["neg_gold"].item()
            neg_neg = out["neg_neg"].item()
            main_pref = main_gold < main_neg
            neg_pref = neg_neg < neg_gold
            running.append(obj)
            log_data["step_losses"].append({
                "step": global_step,
                "objective": round(obj, 4),
                "rank": round(rank, 4),
                "sep": round(sep, 4),
                "div": round(div, 4),
                "k_cos": round(out["k_cos"].item(), 4),
                "v_cos": round(out["v_cos"].item(), 4),
                "main_pref": bool(main_pref),
                "neg_pref": bool(neg_pref),
            })

            if global_step % LOG_EVERY == 0:
                elapsed = (time.time() - start_time) / 60
                print(
                    f"  Step {global_step}/{total_steps} | obj={obj:.4f} | "
                    f"avg={sum(running) / len(running):.4f} | rank={rank:.4f} | "
                    f"sep={sep:.4f} | div={div:.4f} | "
                    f"main={main_gold:.3f}/{main_neg:.3f}:{main_pref} | "
                    f"neg={neg_gold:.3f}/{neg_neg:.3f}:{neg_pref} | "
                    f"Kcos={out['k_cos'].item():.4f} | Vcos={out['v_cos'].item():.4f} | "
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
                    "config": run_config,
                }, CHECKPOINT_PATH)
                save_json(LOG_PATH, log_data)
                print(f"  [service_memory:checkpoint] saved step {global_step}")

            if global_step % EVAL_EVERY == 0:
                metrics = evaluate(model, tokenizer, hypernet, target_layer, val_dataset, device)
                print(
                    f"  -- [service_memory:val @ {global_step}] obj={metrics['objective']:.4f} | "
                    f"main={metrics['main_ok']:.3f} | neg={metrics['neg_ok']:.3f} | "
                    f"flip={metrics['flip_ok']:.3f} | Kcos={metrics['k_cos']:.4f} | "
                    f"Vcos={metrics['v_cos']:.4f}"
                )
                log_data["val_evals"].append({"step": global_step, **metrics})
                if metrics["objective"] < best_val:
                    best_val = metrics["objective"]
                    torch.save({
                        "step": global_step,
                        "hypernet": hypernet.state_dict(),
                        "config": run_config,
                        "val_metrics": metrics,
                    }, WEIGHTS_PATH)
                    print("     [service_memory:best] saved weights")

            if torch.cuda.is_available() and global_step % 50 == 0:
                torch.cuda.empty_cache()

        metrics = evaluate(model, tokenizer, hypernet, target_layer, val_dataset, device)
        print(
            f"[service_memory:epoch {epoch + 1}/{EPOCHS}] obj={metrics['objective']:.4f} | "
            f"main={metrics['main_ok']:.3f} | neg={metrics['neg_ok']:.3f} | "
            f"flip={metrics['flip_ok']:.3f} | Kcos={metrics['k_cos']:.4f} | Vcos={metrics['v_cos']:.4f}"
        )
        log_data["val_evals"].append({"step": global_step, **metrics})
        if metrics["objective"] < best_val:
            best_val = metrics["objective"]
            torch.save({
                "step": global_step,
                "hypernet": hypernet.state_dict(),
                "config": run_config,
                "val_metrics": metrics,
            }, WEIGHTS_PATH)
            print("     [service_memory:best] saved weights")

    log_data["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data["total_time_min"] = round((time.time() - start_time) / 60, 1)
    log_data["best_val"] = round(best_val, 4)
    log_data["total_steps"] = global_step
    save_json(LOG_PATH, log_data)
    print(f"[service_memory] done | best_val={best_val:.4f} | weights={WEIGHTS_PATH}")


if __name__ == "__main__":
    train()
