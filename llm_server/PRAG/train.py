"""Train HyperKV memory from PRAG-style augmented passage supervision."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    ALPHA,
    AUGMENTED_TRAIN_PATH,
    AUGMENTED_VALID_PATH,
    CHECKPOINT_PATH,
    EPOCHS,
    EVAL_EVERY,
    EVAL_MAX_SAMPLES,
    HIDDEN_DIM,
    LOG_EVERY,
    LOG_PATH,
    LR,
    LR_MIN,
    MODEL_NAME,
    NUM_KV,
    RANK_MARGIN,
    RANK_WEIGHT,
    SAVE_EVERY,
    WEIGHTS_PATH,
    load_critical_layer,
)
from .data import MemoryExample, jsonl_snapshot, load_augmented_examples
from .memory import HyperKVGenerator, compute_answer_loss, encode_memory, forward_with_memory, tokenize_qa


def load_model():
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
    return HyperKVGenerator(
        d_model=model.config.hidden_size,
        num_kv=NUM_KV,
        hidden_dim=HIDDEN_DIM,
    ).to(device).float()


def example_loss(model, tokenizer, hypernet, target_layer, example: MemoryExample, device, rank_weight: float = RANK_WEIGHT):
    main_mem = encode_memory(model, tokenizer, hypernet, example.passage, device)
    gold_tok = tokenize_qa(tokenizer, example.question, example.answer, device)
    main_gold_logits = forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], gold_tok, alpha=ALPHA)
    main_gold = compute_answer_loss(main_gold_logits, gold_tok["labels"])
    if main_gold is None:
        return None

    zero = main_gold.detach().new_tensor(0.0)
    out = {
        "objective": main_gold,
        "main_gold": main_gold,
        "neg_neg": zero,
        "rank": zero,
        "main_ok": True,
        "neg_ok": False,
    }
    if not (example.negative_passage and example.negative_answer):
        return out

    neg_mem = encode_memory(model, tokenizer, hypernet, example.negative_passage, device)
    neg_tok = tokenize_qa(tokenizer, example.question, example.negative_answer, device)
    main_neg_logits = forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], neg_tok, alpha=ALPHA)
    neg_gold_logits = forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], gold_tok, alpha=ALPHA)
    neg_neg_logits = forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], neg_tok, alpha=ALPHA)
    main_neg = compute_answer_loss(main_neg_logits, neg_tok["labels"])
    neg_gold = compute_answer_loss(neg_gold_logits, gold_tok["labels"])
    neg_neg = compute_answer_loss(neg_neg_logits, neg_tok["labels"])
    if any(loss is None for loss in (main_neg, neg_gold, neg_neg)):
        return None
    rank = F.relu(RANK_MARGIN + main_gold - main_neg) + F.relu(RANK_MARGIN + neg_neg - neg_gold)
    objective = main_gold + neg_neg + rank_weight * rank
    out.update({
        "objective": objective,
        "neg_neg": neg_neg,
        "rank": rank,
        "main_ok": main_gold.item() < main_neg.item(),
        "neg_ok": neg_neg.item() < neg_gold.item(),
    })
    return out


@torch.no_grad()
def evaluate(model, tokenizer, hypernet, target_layer, examples, device):
    hypernet.eval()
    total = 0
    obj = main_ok = neg_ok = flip_ok = 0.0
    for example in examples[:EVAL_MAX_SAMPLES]:
        out = example_loss(model, tokenizer, hypernet, target_layer, example, device)
        if out is None:
            continue
        total += 1
        obj += out["objective"].item()
        main_ok += int(out["main_ok"])
        neg_ok += int(out["neg_ok"])
        flip_ok += int(out["main_ok"] and out["neg_ok"])
    hypernet.train()
    denom = max(total, 1)
    return {
        "count": total,
        "objective": obj / denom,
        "main_ok": main_ok / denom,
        "neg_ok": neg_ok / denom,
        "flip_ok": flip_ok / denom,
    }


def save_checkpoint(path, hypernet, optimizer, scheduler, step, best_val, run_config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "hypernet": hypernet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "config": run_config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(AUGMENTED_TRAIN_PATH))
    parser.add_argument("--valid", default=str(AUGMENTED_VALID_PATH))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--rank-weight", type=float, default=RANK_WEIGHT)
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="Use only the first N expanded memory examples for a quick overfit test.",
    )
    parser.add_argument(
        "--overfit-repeat",
        type=int,
        default=1,
        help="Repeat the selected overfit examples this many times.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    layer_idx = load_critical_layer()
    target_layer = model.model.layers[layer_idx]
    train_snapshot = jsonl_snapshot(args.train)
    valid_snapshot = jsonl_snapshot(args.valid)
    print(
        "[PRAG:train:datafile] "
        f"train_rows={train_snapshot['rows']} last_train_source={train_snapshot['last_source_id']} "
        f"path={train_snapshot['path']}"
    )
    print(
        "[PRAG:train:datafile] "
        f"valid_rows={valid_snapshot['rows']} last_valid_source={valid_snapshot['last_source_id']} "
        f"path={valid_snapshot['path']}"
    )
    train_examples = load_augmented_examples(args.train, max_samples=args.max_samples or None)
    valid_examples = load_augmented_examples(args.valid, max_samples=args.max_val_samples or None)
    if args.overfit_samples > 0:
        selected = train_examples[: args.overfit_samples]
        train_examples = selected * max(args.overfit_repeat, 1)
        valid_examples = selected
        print(
            f"[PRAG:train] overfit mode: examples={len(selected)} "
            f"repeat={max(args.overfit_repeat, 1)}"
        )
    if not train_examples or not valid_examples:
        raise RuntimeError("Need non-empty augmented train and valid examples.")

    hypernet = make_hypernet(model, device)
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=args.lr, weight_decay=0.0)
    total_steps = max(1, len(train_examples) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LR_MIN)
    run_config = {
        "model": MODEL_NAME,
        "critical_layer": layer_idx,
        "num_kv": NUM_KV,
        "hidden_dim": HIDDEN_DIM,
        "alpha": ALPHA,
        "objective": "atomic_final_ce_plus_negative_flip",
        "rank_weight": args.rank_weight,
        "train_path": str(args.train),
        "valid_path": str(args.valid),
        "overfit_samples": args.overfit_samples,
        "overfit_repeat": args.overfit_repeat,
    }
    print(f"[PRAG:train] config: {run_config}")
    print(
        f"[PRAG:train] expanded_examples train={len(train_examples)} valid={len(valid_examples)} "
        f"from_train_rows={train_snapshot['rows']} from_valid_rows={valid_snapshot['rows']}"
    )

    best_val = float("inf")
    step = 0
    if args.resume and CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        if ckpt.get("config") == run_config:
            hypernet.load_state_dict(ckpt["hypernet"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            best_val = float(ckpt.get("best_val", best_val))
            step = int(ckpt.get("step", 0))
            print(f"[PRAG:train] resumed step={step} best_val={best_val:.4f}")
        else:
            print("[PRAG:train] checkpoint config mismatch; starting fresh.")

    log = {"config": run_config, "step_losses": [], "val_evals": [], "start": datetime.now().isoformat()}
    start = time.time()
    hypernet.train()
    for _epoch in range(args.epochs):
        indices = list(range(len(train_examples)))
        random.shuffle(indices)
        running = []
        for idx in indices:
            if step >= total_steps:
                break
            out = example_loss(
                model,
                tokenizer,
                hypernet,
                target_layer,
                train_examples[idx],
                device,
                rank_weight=args.rank_weight,
            )
            if out is None:
                continue
            optimizer.zero_grad()
            out["objective"].backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            obj = out["objective"].item()
            running.append(obj)
            log["step_losses"].append({"step": step, "objective": round(obj, 4)})
            if step % LOG_EVERY == 0:
                elapsed = (time.time() - start) / 60
                print(
                    f"  Step {step}/{total_steps} | obj={obj:.4f} | "
                    f"avg={sum(running)/len(running):.4f} | main={out['main_gold'].item():.3f} | "
                    f"neg={out['neg_neg'].item():.3f} | rank={out['rank'].item():.3f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}min"
                )
                running = []
            if step % SAVE_EVERY == 0:
                save_checkpoint(CHECKPOINT_PATH, hypernet, optimizer, scheduler, step, best_val, run_config)
                print(f"  [PRAG:checkpoint] saved step {step}")
            if step % EVAL_EVERY == 0:
                metrics = evaluate(model, tokenizer, hypernet, target_layer, valid_examples, device)
                print(
                    f"  -- [PRAG:val @ {step}] obj={metrics['objective']:.4f} | "
                    f"main={metrics['main_ok']:.3f} neg={metrics['neg_ok']:.3f} flip={metrics['flip_ok']:.3f}"
                )
                log["val_evals"].append({"step": step, **metrics})
                if metrics["objective"] < best_val:
                    best_val = metrics["objective"]
                    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"step": step, "hypernet": hypernet.state_dict(), "config": run_config, "val_metrics": metrics}, WEIGHTS_PATH)
                    print("     [PRAG:best] saved weights")

    metrics = evaluate(model, tokenizer, hypernet, target_layer, valid_examples, device)
    if metrics["objective"] < best_val:
        torch.save({"step": step, "hypernet": hypernet.state_dict(), "config": run_config, "val_metrics": metrics}, WEIGHTS_PATH)
    save_checkpoint(CHECKPOINT_PATH, hypernet, optimizer, scheduler, step, min(best_val, metrics["objective"]), run_config)
    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PRAG:train] done step={step} final_val={metrics}")


if __name__ == "__main__":
    main()
