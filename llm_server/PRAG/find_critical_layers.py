"""Paper-style critical layer scanner for the PRAG HyperKV path.

This follows the MergePRAG author-code idea:

1. For each decoder layer, initialize a fresh HyperKV generator.
2. Briefly train that HyperKV on passage -> K/V -> target-layer injection.
3. Score validation answer CE/PPL for that layer.
4. Save the lowest-loss layers to ``critical_layers.json``.

Unlike the full PRAG training script, this scanner intentionally uses a
paper-like positive CE objective only. It does not use hard negatives, ranking
loss, prefix loss, or phrase loss. The goal is to find where K/V injection is
most useful before running the real service-oriented training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    ALPHA,
    CRITICAL_LAYERS_PATH,
    HIDDEN_DIM,
    LR_MIN,
    MODEL_NAME,
    MULTIFACT_AUGMENTED_TRAIN_PATH,
    MULTIFACT_AUGMENTED_VALID_PATH,
    NUM_KV,
    QUESTION_CONDITIONED_MEMORY,
    USE_CONTEXTUAL_MEMORY,
)
from .data import MemoryExample, load_augmented_examples
from .memory import HyperKVGenerator, encode_memory, forward_with_memory, tokenize_qa
from .train import example_has_hangul, example_is_clean_korean


DEFAULT_SCAN_STEPS = 60
DEFAULT_TRAIN_SAMPLES = 96
DEFAULT_VALID_SAMPLES = 48


def parse_layer_spec(spec: str, num_layers: int) -> list[int]:
    spec = str(spec or "all").strip().lower()
    if spec in {"all", "*"}:
        return list(range(num_layers))
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending layer range: {part}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    deduped = sorted(dict.fromkeys(layers))
    invalid = [layer for layer in deduped if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(f"Layer out of range 0..{num_layers - 1}: {invalid}")
    if not deduped:
        raise ValueError("No layers selected.")
    return deduped


def safe_ppl(loss: float) -> float:
    if not math.isfinite(loss):
        return float("inf")
    return math.exp(min(loss, 20.0))


def load_model():
    print(f"[PRAG:layer-scan] loading model={MODEL_NAME}")
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


def make_scan_hypernet(model, device, *, legacy_hypernet: bool = False) -> HyperKVGenerator:
    d_model = int(model.config.hidden_size)
    feature_dim = d_model * (2 if USE_CONTEXTUAL_MEMORY else 1)
    if legacy_hypernet and feature_dim != d_model:
        raise ValueError("Author-style legacy hypernet requires PRAG_USE_CONTEXTUAL_MEMORY=0.")
    return HyperKVGenerator(
        d_model=d_model,
        num_kv=NUM_KV,
        hidden_dim=HIDDEN_DIM,
        feature_dim=feature_dim,
        legacy=legacy_hypernet,
    ).to(device).float()


def sample_objective(
    model,
    tokenizer,
    hypernet,
    target_layer,
    example: MemoryExample,
    device,
    *,
    answer_target: str,
    final_weight: float,
    injection_mode: str,
):
    memory = encode_memory(model, tokenizer, hypernet, example.passage, device, question=example.question)
    answer = example.target_answer(answer_target)
    tok = tokenize_qa(tokenizer, example.question, answer, device)
    logits = forward_with_memory(
        model,
        target_layer,
        memory["K"],
        memory["V"],
        tok,
        alpha=ALPHA,
        injection_mode=injection_mode,
    )
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tok["labels"][:, 1:].contiguous()
    valid = shift_labels != -100
    if valid.sum() == 0:
        return None
    loss = torch.nn.functional.cross_entropy(shift_logits[valid], shift_labels[valid])
    if example.qa_type == "final":
        loss = loss * final_weight
    return loss


@torch.no_grad()
def evaluate_layer(
    model,
    tokenizer,
    hypernet,
    layer_idx: int,
    valid_examples: list[MemoryExample],
    device,
    *,
    answer_target: str,
    final_weight: float,
    injection_mode: str,
):
    target_layer = model.model.layers[layer_idx]
    hypernet.eval()
    total = 0.0
    count = 0
    for example in valid_examples:
        loss = sample_objective(
            model,
            tokenizer,
            hypernet,
            target_layer,
            example,
            device,
            answer_target=answer_target,
            final_weight=final_weight,
            injection_mode=injection_mode,
        )
        if loss is None:
            continue
        total += float(loss.item())
        count += 1
    hypernet.train()
    avg = total / max(count, 1)
    return {"count": count, "val_loss": avg, "val_ppl": safe_ppl(avg)}


def train_and_score_layer(
    model,
    tokenizer,
    layer_idx: int,
    train_examples: list[MemoryExample],
    valid_examples: list[MemoryExample],
    device,
    *,
    steps: int,
    lr: float,
    answer_target: str,
    final_weight: float,
    injection_mode: str,
    legacy_hypernet: bool,
):
    target_layer = model.model.layers[layer_idx]
    hypernet = make_scan_hypernet(model, device, legacy_hypernet=legacy_hypernet)
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, steps),
        eta_min=LR_MIN,
    )

    hypernet.train()
    train_total = 0.0
    train_count = 0
    max_attempts = max(steps * 3, steps + 10)
    for attempt in range(max_attempts):
        if train_count >= steps:
            break
        example = train_examples[attempt % len(train_examples)]
        loss = sample_objective(
            model,
            tokenizer,
            hypernet,
            target_layer,
            example,
            device,
            answer_target=answer_target,
            final_weight=final_weight,
            injection_mode=injection_mode,
        )
        if loss is None:
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_total += float(loss.item())
        train_count += 1

    eval_metrics = evaluate_layer(
        model,
        tokenizer,
        hypernet,
        layer_idx,
        valid_examples,
        device,
        answer_target=answer_target,
        final_weight=final_weight,
        injection_mode=injection_mode,
    )
    train_loss = train_total / max(train_count, 1)
    result = {
        "layer": layer_idx,
        "train_steps": train_count,
        "train_loss": train_loss,
        "train_ppl": safe_ppl(train_loss),
        **eval_metrics,
    }
    del hypernet, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def filter_examples(examples: list[MemoryExample], *, ko_only: bool, clean_ko_only: bool) -> list[MemoryExample]:
    if clean_ko_only:
        return [example for example in examples if example_is_clean_korean(example)]
    if ko_only:
        return [example for example in examples if example_has_hangul(example)]
    return examples


def load_existing_results(output_path: Path, scan_config: dict, resume: bool) -> dict[int, dict]:
    if not resume or not output_path.exists():
        return {}
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if data.get("model") != MODEL_NAME:
        return {}
    previous_config = data.get("scan_config") or {}
    comparable_keys = [
        "answer_target",
        "final_weight",
        "injection_mode",
        "legacy_hypernet",
        "ko_only",
        "clean_ko_only",
        "train_path",
        "valid_path",
    ]
    if any(previous_config.get(key) != scan_config.get(key) for key in comparable_keys):
        return {}
    return {
        int(item["layer"]): item
        for item in data.get("all_layers", [])
        if isinstance(item, dict) and "layer" in item and "val_loss" in item
    }


def write_scan_output(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Find PRAG critical layers with a MergePRAG paper-style CE scan.")
    parser.add_argument("--train", default=str(MULTIFACT_AUGMENTED_TRAIN_PATH))
    parser.add_argument("--valid", default=str(MULTIFACT_AUGMENTED_VALID_PATH))
    parser.add_argument("--multifact", action="store_true", help="Use the default multifact augmented files.")
    parser.add_argument("--ko-only", action="store_true", help="Scan only Korean examples.")
    parser.add_argument("--clean-ko-only", action="store_true", help="Scan only clean Korean examples.")
    parser.add_argument("--max-train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--max-valid-samples", type=int, default=DEFAULT_VALID_SAMPLES)
    parser.add_argument("--steps", type=int, default=DEFAULT_SCAN_STEPS)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", default="all", help="Layer list/range, e.g. all, 0-27, or 7,8,9,10.")
    parser.add_argument("--output", default=str(CRITICAL_LAYERS_PATH))
    parser.add_argument("--answer-target", choices=("answer", "full_answer"), default="full_answer")
    parser.add_argument("--final-weight", type=float, default=0.25)
    parser.add_argument("--injection-mode", choices=("attention", "add_all", "add_last", "hybrid"), default="attention")
    parser.add_argument(
        "--legacy-hypernet",
        action="store_true",
        help="Use the author-code style single-pool HyperKV head for the scan.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.multifact:
        args.train = str(MULTIFACT_AUGMENTED_TRAIN_PATH)
        args.valid = str(MULTIFACT_AUGMENTED_VALID_PATH)
    if args.clean_ko_only:
        args.ko_only = True

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    num_layers = len(model.model.layers)
    selected_layers = parse_layer_spec(args.layers, num_layers)

    train_examples = load_augmented_examples(args.train)
    valid_examples = load_augmented_examples(args.valid)
    before = (len(train_examples), len(valid_examples))
    train_examples = filter_examples(train_examples, ko_only=args.ko_only, clean_ko_only=args.clean_ko_only)
    valid_examples = filter_examples(valid_examples, ko_only=args.ko_only, clean_ko_only=args.clean_ko_only)
    random.shuffle(train_examples)
    random.shuffle(valid_examples)
    train_examples = train_examples[: args.max_train_samples]
    valid_examples = valid_examples[: args.max_valid_samples]
    if not train_examples or not valid_examples:
        raise RuntimeError("Layer scan requires non-empty train and valid examples after filtering.")

    output_path = Path(args.output)
    scan_config = {
        "train_path": str(args.train),
        "valid_path": str(args.valid),
        "ko_only": bool(args.ko_only),
        "clean_ko_only": bool(args.clean_ko_only),
        "max_train_samples": args.max_train_samples,
        "max_valid_samples": args.max_valid_samples,
        "steps": args.steps,
        "lr": args.lr,
        "answer_target": args.answer_target,
        "final_weight": args.final_weight,
        "injection_mode": args.injection_mode,
        "legacy_hypernet": bool(args.legacy_hypernet),
        "seed": args.seed,
        "num_kv": NUM_KV,
        "hidden_dim": HIDDEN_DIM,
        "alpha": ALPHA,
        "use_contextual_memory": USE_CONTEXTUAL_MEMORY,
        "question_conditioned_memory": QUESTION_CONDITIONED_MEMORY,
    }
    existing = load_existing_results(output_path, scan_config, args.resume)
    results: dict[int, dict] = dict(existing)

    print(
        "[PRAG:layer-scan] "
        f"layers={selected_layers} model={MODEL_NAME} d_model={model.config.hidden_size} "
        f"train={before[0]}->{len(train_examples)} valid={before[1]}->{len(valid_examples)} "
        f"steps={args.steps} lr={args.lr} answer_target={args.answer_target}"
    )
    print(
        "[PRAG:layer-scan] "
        f"num_kv={NUM_KV} alpha={ALPHA} contextual={USE_CONTEXTUAL_MEMORY} "
        f"question_conditioned={QUESTION_CONDITIONED_MEMORY} legacy_hypernet={args.legacy_hypernet}"
    )
    if existing:
        print(f"[PRAG:layer-scan] resume: found {len(existing)} existing layer results in {output_path}")

    start_time = time.time()
    for layer_idx in selected_layers:
        if layer_idx in results:
            print(f"[PRAG:layer-scan] layer {layer_idx}: skip existing val_loss={results[layer_idx]['val_loss']:.4f}")
            continue
        layer_start = time.time()
        print(f"[PRAG:layer-scan] layer {layer_idx}: short train/eval...")
        metrics = train_and_score_layer(
            model,
            tokenizer,
            layer_idx,
            train_examples,
            valid_examples,
            device,
            steps=args.steps,
            lr=args.lr,
            answer_target=args.answer_target,
            final_weight=args.final_weight,
            injection_mode=args.injection_mode,
            legacy_hypernet=args.legacy_hypernet,
        )
        metrics = {
            "layer": metrics["layer"],
            "train_steps": metrics["train_steps"],
            "train_loss": round(metrics["train_loss"], 6),
            "train_ppl": round(metrics["train_ppl"], 4),
            "val_count": metrics["count"],
            "val_loss": round(metrics["val_loss"], 6),
            "val_ppl": round(metrics["val_ppl"], 4),
            "elapsed_sec": round(time.time() - layer_start, 3),
        }
        results[layer_idx] = metrics
        print(
            f"  -> val_loss={metrics['val_loss']:.4f} val_ppl={metrics['val_ppl']:.2f} "
            f"train_loss={metrics['train_loss']:.4f} elapsed={metrics['elapsed_sec']:.1f}s"
        )

        sorted_layers = sorted(results.values(), key=lambda item: (item["val_loss"], item["layer"]))
        payload = {
            "scanner": "prag_paper_ce_layer_scan",
            "model": MODEL_NAME,
            "num_layers": num_layers,
            "d_model": int(model.config.hidden_size),
            "critical_layers": [item["layer"] for item in sorted_layers[: args.top_n]],
            "all_layers": sorted_layers,
            "scan_config": scan_config,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(time.time() - start_time, 3),
        }
        write_scan_output(output_path, payload)
        print(f"[PRAG:layer-scan] partial saved: {output_path}")

    sorted_layers = sorted(results.values(), key=lambda item: (item["val_loss"], item["layer"]))
    critical = [item["layer"] for item in sorted_layers[: args.top_n]]
    payload = {
        "scanner": "prag_paper_ce_layer_scan",
        "model": MODEL_NAME,
        "num_layers": num_layers,
        "d_model": int(model.config.hidden_size),
        "critical_layers": critical,
        "all_layers": sorted_layers,
        "scan_config": scan_config,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - start_time, 3),
    }
    write_scan_output(output_path, payload)
    print(f"[PRAG:layer-scan] top-{args.top_n}: {critical}")
    print(f"[PRAG:layer-scan] saved: {output_path}")


if __name__ == "__main__":
    main()
