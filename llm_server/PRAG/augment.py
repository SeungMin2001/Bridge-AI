"""Generate PRAG-style augmented supervision from raw professor passages."""

from __future__ import annotations

import argparse
import json
import random
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import AUGMENT_MODEL_NAME, SOURCE_DATA_PATH, AUGMENTED_TRAIN_PATH, AUGMENTED_VALID_PATH
from .data import extract_answer, get_passage, iter_json_records, jsonl_snapshot, write_jsonl
from .prompts import augmentation_prompt


def extract_json_object(text: str) -> dict | None:
    text = str(text or "").strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def load_local_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def first_hard_negative(source: dict) -> dict:
    values = source.get("hard_negatives")
    if isinstance(values, list) and values:
        item = values[0]
        if isinstance(item, dict):
            return {
                "passage": get_passage(item),
                "answer": extract_answer(item),
            }
    return {"passage": "", "answer": ""}


@torch.no_grad()
def generate_json(model, tokenizer, source: dict, passage: str, max_new_tokens: int = 768) -> dict | None:
    negative = first_hard_negative(source)
    messages = [{
        "role": "user",
        "content": augmentation_prompt(
            passage,
            question=str(source.get("question") or ""),
            answer=extract_answer(source),
            negative_passage=negative["passage"],
            negative_answer=negative["answer"],
        ),
    }]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    return extract_json_object(text)


def normalize_augmented(raw: dict, source: dict, passage: str, source_id: str) -> dict | None:
    atomic = raw.get("atomic_qas")
    final = raw.get("final_qas")
    negatives = raw.get("hard_negatives")
    if not isinstance(atomic, list) or not isinstance(final, list) or not isinstance(negatives, list):
        return None
    if not atomic or not final or not negatives:
        return None
    source_negative = first_hard_negative(source)
    if source_negative["passage"] and negatives and isinstance(negatives[0], dict):
        negatives[0]["passage"] = negatives[0].get("passage") or source_negative["passage"]
        if source_negative["answer"]:
            negatives[0].setdefault("answer", source_negative["answer"])
    return {
        "source_id": source_id,
        "speaker": source.get("speaker", ""),
        "passage": passage,
        "rewrite": str(raw.get("rewrite") or "").strip(),
        "atomic_qas": atomic,
        "final_qas": final,
        "hard_negatives": negatives,
    }


def split_rows(rows: list[dict], valid_every: int) -> tuple[list[dict], list[dict]]:
    train, valid = [], []
    for idx, row in enumerate(rows):
        (valid if valid_every > 0 and idx % valid_every == valid_every - 1 else train).append(row)
    return train, valid


def load_existing_outputs(train_path: str, valid_path: str) -> tuple[set[str], int, int]:
    seen: set[str] = set()
    train_count = valid_count = 0
    for path, is_train in ((train_path, True), (valid_path, False)):
        try:
            for row in iter_json_records(path):
                source_id = str(row.get("source_id") or "").strip()
                if source_id:
                    seen.add(source_id)
                if is_train:
                    train_count += 1
                else:
                    valid_count += 1
        except FileNotFoundError:
            continue
    return seen, train_count, valid_count


def append_jsonl(path: str, row: dict) -> None:
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def count_input_records(path: str) -> int:
    return sum(1 for _ in iter_json_records(path))


def format_progress(done: int, total: int) -> str:
    progress = f"{done}/{total}"
    if total:
        progress += f" ({done / total * 100:.1f}%)"
    return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(SOURCE_DATA_PATH))
    parser.add_argument("--train-output", default=str(AUGMENTED_TRAIN_PATH))
    parser.add_argument("--valid-output", default=str(AUGMENTED_VALID_PATH))
    parser.add_argument("--model", default=AUGMENT_MODEL_NAME)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--valid-every", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle input rows before augmentation so topics are mixed instead of processed sequentially.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing output JSONL files by skipping already written source_id rows.",
    )
    args = parser.parse_args()

    if args.resume:
        seen_ids, train_count, valid_count = load_existing_outputs(args.train_output, args.valid_output)
        existing_count = len(seen_ids)
        if seen_ids:
            print(
                f"[PRAG:augment] resume enabled: existing train={train_count} "
                f"valid={valid_count} seen={len(seen_ids)}"
            )
    else:
        seen_ids, train_count, valid_count = set(), 0, 0
        existing_count = 0
        write_jsonl(args.train_output, [])
        write_jsonl(args.valid_output, [])
        print("[PRAG:augment] resume disabled: output files were reset.")

    input_rows = list(iter_json_records(args.input))
    if args.shuffle:
        random.Random(args.seed).shuffle(input_rows)
    if args.max_samples:
        input_rows = input_rows[: args.max_samples]
    input_total = len(input_rows)
    print(f"[PRAG:augment] input={args.input} total_target={input_total}")
    print(f"[PRAG:augment] shuffle={args.shuffle} seed={args.seed}")
    print(
        f"[PRAG:augment] cumulative_start existing={existing_count}/{input_total} "
        f"({existing_count / input_total * 100:.1f}%) train={train_count} valid={valid_count}"
        if input_total
        else f"[PRAG:augment] cumulative_start existing={existing_count}/0 train={train_count} valid={valid_count}"
    )

    model, tokenizer = load_local_model(args.model)
    made = skipped_seen = skipped_invalid = 0
    started_at = time.time()
    for idx, source in enumerate(input_rows):
        passage = get_passage(source)
        if not passage:
            continue
        source_id = str(source.get("source_id") or source.get("id") or f"raw_{idx}")
        if source_id in seen_ids:
            skipped_seen += 1
            if skipped_seen % 50 == 0:
                processed = idx + 1
                done_total = min(existing_count + made, input_total)
                print(
                    f"[PRAG:augment] resume skip scan={format_progress(processed, input_total)} "
                    f"done_total={format_progress(done_total, input_total)} "
                    f"existing_start={existing_count}, new={made}, skipped_seen={skipped_seen}, "
                    f"train={train_count}, valid={valid_count}"
                )
            continue
        generated = generate_json(model, tokenizer, source, passage, max_new_tokens=args.max_new_tokens)
        if generated is None:
            skipped_invalid += 1
            print(f"[PRAG:augment] skip {source_id}: invalid JSON")
            continue
        row = normalize_augmented(generated, source, passage, source_id)
        if row is None:
            skipped_invalid += 1
            print(f"[PRAG:augment] skip {source_id}: missing required fields")
            continue
        is_valid = args.valid_every > 0 and idx % args.valid_every == args.valid_every - 1
        saved_to = args.valid_output if is_valid else args.train_output
        append_jsonl(saved_to, row)
        seen_ids.add(source_id)
        made += 1
        if is_valid:
            valid_count += 1
        else:
            train_count += 1
        processed = idx + 1
        done_total = min(existing_count + made, input_total)
        elapsed_min = max((time.time() - started_at) / 60, 1e-6)
        rate = processed / elapsed_min
        eta = ""
        if input_total and rate > 0:
            remaining = max(input_total - processed, 0)
            eta = f", eta={remaining / rate:.1f}min"
        print(
            f"[PRAG:augment] ok {source_id}: atomic={len(row['atomic_qas'])} "
            f"final={len(row['final_qas'])} -> {'valid' if is_valid else 'train'} "
            f"saved_to={saved_to} "
            f"(scan={format_progress(processed, input_total)}, "
            f"done_total={format_progress(done_total, input_total)}, "
            f"existing_start={existing_count}, new={made}, skipped_seen={skipped_seen}, "
            f"invalid={skipped_invalid}, train={train_count}, valid={valid_count}, "
            f"rate={rate:.2f}/min{eta})"
        )

    print(
        f"[PRAG:augment] done new={made} skipped_seen={skipped_seen} "
        f"skipped_invalid={skipped_invalid}"
    )
    print(f"[PRAG:augment] train={train_count} -> {args.train_output}")
    print(f"[PRAG:augment] valid={valid_count} -> {args.valid_output}")
    print(f"[PRAG:augment] train_snapshot={jsonl_snapshot(args.train_output)}")
    print(f"[PRAG:augment] valid_snapshot={jsonl_snapshot(args.valid_output)}")


if __name__ == "__main__":
    main()
