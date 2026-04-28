"""Generate PRAG-style augmented supervision from raw professor passages."""

from __future__ import annotations

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import AUGMENT_MODEL_NAME, RAW_PASSAGES_PATH, AUGMENTED_TRAIN_PATH, AUGMENTED_VALID_PATH
from .data import extract_answer, get_passage, iter_json_records, write_jsonl
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
    templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if isinstance(templated, dict):
        input_ids = templated["input_ids"]
    else:
        input_ids = templated
    input_ids = input_ids.to(model.device)
    output = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RAW_PASSAGES_PATH))
    parser.add_argument("--train-output", default=str(AUGMENTED_TRAIN_PATH))
    parser.add_argument("--valid-output", default=str(AUGMENTED_VALID_PATH))
    parser.add_argument("--model", default=AUGMENT_MODEL_NAME)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--valid-every", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()

    model, tokenizer = load_local_model(args.model)
    rows = []
    for idx, source in enumerate(iter_json_records(args.input)):
        if args.max_samples and idx >= args.max_samples:
            break
        passage = get_passage(source)
        if not passage:
            continue
        source_id = str(source.get("source_id") or source.get("id") or f"raw_{idx}")
        generated = generate_json(model, tokenizer, source, passage, max_new_tokens=args.max_new_tokens)
        if generated is None:
            print(f"[PRAG:augment] skip {source_id}: invalid JSON")
            continue
        row = normalize_augmented(generated, source, passage, source_id)
        if row is None:
            print(f"[PRAG:augment] skip {source_id}: missing required fields")
            continue
        rows.append(row)
        print(f"[PRAG:augment] ok {source_id}: atomic={len(row['atomic_qas'])} final={len(row['final_qas'])}")

    train, valid = split_rows(rows, args.valid_every)
    write_jsonl(args.train_output, train)
    write_jsonl(args.valid_output, valid)
    print(f"[PRAG:augment] train={len(train)} -> {args.train_output}")
    print(f"[PRAG:augment] valid={len(valid)} -> {args.valid_output}")


if __name__ == "__main__":
    main()
