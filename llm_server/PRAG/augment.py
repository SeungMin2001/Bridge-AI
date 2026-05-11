"""Generate PRAG-style augmented supervision from raw professor passages."""

from __future__ import annotations

import argparse
import json
import random
import re
import time

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import AUGMENT_MODEL_NAME, SOURCE_DATA_PATH, AUGMENTED_TRAIN_PATH, AUGMENTED_VALID_PATH
from .data import contains_hangul, default_full_answer, extract_answer, get_passage, iter_json_records, jsonl_snapshot, write_jsonl
from .prompts import augmentation_prompt


def contains_text(needle: str, haystack: str) -> bool:
    return str(needle or "").strip().casefold() in str(haystack or "").strip().casefold()


def extract_json_object(text: str) -> dict | None:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    first_object = None
    for match in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            if first_object is None:
                first_object = value
            if {"atomic_qas", "final_qas", "hard_negatives"} <= set(value):
                return value
    # Avoid treating a nested atomic QA as a complete augmentation row when the
    # model output was truncated before the outer JSON object closed.
    return first_object if first_object and {"atomic_qas", "final_qas"} <= set(first_object) else None


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


def apply_chat_template_no_thinking(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_messages(source: dict, passage: str) -> list[dict]:
    negative = first_hard_negative(source)
    return [
        {
            "role": "system",
            "content": "You generate strict raw JSON only. Do not write analysis, markdown, or code fences.",
        },
        {
            "role": "user",
            "content": augmentation_prompt(
                passage,
                question=str(source.get("question") or ""),
                answer=extract_answer(source),
                negative_passage=negative["passage"],
                negative_answer=negative["answer"],
            ),
        },
    ]


@torch.no_grad()
def generate_json(model, tokenizer, source: dict, passage: str, max_new_tokens: int = 768) -> tuple[dict | None, str]:
    messages = build_messages(source, passage)
    prompt = apply_chat_template_no_thinking(tokenizer, messages)
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
    return extract_json_object(text), text


def generate_json_vllm(
    source: dict,
    passage: str,
    model_name: str,
    url: str,
    max_new_tokens: int = 768,
    timeout: float = 300.0,
    json_mode: bool = True,
) -> tuple[dict | None, str]:
    payload = {
        "model": model_name,
        "messages": build_messages(source, passage),
        "temperature": 0,
        "max_tokens": max_new_tokens,
    }
    if json_mode:
        # vLLM's OpenAI-compatible server supports JSON-object guided output on
        # recent versions. This dramatically reduces invalid augmentation rows.
        payload["response_format"] = {"type": "json_object"}
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    try:
        message = data["choices"][0]["message"]
        text = message.get("content") or message.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError):
        text = json.dumps(data, ensure_ascii=False)
    return extract_json_object(text), text


def build_synthetic_negative(atomic: list[dict], final: list[dict]) -> dict | None:
    answers = [str(qa.get("answer") or "").strip() for qa in atomic]
    if not answers:
        return None
    unique_answers = list(dict.fromkeys(answer for answer in answers if answer))
    if len(unique_answers) < 2:
        return None

    neg_atomic = []
    neg_passage_parts = []
    for idx, qa in enumerate(atomic):
        question = str(qa.get("question") or "").strip()
        answer = str(qa.get("answer") or "").strip()
        sub_passage = str(qa.get("sub_passage") or "").strip()
        if not (question and answer and sub_passage):
            return None
        replacement = unique_answers[(unique_answers.index(answer) + 1) % len(unique_answers)] if answer in unique_answers else unique_answers[idx % len(unique_answers)]
        if replacement == answer:
            return None
        neg_sub_passage = sub_passage.replace(answer, replacement, 1)
        if neg_sub_passage == sub_passage:
            neg_sub_passage = f"{question} {replacement}"
        neg_atomic.append({
            "sub_passage": neg_sub_passage,
            "question": question,
            "answer": replacement,
            "full_answer": default_full_answer(question, replacement),
        })
        neg_passage_parts.append(neg_sub_passage)

    neg_final = []
    joined_answers = "; ".join(qa["answer"] for qa in neg_atomic)
    for qa in final:
        question = str(qa.get("question") or "").strip()
        if not question:
            return None
        neg_final.append({
            "question": question,
            "answer": joined_answers,
            "full_answer": default_full_answer(question, joined_answers),
        })

    return {
        "passage": " ".join(neg_passage_parts),
        "answer": joined_answers,
        "atomic_qas": neg_atomic,
        "final_qas": neg_final,
    }


def source_fact_items(source: dict, key: str) -> list[dict]:
    values = source.get(key)
    if not isinstance(values, list):
        return []
    out = []
    for item in values:
        if not isinstance(item, dict):
            continue
        fact_key = str(item.get("key") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if fact_key and answer:
            out.append({"key": fact_key, "answer": answer})
    return out


def answer_span(answer: str, passage: str) -> str:
    answer = str(answer or "").strip()
    passage = str(passage or "").strip()
    if not answer or not passage or not contains_text(answer, passage):
        return ""
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?。！？요다죠])\s+|(?<=[.!?。！？])", passage)
        if item.strip()
    ]
    for sentence in sentences:
        if contains_text(answer, sentence):
            return sentence
    idx = passage.casefold().find(answer.casefold())
    if idx < 0:
        return passage
    start = max(0, idx - 80)
    end = min(len(passage), idx + len(answer) + 80)
    return passage[start:end].strip()


def ensure_source_answers_in_passage(source: dict, passage: str, key: str) -> str:
    passage = str(passage or "").strip()
    additions = []
    is_ko = contains_hangul(passage) or str(source.get("language") or "").strip() == "ko"
    for item in source_fact_items(source, key):
        if contains_text(item["answer"], passage):
            continue
        if is_ko:
            additions.append(f"그리고 {item['key']}은 {item['answer']}입니다.")
        else:
            additions.append(f"And {item['key']} is {item['answer']}.")
    if additions:
        passage = f"{passage} {' '.join(additions)}".strip()
    return passage


def source_question(source: dict, fact_key: str) -> str:
    if str(source.get("language") or "").strip() == "ko" or contains_hangul(fact_key):
        return f"{fact_key}은 무엇인가요?"
    return f"What is {fact_key}?"


def source_full_answer(source: dict, fact_key: str, answer: str) -> str:
    if str(source.get("language") or "").strip() == "ko" or contains_hangul(f"{fact_key}\n{answer}"):
        return f"{fact_key}은 {answer}입니다."
    return f"{fact_key} is {answer}."


def fallback_augmented_from_source(source: dict, passage: str, source_id: str) -> dict | None:
    """Build a deterministic training row when the augmenter returns bad JSON.

    New source builders store exact facts and counterfactual facts. If the LLM
    fails, preserving these exact spans is better than discarding the row,
    especially for service-style dates, locations, owners, and analogies.
    """
    facts = source_fact_items(source, "facts")
    if not facts:
        return None
    passage = ensure_source_answers_in_passage(source, passage, "facts")
    atomic = []
    for item in facts:
        sub_passage = answer_span(item["answer"], passage)
        if not sub_passage:
            return None
        question = source_question(source, item["key"])
        atomic.append({
            "sub_passage": sub_passage,
            "question": question,
            "answer": item["answer"],
            "full_answer": source_full_answer(source, item["key"], item["answer"]),
        })
    final_question = str(source.get("question") or "").strip()
    if not final_question:
        final_question = "핵심 내용은 무엇인가요?" if contains_hangul(passage) else "What are the key points?"
    final_answer = str(source.get("answer") or "").strip() or "; ".join(
        f"{item['key']}: {item['answer']}" for item in facts
    )
    final = [{
        "question": final_question,
        "answer": final_answer,
        "full_answer": default_full_answer(final_question, final_answer),
    }]

    hard_negatives = []
    negative_items = source_fact_items(source, "negative_facts")
    source_negative = first_hard_negative(source)
    neg_passage = source_negative["passage"]
    if negative_items:
        if not neg_passage:
            neg_passage = passage
            for pos, neg in zip(facts, negative_items):
                neg_passage = neg_passage.replace(pos["answer"], neg["answer"], 1)
        neg_passage = ensure_source_answers_in_passage(source, neg_passage, "negative_facts")
        neg_atomic = []
        for pos_qa, neg_item in zip(atomic, negative_items):
            neg_sub_passage = answer_span(neg_item["answer"], neg_passage)
            if not neg_sub_passage:
                return None
            neg_atomic.append({
                "sub_passage": neg_sub_passage,
                "question": pos_qa["question"],
                "answer": neg_item["answer"],
                "full_answer": source_full_answer(source, neg_item["key"], neg_item["answer"]),
            })
        neg_answer = str(source_negative.get("answer") or "").strip() or "; ".join(
            f"{item['key']}: {item['answer']}" for item in negative_items
        )
        hard_negatives.append({
            "passage": neg_passage,
            "answer": neg_answer,
            "atomic_qas": neg_atomic,
            "final_qas": [{
                "question": final_question,
                "answer": neg_answer,
                "full_answer": default_full_answer(final_question, neg_answer),
            }],
        })
    return {
        "source_id": source_id,
        "speaker": source.get("speaker", ""),
        "passage": passage,
        "rewrite": " ".join(qa["full_answer"] for qa in atomic),
        "atomic_qas": atomic,
        "final_qas": final,
        "hard_negatives": hard_negatives,
    }


def normalize_augmented(raw: dict, source: dict, passage: str, source_id: str) -> dict | None:
    atomic = raw.get("atomic_qas")
    final = raw.get("final_qas")
    negatives = raw.get("hard_negatives")
    if not isinstance(atomic, list) or not isinstance(final, list):
        return None
    if not atomic or not final:
        return None
    if not isinstance(negatives, list):
        negatives = []
    if not all(isinstance(qa, dict) for qa in atomic + final):
        return None
    for qa in atomic:
        question = str(qa.get("question") or "").strip()
        answer = str(qa.get("answer") or "").strip()
        sub_passage = str(qa.get("sub_passage") or "").strip()
        if not (question and answer and sub_passage):
            return None
        if not contains_text(answer, sub_passage):
            return None
        qa["full_answer"] = str(qa.get("full_answer") or "").strip() or default_full_answer(question, answer)
    for qa in final:
        question = str(qa.get("question") or "").strip()
        answer = str(qa.get("answer") or "").strip()
        if not (question and answer):
            return None
        qa["full_answer"] = str(qa.get("full_answer") or "").strip() or default_full_answer(question, answer)

    source_negative = first_hard_negative(source)
    require_negative = bool(source_negative["passage"] or source_negative["answer"])
    if source_negative["passage"] and negatives and isinstance(negatives[0], dict):
        # Keep the generated Q/A decomposition, but anchor the counterfactual
        # passage to the deterministic source row so train-time positives and
        # negatives never drift apart because of LLM rewriting.
        negatives[0]["passage"] = source_negative["passage"]
        if source_negative["answer"]:
            negatives[0]["answer"] = source_negative["answer"]

    first_negative = negatives[0] if negatives and isinstance(negatives[0], dict) else None
    if not first_negative or not str(first_negative.get("passage") or "").strip():
        first_negative = build_synthetic_negative(atomic, final) if require_negative else None
    if not first_negative and not require_negative:
        negatives = []
        return {
            "source_id": source_id,
            "speaker": source.get("speaker", ""),
            "passage": passage,
            "rewrite": str(raw.get("rewrite") or "").strip(),
            "atomic_qas": atomic,
            "final_qas": final,
            "hard_negatives": negatives,
        }
    if not first_negative:
        return None
    neg_atomic = first_negative.get("atomic_qas")
    neg_final = first_negative.get("final_qas")
    if not isinstance(neg_atomic, list) or not isinstance(neg_final, list):
        first_negative = build_synthetic_negative(atomic, final)
        if not first_negative:
            return None
        neg_atomic = first_negative.get("atomic_qas")
        neg_final = first_negative.get("final_qas")
    if len(neg_atomic) != len(atomic) or len(neg_final) != len(final):
        first_negative = build_synthetic_negative(atomic, final)
        if not first_negative:
            return None
        neg_atomic = first_negative.get("atomic_qas")
        neg_final = first_negative.get("final_qas")
    for idx, neg_qa in enumerate(neg_atomic):
        if not isinstance(neg_qa, dict):
            return None
        answer = str(neg_qa.get("answer") or "").strip()
        sub_passage = str(neg_qa.get("sub_passage") or "").strip()
        if not (answer and sub_passage):
            return None
        if not contains_text(answer, sub_passage):
            return None
        # Force exact question alignment. The loader can fall back by position,
        # but identical strings keep the contrastive pair unambiguous.
        neg_qa["question"] = str(atomic[idx].get("question") or "").strip()
        neg_qa["full_answer"] = str(neg_qa.get("full_answer") or "").strip() or default_full_answer(neg_qa["question"], answer)
    for idx, neg_qa in enumerate(neg_final):
        if not isinstance(neg_qa, dict) or not str(neg_qa.get("answer") or "").strip():
            return None
        neg_qa["question"] = str(final[idx].get("question") or "").strip()
        neg_qa["full_answer"] = str(neg_qa.get("full_answer") or "").strip() or default_full_answer(
            neg_qa["question"],
            str(neg_qa.get("answer") or "").strip(),
        )
    negatives = [first_negative]

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


def progress_suffix(
    processed: int,
    input_total: int,
    existing_in_input: int,
    made: int,
    skipped_seen: int,
    skipped_invalid: int,
    train_count: int,
    valid_count: int,
    started_at: float,
) -> str:
    done_total = min(existing_in_input + made, input_total)
    elapsed_min = max((time.time() - started_at) / 60, 1e-6)
    rate = processed / elapsed_min
    eta = ""
    if input_total and rate > 0:
        remaining = max(input_total - processed, 0)
        eta = f", eta={remaining / rate:.1f}min"
    return (
        f"scan={format_progress(processed, input_total)}, "
        f"done_total={format_progress(done_total, input_total)}, "
        f"existing_start={existing_in_input}, new={made}, skipped_seen={skipped_seen}, "
        f"invalid={skipped_invalid}, train={train_count}, valid={valid_count}, "
        f"rate={rate:.2f}/min{eta}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(SOURCE_DATA_PATH))
    parser.add_argument("--train-output", default=str(AUGMENTED_TRAIN_PATH))
    parser.add_argument("--valid-output", default=str(AUGMENTED_VALID_PATH))
    parser.add_argument("--model", default=AUGMENT_MODEL_NAME)
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--vllm-url", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--vllm-timeout", type=float, default=300.0)
    parser.add_argument(
        "--vllm-json-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask vLLM for OpenAI response_format=json_object guided output.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--valid-every", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--debug-invalid-raw",
        action="store_true",
        help="Print a short raw model output preview when JSON parsing fails.",
    )
    parser.add_argument("--debug-raw-chars", type=int, default=1200)
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
    parser.add_argument(
        "--fallback-on-invalid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If the LLM returns invalid JSON, build a deterministic QA row from source facts when available.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
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
    input_ids = {
        str(row.get("source_id") or row.get("id") or f"raw_{idx}").strip()
        for idx, row in enumerate(input_rows)
    }
    existing_in_input = len(seen_ids & input_ids)
    print(f"[PRAG:augment] input={args.input} total_target={input_total}")
    print(f"[PRAG:augment] shuffle={args.shuffle} seed={args.seed}")
    print(
        f"[PRAG:augment] cumulative_start existing={existing_in_input}/{input_total} "
        f"({existing_in_input / input_total * 100:.1f}%) "
        f"all_existing_outputs={existing_count} train={train_count} valid={valid_count}"
        if input_total
        else f"[PRAG:augment] cumulative_start existing=0/0 all_existing_outputs={existing_count} train={train_count} valid={valid_count}"
    )

    model = tokenizer = None
    if args.backend == "transformers":
        model, tokenizer = load_local_model(args.model)
    else:
        print(f"[PRAG:augment] backend=vllm url={args.vllm_url} model={args.model}")
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
                done_total = min(existing_in_input + made, input_total)
                print(
                    f"[PRAG:augment] resume skip scan={format_progress(processed, input_total)} "
                    f"done_total={format_progress(done_total, input_total)} "
                    f"existing_start={existing_in_input}, new={made}, skipped_seen={skipped_seen}, "
                    f"train={train_count}, valid={valid_count}"
                )
            continue
        try:
            if args.backend == "vllm":
                generated, raw_text = generate_json_vllm(
                    source,
                    passage,
                    model_name=args.model,
                    url=args.vllm_url,
                    max_new_tokens=args.max_new_tokens,
                    timeout=args.vllm_timeout,
                    json_mode=args.vllm_json_mode,
                )
            else:
                generated, raw_text = generate_json(model, tokenizer, source, passage, max_new_tokens=args.max_new_tokens)
        except requests.RequestException as exc:
            skipped_invalid += 1
            processed = idx + 1
            print(
                f"[PRAG:augment] skip {source_id}: vLLM request failed: {exc} "
                f"({progress_suffix(processed, input_total, existing_in_input, made, skipped_seen, skipped_invalid, train_count, valid_count, started_at)})"
            )
            continue
        row = None
        fallback_reason = ""
        if generated is None:
            if args.fallback_on_invalid:
                row = fallback_augmented_from_source(source, passage, source_id)
                fallback_reason = "invalid JSON"
            if row is None:
                skipped_invalid += 1
                processed = idx + 1
                print(
                    f"[PRAG:augment] skip {source_id}: invalid JSON "
                    f"({progress_suffix(processed, input_total, existing_in_input, made, skipped_seen, skipped_invalid, train_count, valid_count, started_at)})"
                )
                if args.debug_invalid_raw:
                    preview = " ".join(str(raw_text or "").split())
                    print(f"[PRAG:augment:raw] {preview[:args.debug_raw_chars]}")
                continue
            if args.debug_invalid_raw:
                preview = " ".join(str(raw_text or "").split())
                print(f"[PRAG:augment:raw] {preview[:args.debug_raw_chars]}")
        if row is None:
            row = normalize_augmented(generated, source, passage, source_id)
        if row is None:
            if args.fallback_on_invalid:
                row = fallback_augmented_from_source(source, passage, source_id)
                fallback_reason = "missing required fields"
        if row is None:
            skipped_invalid += 1
            processed = idx + 1
            print(
                f"[PRAG:augment] skip {source_id}: missing required fields "
                f"({progress_suffix(processed, input_total, existing_in_input, made, skipped_seen, skipped_invalid, train_count, valid_count, started_at)})"
            )
            if args.debug_invalid_raw:
                passage_preview = " ".join(str(passage or "").split())
                raw_preview = " ".join(str(raw_text or "").split())
                parsed_preview = json.dumps(generated, ensure_ascii=False)[:args.debug_raw_chars] if generated is not None else "none"
                print(f"[PRAG:augment:passage] {passage_preview[:args.debug_raw_chars]}")
                print(f"[PRAG:augment:parsed] {parsed_preview}")
                print(f"[PRAG:augment:raw] {raw_preview[:args.debug_raw_chars]}")
            continue
        if fallback_reason:
            processed = idx + 1
            print(
                f"[PRAG:augment] fallback {source_id}: {fallback_reason} -> source facts "
                f"({progress_suffix(processed, input_total, existing_in_input, made, skipped_seen, skipped_invalid, train_count, valid_count, started_at)})"
            )
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
        print(
            f"[PRAG:augment] ok {source_id}: atomic={len(row['atomic_qas'])} "
            f"final={len(row['final_qas'])} -> {'valid' if is_valid else 'train'} "
            f"saved_to={saved_to} "
            f"({progress_suffix(processed, input_total, existing_in_input, made, skipped_seen, skipped_invalid, train_count, valid_count, started_at)})"
        )

    print(
        f"[PRAG:augment] done new={made} skipped_seen={skipped_seen} "
        f"skipped_invalid={skipped_invalid}"
    )
    print(f"[PRAG:augment] train={train_count} -> {args.train_output}")
    print(f"[PRAG:augment] valid={valid_count} -> {args.valid_output}")
    print(f"[PRAG:augment] train_snapshot={jsonl_snapshot(args.train_output)}")
    print(f"[PRAG:augment] valid_snapshot={jsonl_snapshot(args.valid_output)}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
