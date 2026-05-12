"""LLM rewrite augmentation for related-passage merge PRAG datasets.

Input and output use the existing PRAG augmented JSONL schema. The augmenter
does not invent new answers; it rewrites the evidence sentences, questions, and
full answers while preserving the exact answer strings and the one-topic,
multi-fact row structure needed for orthogonal merge training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

from .augment import extract_json_object
from .data import default_full_answer, iter_json_records


def contains_text(needle: str, haystack: str) -> bool:
    return str(needle or "").strip().casefold() in str(haystack or "").strip().casefold()


def is_marker_style(text: str) -> bool:
    text = str(text or "")
    return "=" in text or ":" in text


def build_messages(row: dict) -> list[dict]:
    source = {
        "source_id": row.get("source_id"),
        "passage": row.get("passage"),
        "atomic_qas": [
            {
                "sub_passage": qa.get("sub_passage"),
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "full_answer": qa.get("full_answer"),
            }
            for qa in (row.get("atomic_qas") or [])
            if isinstance(qa, dict)
        ],
        "final_qas": [
            {
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "full_answer": qa.get("full_answer"),
            }
            for qa in (row.get("final_qas") or [])
            if isinstance(qa, dict)
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You generate strict raw JSON only. Do not write markdown, analysis, or code fences. "
                "Preserve every exact answer string."
            ),
        },
        {
            "role": "user",
            "content": (
                "Rewrite this Korean PRAG memory row for data augmentation.\n"
                "Rules:\n"
                "1. Keep the same JSON schema and the same number/order of atomic_qas and final_qas.\n"
                "2. Preserve every answer exactly. Do not change answer strings.\n"
                "3. Each atomic sub_passage must be one natural Korean sentence about the same topic and must contain its exact answer.\n"
                "4. The row should describe related facts about one coherent topic, not unrelated topics.\n"
                "5. Do not use marker patterns such as A:B or A=B in passage or sub_passage.\n"
                "6. Questions and full_answer may be naturally paraphrased, but full_answer must contain the exact answer.\n"
                "Return only this JSON object:\n"
                "{\n"
                '  "passage": "...",\n'
                '  "atomic_qas": [{"sub_passage": "...", "question": "...", "answer": "...", "full_answer": "..."}],\n'
                '  "final_qas": [{"question": "...", "answer": "...", "full_answer": "..."}]\n'
                "}\n\n"
                f"SOURCE_ROW:\n{json.dumps(source, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def build_completion_prompt(row: dict) -> str:
    messages = build_messages(row)
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}"
        for message in messages
    ) + "\n\nASSISTANT:\n"


def completion_url_from_chat_url(url: str) -> str:
    if url.rstrip("/").endswith("/chat/completions"):
        return url.rstrip("/")[: -len("/chat/completions")] + "/completions"
    parsed = urlsplit(url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = f"{path}/completions"
    elif not path.endswith("/completions"):
        path = f"{path}/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def parse_chat_response(data: dict) -> str:
    try:
        message = data["choices"][0]["message"]
        return message.get("content") or message.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError):
        return json.dumps(data, ensure_ascii=False)


def parse_completion_response(data: dict) -> str:
    try:
        return data["choices"][0].get("text") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return json.dumps(data, ensure_ascii=False)


def post_chat_completion(
    row: dict,
    *,
    model_name: str,
    url: str,
    max_tokens: int,
    timeout: float,
    json_mode: bool,
) -> tuple[dict | None, str]:
    payload = {
        "model": model_name,
        "messages": build_messages(row),
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    text = parse_chat_response(response.json())
    return extract_json_object(text), text


def post_text_completion(
    row: dict,
    *,
    model_name: str,
    url: str,
    max_tokens: int,
    timeout: float,
) -> tuple[dict | None, str]:
    payload = {
        "model": model_name,
        "prompt": build_completion_prompt(row),
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    text = parse_completion_response(response.json())
    return extract_json_object(text), text


def generate_json_vllm(
    row: dict,
    *,
    model_name: str,
    url: str,
    max_tokens: int,
    timeout: float,
    json_mode: bool,
    api_mode: str,
) -> tuple[dict | None, str]:
    url_is_completion = url.rstrip("/").endswith("/completions") and not url.rstrip("/").endswith("/chat/completions")
    if api_mode == "completion" or (api_mode == "auto" and url_is_completion):
        return post_text_completion(row, model_name=model_name, url=url, max_tokens=max_tokens, timeout=timeout)
    if api_mode == "chat":
        return post_chat_completion(
            row,
            model_name=model_name,
            url=url,
            max_tokens=max_tokens,
            timeout=timeout,
            json_mode=json_mode,
        )
    try:
        return post_chat_completion(
            row,
            model_name=model_name,
            url=url,
            max_tokens=max_tokens,
            timeout=timeout,
            json_mode=json_mode,
        )
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code != 404:
            raise
        completion_url = completion_url_from_chat_url(url)
        print(f"[PRAG:related-augment] chat endpoint 404; retrying completions endpoint: {completion_url}")
        return post_text_completion(
            row,
            model_name=model_name,
            url=completion_url,
            max_tokens=max_tokens,
            timeout=timeout,
        )


def safe_text(value: object, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def normalize_augmented_row(raw: dict | None, original: dict, *, model_name: str) -> tuple[dict, str]:
    if not isinstance(raw, dict):
        return original, "fallback_invalid_json"

    original_atomic = [qa for qa in (original.get("atomic_qas") or []) if isinstance(qa, dict)]
    original_final = [qa for qa in (original.get("final_qas") or []) if isinstance(qa, dict)]
    generated_atomic = [qa for qa in (raw.get("atomic_qas") or []) if isinstance(qa, dict)]
    generated_final = [qa for qa in (raw.get("final_qas") or []) if isinstance(qa, dict)]
    if len(generated_atomic) != len(original_atomic) or len(generated_final) != len(original_final):
        return original, "fallback_count_mismatch"

    atomic_qas = []
    fallback_reasons: list[str] = []
    for idx, (src, gen) in enumerate(zip(original_atomic, generated_atomic)):
        answer = str(src.get("answer") or "").strip()
        src_sub = str(src.get("sub_passage") or "").strip()
        sub_passage = safe_text(gen.get("sub_passage"), src_sub)
        if not contains_text(answer, sub_passage) or is_marker_style(sub_passage):
            sub_passage = src_sub
            fallback_reasons.append(f"atomic_{idx}_sub")
        question = safe_text(gen.get("question"), str(src.get("question") or "").strip())
        full_answer = safe_text(gen.get("full_answer"), str(src.get("full_answer") or "").strip())
        if not contains_text(answer, full_answer):
            full_answer = str(src.get("full_answer") or "").strip() or default_full_answer(question, answer)
            fallback_reasons.append(f"atomic_{idx}_full")
        atomic_qas.append({
            "sub_passage": sub_passage,
            "question": question,
            "answer": answer,
            "full_answer": full_answer,
        })

    final_qas = []
    for idx, (src, gen) in enumerate(zip(original_final, generated_final)):
        answer = str(src.get("answer") or "").strip()
        question = safe_text(gen.get("question"), str(src.get("question") or "").strip())
        full_answer = safe_text(gen.get("full_answer"), str(src.get("full_answer") or "").strip())
        if answer and not contains_text(answer, full_answer):
            full_answer = str(src.get("full_answer") or "").strip() or default_full_answer(question, answer)
            fallback_reasons.append(f"final_{idx}_full")
        final_qas.append({
            "question": question,
            "answer": answer,
            "full_answer": full_answer,
        })

    passage = " ".join(qa["sub_passage"] for qa in atomic_qas).strip()
    augmented = {
        **original,
        "passage": passage,
        "rewrite": passage,
        "atomic_qas": atomic_qas,
        "final_qas": final_qas,
        "hard_negatives": original.get("hard_negatives") if isinstance(original.get("hard_negatives"), list) else [],
        "augmentation": {
            "method": "vllm_related_rewrite",
            "model": model_name,
            "fallbacks": fallback_reasons,
        },
    }
    return augmented, "ok" if not fallback_reasons else "ok_with_fallbacks"


def existing_source_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("source_id") or "") for row in iter_json_records(path)}


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_file(
    input_path: Path,
    output_path: Path,
    *,
    model_name: str,
    url: str,
    max_tokens: int,
    timeout: float,
    json_mode: bool,
    api_mode: str,
    limit: int,
    resume: bool,
    sleep_sec: float,
) -> None:
    if not resume:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    seen = existing_source_ids(output_path) if resume else set()
    total = made = skipped = failed = 0
    for row in iter_json_records(input_path):
        source_id = str(row.get("source_id") or "")
        if limit and total >= limit:
            break
        total += 1
        if source_id in seen:
            skipped += 1
            continue
        try:
            raw, raw_text = generate_json_vllm(
                row,
                model_name=model_name,
                url=url,
                max_tokens=max_tokens,
                timeout=timeout,
                json_mode=json_mode,
                api_mode=api_mode,
            )
            augmented, status = normalize_augmented_row(raw, row, model_name=model_name)
        except Exception as exc:
            augmented, status = row, f"fallback_request_error:{exc}"
            raw_text = ""
            failed += 1
        augmented = {
            **augmented,
            "augmentation_status": status,
        }
        append_jsonl(output_path, augmented)
        made += 1
        print(f"[PRAG:related-augment] {input_path.name} {made + skipped}/{total} source={source_id} status={status}")
        if raw_text and status.startswith("fallback"):
            print(f"[PRAG:related-augment:raw] {raw_text[:500]}")
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    print(
        f"[PRAG:related-augment] done input={input_path} output={output_path} "
        f"processed={total} made={made} skipped={skipped} failed={failed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", default="data/PRAG_related_merge_train.jsonl")
    parser.add_argument("--valid-input", default="data/PRAG_related_merge_valid.jsonl")
    parser.add_argument("--test-input", default="data/PRAG_related_merge_test.jsonl")
    parser.add_argument("--train-output", default="data/PRAG_related_merge_augmented_train.jsonl")
    parser.add_argument("--valid-output", default="data/PRAG_related_merge_augmented_valid.jsonl")
    parser.add_argument("--test-output", default="data/PRAG_related_merge_augmented_test.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--vllm-url", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument(
        "--api-mode",
        choices=("auto", "chat", "completion"),
        default="auto",
        help="OpenAI-compatible endpoint mode. auto retries /v1/completions if chat returns 404.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--json-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per split. 0 means all rows.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument(
        "--splits",
        default="train,valid,test",
        help="Comma-separated split names to process from train,valid,test.",
    )
    args = parser.parse_args()

    split_specs = {
        "train": (Path(args.train_input), Path(args.train_output)),
        "valid": (Path(args.valid_input), Path(args.valid_output)),
        "test": (Path(args.test_input), Path(args.test_output)),
    }
    requested = [item.strip() for item in args.splits.split(",") if item.strip()]
    for split in requested:
        if split not in split_specs:
            raise ValueError(f"Unknown split {split!r}; choose from train,valid,test")
        input_path, output_path = split_specs[split]
        process_file(
            input_path,
            output_path,
            model_name=args.model,
            url=args.vllm_url,
            max_tokens=args.max_new_tokens,
            timeout=args.timeout,
            json_mode=args.json_mode,
            api_mode=args.api_mode,
            limit=args.limit,
            resume=args.resume,
            sleep_sec=args.sleep_sec,
        )


if __name__ == "__main__":
    main()
