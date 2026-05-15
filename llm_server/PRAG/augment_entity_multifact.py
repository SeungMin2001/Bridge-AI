"""LLM augmentation for entity-level multi-fact PRAG rows."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from entity_multifact_bank import build_full_answer
except ModuleNotFoundError:
    from llm_server.PRAG.entity_multifact_bank import build_full_answer


DEFAULT_INPUT = Path("data/PRAG_entity_multifact_seed.jsonl")
DEFAULT_OUTPUT = Path("data/PRAG_entity_multifact_augmented_all.jsonl")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def existing_source_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    ids.add(str(json.loads(line).get("source_id", "")))
                except json.JSONDecodeError:
                    continue
    return ids


def completion_url_from_chat_url(url: str) -> str:
    if url.endswith("/v1/chat/completions"):
        return url[: -len("/chat/completions")] + "/completions"
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")] + "/completions"
    return url


def chat_url_from_completion_url(url: str) -> str:
    if url.endswith("/v1/completions"):
        return url[: -len("/completions")] + "/chat/completions"
    if url.endswith("/completions"):
        return url[: -len("/completions")] + "/chat/completions"
    return url


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_vllm(
    prompt: str,
    model: str,
    vllm_url: str,
    api_mode: str,
    max_new_tokens: int,
    timeout: float,
) -> str:
    mode = api_mode
    if mode == "auto":
        # The local vLLM server used in this project often exposes only
        # /v1/completions, so try the provided endpoint first and fall back once.
        mode = "chat" if "chat/completions" in vllm_url else "completion"

    errors: list[str] = []
    modes = [mode]
    if api_mode == "auto":
        modes.append("completion" if mode == "chat" else "chat")

    for current_mode in modes:
        try:
            if current_mode == "chat":
                url = chat_url_from_completion_url(vllm_url)
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You output strict raw JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.35,
                    "top_p": 0.9,
                    "max_tokens": max_new_tokens,
                }
                data = post_json(url, payload, timeout)
                return str(data["choices"][0]["message"]["content"])
            url = completion_url_from_chat_url(vllm_url)
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": 0.35,
                "top_p": 0.9,
                "max_tokens": max_new_tokens,
            }
            data = post_json(url, payload, timeout)
            return str(data["choices"][0]["text"])
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
            errors.append(f"{current_mode}:{exc}")
    raise RuntimeError("; ".join(errors))


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    return text.replace("```", "").strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = strip_thinking(text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def contains_artifact(text: str) -> bool:
    # Avoid data styles that made earlier datasets too synthetic.
    return bool(re.search(r"[A-Za-z]\s*[=:]\s*[A-Za-z가-힣0-9]|[가-힣0-9]\s*[=:]\s*[A-Za-z가-힣0-9]", str(text or "")))


def get_atomic_answers(row: dict[str, Any]) -> list[str]:
    return [str(qa.get("answer", "")).strip() for qa in row.get("atomic_qas", []) if str(qa.get("answer", "")).strip()]


def fallback_final_answer(row: dict[str, Any]) -> str:
    entity = str(row.get("entity") or "해당 주제")
    facts = [(str(qa.get("question", "내용")).split("의 ")[-1].split("은")[0], str(qa.get("answer", ""))) for qa in row.get("atomic_qas", [])]
    facts = [(slot.strip() or "내용", value.strip()) for slot, value in facts if value.strip()]
    if facts:
        return build_full_answer(entity, facts)
    final_qas = row.get("final_qas", [])
    if final_qas:
        return str(final_qas[0].get("answer") or final_qas[0].get("full_answer") or "")
    return ""


def build_prompt(row: dict[str, Any]) -> str:
    compact = {
        "source_id": row.get("source_id"),
        "root_source_id": row.get("root_source_id", row.get("source_id")),
        "entity": row.get("entity"),
        "passage": row.get("passage"),
        "atomic_qas": row.get("atomic_qas", []),
        "final_qas": row.get("final_qas", []),
    }
    return (
        "다음 한국어 PRAG entity-multifact JSON row를 데이터 증강용으로 다시 작성하세요.\n"
        "반드시 raw JSON 객체 하나만 출력하세요. 설명, markdown, <think>를 출력하지 마세요.\n\n"
        "규칙:\n"
        "1. 같은 JSON 스키마와 atomic_qas/final_qas 개수 및 순서를 유지합니다.\n"
        "2. entity 문자열과 모든 atomic_qas.answer 문자열은 글자, 띄어쓰기, 숫자를 정확히 보존합니다.\n"
        "3. 각 atomic_qas.sub_passage는 자연스러운 한국어 한 문장이어야 하며 해당 answer 문자열을 그대로 포함해야 합니다.\n"
        "4. final_qas 질문은 특정 한 항목만 묻지 말고 entity 전체 설명을 묻는 넓은 질문으로 둡니다.\n"
        "5. final_qas.answer와 full_answer는 모든 atomic answer 문자열을 정확히 포함하며, 여러 사실을 자연스럽게 통합한 답변이어야 합니다.\n"
        "6. passage는 atomic sub_passage들을 자연스럽게 이어 붙인 문단으로 만듭니다.\n"
        "7. A:B, A=B, 번호만 바꾼 문장, 영어/중국어 답변, 목록형 bullet을 피하고 한국어 서술문으로 작성합니다.\n\n"
        "입력 JSON:\n"
        f"{json.dumps(compact, ensure_ascii=False)}"
    )


def normalize_augmented(original: dict[str, Any], generated: dict[str, Any] | None, source_id: str) -> tuple[dict[str, Any], str]:
    fallback_count = 0
    row = dict(original if generated is None else generated)
    status = "ok"
    if generated is None:
        row = dict(original)
        status = "fallback_invalid_json"
        fallback_count += 1

    row["source_id"] = source_id
    row["root_source_id"] = original.get("root_source_id", original.get("source_id"))
    row["language"] = "ko"
    row["clean_ko"] = True
    row["entity_multifact"] = True
    row["atomic_qas_as_evidence_only"] = True
    row["entity"] = original.get("entity")
    row["domain"] = original.get("domain")
    row["base_entity"] = original.get("base_entity")
    row["hard_negatives"] = []

    orig_atomic = original.get("atomic_qas", [])
    gen_atomic = row.get("atomic_qas", [])
    if not isinstance(gen_atomic, list) or len(gen_atomic) != len(orig_atomic):
        gen_atomic = orig_atomic
        fallback_count += 1

    fixed_atomic = []
    for orig_qa, gen_qa in zip(orig_atomic, gen_atomic):
        qa = dict(gen_qa if isinstance(gen_qa, dict) else orig_qa)
        answer = str(orig_qa.get("answer", "")).strip()
        sub_passage = str(qa.get("sub_passage", "")).strip()
        full_answer = str(qa.get("full_answer", "")).strip()
        if not sub_passage or answer not in sub_passage or contains_artifact(sub_passage):
            sub_passage = str(orig_qa.get("sub_passage", "")).strip()
            fallback_count += 1
        if not full_answer or answer not in full_answer or contains_artifact(full_answer):
            full_answer = str(orig_qa.get("full_answer", "")).strip()
            fallback_count += 1
        qa["sub_passage"] = sub_passage
        qa["question"] = str(qa.get("question") or orig_qa.get("question") or "").strip()
        qa["answer"] = answer
        qa["full_answer"] = full_answer
        fixed_atomic.append(qa)
    row["atomic_qas"] = fixed_atomic

    atomic_answers = get_atomic_answers(row)
    orig_final = original.get("final_qas", [])
    gen_final = row.get("final_qas", [])
    if not isinstance(gen_final, list) or len(gen_final) != len(orig_final):
        gen_final = orig_final
        fallback_count += 1

    default_final = fallback_final_answer(row)
    fixed_final = []
    for idx, orig_qa in enumerate(orig_final):
        gen_qa = gen_final[idx] if idx < len(gen_final) and isinstance(gen_final[idx], dict) else orig_qa
        qa = dict(gen_qa)
        question = str(qa.get("question") or orig_qa.get("question") or "").strip()
        answer = str(qa.get("answer") or qa.get("full_answer") or "").strip()
        if not answer or any(target not in answer for target in atomic_answers) or contains_artifact(answer):
            answer = str(orig_qa.get("answer") or orig_qa.get("full_answer") or default_final).strip()
            fallback_count += 1
        full_answer = str(qa.get("full_answer") or answer).strip()
        if any(target not in full_answer for target in atomic_answers) or contains_artifact(full_answer):
            full_answer = answer
            fallback_count += 1
        qa["question"] = question
        qa["answer"] = answer
        qa["full_answer"] = full_answer
        fixed_final.append(qa)
    row["final_qas"] = fixed_final

    passage = " ".join(qa["sub_passage"] for qa in fixed_atomic)
    row["passage"] = passage
    row["rewrite"] = passage
    row["augmentation_meta"] = {
        "source": original.get("source_id"),
        "status": status,
        "fallback_count": fallback_count,
    }
    if status == "ok" and fallback_count:
        status = "ok_with_fallbacks"
    return row, status


def source_id_for_variant(row: dict[str, Any], variant_idx: int, variants_per_row: int) -> str:
    base = str(row.get("source_id"))
    if variants_per_row <= 1:
        return base
    return f"{base}_aug{variant_idx}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment entity-level multi-fact PRAG rows with a local vLLM server.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--vllm-url", default="http://localhost:8001/v1/completions")
    parser.add_argument("--api-mode", choices=["auto", "completion", "chat"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=2400)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--variants-per-row", type=int, default=1)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    seen = existing_source_ids(args.output) if args.resume else set()
    if not args.resume and args.output.exists():
        args.output.unlink()

    total = len(rows) * max(1, args.variants_per_row)
    made = skipped = failed = processed = 0
    for row_idx, row in enumerate(rows):
        for variant_idx in range(max(1, args.variants_per_row)):
            processed += 1
            out_source_id = source_id_for_variant(row, variant_idx, args.variants_per_row)
            if out_source_id in seen:
                skipped += 1
                continue
            generated: dict[str, Any] | None = None
            status = "ok"
            try:
                raw = call_vllm(
                    build_prompt(row),
                    model=args.model,
                    vllm_url=args.vllm_url,
                    api_mode=args.api_mode,
                    max_new_tokens=args.max_new_tokens,
                    timeout=args.timeout,
                )
                generated = extract_json_object(raw)
                if generated is None:
                    status = "fallback_invalid_json"
            except Exception as exc:  # noqa: BLE001 - keep batch augmentation resilient.
                status = f"fallback_request_error:{exc}"
                failed += 1
            fixed, normalized_status = normalize_augmented(row, generated, out_source_id)
            if status.startswith("fallback_request_error"):
                fixed["augmentation_meta"]["status"] = status
            else:
                status = normalized_status
            append_jsonl(args.output, fixed)
            made += 1
            print(f"[PRAG:entity-augment] {args.input.name} {processed}/{total} source={out_source_id} status={status}", flush=True)
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    print(
        f"[PRAG:entity-augment] done input={args.input} output={args.output} "
        f"processed={processed} made={made} skipped={skipped} failed={failed}"
    )


if __name__ == "__main__":
    main()
