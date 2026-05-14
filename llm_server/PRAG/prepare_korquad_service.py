"""Rewrite KorQuAD into service-style professor transcript PRAG data.

KorQuAD already gives Korean context/question/answer triples. This converter
keeps the factual answer span, but asks a local/vLLM model to rewrite the
evidence into a lecture-style transcript with explanation, example, analogy, or
comparison. The output schema matches the existing PRAG augmented JSONL format.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import requests

from .augment import extract_json_object
from .config import KORQUAD_SERVICE_AUGMENTED_TRAIN_PATH, KORQUAD_SERVICE_AUGMENTED_VALID_PATH
from .prepare_korquad import iter_hf_records, iter_local_records, sentence_for_answer, service_full_answer, write_jsonl


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def contains_text(needle: str, haystack: str) -> bool:
    return normalize_text(needle).casefold() in normalize_text(haystack).casefold()


def split_sentences(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?。！？요다죠])\s+|(?<=[.!?。！？])|\n+", str(text or ""))
        if item.strip()
    ]


def answer_span(answer: str, passage: str, max_chars: int = 320) -> str:
    answer = normalize_text(answer)
    passage = normalize_text(passage)
    for sentence in split_sentences(passage):
        if contains_text(answer, sentence):
            return sentence[:max_chars].strip()
    idx = passage.casefold().find(answer.casefold())
    if idx >= 0:
        start = max(0, idx - max_chars // 2)
        end = min(len(passage), idx + len(answer) + max_chars // 2)
        return passage[start:end].strip()
    return ""


def answer_type(answer: str) -> str:
    text = normalize_text(answer)
    if re.search(r"\d+\s*악장|[일이삼사오육칠팔구십한두세네]\s*악장", text):
        return "movement"
    if re.search(r"교향곡\s*\d+\s*번", text):
        return "symphony_number"
    if re.search(r"\d+\s*(년|월|일|시|분|초|개월|주|명|개|권|편|번|회|%)", text):
        return "number_unit"
    if re.search(r"(월요일|화요일|수요일|목요일|금요일|토요일|일요일|주말|평일)", text):
        return "weekday"
    if re.search(r"(대학교|대학|학교|고등학교|중학교|초등학교)$", text):
        return "school"
    if re.search(r"(시|군|구|도|국|나라|공화국|왕국|섬|강|산|궁|성)$", text):
        return "place"
    if re.search(r"(곡|서곡|소나타|협주곡|오페라)$", text):
        return "music_work"
    if re.search(r"(왕|대통령|장군|교수|작가|시인|화가|감독)$", text):
        return "person_title"
    if len(text) <= 12 and re.fullmatch(r"[가-힣A-Za-z0-9\s·.-]+", text):
        return "short_phrase"
    return "other"


def synthetic_distractor(answer: str) -> str | None:
    text = normalize_text(answer)
    match = re.search(r"(\d+)(\s*악장)", text)
    if match:
        value = int(match.group(1))
        replacement = f"{value + 1 if value < 9 else value - 1}{match.group(2)}"
        return text[:match.start()] + replacement + text[match.end():]

    match = re.search(r"(\d+)(\s*번)", text)
    if match and "교향곡" in text:
        value = int(match.group(1))
        replacement = f"{value + 1 if value < 9 else value - 1}{match.group(2)}"
        return text[:match.start()] + replacement + text[match.end():]

    match = re.search(r"(\d+)(\s*(년|월|일|시|분|초|개월|주|명|개|권|편|회|%))", text)
    if match:
        value = int(match.group(1))
        replacement = f"{value + 1 if value < 99 else value - 1}{match.group(2)}"
        return text[:match.start()] + replacement + text[match.end():]

    weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    for idx, day in enumerate(weekdays):
        if day in text:
            return text.replace(day, weekdays[(idx + 1) % len(weekdays)], 1)
    return None


def choose_distractor(answer: str, pool: list[str], rng: random.Random) -> str | None:
    answer = normalize_text(answer)
    expected_type = answer_type(answer)
    synthetic = synthetic_distractor(answer)
    if synthetic and synthetic.casefold() != answer.casefold():
        return synthetic
    choices = [
        item
        for item in pool
        if (
            item
            and item.casefold() != answer.casefold()
            and 1 <= len(item) <= 80
            and answer_type(item) == expected_type
        )
    ]
    if choices:
        return rng.choice(choices)
    return None


def replace_once(text: str, old: str, new: str) -> tuple[str, bool]:
    match = re.search(re.escape(old), text, flags=re.IGNORECASE)
    if not match:
        return text, False
    return text[: match.start()] + new + text[match.end():], True


def ensure_answer_in_passage(passage: str, answer: str, question: str) -> str:
    passage = normalize_text(passage)
    answer = normalize_text(answer)
    if contains_text(answer, passage):
        return passage
    return f"{passage} 정리하면, 이 질문의 핵심 답은 {answer}입니다.".strip()


def fallback_passage(record: dict, idx: int) -> str:
    question = normalize_text(record["question"])
    answer = normalize_text(record["answer"])
    evidence = sentence_for_answer(record["context"], answer, max_chars=360)
    variants = [
        (
            f"자, 여기서 중요한 부분만 짚고 갈게요. {evidence} "
            f"쉽게 말하면 질문에서 찾는 핵심 답은 {answer}입니다."
        ),
        (
            f"이 내용을 예시로 보면, 긴 지문에서 단서를 찾아 답을 고르는 과정과 비슷해요. "
            f"{evidence} 그래서 {question}에 대한 답은 {answer}입니다."
        ),
        (
            f"비교해서 말하면 주변 설명은 배경이고, 우리가 꼭 가져가야 할 값은 따로 있습니다. "
            f"{evidence} 이 부분에서 핵심 답은 {answer}입니다."
        ),
        (
            f"자, 수업에서 이 문장을 만나면 답이 되는 표현을 그대로 잡아야 합니다. "
            f"{evidence} 따라서 답은 {answer}입니다."
        ),
    ]
    return normalize_text(variants[idx % len(variants)])


def build_messages(record: dict, idx: int) -> list[dict]:
    answer = normalize_text(record["answer"])
    question = normalize_text(record["question"])
    evidence = sentence_for_answer(record["context"], answer, max_chars=520)
    title = normalize_text(record.get("title") or "")
    style_hint = [
        "교수님만의 쉬운 예시를 하나 넣어라",
        "비유를 하나 넣어라",
        "비슷하지만 다른 개념과 비교해서 설명하라",
        "학생이 헷갈릴 만한 지점을 바로잡는 식으로 설명하라",
    ][idx % 4]
    prompt = f"""
너는 PRAG K/V 주입 학습용 데이터를 만드는 증강기다.
KorQuAD의 한국어 지문 내용을 우리 서비스에 맞는 실제 강의 전사문으로 바꿔라.

[원본 제목]
{title or "(없음)"}

[원본 근거 문장]
{evidence}

[원본 질문]
{question}

[정답 phrase]
{answer}

[증강 목표]
- 교수님이 강의 중 학생에게 설명하는 말투로 passage를 작성한다.
- "누가 무엇이라고 했다" 같은 보고문 말투를 쓰지 말고, 직접 발화처럼 써라.
- 예: "자, 여기서 중요한 건 ...입니다", "쉽게 말하면 ...", "예를 들어 ...", "비교하면 ..."처럼 쓴다.
- 교수님만의 설명, 예시, 비유, 비교 중 하나 이상을 넣어라: {style_hint}.
- passage는 3~6문장 정도로 만든다.
- 정답 phrase "{answer}"를 passage 안에 글자 하나 바꾸지 말고 그대로 포함한다.
- passage 밖의 새로운 사실을 만들지 않는다.
- atomic_qas는 원본 질문 의미를 유지하되, answer는 반드시 "{answer}" 그대로 둔다.
- full_answer도 학생에게 답하는 AI 선생님 문장처럼 쓰고, "{answer}"를 그대로 포함한다.
- final_qas의 question은 원본 질문 "{question}"을 그대로 사용한다.
- JSON 외 설명, markdown, 코드블록은 절대 쓰지 않는다.

[출력 JSON 스키마]
{{
  "passage": "...교수님 강의 전사문...",
  "rewrite": "...짧은 정리문...",
  "atomic_qas": [
    {{"sub_passage": "...정답 phrase가 포함된 passage 일부...", "question": "...", "answer": "{answer}", "full_answer": "..."}}
  ],
  "final_qas": [
    {{"question": "{question}", "answer": "{answer}", "full_answer": "..."}}
  ]
}}
""".strip()
    return [
        {"role": "system", "content": "You generate strict raw JSON only. Do not write markdown or analysis."},
        {"role": "user", "content": prompt},
    ]


def generate_vllm(record: dict, idx: int, model: str, url: str, max_tokens: int, timeout: float) -> tuple[dict | None, str]:
    payload = {
        "model": model,
        "messages": build_messages(record, idx),
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    try:
        text = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        text = json.dumps(data, ensure_ascii=False)
    return extract_json_object(text), text


def teacher_full_answer(question: str, answer: str) -> str:
    question = normalize_text(question)
    answer = normalize_text(answer)
    if not answer:
        return ""
    if question.endswith("?"):
        return f"교수님 설명에 따르면, 답은 {answer}입니다."
    return f"교수님 설명에 따르면, {answer}입니다."


def normalize_generated(raw: dict | None, record: dict, idx: int, distractor: str | None, source_id: str) -> dict:
    answer = normalize_text(record["answer"])
    question = normalize_text(record["question"])
    passage = normalize_text(raw.get("passage") if isinstance(raw, dict) else "")
    if not passage:
        passage = fallback_passage(record, idx)
    passage = ensure_answer_in_passage(passage, answer, question)

    sub_passage = answer_span(answer, passage)
    if not sub_passage:
        sub_passage = f"핵심 답은 {answer}입니다."
        passage = f"{passage} {sub_passage}".strip()

    generated_atomic = raw.get("atomic_qas") if isinstance(raw, dict) else None
    generated_qa = generated_atomic[0] if isinstance(generated_atomic, list) and generated_atomic and isinstance(generated_atomic[0], dict) else {}
    atomic_question = normalize_text(generated_qa.get("question") or question)
    atomic_full = normalize_text(generated_qa.get("full_answer") or service_full_answer(atomic_question, answer))
    if not contains_text(answer, atomic_full) or atomic_full == service_full_answer(atomic_question, answer):
        atomic_full = teacher_full_answer(atomic_question, answer)

    final_question = question
    final_full = teacher_full_answer(final_question, answer)

    row = {
        "source_id": source_id,
        "task": "korquad_service_transcript_memory",
        "dataset": "korquad",
        "language": "ko",
        "title": normalize_text(record.get("title") or ""),
        "passage": passage,
        "rewrite": normalize_text(raw.get("rewrite") if isinstance(raw, dict) else "") or final_full,
        "atomic_qas": [{
            "sub_passage": sub_passage,
            "question": atomic_question,
            "answer": answer,
            "full_answer": atomic_full,
        }],
        "final_qas": [{
            "question": final_question,
            "answer": answer,
            "full_answer": final_full,
        }],
        "hard_negatives": [],
    }
    if distractor:
        negative_passage, changed = replace_once(passage, answer, distractor)
        if changed:
            negative_sub = answer_span(distractor, negative_passage)
            if not negative_sub:
                negative_sub = f"핵심 답은 {distractor}입니다."
                negative_passage = f"{negative_passage} {negative_sub}".strip()
            row["hard_negatives"] = [{
                "passage": negative_passage,
                "answer": distractor,
                "atomic_qas": [{
                    "sub_passage": negative_sub,
                    "question": atomic_question,
                    "answer": distractor,
                    "full_answer": teacher_full_answer(atomic_question, distractor),
                }],
                "final_qas": [{
                    "question": final_question,
                    "answer": distractor,
                    "full_answer": teacher_full_answer(final_question, distractor),
                }],
            }]
    return row


def load_records(args: argparse.Namespace) -> tuple[list[dict], list[dict], str]:
    if args.input:
        rows = list(iter_local_records(Path(args.input)))
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        valid_count = min(args.max_valid_records, max(1, len(rows) // 10))
        valid_records = rows[:valid_count]
        train_records = rows[valid_count: valid_count + args.max_train_records]
        return train_records, valid_records, args.input
    train_records = list(iter_hf_records(args.dataset, args.train_split))[: args.max_train_records]
    valid_records = list(iter_hf_records(args.dataset, args.valid_split))[: args.max_valid_records]
    return train_records, valid_records, args.dataset


def load_seen(path: Path) -> set[str]:
    seen = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_id = str(row.get("source_id") or "").strip()
            if source_id:
                seen.add(source_id)
    return seen


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def run_split(
    records: list[dict],
    split: str,
    output: Path,
    answer_pool: list[str],
    args: argparse.Namespace,
) -> int:
    rng = random.Random(args.seed + (0 if split == "train" else 1000))
    seen = load_seen(output) if args.resume else set()
    if not args.resume:
        write_jsonl(output, [])
    made = skipped = failed = 0
    started = time.time()
    for idx, record in enumerate(records):
        record_id = normalize_text(record.get("id") or str(idx)).replace(" ", "_")
        source_id = f"korquad_service_{split}_{record_id}_{idx}"
        if source_id in seen:
            skipped += 1
            continue
        distractor = choose_distractor(record["answer"], answer_pool, rng) if args.negative else None
        raw = None
        if args.backend == "vllm":
            try:
                raw, _raw_text = generate_vllm(
                    record,
                    idx,
                    model=args.model,
                    url=args.vllm_url,
                    max_tokens=args.max_new_tokens,
                    timeout=args.vllm_timeout,
                )
            except requests.RequestException as exc:
                failed += 1
                print(f"[PRAG:korquad-service] {split} fallback {source_id}: vLLM failed: {exc}")
        row = normalize_generated(raw, record, idx, distractor, source_id)
        append_jsonl(output, row)
        made += 1
        if made % args.log_every == 0 or made == 1:
            elapsed_min = max((time.time() - started) / 60, 1e-6)
            rate = (idx + 1) / elapsed_min
            remaining = max(len(records) - idx - 1, 0)
            eta = remaining / rate if rate > 0 else 0.0
            print(
                f"[PRAG:korquad-service] {split} made={made} scan={idx + 1}/{len(records)} "
                f"skipped_seen={skipped} fallback_fail={failed} rate={rate:.2f}/min eta={eta:.1f}min -> {output}"
            )
    return made


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="KorQuAD/squad_kor_v1")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-split", default="validation")
    parser.add_argument("--input", default="", help="Optional local KorQuAD/SQuAD-style JSON file.")
    parser.add_argument("--train-output", default=str(KORQUAD_SERVICE_AUGMENTED_TRAIN_PATH))
    parser.add_argument("--valid-output", default=str(KORQUAD_SERVICE_AUGMENTED_VALID_PATH))
    parser.add_argument("--max-train-records", type=int, default=4000)
    parser.add_argument("--max-valid-records", type=int, default=800)
    parser.add_argument("--backend", choices=("vllm", "deterministic"), default="vllm")
    parser.add_argument("--vllm-url", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--vllm-timeout", type=float, default=300.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--negative",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate same-type hard negatives. Default is disabled for positive-only service-generation training.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    train_records, valid_records, source_desc = load_records(args)
    answer_pool = [normalize_text(row["answer"]) for row in train_records + valid_records if normalize_text(row.get("answer"))]
    print(f"[PRAG:korquad-service] source={source_desc}")
    print(
        f"[PRAG:korquad-service] train_records={len(train_records)} "
        f"valid_records={len(valid_records)} backend={args.backend} negative={args.negative}"
    )
    train_made = run_split(train_records, "train", Path(args.train_output), answer_pool, args)
    valid_made = run_split(valid_records, "valid", Path(args.valid_output), answer_pool, args)
    print(f"[PRAG:korquad-service] done train_new={train_made} -> {args.train_output}")
    print(f"[PRAG:korquad-service] done valid_new={valid_made} -> {args.valid_output}")


if __name__ == "__main__":
    main()
