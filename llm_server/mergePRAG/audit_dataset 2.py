"""
Dataset auditor for MergePRAG training data.

목적:
- 같은/거의 같은 passage가 여러 샘플에 재등장하는지 확인
- truncation(기본 512 tokens) 후 서로 다른 passage가 사실상 같은 입력이 되는지 확인
- get_negative_sample 제약(같은 answer 제외 + 같은 passage 제외) 하에서 유효 negative가 충분한지 점검
- answer가 지나치게 많은 passage에 연결되어 generic-V를 유도하는지 확인

Usage:
  python -m llm_server.mergePRAG.audit_dataset
  python -m llm_server.mergePRAG.audit_dataset /path/to/data.jsonl --limit 50000
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import MAX_SEQ_LEN, MODEL_NAME, TRAIN_DATA_PATH
from .train import extract_answer, extract_passage, iter_records, normalize_passage_text


def normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def safe_preview(text: str, limit: int = 120) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class AuditRow:
    index: int
    source_id: str
    question: str
    answer: str
    passage: str
    norm_question: str
    norm_answer: str
    norm_passage: str
    trunc_passage: str
    trunc_len: int
    raw_len: int
    passage_contains_answer: bool


def load_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as exc:
        print(f"[audit] tokenizer load fallback: {exc}")
        return None


def truncate_with_tokenizer(tokenizer, text: str, max_length: int) -> tuple[str, int]:
    if tokenizer is None:
        pieces = text.split()
        clipped = pieces[:max_length]
        return " ".join(clipped), len(clipped)

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    return tokenizer.decode(token_ids, skip_special_tokens=True), len(token_ids)


def iter_audit_rows(dataset_path: Path, max_length: int, limit: int | None) -> Iterable[AuditRow]:
    tokenizer = load_tokenizer()
    kept = 0
    for idx, item in enumerate(iter_records(str(dataset_path))):
        if limit is not None and kept >= limit:
            break

        question = str(item.get("question", "")).strip()
        answer = extract_answer(item)
        passage = extract_passage(item)
        if not (question and answer and passage):
            continue

        norm_question = normalize_text(question)
        norm_answer = normalize_text(answer)
        norm_passage = normalize_passage_text(passage)
        trunc_passage, trunc_len = truncate_with_tokenizer(tokenizer, passage, max_length=max_length)
        source_id = str(item.get("source_id") or item.get("id") or f"row-{idx}")

        yield AuditRow(
            index=idx,
            source_id=source_id,
            question=question,
            answer=answer,
            passage=passage,
            norm_question=norm_question,
            norm_answer=norm_answer,
            norm_passage=norm_passage,
            trunc_passage=normalize_passage_text(trunc_passage),
            trunc_len=trunc_len,
            raw_len=len(passage),
            passage_contains_answer=norm_answer in norm_passage,
        )
        kept += 1


def summarize_top(counter: Counter, title: str, k: int = 10):
    print(f"\n[{title}]")
    for value, count in counter.most_common(k):
        print(f"  {count:6d} | {safe_preview(value)}")


def print_group_examples(title: str, grouped: dict[str, list[AuditRow]], *, preview_attr: str, limit: int = 5):
    print(f"\n[{title}]")
    shown = 0
    for key, rows in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        if len(rows) < 2:
            continue
        shown += 1
        unique_answers = len({r.norm_answer for r in rows})
        unique_passages = len({r.norm_passage for r in rows})
        print(
            f"  count={len(rows):4d} | unique_answers={unique_answers:3d} | "
            f"unique_passages={unique_passages:3d} | {safe_preview(key)}"
        )
        for row in rows[:3]:
            print(
                f"    - src={row.source_id} | ans={safe_preview(row.answer, 48)} | "
                f"{safe_preview(getattr(row, preview_attr), 96)}"
            )
        if shown >= limit:
            break
    if shown == 0:
        print("  no duplicate groups found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default=TRAIN_DATA_PATH)
    parser.add_argument("--max-length", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = list(iter_audit_rows(dataset_path, max_length=args.max_length, limit=args.limit))
    if not rows:
        raise RuntimeError("No valid rows found in dataset")

    by_source = defaultdict(list)
    by_question = defaultdict(list)
    by_answer = defaultdict(list)
    by_passage = defaultdict(list)
    by_trunc_passage = defaultdict(list)
    by_question_passage = defaultdict(list)

    truncated_rows = 0
    contains_answer = 0

    for row in rows:
        by_source[row.source_id].append(row)
        by_question[row.norm_question].append(row)
        by_answer[row.norm_answer].append(row)
        by_passage[row.norm_passage].append(row)
        by_trunc_passage[row.trunc_passage].append(row)
        by_question_passage[(row.norm_question, row.trunc_passage)].append(row)
        if row.trunc_len >= args.max_length:
            truncated_rows += 1
        if row.passage_contains_answer:
            contains_answer += 1

    same_source_same_passage = 0
    same_source_same_question = 0
    same_source_valid_negative = 0
    global_valid_negative = 0
    no_valid_negative = 0

    answer_to_passages = defaultdict(set)
    question_to_answers = defaultdict(set)
    question_to_passages = defaultdict(set)

    for row in rows:
        answer_to_passages[row.norm_answer].add(row.norm_passage)
        question_to_answers[row.norm_question].add(row.norm_answer)
        question_to_passages[row.norm_question].add(row.norm_passage)

    answer_counter = Counter()
    for answer, passages in answer_to_passages.items():
        answer_counter[answer] = len(passages)

    question_counter = Counter()
    for question, passages in question_to_passages.items():
        question_counter[question] = len(passages)

    passage_counter = Counter({k: len(v) for k, v in by_passage.items()})
    trunc_counter = Counter({k: len(v) for k, v in by_trunc_passage.items()})

    all_answers = Counter(row.norm_answer for row in rows)

    answer_passage_counter = Counter((row.norm_answer, row.norm_passage) for row in rows)

    total_rows = len(rows)
    for row in rows:
        source_rows = by_source[row.source_id]
        source_same_passage = [r for r in source_rows if r.norm_passage == row.norm_passage and r.index != row.index]
        source_diff_passage_diff_answer = [
            r
            for r in source_rows
            if r.index != row.index
            and r.norm_passage != row.norm_passage
            and r.norm_answer != row.norm_answer
        ]
        if source_same_passage:
            same_source_same_passage += 1
        if any(r.norm_question == row.norm_question and r.index != row.index for r in source_rows):
            same_source_same_question += 1
        if source_diff_passage_diff_answer:
            same_source_valid_negative += 1
        fallback_count = (
            total_rows
            - all_answers[row.norm_answer]
            - passage_counter[row.norm_passage]
            + answer_passage_counter[(row.norm_answer, row.norm_passage)]
        )
        fallback_exists = fallback_count > 0
        if fallback_exists:
            global_valid_negative += 1
        else:
            no_valid_negative += 1

    trunc_collisions = {
        key: value
        for key, value in by_trunc_passage.items()
        if len({row.norm_passage for row in value}) > 1
    }
    q_trunc_collisions = {
        f"{question} || {passage}": value
        for (question, passage), value in by_question_passage.items()
        if len({row.norm_passage for row in value}) > 1
    }

    print(f"[dataset] {dataset_path}")
    print(f"[rows] {total_rows}")
    print(f"[truncated_at_{args.max_length}] {truncated_rows} ({truncated_rows / total_rows:.2%})")
    print(f"[passage_contains_answer] {contains_answer} ({contains_answer / total_rows:.2%})")
    print(f"[same_source_same_passage] {same_source_same_passage} ({same_source_same_passage / total_rows:.2%})")
    print(f"[same_source_same_question] {same_source_same_question} ({same_source_same_question / total_rows:.2%})")
    print(f"[same_source_valid_negative] {same_source_valid_negative} ({same_source_valid_negative / total_rows:.2%})")
    print(f"[global_valid_negative] {global_valid_negative} ({global_valid_negative / total_rows:.2%})")
    print(f"[no_valid_negative] {no_valid_negative} ({no_valid_negative / total_rows:.2%})")
    print(f"[exact_passage_duplicates] {sum(1 for v in by_passage.values() if len(v) > 1)} groups")
    print(f"[trunc_passage_collisions] {len(trunc_collisions)} groups")
    print(f"[question_plus_trunc_collision] {len(q_trunc_collisions)} groups")

    summarize_top(passage_counter, "Most Reused Passages")
    summarize_top(trunc_counter, f"Most Reused Truncated-{args.max_length} Passages")
    summarize_top(all_answers, "Most Frequent Answers")
    summarize_top(answer_counter, "Answers Linked To Many Distinct Passages")
    summarize_top(question_counter, "Questions Linked To Many Distinct Passages")

    print_group_examples(
        "Exact Passage Duplicate Examples",
        {k: v for k, v in by_passage.items() if len(v) > 1},
        preview_attr="question",
    )
    print_group_examples(
        f"Truncation Collision Examples (same first {args.max_length} tokens, different full passage)",
        trunc_collisions,
        preview_attr="passage",
    )
    print_group_examples(
        "Question + Truncation Collision Examples",
        q_trunc_collisions,
        preview_attr="passage",
    )

    print("\n[interpretation]")
    print("  - exact_passage_duplicates가 많으면 passage 분리 학습이 약해질 수 있습니다.")
    print("  - trunc_passage_collisions가 많으면 raw passage는 달라도 hypernetwork 입력은 같아집니다.")
    print("  - same_source_valid_negative가 낮으면 같은 질문 안에서 hard negative가 부족합니다.")
    print("  - Answers Linked To Many Distinct Passages 상위 answer가 많으면 generic answer prior를 의심해야 합니다.")
    print("  - passage_contains_answer 비율이 높으면 memory가 근거보다 정답 문자열 매칭으로 흐를 수 있습니다.")


if __name__ == "__main__":
    main()
