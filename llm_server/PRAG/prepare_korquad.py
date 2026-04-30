"""Convert Korean MRC datasets such as KorQuAD into PRAG augmented JSONL.

The produced rows intentionally match the output shape of augment.py, so they
can be trained with llm_server.PRAG.train without another LLM augmentation pass.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .config import KORQUAD_AUGMENTED_TRAIN_PATH, KORQUAD_AUGMENTED_VALID_PATH


def normalize_flat_record(item: dict) -> dict | None:
    context = str(item.get("context") or item.get("passage") or "").strip()
    question = str(item.get("question") or "").strip()
    answer = str(item.get("answer") or "").strip()
    answers = item.get("answers")
    if not answer and isinstance(answers, dict):
        texts = answers.get("text") or []
        if texts:
            answer = str(texts[0]).strip()
    elif not answer and isinstance(answers, list):
        if answers and isinstance(answers[0], dict):
            answer = str(answers[0].get("text") or "").strip()
        elif answers and isinstance(answers[0], str):
            answer = answers[0].strip()
    if context and question and answer:
        return {
            "title": str(item.get("title") or "").strip(),
            "context": context,
            "question": question,
            "answer": answer,
            "id": str(item.get("id") or ""),
        }
    return None


def iter_local_records(path: Path) -> Iterable[dict]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = normalize_flat_record(json.loads(line))
                if record:
                    yield record
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for item in data:
            record = normalize_flat_record(item)
            if record:
                yield record
        return

    # Official SQuAD/KorQuAD style:
    # {"data": [{"title": ..., "paragraphs": [{"context": ..., "qas": [...]}]}]}
    for article in data.get("data", []):
        title = str(article.get("title") or "").strip()
        for paragraph in article.get("paragraphs", []):
            context = str(paragraph.get("context") or "").strip()
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers") or []
                answer = ""
                if answers and isinstance(answers[0], dict):
                    answer = str(answers[0].get("text") or "").strip()
                elif answers and isinstance(answers[0], str):
                    answer = answers[0].strip()
                if context and qa.get("question") and answer:
                    yield {
                        "title": title,
                        "context": context,
                        "question": str(qa["question"]).strip(),
                        "answer": answer,
                        "id": str(qa.get("id") or ""),
                    }


def iter_hf_records(dataset_name: str, split: str) -> Iterable[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the Hugging Face datasets package or pass --input with a local KorQuAD JSON file.") from exc

    ds = load_dataset(dataset_name, split=split)
    for row in ds:
        answer = ""
        answers = row.get("answers")
        if isinstance(answers, dict):
            texts = answers.get("text") or []
            if texts:
                answer = str(texts[0]).strip()
        elif isinstance(row.get("answer"), str):
            answer = row["answer"].strip()
        if row.get("context") and row.get("question") and answer:
            yield {
                "title": str(row.get("title") or "").strip(),
                "context": str(row["context"]).strip(),
                "question": str(row["question"]).strip(),
                "answer": answer,
                "id": str(row.get("id") or ""),
            }


def sentence_for_answer(context: str, answer: str, max_chars: int = 260) -> str:
    parts = re.split(r"(?<=[.!?。！？])\s+|(?<=[다요죠음임됨함])\.\s*|\n+", context)
    for part in parts:
        part = part.strip()
        if answer in part:
            return part[:max_chars].strip()

    idx = context.find(answer)
    if idx >= 0:
        start = max(0, idx - max_chars // 2)
        end = min(len(context), idx + len(answer) + max_chars // 2)
        return context[start:end].strip()
    return context[:max_chars].strip()


def compact_context(context: str, qas: list[dict], max_chars: int) -> str:
    if len(context) <= max_chars:
        return context
    snippets = []
    for qa in qas:
        snippets.append(sentence_for_answer(context, qa["answer"]))
    compact = " ".join(dict.fromkeys(s for s in snippets if s))
    if compact and len(compact) <= max_chars:
        return compact
    return compact[:max_chars].strip() if compact else context[:max_chars].strip()


def replace_once(text: str, old: str, new: str) -> tuple[str, bool]:
    if old and old in text:
        return text.replace(old, new, 1), True
    return text, False


def build_negative_passage(passage: str, qas: list[dict], distractors: list[str]) -> str:
    negative = passage
    missing = []
    for qa, distractor in zip(qas, distractors):
        negative, changed = replace_once(negative, qa["answer"], distractor)
        if not changed:
            missing.append((qa["question"], distractor))
    if missing:
        extra = " ".join(f"혼동용 정보: {question} 답은 {answer}입니다." for question, answer in missing)
        negative = f"{negative} {extra}".strip()
    return negative


def make_final_qa(qas: list[dict], *, negative: bool = False) -> dict:
    pairs = [f"{qa['question']} -> {qa['answer']}" for qa in qas]
    return {
        "question": "위 문단의 핵심 질문과 답을 정리하면?",
        "answer": "; ".join(pairs),
        "full_answer": "; ".join(pairs),
        "sub_passage": "",
    }


def build_rows(records: list[dict], *, split_name: str, qas_per_row: int, max_context_chars: int, include_final: bool, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_context: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_context[record["context"]].append(record)

    answer_pool = [record["answer"] for record in records if record.get("answer")]
    rows = []
    for ctx_idx, (context, qas_all) in enumerate(by_context.items()):
        rng.shuffle(qas_all)
        for chunk_idx in range(0, len(qas_all), qas_per_row):
            chunk = qas_all[chunk_idx: chunk_idx + qas_per_row]
            if not chunk:
                continue
            passage = compact_context(context, chunk, max_context_chars)
            atomic_qas = []
            distractors = []
            for qa in chunk:
                answer = qa["answer"]
                choices = [item for item in answer_pool if item and item != answer]
                distractor = rng.choice(choices) if choices else f"{answer}_반례"
                distractors.append(distractor)
                sub_passage = sentence_for_answer(passage, answer)
                atomic_qas.append({
                    "sub_passage": sub_passage,
                    "question": qa["question"],
                    "answer": answer,
                    "full_answer": answer,
                })

            neg_passage = build_negative_passage(passage, chunk, distractors)
            neg_atomic = []
            for qa, distractor in zip(chunk, distractors):
                neg_sub = sentence_for_answer(neg_passage, distractor)
                neg_atomic.append({
                    "sub_passage": neg_sub,
                    "question": qa["question"],
                    "answer": distractor,
                    "full_answer": distractor,
                })

            final_qas = [make_final_qa(atomic_qas)] if include_final and len(atomic_qas) > 1 else []
            neg_final_qas = [make_final_qa(neg_atomic, negative=True)] if include_final and len(neg_atomic) > 1 else []
            rows.append({
                "source_id": f"korquad_{split_name}_{ctx_idx}_{chunk_idx // max(qas_per_row, 1)}",
                "passage": passage,
                "rewrite": "KorQuAD Korean MRC paragraph converted for PRAG memory training.",
                "atomic_qas": atomic_qas,
                "final_qas": final_qas,
                "hard_negatives": [{
                    "passage": neg_passage,
                    "answer": distractors[0],
                    "atomic_qas": neg_atomic,
                    "final_qas": neg_final_qas,
                }],
            })
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="KorQuAD/squad_kor_v1", help="Hugging Face dataset name.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-split", default="validation")
    parser.add_argument("--input", default="", help="Optional local KorQuAD/SQuAD-style JSON file.")
    parser.add_argument("--train-output", default=str(KORQUAD_AUGMENTED_TRAIN_PATH))
    parser.add_argument("--valid-output", default=str(KORQUAD_AUGMENTED_VALID_PATH))
    parser.add_argument("--max-train-records", type=int, default=2000)
    parser.add_argument("--max-valid-records", type=int, default=400)
    parser.add_argument("--qas-per-row", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=900)
    parser.add_argument("--include-final", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.input:
        all_records = list(iter_local_records(Path(args.input)))
        rng = random.Random(args.seed)
        rng.shuffle(all_records)
        valid_count = min(args.max_valid_records, max(1, len(all_records) // 10))
        valid_records = all_records[:valid_count]
        train_records = all_records[valid_count: valid_count + args.max_train_records]
        source_desc = args.input
    else:
        train_records = list(iter_hf_records(args.dataset, args.train_split))[: args.max_train_records]
        valid_records = list(iter_hf_records(args.dataset, args.valid_split))[: args.max_valid_records]
        source_desc = args.dataset

    train_rows = build_rows(
        train_records,
        split_name="train",
        qas_per_row=args.qas_per_row,
        max_context_chars=args.max_context_chars,
        include_final=args.include_final,
        seed=args.seed,
    )
    valid_rows = build_rows(
        valid_records,
        split_name="valid",
        qas_per_row=args.qas_per_row,
        max_context_chars=args.max_context_chars,
        include_final=args.include_final,
        seed=args.seed + 1,
    )
    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.valid_output), valid_rows)
    print(f"[PRAG:korquad] source={source_desc}")
    print(f"[PRAG:korquad] train_records={len(train_records)} train_rows={len(train_rows)} -> {args.train_output}")
    print(f"[PRAG:korquad] valid_records={len(valid_records)} valid_rows={len(valid_rows)} -> {args.valid_output}")
    print(f"[PRAG:korquad] include_final={args.include_final} qas_per_row={args.qas_per_row} max_context_chars={args.max_context_chars}")


if __name__ == "__main__":
    main()
