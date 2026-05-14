"""Prepare external QA datasets as PRAG augmented memory supervision.

This converter keeps the output schema identical to the LLM-augmented
multi-fact JSONL rows:

{
  "source_id": ...,
  "passage": ...,
  "rewrite": ...,
  "atomic_qas": [{"sub_passage", "question", "answer", "full_answer"}],
  "final_qas": [...],
  "hard_negatives": [{"passage", "answer", "atomic_qas", "final_qas"}]
}

HotpotQA and KorQuAD already provide passage/question/answer supervision, so
this file converts them directly instead of calling a local LLM again. Hard
negatives are counterfactual rows made by replacing the gold answer with a
distractor answer from another example.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .config import EXTERNAL_QA_AUGMENTED_TRAIN_PATH, EXTERNAL_QA_AUGMENTED_VALID_PATH


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def contains_hangul(text: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in str(text or ""))


def service_full_answer(question: str, answer: str) -> str:
    answer = normalize_text(answer)
    if not answer:
        return ""
    if answer.endswith((".", "?", "!", "다", "요", "임", "함", "음")):
        return answer
    if contains_hangul(question) or contains_hangul(answer):
        return f"{answer}입니다."
    return f"{answer}."


def contains_ci(needle: str, haystack: str) -> bool:
    return normalize_text(needle).casefold() in normalize_text(haystack).casefold()


def replace_ci_once(text: str, old: str, new: str) -> tuple[str, bool]:
    if not old:
        return text, False
    match = re.search(re.escape(old), text, flags=re.IGNORECASE)
    if not match:
        return text, False
    return text[: match.start()] + new + text[match.end():], True


def sentence_for_answer(context: str, answer: str, max_chars: int = 320) -> str:
    context = normalize_text(context)
    answer = normalize_text(answer)
    if not context:
        return ""
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", context)
    for part in parts:
        part = normalize_text(part)
        if part and contains_ci(answer, part):
            return part[:max_chars].strip()

    idx = context.casefold().find(answer.casefold())
    if idx >= 0:
        start = max(0, idx - max_chars // 2)
        end = min(len(context), idx + len(answer) + max_chars // 2)
        return context[start:end].strip()
    return context[:max_chars].strip()


def compact_passage(context: str, qas: list[dict], max_chars: int) -> str:
    context = normalize_text(context)
    if len(context) <= max_chars:
        return context
    snippets = []
    for qa in qas:
        snippet = sentence_for_answer(context, qa["answer"])
        if snippet:
            snippets.append(snippet)
    compact = " ".join(dict.fromkeys(snippets))
    if compact:
        return compact[:max_chars].strip()
    return context[:max_chars].strip()


def load_dataset_checked(name: str, *args, split: str, trust_remote_code: bool = False):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face 'datasets' package is required. Install it in the active environment "
            "or pass local JSON/JSONL input through the KorQuAD converter."
        ) from exc
    return load_dataset(name, *args, split=split, trust_remote_code=trust_remote_code)


def iter_korquad_records(dataset_name: str, split: str, limit: int) -> Iterable[dict]:
    ds = load_dataset_checked(dataset_name, split=split)
    count = 0
    for row in ds:
        answer = ""
        answers = row.get("answers")
        if isinstance(answers, dict):
            texts = answers.get("text") or []
            answer = normalize_text(texts[0]) if texts else ""
        elif isinstance(answers, list) and answers:
            first = answers[0]
            answer = normalize_text(first.get("text")) if isinstance(first, dict) else normalize_text(first)
        elif isinstance(row.get("answer"), str):
            answer = normalize_text(row["answer"])
        context = normalize_text(row.get("context") or row.get("passage") or "")
        question = normalize_text(row.get("question") or "")
        if context and question and answer and contains_ci(answer, context):
            yield {
                "dataset": "korquad",
                "lang": "ko",
                "id": str(row.get("id") or count),
                "context": context,
                "question": question,
                "answer": answer,
            }
            count += 1
            if limit and count >= limit:
                break


def hotpot_sentence_map(row: dict) -> dict[str, list[str]]:
    context = row.get("context") or {}
    titles = context.get("title") if isinstance(context, dict) else None
    sentences = context.get("sentences") if isinstance(context, dict) else None
    mapping: dict[str, list[str]] = {}
    if isinstance(titles, list) and isinstance(sentences, list):
        for title, sent_list in zip(titles, sentences):
            if isinstance(sent_list, list):
                mapping[str(title)] = [normalize_text(item) for item in sent_list if normalize_text(item)]
        return mapping
    if isinstance(context, list):
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0])
                sent_list = item[1] if isinstance(item[1], list) else []
                mapping[title] = [normalize_text(value) for value in sent_list if normalize_text(value)]
    return mapping


def hotpot_supporting_sentences(row: dict) -> list[str]:
    mapping = hotpot_sentence_map(row)
    supporting = row.get("supporting_facts") or {}
    titles = supporting.get("title") if isinstance(supporting, dict) else None
    sent_ids = supporting.get("sent_id") if isinstance(supporting, dict) else None
    output = []
    if isinstance(titles, list) and isinstance(sent_ids, list):
        for title, sent_id in zip(titles, sent_ids):
            sent_list = mapping.get(str(title), [])
            try:
                idx = int(sent_id)
            except (TypeError, ValueError):
                idx = -1
            if 0 <= idx < len(sent_list):
                output.append(sent_list[idx])
    if output:
        return list(dict.fromkeys(output))

    flattened = []
    for sent_list in mapping.values():
        flattened.extend(sent_list)
    return flattened[:6]


def iter_hotpot_records(dataset_name: str, config_name: str, split: str, limit: int) -> Iterable[dict]:
    ds = load_dataset_checked(dataset_name, config_name, split=split)
    count = 0
    for row in ds:
        answer = normalize_text(row.get("answer") or "")
        question = normalize_text(row.get("question") or "")
        if not answer or answer.casefold() in {"yes", "no"}:
            continue
        support = hotpot_supporting_sentences(row)
        passage = normalize_text(" ".join(support))
        if not (question and passage and contains_ci(answer, passage)):
            continue
        yield {
            "dataset": "hotpotqa",
            "lang": "en",
            "id": str(row.get("id") or count),
            "context": passage,
            "question": question,
            "answer": answer,
        }
        count += 1
        if limit and count >= limit:
            break


def group_records(records: list[dict], qas_per_row: int, max_chars: int) -> list[dict]:
    by_context: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        key = (record["dataset"], record["lang"], record["context"])
        by_context[key].append(record)

    groups = []
    for (dataset, lang, context), values in by_context.items():
        for start in range(0, len(values), max(1, qas_per_row)):
            qas = values[start: start + max(1, qas_per_row)]
            passage = compact_passage(context, qas, max_chars)
            groups.append({
                "dataset": dataset,
                "lang": lang,
                "context": passage,
                "qas": qas,
            })
    return groups


def choose_distractor(answer: str, pool: list[str], rng: random.Random) -> str:
    candidates = [
        item
        for item in pool
        if item and item.casefold() != answer.casefold() and 1 <= len(item) <= 80
    ]
    if not candidates:
        return f"{answer}_counterfactual"
    return rng.choice(candidates)


def make_final_qa(qas: list[dict], lang: str) -> dict:
    if lang == "ko":
        question = "위 passage의 핵심 질문과 답을 정리하면?"
    else:
        question = "What are the key questions and answers from the passage?"
    pairs = [f"{qa['question']} -> {qa['answer']}" for qa in qas]
    answer = "; ".join(pairs)
    return {
        "question": question,
        "answer": answer,
        "full_answer": answer,
    }


def make_row(group: dict, split: str, idx: int, answer_pool: list[str], rng: random.Random, include_final: bool) -> dict | None:
    passage = group["context"]
    lang = group["lang"]
    atomic_qas = []
    neg_atomic_qas = []
    neg_passage = passage
    distractors = []

    for qa in group["qas"]:
        question = normalize_text(qa["question"])
        answer = normalize_text(qa["answer"])
        sub_passage = sentence_for_answer(passage, answer)
        if not (question and answer and sub_passage and contains_ci(answer, sub_passage)):
            return None
        distractor = choose_distractor(answer, answer_pool, rng)
        neg_sub, changed_sub = replace_ci_once(sub_passage, answer, distractor)
        neg_passage, changed_passage = replace_ci_once(neg_passage, answer, distractor)
        if not (changed_sub and changed_passage and contains_ci(distractor, neg_sub)):
            return None
        atomic_qas.append({
            "sub_passage": sub_passage,
            "question": question,
            "answer": answer,
            "full_answer": service_full_answer(question, answer),
        })
        neg_atomic_qas.append({
            "sub_passage": neg_sub,
            "question": question,
            "answer": distractor,
            "full_answer": service_full_answer(question, distractor),
        })
        distractors.append(distractor)

    final_qas = [make_final_qa(atomic_qas, lang)] if include_final else []
    neg_final_qas = [make_final_qa(neg_atomic_qas, lang)] if include_final else []
    return {
        "source_id": f"externalqa_{group['dataset']}_{lang}_{split}_{idx}",
        "passage": passage,
        "rewrite": f"{group['dataset']} external QA converted for PRAG memory training.",
        "atomic_qas": atomic_qas,
        "final_qas": final_qas,
        "hard_negatives": [{
            "passage": neg_passage,
            "answer": distractors[0] if distractors else "",
            "atomic_qas": neg_atomic_qas,
            "final_qas": neg_final_qas,
        }],
    }


def build_rows(records: list[dict], *, split: str, qas_per_row: int, max_chars: int, include_final: bool, seed: int) -> list[dict]:
    rng = random.Random(seed)
    answer_pool = [record["answer"] for record in records]
    answer_pool_by_lang: dict[str, list[str]] = defaultdict(list)
    for record in records:
        answer_pool_by_lang[record["lang"]].append(record["answer"])
    groups = group_records(records, qas_per_row, max_chars)
    rng.shuffle(groups)
    rows = []
    skipped = 0
    for idx, group in enumerate(groups):
        pool = answer_pool_by_lang.get(group["lang"]) or answer_pool
        row = make_row(group, split, idx, pool, rng, include_final)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    print(f"[PRAG:external-qa] split={split} groups={len(groups)} rows={len(rows)} skipped={skipped}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="hotpotqa,korquad", help="Comma-separated: hotpotqa,korquad")
    parser.add_argument("--hotpot-dataset", default="hotpot_qa")
    parser.add_argument("--hotpot-config", default="distractor")
    parser.add_argument("--hotpot-train-split", default="train")
    parser.add_argument("--hotpot-valid-split", default="validation")
    parser.add_argument("--korquad-dataset", default="KorQuAD/squad_kor_v1")
    parser.add_argument("--korquad-train-split", default="train")
    parser.add_argument("--korquad-valid-split", default="validation")
    parser.add_argument("--max-hotpot-train-records", type=int, default=3000)
    parser.add_argument("--max-hotpot-valid-records", type=int, default=600)
    parser.add_argument("--max-korquad-train-records", type=int, default=3000)
    parser.add_argument("--max-korquad-valid-records", type=int, default=600)
    parser.add_argument("--qas-per-row", type=int, default=3)
    parser.add_argument("--max-passage-chars", type=int, default=900)
    parser.add_argument("--include-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-output", default=str(EXTERNAL_QA_AUGMENTED_TRAIN_PATH))
    parser.add_argument("--valid-output", default=str(EXTERNAL_QA_AUGMENTED_VALID_PATH))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    requested = {item.strip().lower() for item in args.datasets.split(",") if item.strip()}
    train_records: list[dict] = []
    valid_records: list[dict] = []

    if "hotpotqa" in requested or "hotpot" in requested:
        train_records.extend(iter_hotpot_records(
            args.hotpot_dataset,
            args.hotpot_config,
            args.hotpot_train_split,
            args.max_hotpot_train_records,
        ))
        valid_records.extend(iter_hotpot_records(
            args.hotpot_dataset,
            args.hotpot_config,
            args.hotpot_valid_split,
            args.max_hotpot_valid_records,
        ))
    if "korquad" in requested or "ko" in requested:
        train_records.extend(iter_korquad_records(args.korquad_dataset, args.korquad_train_split, args.max_korquad_train_records))
        valid_records.extend(iter_korquad_records(args.korquad_dataset, args.korquad_valid_split, args.max_korquad_valid_records))

    rng = random.Random(args.seed)
    rng.shuffle(train_records)
    rng.shuffle(valid_records)
    train_rows = build_rows(
        train_records,
        split="train",
        qas_per_row=args.qas_per_row,
        max_chars=args.max_passage_chars,
        include_final=args.include_final,
        seed=args.seed,
    )
    valid_rows = build_rows(
        valid_records,
        split="valid",
        qas_per_row=args.qas_per_row,
        max_chars=args.max_passage_chars,
        include_final=args.include_final,
        seed=args.seed + 1,
    )
    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.valid_output), valid_rows)
    print(f"[PRAG:external-qa] train_records={len(train_records)} train_rows={len(train_rows)} -> {args.train_output}")
    print(f"[PRAG:external-qa] valid_records={len(valid_records)} valid_rows={len(valid_rows)} -> {args.valid_output}")
    print(f"[PRAG:external-qa] include_final={args.include_final} qas_per_row={args.qas_per_row}")


if __name__ == "__main__":
    main()
