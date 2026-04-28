"""Dataset utilities for augmented single-passage PRAG memory training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class MemoryExample:
    source_id: str
    passage: str
    question: str
    answer: str
    negative_passage: str | None = None
    negative_answer: str | None = None
    qa_type: str = "qa"


def iter_json_records(path: str | Path) -> Iterable[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array: {path}")
        yield from data
        return
    raise ValueError(f"Unsupported dataset format: {path}")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_passage(row: dict) -> str:
    for key in ("passage", "utterance", "text", "content"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            speaker = str(row.get("speaker") or "").strip()
            text = value.strip()
            return f"{speaker}: {text}" if speaker and key != "passage" else text
    return ""


def extract_answer(row: dict) -> str:
    for key in ("answer", "target", "output", "response", "negative_answer", "counterfactual_answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    answers = row.get("answers")
    if isinstance(answers, list):
        for item in answers:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def normalize_qas(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    qas = []
    for item in value:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or item.get("sub_question") or "").strip()
        answer = str(item.get("answer") or item.get("sub_answer") or "").strip()
        full_answer = str(item.get("full_answer") or "").strip()
        sub_passage = str(item.get("sub_passage") or item.get("evidence") or item.get("passage") or "").strip()
        if question and answer:
            qas.append({
                "question": question,
                "answer": answer,
                "full_answer": full_answer,
                "sub_passage": sub_passage,
            })
    return qas


def load_augmented_examples(path: str | Path, max_samples: int | None = None) -> list[MemoryExample]:
    examples: list[MemoryExample] = []
    for row_idx, row in enumerate(iter_json_records(path)):
        if max_samples and row_idx >= max_samples:
            break
        passage = get_passage(row)
        source_id = str(row.get("source_id") or row.get("id") or f"row_{row_idx}")
        if not passage:
            continue

        qas = []
        qas.extend((qa, "atomic", qa.get("sub_passage") or passage) for qa in normalize_qas(row.get("atomic_qas")))
        qas.extend((qa, "final", passage) for qa in normalize_qas(row.get("final_qas")))
        if not qas and row.get("question") and row.get("answer"):
            qas.append(({"question": str(row["question"]), "answer": str(row["answer"])}, "direct", passage))

        negatives = row.get("hard_negatives") if isinstance(row.get("hard_negatives"), list) else []
        neg_by_question = {}
        fallback_neg = None
        for neg in negatives:
            if not isinstance(neg, dict):
                continue
            neg_passage = get_passage(neg)
            neg_atomic = normalize_qas(neg.get("atomic_qas")) + normalize_qas(neg.get("qas"))
            neg_final = normalize_qas(neg.get("final_qas"))
            neg_qas = [(qa, qa.get("sub_passage") or neg_passage) for qa in neg_atomic]
            neg_qas.extend((qa, neg_passage) for qa in neg_final)
            if neg_passage and neg.get("answer"):
                fallback_neg = {"passage": neg_passage, "answer": str(neg["answer"]).strip()}
            for neg_qa, neg_memory_passage in neg_qas:
                if neg_memory_passage:
                    neg_by_question[neg_qa["question"]] = {
                        "passage": neg_memory_passage,
                        "answer": neg_qa["answer"],
                    }

        for qa_idx, (qa, qa_type, memory_passage) in enumerate(qas):
            neg = neg_by_question.get(qa["question"]) or fallback_neg
            examples.append(
                MemoryExample(
                    source_id=f"{source_id}:{qa_type}:{qa_idx}",
                    passage=memory_passage,
                    question=qa["question"],
                    answer=qa["answer"],
                    negative_passage=neg["passage"] if neg else None,
                    negative_answer=neg["answer"] if neg else None,
                    qa_type=qa_type,
                )
            )
    print(f"[PRAG:data] loaded {len(examples)} memory QA examples from {path}")
    return examples
