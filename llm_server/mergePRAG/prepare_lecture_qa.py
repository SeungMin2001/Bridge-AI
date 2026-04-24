"""
Convert lecture/service QA JSON into MergePRAG training JSONL.

이 스크립트는 QA를 생성하지 않고, 이미 만든 lecture QA를 학습 포맷으로 정규화한다.
서비스 목표에 맞는 권장 입력 구조는 다음 둘 중 하나다.

Flat row:
  {
    "course_id": "os-2026",
    "chunk_id": "week01-003",
    "speaker": "교수",
    "utterance": "프로세스는 실행 중인 프로그램입니다.",
    "question": "교수님은 프로세스를 뭐라고 설명했어?",
    "answer": "교수님은 프로세스를 실행 중인 프로그램이라고 설명했습니다.",
    "contrast_id": "process-definition",
    "hard_negatives": [
      {"passage": "교수: 스레드는 프로세스 안의 실행 흐름입니다.", "answer": "스레드"}
    ]
  }

Nested row:
  {
    "course_id": "os-2026",
    "chunk_id": "week01-003",
    "speaker": "교수",
    "utterance": "프로세스는 실행 중인 프로그램입니다.",
    "qas": [
      {"question": "...", "answer": "...", "hard_negatives": [...]}
    ]
  }

Output row:
  {"source_id": ..., "task": "final_qa", "question": ..., "answer": ..., "passage": ...}

Usage:
  python -m llm_server.mergePRAG.prepare_lecture_qa input.jsonl output.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .train import extract_answer, extract_passage


META_KEYS = (
    "course_id",
    "lecture_id",
    "chunk_id",
    "speaker",
    "timestamp",
    "start_time",
    "end_time",
    "title",
    "source",
    "contrast_id",
    "group_id",
)


def iter_records(path: Path) -> Iterable[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                yield item
            return
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            for item in data["data"]:
                yield item
            return
        raise ValueError("JSON input must be a list or {'data': [...]} object")

    raise ValueError(f"Unsupported input format: {path.suffix}")


def pick_metadata(record: dict, qa: dict | None = None) -> dict:
    qa = qa or {}
    metadata = {}
    for key in META_KEYS:
        value = qa.get(key, record.get(key))
        if value is not None:
            metadata[key] = value
    return metadata


def merge_hard_negatives(record: dict, qa: dict | None = None) -> list:
    qa = qa or {}
    merged = []
    for source in (record, qa):
        for key in ("hard_negatives", "negative_passages", "counterfactuals", "distractors"):
            value = source.get(key)
            if isinstance(value, list):
                merged.extend(value)
    return merged


def make_source_id(record: dict, qa: dict | None, row_idx: int, qa_idx: int) -> str:
    qa = qa or {}
    explicit = qa.get("source_id") or qa.get("id") or record.get("source_id") or record.get("id")
    if explicit:
        return f"{explicit}:{qa_idx}" if qa is not None else str(explicit)
    course_id = record.get("course_id", "course")
    chunk_id = record.get("chunk_id") or record.get("lecture_id") or row_idx
    return f"{course_id}:{chunk_id}:{qa_idx}"


def build_row(record: dict, row_idx: int, qa: dict | None = None, qa_idx: int = 0) -> dict | None:
    qa = qa or record
    question = str(qa.get("question", "")).strip()
    answer = extract_answer(qa)
    passage = extract_passage(qa) or extract_passage(record)
    if not (question and answer and passage):
        return None

    output = {
        **pick_metadata(record, qa),
        "source_id": make_source_id(record, qa if qa is not record else None, row_idx, qa_idx),
        "task": str(qa.get("task") or record.get("task") or "final_qa"),
        "question": question,
        "answer": answer,
        "passage": passage,
    }

    hard_negatives = merge_hard_negatives(record, qa)
    if hard_negatives:
        output["hard_negatives"] = hard_negatives

    for key in ("negative_passage", "counterfactual_passage", "hard_negative_passage"):
        value = qa.get(key, record.get(key))
        if value:
            output[key] = value

    return output


def iter_output_rows(input_path: Path) -> Iterable[dict]:
    for row_idx, record in enumerate(iter_records(input_path)):
        qas = record.get("qas")
        if isinstance(qas, list) and qas:
            for qa_idx, qa in enumerate(qas):
                if not isinstance(qa, dict):
                    continue
                row = build_row(record, row_idx, qa=qa, qa_idx=qa_idx)
                if row is not None:
                    yield row
            continue

        row = build_row(record, row_idx)
        if row is not None:
            yield row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="lecture QA json/jsonl path")
    parser.add_argument("output", help="processed MergePRAG jsonl path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in iter_output_rows(input_path):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[prepare_lecture_qa] saved {kept} rows -> {output_path}")


if __name__ == "__main__":
    main()
