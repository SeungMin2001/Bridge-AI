"""
Download SQuAD v1.1 from Hugging Face and save processed JSONL for MergePRAG training.

왜 SQuAD인가:
  논문 저자는 HotpotQA를 decompose해서 (질문, 단일 passage, 답) 삼중항으로 학습함 (passage→answer 결정성 확보).
  HotpotQA를 그대로 쓰면 multi-hop 질문에 single-hop passage를 주어 task CE가 passage별 K/V를 학습시키지 못함.
  SQuAD는 원래부터 (question, passage, answer) 삼중항이라 decompose 없이 조건을 만족한다.

Output rows (MergePRAGDataset 포맷과 호환):
  {"source_id": ..., "task": "final_qa", "question": ..., "answer": ..., "passage": ...}

Usage:
  python -m llm_server.mergePRAG.prepare_squad
"""
from __future__ import annotations

import json
from pathlib import Path


OUT_TRAIN_PATH = Path(r"C:\Users\user\Documents\last_project\data\SQuAD_train_processed.jsonl")
OUT_VALID_PATH = Path(r"C:\Users\user\Documents\last_project\data\SQuAD_valid_processed.jsonl")


def _to_training_row(example: dict) -> dict | None:
    question = str(example.get("question", "")).strip()
    context = str(example.get("context", "")).strip()
    answers = example.get("answers") or {}
    texts = answers.get("text") if isinstance(answers, dict) else None
    answer = ""
    if isinstance(texts, list):
        for text in texts:
            if isinstance(text, str) and text.strip():
                answer = text.strip()
                break
    if not question or not context or not answer:
        return None
    return {
        "source_id": example.get("id"),
        "task": "final_qa",
        "question": question,
        "answer": answer,
        "passage": context,
        "title": example.get("title"),
    }


def _dump_split(dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with output_path.open("w", encoding="utf-8") as f:
        for item in dataset:
            row = _to_training_row(item)
            if row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[prepare_squad] saved {kept} rows -> {output_path}")


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets package is required. Install it with: pip install datasets"
        ) from exc

    print("[prepare_squad] loading SQuAD v1.1 from Hugging Face...")
    train_dataset = load_dataset("rajpurkar/squad", split="train")
    valid_dataset = load_dataset("rajpurkar/squad", split="validation")

    _dump_split(train_dataset, OUT_TRAIN_PATH)
    _dump_split(valid_dataset, OUT_VALID_PATH)
    print("[prepare_squad] done")


if __name__ == "__main__":
    main()
