"""
Download HotpotQA from Hugging Face and save raw JSONL files locally.

Usage:
  python -m llm_server.mergePRAG.download_hotpotqa
"""
from __future__ import annotations

import json
from pathlib import Path


RAW_TRAIN_PATH = Path(r"C:\Users\user\Documents\last_project\data\HotPot_train_raw.jsonl")
RAW_VALID_PATH = Path(r"C:\Users\user\Documents\last_project\data\HotPot_valid_raw.jsonl")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_plain_record(example: dict) -> dict:
    supporting = example.get("supporting_facts") or {}
    context = example.get("context") or {}
    return {
        "id": example.get("id"),
        "question": example.get("question"),
        "answer": example.get("answer"),
        "type": example.get("type"),
        "level": example.get("level"),
        "supporting_facts": {
            "title": list(supporting.get("title", [])),
            "sent_id": list(supporting.get("sent_id", [])),
        },
        "context": {
            "title": list(context.get("title", [])),
            "sentences": list(context.get("sentences", [])),
        },
    }


def _dump_split(dataset, output_path: Path) -> None:
    _ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(_to_plain_record(item), ensure_ascii=False) + "\n")
    print(f"[download] saved {len(dataset)} rows -> {output_path}")


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets package is required. Install it with: pip install datasets"
        ) from exc

    print("[download] loading HotpotQA (distractor) from Hugging Face...")
    train_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    valid_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    _dump_split(train_dataset, RAW_TRAIN_PATH)
    _dump_split(valid_dataset, RAW_VALID_PATH)
    print("[download] done")


if __name__ == "__main__":
    main()
