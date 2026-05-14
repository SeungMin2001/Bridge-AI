"""Single-hop service-memory dataset helpers.

The paper code supports multi-hop QA datasets such as HotPotQA. Our product
goal is different: each professor utterance is one service memory item, and the
hypernetwork should learn one passage -> one K/V memory for later QA.

This loader intentionally keeps that contract narrow. It accepts rows with:
  question, answer, passage, hard_negatives[0].passage, hard_negatives[0].answer
and skips rows that do not contain a usable single-hop hard pair.
"""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset


def iter_records(dataset_path: str):
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

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
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array dataset: {dataset_path}")
        yield from data
        return

    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def extract_answer(sample: dict) -> str:
    for key in ("answer", "target", "output", "response"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    answers = sample.get("answers")
    if isinstance(answers, list):
        for item in answers:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def extract_passage(sample: dict) -> str:
    passage = sample.get("passage")
    if isinstance(passage, str) and passage.strip():
        return passage.strip()

    for key in ("utterance", "text", "content", "transcript", "chunk_text"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            speaker = str(sample.get("speaker") or "").strip()
            text = value.strip()
            return f"{speaker}: {text}" if speaker else text
    return ""


def extract_first_hard_negative(sample: dict) -> dict | None:
    for key in ("hard_negatives", "negative_passages", "counterfactuals", "distractors"):
        values = sample.get(key)
        if not isinstance(values, list) or not values:
            continue
        item = values[0]
        if isinstance(item, dict):
            passage = extract_passage(item)
            answer = extract_answer(item)
            if passage and answer:
                return {"passage": passage, "answer": answer}
        elif isinstance(item, str) and item.strip():
            answer = str(
                sample.get("negative_answer")
                or sample.get("counterfactual_answer")
                or sample.get("distractor_answer")
                or ""
            ).strip()
            if answer:
                return {"passage": item.strip(), "answer": answer}

    for key in ("hard_negative_passage", "negative_passage", "counterfactual_passage", "distractor_passage"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            answer = str(
                sample.get("negative_answer")
                or sample.get("counterfactual_answer")
                or sample.get("distractor_answer")
                or ""
            ).strip()
            if answer:
                return {"passage": value.strip(), "answer": answer}
    return None


class ServiceMemoryDataset(Dataset):
    """Strict single-hop hard-pair dataset for train_service_memory.py."""

    def __init__(self, dataset_path: str, max_samples: int | None = None):
        self.data = []
        skipped = 0
        for i, item in enumerate(iter_records(dataset_path)):
            if max_samples and i >= max_samples:
                break

            question = str(item.get("question", "")).strip()
            answer = extract_answer(item)
            passage = extract_passage(item)
            negative = extract_first_hard_negative(item)
            if not (question and answer and passage and negative):
                skipped += 1
                continue
            if not (negative.get("passage") and negative.get("answer")):
                skipped += 1
                continue

            self.data.append({
                **item,
                "question": question,
                "answer": answer,
                "passage": passage,
                "hard_negatives": [negative],
                "num_hops": 1,
            })

        print(f"[service:data] loaded {len(self.data)} single-hop hard pairs from {dataset_path}")
        if skipped:
            print(f"[service:data] skipped {skipped} rows without a complete single-hop hard pair")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
