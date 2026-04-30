"""
Prepare HotpotQA raw JSONL into a weak-MergePRAG training format.

Output rows:
  - hop_fact supervision:
    {"source_id": ..., "task": "hop_fact", "question": ..., "answer": latest_fact, "passage": cumulative_facts, ...}
  - final_qa supervision:
    {"source_id": ..., "task": "final_qa", "question": original_question, "answer": original_answer, "passage": full_facts, ...}

Usage:
  python -m llm_server.mergePRAG.prepare_hotpotqa
"""
from __future__ import annotations

import json
from pathlib import Path


RAW_TRAIN_PATH = Path(r"C:\Users\user\Documents\last_project\data\HotPot_train_raw.jsonl")
RAW_VALID_PATH = Path(r"C:\Users\user\Documents\last_project\data\HotPot_valid_raw.jsonl")
OUT_TRAIN_PATH = Path(r"C:\Users\user\Documents\last_project\data\HotPot_train_processed.jsonl")
OUT_VALID_PATH = Path(r"C:\Users\user\Documents\last_project\data\HotPot_valid_processed.jsonl")
MAX_FACTS = 6


def _build_facts(record: dict) -> list[str]:
    supporting = record.get("supporting_facts") or {}
    context = record.get("context") or {}
    titles = list(supporting.get("title", []))
    sent_ids = list(supporting.get("sent_id", []))
    context_titles = list(context.get("title", []))
    context_sentences = list(context.get("sentences", []))

    context_map = {
        title: sentences
        for title, sentences in zip(context_titles, context_sentences)
        if isinstance(title, str) and isinstance(sentences, list)
    }

    facts: list[str] = []
    seen: set[tuple[str, int]] = set()
    for title, sent_id in zip(titles, sent_ids):
        if not isinstance(title, str) or not isinstance(sent_id, int):
            continue
        sentences = context_map.get(title)
        if not sentences or not (0 <= sent_id < len(sentences)):
            continue
        key = (title, sent_id)
        if key in seen:
            continue
        seen.add(key)
        sentence = str(sentences[sent_id]).strip()
        if sentence:
            facts.append(f"{title}: {sentence}")
        if len(facts) >= MAX_FACTS:
            break
    return facts


def _iter_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_training_rows(record: dict, facts: list[str]) -> list[dict]:
    rows: list[dict] = []
    source_id = record.get("id")
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    task_type = record.get("type")
    level = record.get("level")

    running: list[str] = []
    for hop_idx, fact in enumerate(facts, start=1):
        fact = str(fact).strip()
        if not fact:
            continue
        running.append(fact)
        rows.append({
            "source_id": source_id,
            "task": "hop_fact",
            "question": question,
            "answer": fact,
            "passage": "\n".join(running),
            "facts": list(running),
            "hop_index": hop_idx,
            "num_hops": len(facts),
            "type": task_type,
            "level": level,
        })

    rows.append({
        "source_id": source_id,
        "task": "final_qa",
        "question": question,
        "answer": answer,
        "passage": "\n".join(facts),
        "facts": list(facts),
        "hop_index": len(facts),
        "num_hops": len(facts),
        "type": task_type,
        "level": level,
    })
    return rows


def _prepare_split(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for item in _iter_jsonl(input_path):
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            facts = _build_facts(item)
            if not question or not answer or not facts:
                continue

            for prepared in _build_training_rows(item, facts):
                fout.write(json.dumps(prepared, ensure_ascii=False) + "\n")
                kept += 1
    print(f"[prepare] saved {kept} rows -> {output_path}")


def main() -> None:
    _prepare_split(RAW_TRAIN_PATH, OUT_TRAIN_PATH)
    _prepare_split(RAW_VALID_PATH, OUT_VALID_PATH)
    print("[prepare] done")


if __name__ == "__main__":
    main()
