"""Split augmented entity-multifact rows by root_source_id."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_by_root(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        root = str(row.get("root_source_id") or row.get("source_id"))
        groups[root].append(row)
    return groups


def qa_pairs(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    pairs = set()
    for row in rows:
        if row.get("question") and row.get("answer"):
            pairs.add((str(row["question"]).strip(), str(row["answer"]).strip()))
        for qa in row.get("final_qas", []):
            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer") or qa.get("full_answer") or "").strip()
            if question or answer:
                pairs.add((question, answer))
    return pairs


def source_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("source_id", "")) for row in rows}


def flatten(root_keys: list[str], groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in root_keys:
        out.extend(groups[key])
    return out


def split_roots(
    roots: list[str],
    train_roots: int,
    valid_roots: int,
    test_roots: int,
) -> tuple[list[str], list[str], list[str]]:
    needed = train_roots + valid_roots + test_roots
    if needed > len(roots):
        raise SystemExit(f"requested roots={needed} but only {len(roots)} root groups exist")
    train = roots[:train_roots]
    valid = roots[train_roots : train_roots + valid_roots]
    test = roots[train_roots + valid_roots : needed]
    return train, valid, test


def print_stats(name: str, rows: list[dict[str, Any]]) -> None:
    simple_qas = sum(1 for row in rows if row.get("question") and row.get("answer"))
    final_qas = simple_qas + sum(len(row.get("final_qas", [])) for row in rows)
    atomic_qas = sum(len(row.get("atomic_qas", [])) for row in rows)
    passages = sum(len(row.get("passages", [])) for row in rows if isinstance(row.get("passages"), list))
    roots = {str(row.get("root_source_id") or row.get("source_id")) for row in rows}
    print(
        f"[PRAG:entity-split] {name}: rows={len(rows)} roots={len(roots)} "
        f"passages={passages} atomic_qas={atomic_qas} qas={final_qas}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split entity-multifact rows without root leakage.")
    parser.add_argument("--input", type=Path, default=Path("data/PRAG_entity_multifact_augmented_all.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("data/PRAG_entity_multifact_augmented_train.jsonl"))
    parser.add_argument("--valid-output", type=Path, default=Path("data/PRAG_entity_multifact_augmented_valid.jsonl"))
    parser.add_argument("--test-output", type=Path, default=Path("data/PRAG_entity_multifact_augmented_test.jsonl"))
    parser.add_argument("--train-roots", type=int, default=3200)
    parser.add_argument("--valid-roots", type=int, default=400)
    parser.add_argument("--test-roots", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    groups = group_by_root(rows)
    roots = sorted(groups)
    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(roots)

    train_keys, valid_keys, test_keys = split_roots(roots, args.train_roots, args.valid_roots, args.test_roots)
    train_rows = flatten(train_keys, groups)
    valid_rows = flatten(valid_keys, groups)
    test_rows = flatten(test_keys, groups)

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.valid_output, valid_rows)
    write_jsonl(args.test_output, test_rows)

    print_stats("train", train_rows)
    print_stats("valid", valid_rows)
    print_stats("test", test_rows)

    train_sources = source_ids(train_rows)
    valid_sources = source_ids(valid_rows)
    test_sources = source_ids(test_rows)
    train_qas = qa_pairs(train_rows)
    valid_qas = qa_pairs(valid_rows)
    test_qas = qa_pairs(test_rows)
    print(f"[PRAG:entity-split] source overlap test-train={len(test_sources & train_sources)}")
    print(f"[PRAG:entity-split] source overlap test-valid={len(test_sources & valid_sources)}")
    print(f"[PRAG:entity-split] QA overlap test-train={len(test_qas & train_qas)}")
    print(f"[PRAG:entity-split] QA overlap test-valid={len(test_qas & valid_qas)}")


if __name__ == "__main__":
    main()
