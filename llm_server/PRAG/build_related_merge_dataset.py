"""Build related-passage PRAG train/valid/test splits for merge-aware training.

The resulting JSONL schema intentionally matches the existing augmented PRAG
format, but each row is organized as one coherent topic with several related
atomic facts. This makes orthogonal merge training/evaluation reflect the
intended setting: multiple passages about the same subject are fused, then a
question selects the relevant fact.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .build_natural_testset import build_rows
from .data import write_jsonl


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_PREFIX = "PRAG_related_merge"


def split_rows(rows: list[dict], train_count: int, valid_count: int, test_count: int) -> tuple[list[dict], list[dict], list[dict]]:
    expected = train_count + valid_count + test_count
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, got {len(rows)} rows")
    train = rows[:train_count]
    valid = rows[train_count: train_count + valid_count]
    test = rows[train_count + valid_count:]
    return train, valid, test


def write_split_summary(path: Path, rows: list[dict]) -> None:
    atomic_qas = sum(len(row.get("atomic_qas") or []) for row in rows)
    final_qas = sum(len(row.get("final_qas") or []) for row in rows)
    print(f"[PRAG:related-merge] {path.name}: rows={len(rows)} atomic={atomic_qas} final={final_qas} qa_total={atomic_qas + final_qas}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=3200)
    parser.add_argument("--valid-count", type=int, default=400)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--output-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle complete topic rows before splitting. Source ids remain unique.",
    )
    args = parser.parse_args()

    total = args.train_count + args.valid_count + args.test_count
    rows = build_rows(total)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    train, valid, test = split_rows(rows, args.train_count, args.valid_count, args.test_count)

    output_dir = Path(args.output_dir)
    train_path = output_dir / f"{args.prefix}_train.jsonl"
    valid_path = output_dir / f"{args.prefix}_valid.jsonl"
    test_path = output_dir / f"{args.prefix}_test.jsonl"

    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    write_jsonl(test_path, test)

    print(
        "[PRAG:related-merge] wrote coherent related-passage splits "
        f"train={args.train_count} valid={args.valid_count} test={args.test_count}"
    )
    write_split_summary(train_path, train)
    write_split_summary(valid_path, valid)
    write_split_summary(test_path, test)
    print(f"[PRAG:related-merge] train={train_path}")
    print(f"[PRAG:related-merge] valid={valid_path}")
    print(f"[PRAG:related-merge] test={test_path}")


if __name__ == "__main__":
    main()
