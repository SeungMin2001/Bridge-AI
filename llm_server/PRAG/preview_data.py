"""Print human-readable samples from augmented PRAG data."""

from __future__ import annotations

import argparse
import random
import textwrap

from .config import AUGMENTED_TRAIN_PATH
from .data import get_passage, iter_json_records, normalize_qas


def clip(text: str, width: int = 140) -> str:
    text = " ".join(str(text or "").split())
    return textwrap.shorten(text, width=width, placeholder="...")


def print_qa_list(title: str, qas: list[dict], max_items: int, width: int) -> None:
    print(f"\n  [{title}] count={len(qas)}")
    for idx, qa in enumerate(qas[:max_items], start=1):
        sub_passage = qa.get("sub_passage")
        if sub_passage:
            print(f"    {idx}. sub_passage: {clip(sub_passage, width)}")
        print(f"       Q: {clip(qa.get('question'), width)}")
        print(f"       A: {clip(qa.get('answer'), width)}")
        full_answer = qa.get("full_answer")
        if full_answer:
            print(f"       full: {clip(full_answer, width)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=str(AUGMENTED_TRAIN_PATH))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-qas", type=int, default=4)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    rows = list(iter_json_records(args.path))
    if args.random:
        random.shuffle(rows)
    rows = rows[: args.samples]

    for row_idx, row in enumerate(rows, start=1):
        print("\n" + "=" * 90)
        print(f"[Sample {row_idx}] source_id={row.get('source_id')}")
        print(f"passage: {clip(get_passage(row), args.width)}")
        if row.get("rewrite"):
            print(f"rewrite: {clip(row.get('rewrite'), args.width)}")

        print_qa_list("atomic_qas", normalize_qas(row.get("atomic_qas")), args.max_qas, args.width)
        print_qa_list("final_qas", normalize_qas(row.get("final_qas")), args.max_qas, args.width)

        negatives = row.get("hard_negatives") if isinstance(row.get("hard_negatives"), list) else []
        print(f"\n  [hard_negatives] count={len(negatives)}")
        for neg_idx, neg in enumerate(negatives[:2], start=1):
            if not isinstance(neg, dict):
                continue
            print(f"    negative {neg_idx} passage: {clip(get_passage(neg), args.width)}")
            if neg.get("answer"):
                print(f"    negative {neg_idx} fallback answer: {clip(neg.get('answer'), args.width)}")
            neg_atomic = normalize_qas(neg.get("atomic_qas")) + normalize_qas(neg.get("qas"))
            neg_final = normalize_qas(neg.get("final_qas"))
            print_qa_list(f"negative {neg_idx} atomic_qas", neg_atomic, args.max_qas, args.width)
            print_qa_list(f"negative {neg_idx} final_qas", neg_final, args.max_qas, args.width)


if __name__ == "__main__":
    main()
