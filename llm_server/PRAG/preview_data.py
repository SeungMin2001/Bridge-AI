"""Print human-readable samples from augmented PRAG data."""

from __future__ import annotations

import argparse
import random
import textwrap

from .config import (
    AUGMENTED_TRAIN_PATH,
    AUGMENTED_VALID_PATH,
    EXTERNAL_QA_AUGMENTED_TRAIN_PATH,
    EXTERNAL_QA_AUGMENTED_VALID_PATH,
    MULTIFACT_AUGMENTED_TRAIN_PATH,
    MULTIFACT_AUGMENTED_VALID_PATH,
    TRANSCRIPT_AUGMENTED_TRAIN_PATH,
    TRANSCRIPT_AUGMENTED_VALID_PATH,
)
from .data import contains_hangul, get_passage, iter_json_records, normalize_qas


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


def row_text(row: dict) -> str:
    fields = [get_passage(row), str(row.get("rewrite") or "")]
    for key in ("atomic_qas", "final_qas"):
        for qa in normalize_qas(row.get(key)):
            fields.extend([
                qa.get("sub_passage", ""),
                qa.get("question", ""),
                qa.get("answer", ""),
                qa.get("full_answer", ""),
            ])
    negatives = row.get("hard_negatives") if isinstance(row.get("hard_negatives"), list) else []
    for neg in negatives:
        if not isinstance(neg, dict):
            continue
        fields.append(get_passage(neg))
        for key in ("atomic_qas", "qas", "final_qas"):
            for qa in normalize_qas(neg.get(key)):
                fields.extend([
                    qa.get("sub_passage", ""),
                    qa.get("question", ""),
                    qa.get("answer", ""),
                    qa.get("full_answer", ""),
                ])
    return "\n".join(str(item or "") for item in fields)


def contains_cjk_ideograph(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def has_korean_artifact(text: str) -> bool:
    lowered = str(text or "").casefold()
    english_list_markers = ("part 1", "part 2", "part 3", "part 1:", "part 2:", "part 3:")
    return contains_cjk_ideograph(lowered) or any(marker in lowered for marker in english_list_markers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=str(AUGMENTED_TRAIN_PATH))
    parser.add_argument(
        "--split",
        choices=("train", "valid"),
        help="Use the default augmented train or valid dataset path.",
    )
    parser.add_argument(
        "--multifact",
        action="store_true",
        help="Preview the default multi-fact augmented train or valid dataset.",
    )
    parser.add_argument(
        "--external-qa",
        action="store_true",
        help="Preview the HotpotQA/KorQuAD external-QA augmented train or valid dataset.",
    )
    parser.add_argument(
        "--transcript",
        action="store_true",
        help="Preview the transcript-style augmented train or valid dataset.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based row index to preview when --random is not used.",
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-qas", type=int, default=4)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--ko-only", action="store_true", help="Preview only rows containing Korean text.")
    parser.add_argument(
        "--clean-ko-only",
        action="store_true",
        help="Preview Korean rows after dropping obvious artifacts such as CJK ideographs or 'part 1' labels.",
    )
    parser.add_argument("--en-only", action="store_true", help="Preview only rows without Korean text.")
    args = parser.parse_args()
    if sum(bool(flag) for flag in (args.ko_only, args.clean_ko_only, args.en_only)) > 1:
        raise ValueError("Use only one of --ko-only, --clean-ko-only, or --en-only.")

    path = args.path
    train_default = AUGMENTED_TRAIN_PATH
    valid_default = AUGMENTED_VALID_PATH
    if args.multifact:
        train_default = MULTIFACT_AUGMENTED_TRAIN_PATH
        valid_default = MULTIFACT_AUGMENTED_VALID_PATH
    if args.external_qa:
        train_default = EXTERNAL_QA_AUGMENTED_TRAIN_PATH
        valid_default = EXTERNAL_QA_AUGMENTED_VALID_PATH
    if args.transcript:
        train_default = TRANSCRIPT_AUGMENTED_TRAIN_PATH
        valid_default = TRANSCRIPT_AUGMENTED_VALID_PATH
    if args.split == "train":
        path = str(train_default)
    elif args.split == "valid":
        path = str(valid_default)
    elif args.multifact or args.external_qa or args.transcript:
        path = str(train_default)

    rows = list(iter_json_records(path))
    ko_rows = [row for row in rows if contains_hangul(row_text(row))]
    clean_ko_rows = [
        row
        for row in ko_rows
        if not has_korean_artifact(row_text(row))
    ]
    en_rows = [row for row in rows if row not in ko_rows]
    print(
        f"[PRAG:preview] path={path} rows={len(rows)} "
        f"ko_like={len(ko_rows)} clean_ko_like={len(clean_ko_rows)} non_ko_like={len(en_rows)}"
    )
    if args.ko_only:
        rows = ko_rows
        print(f"[PRAG:preview] filter=ko_only rows={len(rows)}")
    elif args.clean_ko_only:
        rows = clean_ko_rows
        print(f"[PRAG:preview] filter=clean_ko_only rows={len(rows)}")
    elif args.en_only:
        rows = en_rows
        print(f"[PRAG:preview] filter=en_only rows={len(rows)}")
    if args.random:
        random.shuffle(rows)
        rows = rows[: args.samples]
    else:
        start = max(args.index, 0)
        rows = rows[start : start + args.samples]

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
