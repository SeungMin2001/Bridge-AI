"""Validate augmented PRAG service-memory data before training."""

from __future__ import annotations

import argparse
from collections import Counter

from .config import (
    AIHUB_LECTURE_AUGMENTED_TRAIN_PATH,
    AIHUB_LECTURE_AUGMENTED_VALID_PATH,
    AUGMENTED_TRAIN_PATH,
    AUGMENTED_VALID_PATH,
    KO_CONTENT_AUGMENTED_TRAIN_PATH,
    KO_CONTENT_AUGMENTED_VALID_PATH,
    MULTIFACT_AUGMENTED_TRAIN_PATH,
    MULTIFACT_AUGMENTED_VALID_PATH,
)
from .data import contains_hangul, get_passage, iter_json_records, normalize_qas


def raw_qas(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def add_raw_qa_stats(stats: Counter, prefix: str, qas: list[dict]) -> None:
    for qa in qas:
        stats[f"{prefix}_raw_qas"] += 1
        stats[f"{prefix}_missing_question"] += int(not str(qa.get("question") or qa.get("sub_question") or "").strip())
        stats[f"{prefix}_missing_answer"] += int(not str(qa.get("answer") or qa.get("sub_answer") or "").strip())
        stats[f"{prefix}_missing_full_answer"] += int(not str(qa.get("full_answer") or "").strip())
        stats[f"{prefix}_missing_sub_passage"] += int(
            not str(qa.get("sub_passage") or qa.get("evidence") or qa.get("passage") or "").strip()
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=[str(AUGMENTED_TRAIN_PATH), str(AUGMENTED_VALID_PATH)])
    parser.add_argument(
        "--multifact",
        action="store_true",
        help="Validate the default multi-fact augmented train/valid files.",
    )
    parser.add_argument(
        "--aihub-lecture",
        action="store_true",
        help="Validate the AIHub university lecture augmented train/valid files.",
    )
    parser.add_argument(
        "--ko-content",
        action="store_true",
        help="Validate the Korean content-inspired augmented train/valid files.",
    )
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()
    if args.multifact:
        args.paths = [str(MULTIFACT_AUGMENTED_TRAIN_PATH), str(MULTIFACT_AUGMENTED_VALID_PATH)]
    if args.aihub_lecture:
        args.paths = [str(AIHUB_LECTURE_AUGMENTED_TRAIN_PATH), str(AIHUB_LECTURE_AUGMENTED_VALID_PATH)]
    if args.ko_content:
        args.paths = [str(KO_CONTENT_AUGMENTED_TRAIN_PATH), str(KO_CONTENT_AUGMENTED_VALID_PATH)]

    for path in args.paths:
        rows = list(iter_json_records(path))
        stats = Counter()
        shown = 0
        for row in rows:
            passage = get_passage(row)
            raw_atomic = raw_qas(row.get("atomic_qas"))
            raw_final = raw_qas(row.get("final_qas"))
            atomic = normalize_qas(row.get("atomic_qas"))
            final = normalize_qas(row.get("final_qas"))
            negatives = row.get("hard_negatives") if isinstance(row.get("hard_negatives"), list) else []
            stats["rows"] += 1
            stats["ko_rows"] += int(contains_hangul(passage))
            stats["en_rows"] += int(bool(passage) and not contains_hangul(passage))
            stats["missing_passage"] += int(not passage)
            stats["missing_atomic"] += int(not atomic)
            stats["missing_final"] += int(not final)
            stats["missing_negative"] += int(not negatives)
            add_raw_qa_stats(stats, "atomic", raw_atomic)
            add_raw_qa_stats(stats, "final", raw_final)
            for qa in atomic:
                stats["atomic_qas"] += 1
                stats["atomic_normalized_missing_sub_passage"] += int(not qa.get("sub_passage"))
                if qa.get("answer") and qa.get("sub_passage"):
                    stats["atomic_answer_in_sub_passage"] += int(qa["answer"].lower() in qa["sub_passage"].lower())
            for qa in final:
                stats["final_qas"] += 1
            row_neg_mismatch = False
            row_question_mismatch = False
            for neg in negatives:
                if not isinstance(neg, dict):
                    continue
                neg_passage = get_passage(neg)
                raw_neg_atomic = raw_qas(neg.get("atomic_qas")) + raw_qas(neg.get("qas"))
                raw_neg_final = raw_qas(neg.get("final_qas"))
                neg_atomic = normalize_qas(neg.get("atomic_qas")) + normalize_qas(neg.get("qas"))
                neg_final = normalize_qas(neg.get("final_qas"))
                add_raw_qa_stats(stats, "negative_atomic", raw_neg_atomic)
                add_raw_qa_stats(stats, "negative_final", raw_neg_final)
                stats["negative_passages"] += int(bool(neg_passage))
                stats["negative_atomic_qas"] += len(neg_atomic)
                stats["negative_final_qas"] += len(neg_final)
                atomic_count_mismatch = len(neg_atomic) != len(atomic)
                final_count_mismatch = len(neg_final) != len(final)
                row_neg_mismatch = row_neg_mismatch or atomic_count_mismatch or final_count_mismatch
                stats["negative_atomic_count_mismatch"] += int(atomic_count_mismatch)
                stats["negative_final_count_mismatch"] += int(final_count_mismatch)
                for idx, qa in enumerate(neg_atomic[:len(atomic)]):
                    question_mismatch = qa.get("question") != atomic[idx].get("question")
                    row_question_mismatch = row_question_mismatch or question_mismatch
                    stats["negative_atomic_question_mismatch"] += int(question_mismatch)
                    if qa.get("answer") and qa.get("sub_passage"):
                        stats["negative_atomic_answer_in_sub_passage"] += int(qa["answer"].lower() in qa["sub_passage"].lower())
                for idx, qa in enumerate(neg_final[:len(final)]):
                    question_mismatch = qa.get("question") != final[idx].get("question")
                    row_question_mismatch = row_question_mismatch or question_mismatch
                    stats["negative_final_question_mismatch"] += int(question_mismatch)
            warn = (
                not atomic
                or any(not qa.get("sub_passage") for qa in atomic)
                or not negatives
                or row_neg_mismatch
                or row_question_mismatch
            )
            if shown < args.show and warn:
                shown += 1
                print(f"\n[warn example] {path}")
                print(f"  source_id={row.get('source_id')}")
                print(f"  passage={passage[:160]}")
                print(f"  atomic={atomic[:2]}")
                print(f"  negatives={negatives[:1]}")

        print(f"\n[PRAG:validate] {path}")
        for key in sorted(stats):
            print(f"  {key}: {stats[key]}")
        if stats["rows"]:
            missing_atomic_ratio = stats["missing_atomic"] / stats["rows"]
            missing_negative_ratio = stats["missing_negative"] / stats["rows"]
            raw_qa_total = stats["atomic_raw_qas"] + stats["final_raw_qas"]
            raw_full_missing = stats["atomic_missing_full_answer"] + stats["final_missing_full_answer"]
            neg_raw_qa_total = stats["negative_atomic_raw_qas"] + stats["negative_final_raw_qas"]
            neg_raw_full_missing = stats["negative_atomic_missing_full_answer"] + stats["negative_final_missing_full_answer"]
            print(f"  missing_atomic_ratio: {missing_atomic_ratio:.3f}")
            print(f"  missing_negative_ratio: {missing_negative_ratio:.3f}")
            print(f"  raw_full_answer_missing_ratio: {raw_full_missing / max(raw_qa_total, 1):.3f}")
            print(f"  negative_raw_full_answer_missing_ratio: {neg_raw_full_missing / max(neg_raw_qa_total, 1):.3f}")
            print(f"  ko_row_ratio: {stats['ko_rows'] / stats['rows']:.3f}")


if __name__ == "__main__":
    main()
