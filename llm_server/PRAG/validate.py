"""Validate augmented PRAG service-memory data before training."""

from __future__ import annotations

import argparse
from collections import Counter

from .config import AUGMENTED_TRAIN_PATH, AUGMENTED_VALID_PATH, MULTIFACT_AUGMENTED_TRAIN_PATH, MULTIFACT_AUGMENTED_VALID_PATH
from .data import get_passage, iter_json_records, normalize_qas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=[str(AUGMENTED_TRAIN_PATH), str(AUGMENTED_VALID_PATH)])
    parser.add_argument(
        "--multifact",
        action="store_true",
        help="Validate the default multi-fact augmented train/valid files.",
    )
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()
    if args.multifact:
        args.paths = [str(MULTIFACT_AUGMENTED_TRAIN_PATH), str(MULTIFACT_AUGMENTED_VALID_PATH)]

    for path in args.paths:
        rows = list(iter_json_records(path))
        stats = Counter()
        shown = 0
        for row in rows:
            passage = get_passage(row)
            atomic = normalize_qas(row.get("atomic_qas"))
            final = normalize_qas(row.get("final_qas"))
            negatives = row.get("hard_negatives") if isinstance(row.get("hard_negatives"), list) else []
            stats["rows"] += 1
            stats["missing_passage"] += int(not passage)
            stats["missing_atomic"] += int(not atomic)
            stats["missing_final"] += int(not final)
            stats["missing_negative"] += int(not negatives)
            for qa in atomic:
                stats["atomic_qas"] += 1
                stats["atomic_missing_sub_passage"] += int(not qa.get("sub_passage"))
                if qa.get("answer") and qa.get("sub_passage"):
                    stats["atomic_answer_in_sub_passage"] += int(qa["answer"].lower() in qa["sub_passage"].lower())
            for qa in final:
                stats["final_qas"] += 1
            for neg in negatives:
                if not isinstance(neg, dict):
                    continue
                neg_passage = get_passage(neg)
                neg_atomic = normalize_qas(neg.get("atomic_qas")) + normalize_qas(neg.get("qas"))
                neg_final = normalize_qas(neg.get("final_qas"))
                stats["negative_passages"] += int(bool(neg_passage))
                stats["negative_atomic_qas"] += len(neg_atomic)
                stats["negative_final_qas"] += len(neg_final)
            if shown < args.show and (not atomic or any(not qa.get("sub_passage") for qa in atomic) or not negatives):
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
            print(f"  missing_atomic_ratio: {missing_atomic_ratio:.3f}")
            print(f"  missing_negative_ratio: {missing_negative_ratio:.3f}")


if __name__ == "__main__":
    main()
