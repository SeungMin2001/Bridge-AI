"""Validate augmented PRAG service-memory data before training."""

from __future__ import annotations

import argparse
from collections import Counter

from .config import (
    AIHUB_LECTURE_AUGMENTED_TRAIN_PATH,
    AIHUB_LECTURE_AUGMENTED_VALID_PATH,
    AUGMENTED_TRAIN_PATH,
    AUGMENTED_VALID_PATH,
    EXTERNAL_QA_AUGMENTED_TRAIN_PATH,
    EXTERNAL_QA_AUGMENTED_VALID_PATH,
    KORQUAD_SERVICE_AUGMENTED_TRAIN_PATH,
    KORQUAD_SERVICE_AUGMENTED_VALID_PATH,
    MULTIFACT_AUGMENTED_TRAIN_PATH,
    MULTIFACT_AUGMENTED_VALID_PATH,
    TRANSCRIPT_AUGMENTED_TRAIN_PATH,
    TRANSCRIPT_AUGMENTED_VALID_PATH,
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


def contains_answer(answer: str, text: str) -> bool:
    answer = str(answer or "").strip().casefold()
    text = str(text or "").strip().casefold()
    return bool(answer and text and answer in text)


def starts_with_answer(answer: str, text: str) -> bool:
    answer = str(answer or "").strip().casefold()
    text = str(text or "").strip().casefold()
    return bool(answer and text and text.startswith(answer))


def add_normalized_qa_stats(stats: Counter, prefix: str, qa: dict) -> None:
    answer = qa.get("answer", "")
    full_answer = qa.get("full_answer", "")
    stats[f"{prefix}_answer_in_full_answer"] += int(contains_answer(answer, full_answer))
    stats[f"{prefix}_full_answer_starts_answer"] += int(starts_with_answer(answer, full_answer))


def contains_cjk_ideograph(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def has_korean_artifact(text: str) -> bool:
    lowered = str(text or "").casefold()
    english_list_markers = ("part 1", "part 2", "part 3", "part 1:", "part 2:", "part 3:")
    return contains_cjk_ideograph(lowered) or any(marker in lowered for marker in english_list_markers)


def collect_row_text(row: dict) -> str:
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
        "--external-qa",
        action="store_true",
        help="Validate the HotpotQA/KorQuAD external-QA augmented train/valid files.",
    )
    parser.add_argument(
        "--korquad-service",
        action="store_true",
        help="Validate KorQuAD rewritten as professor-style service transcript data.",
    )
    parser.add_argument(
        "--transcript",
        action="store_true",
        help="Validate the transcript-style augmented train/valid files.",
    )
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()
    if args.multifact:
        args.paths = [str(MULTIFACT_AUGMENTED_TRAIN_PATH), str(MULTIFACT_AUGMENTED_VALID_PATH)]
    if args.aihub_lecture:
        args.paths = [str(AIHUB_LECTURE_AUGMENTED_TRAIN_PATH), str(AIHUB_LECTURE_AUGMENTED_VALID_PATH)]
    if args.external_qa:
        args.paths = [str(EXTERNAL_QA_AUGMENTED_TRAIN_PATH), str(EXTERNAL_QA_AUGMENTED_VALID_PATH)]
    if args.korquad_service:
        args.paths = [str(KORQUAD_SERVICE_AUGMENTED_TRAIN_PATH), str(KORQUAD_SERVICE_AUGMENTED_VALID_PATH)]
    if args.transcript:
        args.paths = [str(TRANSCRIPT_AUGMENTED_TRAIN_PATH), str(TRANSCRIPT_AUGMENTED_VALID_PATH)]

    for path in args.paths:
        rows = list(iter_json_records(path))
        stats = Counter()
        shown = 0
        for row in rows:
            passage = get_passage(row)
            full_text = collect_row_text(row)
            raw_atomic = raw_qas(row.get("atomic_qas"))
            raw_final = raw_qas(row.get("final_qas"))
            atomic = normalize_qas(row.get("atomic_qas"))
            final = normalize_qas(row.get("final_qas"))
            negatives = row.get("hard_negatives") if isinstance(row.get("hard_negatives"), list) else []
            stats["rows"] += 1
            row_has_hangul = contains_hangul(full_text)
            row_has_artifact = has_korean_artifact(full_text)
            stats["ko_rows"] += int(row_has_hangul)
            stats["en_rows"] += int(bool(passage) and not row_has_hangul)
            stats["artifact_rows"] += int(row_has_artifact)
            stats["clean_ko_rows"] += int(row_has_hangul and not row_has_artifact)
            stats["missing_passage"] += int(not passage)
            stats["missing_atomic"] += int(not atomic)
            stats["missing_final"] += int(not final)
            stats["missing_negative"] += int(not negatives)
            add_raw_qa_stats(stats, "atomic", raw_atomic)
            add_raw_qa_stats(stats, "final", raw_final)
            for qa in atomic:
                stats["atomic_qas"] += 1
                add_normalized_qa_stats(stats, "atomic", qa)
                stats["atomic_normalized_missing_sub_passage"] += int(not qa.get("sub_passage"))
                if qa.get("answer") and qa.get("sub_passage"):
                    stats["atomic_answer_in_sub_passage"] += int(qa["answer"].lower() in qa["sub_passage"].lower())
            for qa in final:
                stats["final_qas"] += 1
                add_normalized_qa_stats(stats, "final", qa)
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
                    add_normalized_qa_stats(stats, "negative_atomic", qa)
                    question_mismatch = qa.get("question") != atomic[idx].get("question")
                    row_question_mismatch = row_question_mismatch or question_mismatch
                    stats["negative_atomic_question_mismatch"] += int(question_mismatch)
                    if qa.get("answer") and qa.get("sub_passage"):
                        stats["negative_atomic_answer_in_sub_passage"] += int(qa["answer"].lower() in qa["sub_passage"].lower())
                for idx, qa in enumerate(neg_final[:len(final)]):
                    add_normalized_qa_stats(stats, "negative_final", qa)
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
            answer_in_full = stats["atomic_answer_in_full_answer"] + stats["final_answer_in_full_answer"]
            normalized_qa_total = stats["atomic_qas"] + stats["final_qas"]
            neg_answer_in_full = stats["negative_atomic_answer_in_full_answer"] + stats["negative_final_answer_in_full_answer"]
            neg_normalized_qa_total = stats["negative_atomic_qas"] + stats["negative_final_qas"]
            print(f"  answer_in_full_answer_ratio: {answer_in_full / max(normalized_qa_total, 1):.3f}")
            print(f"  negative_answer_in_full_answer_ratio: {neg_answer_in_full / max(neg_normalized_qa_total, 1):.3f}")
            print(f"  ko_row_ratio: {stats['ko_rows'] / stats['rows']:.3f}")
            print(f"  clean_ko_row_ratio: {stats['clean_ko_rows'] / stats['rows']:.3f}")
            print(f"  artifact_row_ratio: {stats['artifact_rows'] / stats['rows']:.3f}")


if __name__ == "__main__":
    main()
