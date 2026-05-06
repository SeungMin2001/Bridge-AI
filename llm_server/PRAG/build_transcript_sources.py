"""Build diverse fact bundles for transcript-style PRAG augmentation.

The output is not the final training data. Each row is a compact set of facts
that a stronger LLM rewrites into realistic lecture/meeting transcript speech.
Keeping this source file separate lets us regenerate transcript-style data
without touching the existing multi-fact augmented dataset.
"""

from __future__ import annotations

import argparse
import random

from .build_multifact_sources import DOMAIN_FACTS, SPEAKERS
from .config import TRANSCRIPT_SOURCE_PATH
from .data import write_jsonl


SPEECH_PATTERNS = [
    "concept_explanation",
    "analogy",
    "assignment_notice",
    "exam_notice",
    "lab_instruction",
    "project_guidance",
    "meeting_decision",
    "operation_notice",
    "definition_then_example",
    "common_mistake_correction",
]


def ko_template(kind: str, subject: str, facts: list[tuple[str, str]], speaker: str) -> str:
    joined = "; ".join(f"{key}: {value}" for key, value in facts)
    if kind == "analogy":
        return f"{speaker}는 {subject} 설명에서 다음 핵심 내용을 실제 비유를 섞어 말해야 한다: {joined}"
    if kind in {"assignment_notice", "exam_notice", "lab_instruction"}:
        return f"{speaker}는 {subject} 수업 공지처럼 말하며 다음 내용을 알려야 한다: {joined}"
    if kind in {"meeting_decision", "operation_notice"}:
        return f"{speaker}는 {subject} 회의 발화처럼 말하며 결정사항을 알려야 한다: {joined}"
    if kind == "common_mistake_correction":
        return f"{speaker}는 {subject}에서 헷갈리기 쉬운 내용을 바로잡으며 다음 내용을 설명해야 한다: {joined}"
    return f"{speaker}는 {subject} 수업에서 실제 강의처럼 다음 내용을 설명해야 한다: {joined}"


def en_template(kind: str, subject: str, facts: list[tuple[str, str]], speaker: str) -> str:
    joined = "; ".join(f"{key}: {value}" for key, value in facts)
    if kind == "analogy":
        return f"{speaker} should explain these {subject} points with a natural classroom analogy: {joined}"
    if kind in {"assignment_notice", "exam_notice", "lab_instruction"}:
        return f"{speaker} should announce these {subject} class details in a realistic spoken style: {joined}"
    if kind in {"meeting_decision", "operation_notice"}:
        return f"{speaker} should state these {subject} meeting decisions in a realistic spoken style: {joined}"
    if kind == "common_mistake_correction":
        return f"{speaker} should correct a common misconception in {subject} while explaining: {joined}"
    return f"{speaker} should explain these {subject} points like a real lecture transcript: {joined}"


def build_rows(rows_per_domain: int, facts_per_passage: int, seed: int, languages: list[str]) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for domain_idx, (scene, domain, ko_subject, en_subject, facts) in enumerate(DOMAIN_FACTS):
        for row_idx in range(rows_per_domain):
            lang = languages[(domain_idx + row_idx) % len(languages)]
            speaker = rng.choice(SPEAKERS[lang])
            pattern = rng.choice(SPEECH_PATTERNS)
            sampled = rng.sample(facts, k=min(facts_per_passage, len(facts)))
            positive_facts = []
            negative_facts = []
            for ko_key, en_key, ko_answer, en_answer, ko_neg, en_neg in sampled:
                if lang == "ko":
                    positive_facts.append((ko_key, ko_answer))
                    negative_facts.append((ko_key, ko_neg))
                else:
                    positive_facts.append((en_key, en_answer))
                    negative_facts.append((en_key, en_neg))
            subject = ko_subject if lang == "ko" else en_subject
            passage = (
                ko_template(pattern, subject, positive_facts, speaker)
                if lang == "ko"
                else en_template(pattern, subject, positive_facts, speaker)
            )
            negative_passage = (
                ko_template(pattern, subject, negative_facts, speaker)
                if lang == "ko"
                else en_template(pattern, subject, negative_facts, speaker)
            )
            rows.append({
                "source_id": f"transcript_{scene}_{lang}_{domain}_{pattern}_{row_idx}",
                "scene": scene,
                "domain": domain,
                "subject": subject,
                "language": lang,
                "speaker": speaker,
                "speech_pattern": pattern,
                "facts": [{"key": key, "answer": answer} for key, answer in positive_facts],
                "negative_facts": [{"key": key, "answer": answer} for key, answer in negative_facts],
                "passage": passage,
                "hard_negatives": [{"passage": negative_passage}],
            })
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(TRANSCRIPT_SOURCE_PATH))
    parser.add_argument("--rows-per-domain", type=int, default=80)
    parser.add_argument("--facts-per-passage", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--languages",
        default="ko,en",
        help="Comma-separated language mix. Use 'ko' for Korean-only.",
    )
    args = parser.parse_args()
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    if not languages:
        raise ValueError("--languages must contain at least one language code")
    unsupported = sorted(set(languages) - {"ko", "en"})
    if unsupported:
        raise ValueError(f"Unsupported languages: {unsupported}")
    rows = build_rows(args.rows_per_domain, args.facts_per_passage, args.seed, languages)
    write_jsonl(args.output, rows)
    ko_rows = sum(1 for row in rows if row["language"] == "ko")
    en_rows = len(rows) - ko_rows
    print(f"[PRAG:transcript-source] rows={len(rows)} ko={ko_rows} en={en_rows} -> {args.output}")


if __name__ == "__main__":
    main()
