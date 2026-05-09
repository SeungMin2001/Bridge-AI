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


def ko_sentence(key: str, value: str) -> str:
    return f"{key}은 {value}입니다"


def en_sentence(key: str, value: str) -> str:
    return f"{key} is {value}"


def ko_template(kind: str, subject: str, facts: list[tuple[str, str]], speaker: str) -> str:
    first = facts[0] if facts else ("핵심 내용", "")
    rest = facts[1:]
    rest_text = " ".join(f"그리고 {ko_sentence(key, value)}." for key, value in rest)
    if kind == "analogy":
        return (
            f"자, {subject}를 비유로 한번 설명해볼게요. "
            f"{first[0]}은 쉽게 말하면 {first[1]}라고 보면 됩니다. {rest_text}"
        ).strip()
    if kind in {"assignment_notice", "exam_notice", "lab_instruction"}:
        return (
            f"자, {subject} 수업 공지 잠깐 할게요. "
            f"{ko_sentence(first[0], first[1])}. {rest_text} "
            f"이 부분은 헷갈리지 않게 꼭 확인해 주세요."
        ).strip()
    if kind in {"meeting_decision", "operation_notice"}:
        return (
            f"오늘 {subject} 회의에서 정리된 내용부터 말할게요. "
            f"{ko_sentence(first[0], first[1])}. {rest_text} "
            f"회의록에도 이 내용 그대로 남기면 됩니다."
        ).strip()
    if kind == "common_mistake_correction":
        return (
            f"{subject}에서 많이 헷갈리는 부분을 바로잡을게요. "
            f"{ko_sentence(first[0], first[1])}. {rest_text} "
            f"다른 값으로 기억하면 안 됩니다."
        ).strip()
    return (
        f"자, 오늘 {subject}에서 꼭 기억해야 할 내용을 보겠습니다. "
        f"{ko_sentence(first[0], first[1])}. {rest_text}"
    ).strip()


def en_template(kind: str, subject: str, facts: list[tuple[str, str]], speaker: str) -> str:
    first = facts[0] if facts else ("key point", "")
    rest = facts[1:]
    rest_text = " ".join(f"Also, {en_sentence(key, value)}." for key, value in rest)
    if kind == "analogy":
        return (
            f"Okay, let me explain {subject} with a simple analogy. "
            f"{en_sentence(first[0], first[1])}. {rest_text}"
        ).strip()
    if kind in {"assignment_notice", "exam_notice", "lab_instruction"}:
        return (
            f"Okay, quick {subject} class notice. "
            f"{en_sentence(first[0], first[1])}. {rest_text} Please check this carefully."
        ).strip()
    if kind in {"meeting_decision", "operation_notice"}:
        return (
            f"Let me summarize what we decided in the {subject} meeting. "
            f"{en_sentence(first[0], first[1])}. {rest_text}"
        ).strip()
    if kind == "common_mistake_correction":
        return (
            f"Here is the part people often mix up in {subject}. "
            f"{en_sentence(first[0], first[1])}. {rest_text}"
        ).strip()
    return (
        f"Okay, here are the key points for {subject}. "
        f"{en_sentence(first[0], first[1])}. {rest_text}"
    ).strip()


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
