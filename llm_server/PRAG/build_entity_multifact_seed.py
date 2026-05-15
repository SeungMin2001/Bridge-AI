"""Build seed rows for entity-level multi-fact PRAG training.

The seed file is intentionally not the final dataset.  It gives the LLM
augmentation step a clean, structured starting point where each row contains
several related facts about one entity and final questions ask for the combined
description rather than one isolated fact.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

try:
    from entity_multifact_bank import (
        ENTITY_BANK,
        QUESTION_TEMPLATES,
        SENTENCE_TEMPLATES,
        SPEAKERS,
        build_full_answer,
        context_for_index,
        quoted,
    )
except ModuleNotFoundError:
    from llm_server.PRAG.entity_multifact_bank import (
        ENTITY_BANK,
        QUESTION_TEMPLATES,
        SENTENCE_TEMPLATES,
        SPEAKERS,
        build_full_answer,
        context_for_index,
        quoted,
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fact_answer(entity: str, slot: str, value: str) -> str:
    return f"{entity} 설명에서 {slot} 관련 내용은 {quoted(value)} 설명했습니다."


def build_entity_name(base_entity: str, index: int) -> str:
    return f"{context_for_index(index)} {base_entity}"


def rotate_facts(facts: list[tuple[str, str]], index: int, facts_per_row: int) -> list[tuple[str, str]]:
    if facts_per_row >= len(facts):
        return list(facts)
    offset = index % len(facts)
    return [facts[(offset + idx) % len(facts)] for idx in range(facts_per_row)]


def build_row(index: int, source_prefix: str, facts_per_row: int, final_variants: int) -> dict[str, Any]:
    domain, base_entity, facts = ENTITY_BANK[index % len(ENTITY_BANK)]
    entity = build_entity_name(base_entity, index)
    selected_facts = rotate_facts(facts, index // len(ENTITY_BANK), facts_per_row)
    source_id = f"{source_prefix}_{domain}_{index:05d}"

    atomic_qas: list[dict[str, str]] = []
    sub_passages: list[str] = []
    for fact_idx, (slot, value) in enumerate(selected_facts):
        speaker = SPEAKERS[(index + fact_idx) % len(SPEAKERS)]
        template = SENTENCE_TEMPLATES[(index + fact_idx) % len(SENTENCE_TEMPLATES)]
        sub_passage = template.format(
            speaker=speaker,
            entity=entity,
            slot=slot,
            quoted_value=quoted(value),
        )
        sub_passages.append(sub_passage)
        atomic_qas.append(
            {
                "sub_passage": sub_passage,
                "question": f"{entity} 설명에서 {slot} 관련 내용은 무엇이라고 했어?",
                "answer": value,
                "full_answer": fact_answer(entity, slot, value),
            }
        )

    full_answer = build_full_answer(entity, selected_facts)
    final_qas = []
    for q_idx in range(final_variants):
        question = QUESTION_TEMPLATES[q_idx % len(QUESTION_TEMPLATES)].format(entity=entity)
        final_qas.append(
            {
                "question": question,
                "answer": full_answer,
                "full_answer": full_answer,
            }
        )

    passage = " ".join(sub_passages)
    return {
        "source_id": source_id,
        "root_source_id": source_id,
        "language": "ko",
        "clean_ko": True,
        "entity_multifact": True,
        "atomic_qas_as_evidence_only": True,
        "entity": entity,
        "domain": domain,
        "base_entity": base_entity,
        "passage": passage,
        "rewrite": passage,
        "atomic_qas": atomic_qas,
        "final_qas": final_qas,
        "hard_negatives": [],
        "seed_meta": {
            "facts_per_row": len(selected_facts),
            "final_variants": len(final_qas),
            "selected_slots": [slot for slot, _ in selected_facts],
        },
    }


def build_rows(count: int, facts_per_row: int, final_variants: int, source_prefix: str, seed: int) -> list[dict[str, Any]]:
    random.seed(seed)
    indices = list(range(count))
    # Keep deterministic output while avoiding one-domain blocks.
    random.shuffle(indices)
    return [build_row(index, source_prefix, facts_per_row, final_variants) for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build seed rows for entity-level multi-fact PRAG data.")
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=Path("data/PRAG_entity_multifact_seed.jsonl"))
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--facts-per-row", type=int, default=3)
    parser.add_argument("--final-variants", type=int, default=2)
    parser.add_argument("--source-prefix", default="entity_multifact_seed")
    args = parser.parse_args()

    if args.facts_per_row < 2:
        raise SystemExit("--facts-per-row must be at least 2 for multi-fact questions.")
    if args.final_variants < 1:
        raise SystemExit("--final-variants must be at least 1.")

    rows = build_rows(
        count=args.count,
        facts_per_row=args.facts_per_row,
        final_variants=args.final_variants,
        source_prefix=args.source_prefix,
        seed=args.seed,
    )
    write_jsonl(args.output, rows)
    final_qas = sum(len(row.get("final_qas", [])) for row in rows)
    atomic_qas = sum(len(row.get("atomic_qas", [])) for row in rows)
    print(f"[PRAG:entity-seed] rows={len(rows)} atomic_qas={atomic_qas} final_qas={final_qas} output={args.output}")


if __name__ == "__main__":
    main()
