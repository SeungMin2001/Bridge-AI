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
        topic_particle,
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
        topic_particle,
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fact_answer(entity: str, slot: str, value: str) -> str:
    return f"{entity} 설명에서 {slot}{topic_particle(slot)} {quoted(value)} 했습니다."


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

    sub_passages: list[str] = []
    for fact_idx, (slot, value) in enumerate(selected_facts):
        speaker = SPEAKERS[(index + fact_idx) % len(SPEAKERS)]
        template = SENTENCE_TEMPLATES[(index + fact_idx) % len(SENTENCE_TEMPLATES)]
        sub_passage = template.format(
            speaker=speaker,
            speaker_topic=topic_particle(speaker),
            entity=entity,
            slot=slot,
            slot_topic=topic_particle(slot),
            quoted_value=quoted(value),
        )
        sub_passages.append(sub_passage)

    full_answer = build_full_answer(entity, selected_facts)
    question = QUESTION_TEMPLATES[index % len(QUESTION_TEMPLATES)].format(entity=entity, topic=topic_particle(entity))
    return {
        "source_id": source_id,
        "root_source_id": source_id,
        "language": "ko",
        "clean_ko": True,
        "entity_multifact_simple": True,
        "entity": entity,
        "domain": domain,
        "base_entity": base_entity,
        "question": question,
        "passages": sub_passages,
        "answer": full_answer,
        "seed_meta": {
            "facts_per_row": len(selected_facts),
            "final_variants": 1,
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
    parser.add_argument("--final-variants", type=int, default=1)
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
    passages = sum(len(row.get("passages", [])) for row in rows)
    print(f"[PRAG:entity-seed] rows={len(rows)} passages={passages} qas={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
