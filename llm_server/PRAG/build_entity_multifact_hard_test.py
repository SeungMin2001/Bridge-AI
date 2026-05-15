"""Build hard entity-level multi-fact PRAG test rows.

The simple entity dataset asks broad summary questions, so both passage-only
and question-aware memories can often answer by summarizing every injected
passage.  This builder keeps the same top-level schema but asks for only a
subset of facts while injecting nearby distractor facts.
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
        SENTENCE_TEMPLATES,
        SPEAKERS,
        build_full_answer,
        context_for_index,
        has_batchim,
        quoted,
        topic_particle,
    )
except ModuleNotFoundError:
    from llm_server.PRAG.entity_multifact_bank import (
        ENTITY_BANK,
        SENTENCE_TEMPLATES,
        SPEAKERS,
        build_full_answer,
        context_for_index,
        has_batchim,
        quoted,
        topic_particle,
    )


QUESTION_TEMPLATES = [
    "{entity} 설명에서 {slot_pair}만 알려줘.",
    "{entity}에서 {slot_pair_object} 함께 정리해줘.",
    "교수님이 {entity}의 {slot_pair_object} 어떻게 말했어?",
    "{entity} 설명 중 {slot_pair}에 해당하는 내용만 답해줘.",
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_entity_name(base_entity: str, index: int) -> str:
    return f"{context_for_index(index)} {base_entity}"


def and_pair(left: str, right: str) -> str:
    return f"{left}{'과' if has_batchim(left) else '와'} {right}"


def object_particle(text: str) -> str:
    return "을" if has_batchim(text) else "를"


def and_pair_object(left: str, right: str) -> str:
    return f"{and_pair(left, right)}{object_particle(right)}"


def build_fact_sentence(entity: str, slot: str, value: str, index: int) -> str:
    speaker = SPEAKERS[index % len(SPEAKERS)]
    template = SENTENCE_TEMPLATES[index % len(SENTENCE_TEMPLATES)]
    return template.format(
        speaker=speaker,
        speaker_topic=topic_particle(speaker),
        entity=entity,
        slot=slot,
        slot_topic=topic_particle(slot),
        quoted_value=quoted(value),
    )


def select_facts(facts: list[tuple[str, str]], index: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if len(facts) < 4:
        raise ValueError("Hard test requires at least four facts per entity.")
    offset = index % len(facts)
    rotated = [facts[(offset + idx) % len(facts)] for idx in range(len(facts))]
    target_facts = [rotated[0], rotated[2]]
    distractor_facts = [rotated[1], rotated[3]]
    return target_facts, distractor_facts


def neighboring_entity(index: int, domain: str, base_entity: str) -> tuple[str, str, list[tuple[str, str]]]:
    for step in range(1, len(ENTITY_BANK) + 1):
        cand_domain, cand_base, cand_facts = ENTITY_BANK[(index + step) % len(ENTITY_BANK)]
        if cand_domain == domain and cand_base != base_entity:
            return cand_domain, cand_base, cand_facts
    return ENTITY_BANK[(index + 1) % len(ENTITY_BANK)]


def build_row(index: int, source_prefix: str, include_neighbor_distractor: bool) -> dict[str, Any]:
    domain, base_entity, facts = ENTITY_BANK[index % len(ENTITY_BANK)]
    entity = build_entity_name(base_entity, index)
    source_id = f"{source_prefix}_{domain}_{index:05d}"
    target_facts, distractor_facts = select_facts(facts, index // len(ENTITY_BANK))

    passages = [
        build_fact_sentence(entity, target_facts[0][0], target_facts[0][1], index),
        build_fact_sentence(entity, distractor_facts[0][0], distractor_facts[0][1], index + 1),
        build_fact_sentence(entity, target_facts[1][0], target_facts[1][1], index + 2),
    ]

    if include_neighbor_distractor:
        _, neighbor_base, neighbor_facts = neighboring_entity(index, domain, base_entity)
        neighbor_entity = build_entity_name(neighbor_base, index)
        distractor_slot, distractor_value = neighbor_facts[(index // len(ENTITY_BANK)) % len(neighbor_facts)]
        passages.append(build_fact_sentence(neighbor_entity, distractor_slot, distractor_value, index + 3))
    else:
        passages.append(build_fact_sentence(entity, distractor_facts[1][0], distractor_facts[1][1], index + 3))

    question = QUESTION_TEMPLATES[index % len(QUESTION_TEMPLATES)].format(
        entity=entity,
        slot_pair=and_pair(target_facts[0][0], target_facts[1][0]),
        slot_pair_object=and_pair_object(target_facts[0][0], target_facts[1][0]),
    )
    answer = build_full_answer(entity, target_facts)
    return {
        "source_id": source_id,
        "root_source_id": source_id,
        "language": "ko",
        "clean_ko": True,
        "entity_multifact_simple": True,
        "entity_multifact_hard": True,
        "entity": entity,
        "domain": domain,
        "base_entity": base_entity,
        "question": question,
        "passages": passages,
        "answer": answer,
        "hard_meta": {
            "target_slots": [slot for slot, _ in target_facts],
            "distractor_slots": [slot for slot, _ in distractor_facts],
            "neighbor_distractor": include_neighbor_distractor,
        },
    }


def build_rows(count: int, seed: int, source_prefix: str, include_neighbor_distractor: bool) -> list[dict[str, Any]]:
    random.seed(seed)
    indices = list(range(count))
    random.shuffle(indices)
    return [build_row(index, source_prefix, include_neighbor_distractor) for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard simple-schema entity multifact test rows.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/PRAG_entity_multifact_simple_hard_test100.jsonl"))
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--source-prefix", default="entity_simple_hard_test")
    parser.add_argument(
        "--neighbor-distractor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use one same-domain neighboring entity passage as an extra distractor.",
    )
    args = parser.parse_args()

    rows = build_rows(
        count=args.count,
        seed=args.seed,
        source_prefix=args.source_prefix,
        include_neighbor_distractor=args.neighbor_distractor,
    )
    write_jsonl(args.output, rows)
    passages = sum(len(row.get("passages", [])) for row in rows)
    print(f"[PRAG:entity-hard-test] rows={len(rows)} passages={passages} qas={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
