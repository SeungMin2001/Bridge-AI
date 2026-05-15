"""Normalize answers in simple entity-multifact PRAG rows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from entity_multifact_bank import build_full_answer
except ModuleNotFoundError:
    from llm_server.PRAG.entity_multifact_bank import build_full_answer


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def unquote_value(value: str) -> str:
    value = str(value or "").strip().rstrip(".")
    for suffix in ("이라고", "라고"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    if value.endswith("다고"):
        value = value[: -len("다고")] + "다"
    return value.strip()


def split_fact_part(part: str) -> tuple[str, str] | None:
    part = re.sub(r"^\s*그리고\s+", "", str(part or "").strip())
    for marker in ("은 ", "는 "):
        if marker in part:
            slot, value = part.split(marker, 1)
            slot = slot.strip()
            value = unquote_value(value)
            if slot and value:
                return slot, value
    return None


def answer_body(row: dict[str, Any]) -> str:
    answer = str(row.get("answer") or "").strip()
    entity = str(row.get("entity") or "").strip()
    if entity and answer.startswith(entity):
        answer = answer[len(entity) :].strip()
    if answer.startswith("설명은"):
        answer = answer[len("설명은") :].strip()
    elif answer.startswith("에 대해"):
        answer = answer[len("에 대해") :].strip()
    answer = re.sub(r"\s*설명했습니다\.?$", "", answer).strip()
    answer = re.sub(r"\s*입니다\.?$", "", answer).strip()
    return answer


def parse_facts(row: dict[str, Any]) -> list[tuple[str, str]]:
    body = answer_body(row)
    parts = [part.strip() for part in re.split(r",\s*그리고\s*|,\s*", body) if part.strip()]
    facts: list[tuple[str, str]] = []
    for part in parts:
        fact = split_fact_part(part)
        if fact:
            facts.append(fact)
    return facts


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    if not (row.get("question") and row.get("answer") and isinstance(row.get("passages"), list)):
        return row
    facts = parse_facts(row)
    if not facts:
        return row
    out = dict(row)
    entity = str(row.get("entity") or "").strip()
    out["answer"] = build_full_answer(entity, facts) if entity else row["answer"]
    out["full_answer"] = out["answer"] if row.get("full_answer") else row.get("full_answer", "")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite simple entity-multifact answers into the compact natural style.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    if args.in_place:
        output = args.input
    elif args.output:
        output = args.output
    else:
        raise SystemExit("Provide --output or --in-place.")

    rows = load_jsonl(args.input)
    normalized = [normalize_row(row) for row in rows]
    write_jsonl(output, normalized)
    changed = sum(1 for old, new in zip(rows, normalized) if old.get("answer") != new.get("answer"))
    print(f"[PRAG:entity-normalize] rows={len(rows)} changed={changed} output={output}")


if __name__ == "__main__":
    main()
