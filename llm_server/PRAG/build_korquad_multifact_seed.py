"""Build KorQuAD-based simple multi-fact rows for BridgePRAG.

The final BridgePRAG entity-multifact experiments use rows shaped as:

    question + passages[] -> answer

This builder keeps that same schema, but uses KorQuAD question/answer/context
content as the factual source instead of the synthetic entity fact bank.  Each
row groups several QA pairs from the same KorQuAD context, so a broad final
question requires multiple related facts from the same passage family.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from .prepare_korquad import iter_hf_records, iter_local_records, sentence_for_answer
except ImportError:  # pragma: no cover - direct script execution fallback.
    from prepare_korquad import iter_hf_records, iter_local_records, sentence_for_answer


ORDINAL_LABELS = [
    "첫 번째 답",
    "두 번째 답",
    "세 번째 답",
    "네 번째 답",
    "다섯 번째 답",
    "여섯 번째 답",
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_batchim(text: str) -> bool:
    for ch in reversed(str(text or "").strip()):
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            return (code - 0xAC00) % 28 != 0
    return False


def quoted(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return "없다고"
    if value.endswith("다"):
        return f"{value[:-1]}다고"
    return f"{value}{'이라고' if has_batchim(value) else '라고'}"


def topic_particle(text: str) -> str:
    return "은" if has_batchim(text) else "는"


def compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def safe_title(record: dict[str, Any], group_idx: int) -> str:
    title = compact_text(record.get("title") or "", 28)
    if title:
        return title
    record_id = str(record.get("id") or "").strip()
    if record_id:
        return f"KorQuAD 지문 {record_id.split('-')[0]}"
    return f"KorQuAD 지문 {group_idx + 1}"


def context_key(record: dict[str, Any]) -> str:
    return str(record.get("context") or "").strip()


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input:
        records = list(iter_local_records(Path(args.input)))
        source_desc = args.input
    else:
        records = list(iter_hf_records(args.dataset, args.split))
        source_desc = f"{args.dataset}:{args.split}"
    if args.max_source_records > 0:
        records = records[: args.max_source_records]
    print(f"[PRAG:korquad-multifact-seed] source={source_desc} records={len(records)}")
    return records


def grouped_records(records: Iterable[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = context_key(record)
        if key:
            groups[key].append(record)
    # Keep only contexts that can actually form multi-fact rows.
    return [(ctx, qas) for ctx, qas in groups.items() if len(qas) >= 2]


def build_answer(entity: str, qas: list[dict[str, Any]]) -> str:
    parts = []
    for idx, qa in enumerate(qas):
        label = ORDINAL_LABELS[idx] if idx < len(ORDINAL_LABELS) else f"{idx + 1}번째 답"
        parts.append(f"{label}{topic_particle(label)} {quoted(str(qa['answer']).strip())}")
    body = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f", 그리고 {parts[-1]}"
    return f"{entity}에 대해 {body} 설명했습니다."


def build_passage(entity: str, qa: dict[str, Any], idx: int, evidence_chars: int) -> str:
    label = ORDINAL_LABELS[idx] if idx < len(ORDINAL_LABELS) else f"{idx + 1}번째 답"
    question = compact_text(qa.get("question") or "", 52)
    answer = str(qa.get("answer") or "").strip()
    evidence = compact_text(sentence_for_answer(str(qa.get("context") or ""), answer, max_chars=evidence_chars), evidence_chars)
    if evidence and answer in evidence:
        return f"{entity}의 {label}은 {quoted(answer)} 설명했습니다. 근거 문장은 \"{evidence}\"입니다."
    return f"{entity}의 {label}은 {quoted(answer)} 설명했습니다. 원문 질문은 \"{question}\"입니다."


def build_rows(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    groups = grouped_records(records)
    rng.shuffle(groups)

    rows: list[dict[str, Any]] = []
    for group_idx, (context, qas_all) in enumerate(groups):
        qas = list(qas_all)
        rng.shuffle(qas)
        for chunk_idx in range(0, len(qas), args.facts_per_row):
            chunk = qas[chunk_idx : chunk_idx + args.facts_per_row]
            if len(chunk) < args.min_facts_per_row:
                continue
            entity_base = safe_title(chunk[0], group_idx)
            entity = f"{entity_base} {group_idx + 1}번 묶음"
            source_id = f"{args.source_prefix}_{group_idx:05d}_{chunk_idx // max(args.facts_per_row, 1):02d}"
            passages = [build_passage(entity, qa, idx, args.evidence_chars) for idx, qa in enumerate(chunk)]
            answer = build_answer(entity, chunk)
            question = f"{entity}에서 KorQuAD 질문들이 묻는 핵심 답을 정리해줘."
            rows.append(
                {
                    "source_id": source_id,
                    "root_source_id": source_id,
                    "language": "ko",
                    "clean_ko": True,
                    "entity_multifact_simple": True,
                    "dataset": "korquad",
                    "entity": entity,
                    "domain": "korquad",
                    "base_entity": entity_base,
                    "question": question,
                    "passages": passages,
                    "answer": answer,
                    "korquad_qas": [
                        {
                            "id": str(qa.get("id") or ""),
                            "title": str(qa.get("title") or ""),
                            "question": str(qa.get("question") or ""),
                            "answer": str(qa.get("answer") or ""),
                        }
                        for qa in chunk
                    ],
                    "seed_meta": {
                        "source_dataset": args.dataset,
                        "source_split": args.split,
                        "facts_per_row": len(chunk),
                        "context_chars": len(context),
                    },
                }
            )
            if args.count > 0 and len(rows) >= args.count:
                return rows
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KorQuAD-based simple multi-fact seed rows.")
    parser.add_argument("--dataset", default="KorQuAD/squad_kor_v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--input", default="", help="Optional local KorQuAD/SQuAD-style JSON or JSONL file.")
    parser.add_argument("--output", type=Path, default=Path("data/PRAG_korquad_multifact_simple_seed.jsonl"))
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--max-source-records", type=int, default=0)
    parser.add_argument("--facts-per-row", type=int, default=3)
    parser.add_argument("--min-facts-per-row", type=int, default=3)
    parser.add_argument("--evidence-chars", type=int, default=110)
    parser.add_argument("--source-prefix", default="korquad_multifact_seed")
    parser.add_argument("--seed", type=int, default=20260519)
    args = parser.parse_args()

    if args.facts_per_row < 2:
        raise SystemExit("--facts-per-row must be at least 2.")
    if args.min_facts_per_row < 2 or args.min_facts_per_row > args.facts_per_row:
        raise SystemExit("--min-facts-per-row must be between 2 and --facts-per-row.")

    records = load_records(args)
    rows = build_rows(args, records)
    if args.count > 0 and len(rows) < args.count:
        print(
            f"[PRAG:korquad-multifact-seed] warning: requested count={args.count} but built rows={len(rows)}. "
            "Increase --max-source-records or lower --min-facts-per-row if needed.",
            flush=True,
        )
    write_jsonl(args.output, rows)
    passages = sum(len(row.get("passages", [])) for row in rows)
    print(f"[PRAG:korquad-multifact-seed] rows={len(rows)} passages={passages} output={args.output}")


if __name__ == "__main__":
    main()
