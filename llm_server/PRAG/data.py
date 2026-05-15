"""Dataset utilities for augmented single-passage PRAG memory training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class MemoryExample:
    source_id: str
    passage: str
    question: str
    answer: str
    full_answer: str = ""
    negative_passage: str | None = None
    negative_answer: str | None = None
    negative_full_answer: str | None = None
    qa_type: str = "qa"

    def target_answer(self, mode: str = "answer") -> str:
        if mode == "full_answer" and self.full_answer:
            return self.full_answer
        return self.answer

    def target_negative_answer(self, mode: str = "answer") -> str | None:
        if mode == "full_answer" and self.negative_full_answer:
            return self.negative_full_answer
        return self.negative_answer


@dataclass
class MemoryGroup:
    source_id: str
    passage: str
    qas: list[MemoryExample]
    negative_passage: str | None = None


def iter_json_records(path: str | Path) -> Iterable[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array: {path}")
        yield from data
        return
    raise ValueError(f"Unsupported dataset format: {path}")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def jsonl_snapshot(path: str | Path) -> dict:
    count = 0
    last_source_id = ""
    try:
        for row in iter_json_records(path):
            count += 1
            last_source_id = str(row.get("source_id") or row.get("id") or "").strip()
    except FileNotFoundError:
        pass
    return {
        "path": str(path),
        "rows": count,
        "last_source_id": last_source_id or "none",
    }


def get_passage(row: dict) -> str:
    passages = row.get("passages")
    if isinstance(passages, list):
        chunks = [str(item).strip() for item in passages if str(item or "").strip()]
        if chunks:
            return "\n".join(chunks)
    for key in ("passage", "utterance", "text", "content"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            speaker = str(row.get("speaker") or "").strip()
            text = value.strip()
            return f"{speaker}: {text}" if speaker and key != "passage" else text
    return ""


def extract_answer(row: dict) -> str:
    for key in ("answer", "target", "output", "response", "negative_answer", "counterfactual_answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    answers = row.get("answers")
    if isinstance(answers, list):
        for item in answers:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def normalize_qas(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    qas = []
    for item in value:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or item.get("sub_question") or "").strip()
        answer = str(item.get("answer") or item.get("sub_answer") or "").strip()
        full_answer = str(item.get("full_answer") or "").strip()
        sub_passage = str(item.get("sub_passage") or item.get("evidence") or item.get("passage") or "").strip()
        if question and answer:
            qas.append({
                "question": question,
                "answer": answer,
                "full_answer": full_answer or default_full_answer(question, answer),
                "sub_passage": sub_passage,
            })
    return qas


def contains_hangul(text: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in str(text or ""))


def default_full_answer(question: str, answer: str) -> str:
    """Backfill service-style answers for older augmented data.

    Earlier augmentation rows sometimes only contain a compact answer span.
    Free-generation training needs an assistant-like sentence, but the sentence
    must still start with the answer phrase so token scoring and generation stay
    aligned.
    """
    answer = str(answer or "").strip()
    if not answer:
        return ""
    if answer.endswith((".", "?", "!", "다", "요", "임", "함", "음")):
        return answer
    if contains_hangul(question) or contains_hangul(answer):
        return f"{answer}입니다."
    return f"{answer}."


def answer_in_text(answer: str, text: str) -> bool:
    return answer.strip().casefold() in text.strip().casefold()


def select_evidence_passage(qa: dict, atomic_qas: list[dict], fallback_passage: str) -> str:
    """Use the smallest evidence span that supports a QA item.

    Final QAs often ask a paraphrased version of an atomic fact. Feeding the
    full passage can reintroduce competing facts into passage-only memory, so
    prefer matching atomic sub-passages when possible.
    """
    explicit = qa.get("sub_passage") or ""
    if explicit:
        return explicit

    answer = qa.get("answer") or ""
    matched = [
        atomic["sub_passage"]
        for atomic in atomic_qas
        if atomic.get("sub_passage")
        and atomic.get("answer")
        and answer_in_text(atomic["answer"], answer)
    ]
    if matched:
        return "\n".join(dict.fromkeys(matched))

    if len(atomic_qas) == 1 and atomic_qas[0].get("sub_passage"):
        return atomic_qas[0]["sub_passage"]

    evidence_chunks = [atomic["sub_passage"] for atomic in atomic_qas if atomic.get("sub_passage")]
    if evidence_chunks:
        return "\n".join(dict.fromkeys(evidence_chunks))

    return fallback_passage


def load_augmented_examples(path: str | Path, max_samples: int | None = None) -> list[MemoryExample]:
    examples: list[MemoryExample] = []
    for row_idx, row in enumerate(iter_json_records(path)):
        if max_samples and row_idx >= max_samples:
            break
        passage = get_passage(row)
        source_id = str(row.get("source_id") or row.get("id") or f"row_{row_idx}")
        if not passage:
            continue

        simple_multifact_row = isinstance(row.get("passages"), list) and row.get("question") and row.get("answer")
        atomic_qas = [] if simple_multifact_row else normalize_qas(row.get("atomic_qas"))
        final_qas = [] if simple_multifact_row else normalize_qas(row.get("final_qas"))
        use_row_passage_for_examples = row.get("task") == "korquad_service_transcript_memory"
        atomic_qas_as_evidence_only = bool(
            row.get("atomic_qas_as_evidence_only")
            or row.get("evidence_only_atomic_qas")
            or row.get("entity_multifact")
        )

        qas = []
        if simple_multifact_row:
            qa = {
                "question": str(row["question"]).strip(),
                "answer": str(row["answer"]).strip(),
                "full_answer": str(row.get("full_answer") or row["answer"]).strip(),
            }
            qas.append((qa, "final", passage))
        else:
            qas.extend(
                (qa, "atomic", passage if use_row_passage_for_examples else qa.get("sub_passage") or passage)
                for qa in ([] if atomic_qas_as_evidence_only else atomic_qas)
            )
            qas.extend(
                (
                    qa,
                    "final",
                    passage if use_row_passage_for_examples else select_evidence_passage(qa, atomic_qas, passage),
                )
                for qa in final_qas
            )
        if not qas and row.get("question") and row.get("answer"):
            qas.append(({"question": str(row["question"]), "answer": str(row["answer"])}, "direct", passage))

        negatives = row.get("hard_negatives") if isinstance(row.get("hard_negatives"), list) else []
        neg_by_question = {}
        neg_by_position = {}
        fallback_neg = None
        for neg in negatives:
            if not isinstance(neg, dict):
                continue
            neg_passage = get_passage(neg)
            neg_atomic = normalize_qas(neg.get("atomic_qas")) + normalize_qas(neg.get("qas"))
            neg_final = normalize_qas(neg.get("final_qas"))
            neg_qas = [
                ("atomic", idx, qa, qa.get("sub_passage") or neg_passage)
                for idx, qa in enumerate(neg_atomic)
            ]
            neg_qas.extend(
                ("final", idx, qa, select_evidence_passage(qa, neg_atomic, neg_passage))
                for idx, qa in enumerate(neg_final)
            )
            if neg_passage and neg.get("answer"):
                fallback_neg = {"passage": neg_passage, "answer": str(neg["answer"]).strip()}
            for neg_type, neg_idx, neg_qa, neg_memory_passage in neg_qas:
                if neg_memory_passage:
                    neg_item = {
                        "passage": neg_memory_passage,
                        "answer": neg_qa["answer"],
                        "full_answer": neg_qa.get("full_answer") or default_full_answer(neg_qa["question"], neg_qa["answer"]),
                    }
                    neg_by_question[neg_qa["question"]] = neg_item
                    neg_by_position[(neg_type, neg_idx)] = neg_item

        for qa_idx, (qa, qa_type, memory_passage) in enumerate(qas):
            # Prefer an exact same-question counterfactual. If the local LLM
            # paraphrases the negative question, fall back to the same
            # atomic/final position. Do not attach a generic fallback to
            # generated QA rows because it can pair a final explanatory answer
            # with an unrelated short negative answer.
            neg = neg_by_question.get(qa["question"]) or neg_by_position.get((qa_type, qa_idx))
            if neg is None and qa_type == "direct":
                neg = fallback_neg
            examples.append(
                MemoryExample(
                    source_id=f"{source_id}:{qa_type}:{qa_idx}",
                    passage=memory_passage,
                    question=qa["question"],
                    answer=qa["answer"],
                    full_answer=qa.get("full_answer") or default_full_answer(qa["question"], qa["answer"]),
                    negative_passage=neg["passage"] if neg else None,
                    negative_answer=neg["answer"] if neg else None,
                    negative_full_answer=neg.get("full_answer") if neg else None,
                    qa_type=qa_type,
                )
            )
    print(f"[PRAG:data] loaded {len(examples)} memory QA examples from {path}")
    return examples


def load_augmented_groups(path: str | Path, max_samples: int | None = None) -> list[MemoryGroup]:
    groups: list[MemoryGroup] = []
    for row_idx, row in enumerate(iter_json_records(path)):
        if max_samples and row_idx >= max_samples:
            break
        passage = get_passage(row)
        source_id = str(row.get("source_id") or row.get("id") or f"row_{row_idx}")
        if not passage:
            continue

        simple_multifact_row = isinstance(row.get("passages"), list) and row.get("question") and row.get("answer")
        atomic_qas = [] if simple_multifact_row else normalize_qas(row.get("atomic_qas"))
        final_qas = [] if simple_multifact_row else normalize_qas(row.get("final_qas"))
        use_row_passage_for_examples = row.get("task") == "korquad_service_transcript_memory"
        atomic_qas_as_evidence_only = bool(
            row.get("atomic_qas_as_evidence_only")
            or row.get("evidence_only_atomic_qas")
            or row.get("entity_multifact")
        )
        qas = []
        if simple_multifact_row:
            qas.append(
                (
                    0,
                    {
                        "question": str(row["question"]).strip(),
                        "answer": str(row["answer"]).strip(),
                        "full_answer": str(row.get("full_answer") or row["answer"]).strip(),
                    },
                    "final",
                )
            )
        else:
            if not atomic_qas_as_evidence_only:
                qas.extend((idx, qa, "atomic") for idx, qa in enumerate(atomic_qas))
            qas.extend((idx, qa, "final") for idx, qa in enumerate(final_qas))
        if not qas and row.get("question") and row.get("answer"):
            qas.append((0, {"question": str(row["question"]), "answer": str(row["answer"])}, "direct"))

        negatives = row.get("hard_negatives") if isinstance(row.get("hard_negatives"), list) else []
        negative_passage = None
        neg_by_question = {}
        neg_by_position = {}
        fallback_answer = None
        for neg in negatives:
            if not isinstance(neg, dict):
                continue
            neg_passage = get_passage(neg)
            if not neg_passage:
                continue
            negative_passage = negative_passage or neg_passage
            fallback_text = str(neg.get("answer") or "").strip()
            fallback_answer = {"answer": fallback_text, "full_answer": default_full_answer("", fallback_text)} if fallback_text else None
            neg_atomic = normalize_qas(neg.get("atomic_qas")) + normalize_qas(neg.get("qas"))
            neg_final = normalize_qas(neg.get("final_qas"))
            for neg_idx, neg_qa in enumerate(neg_atomic):
                neg_item = {
                    "answer": neg_qa["answer"],
                    "full_answer": neg_qa.get("full_answer") or default_full_answer(neg_qa["question"], neg_qa["answer"]),
                }
                neg_by_question[neg_qa["question"]] = neg_item
                neg_by_position[("atomic", neg_idx)] = neg_item
            for neg_idx, neg_qa in enumerate(neg_final):
                neg_item = {
                    "answer": neg_qa["answer"],
                    "full_answer": neg_qa.get("full_answer") or default_full_answer(neg_qa["question"], neg_qa["answer"]),
                }
                neg_by_question[neg_qa["question"]] = neg_item
                neg_by_position[("final", neg_idx)] = neg_item
            break

        group_qas: list[MemoryExample] = []
        for qa_idx, qa, qa_type in qas:
            neg_item = neg_by_question.get(qa["question"]) or neg_by_position.get((qa_type, qa_idx))
            if neg_item is None and qa_type == "direct" and fallback_answer:
                neg_item = fallback_answer
            if qa_type == "direct" or use_row_passage_for_examples:
                memory_passage = passage
            elif qa_type == "final":
                memory_passage = select_evidence_passage(qa, atomic_qas, passage)
            else:
                memory_passage = qa.get("sub_passage") or passage
            group_qas.append(
                MemoryExample(
                    source_id=f"{source_id}:{qa_type}:{qa_idx}",
                    passage=memory_passage,
                    question=qa["question"],
                    answer=qa["answer"],
                    full_answer=qa.get("full_answer") or default_full_answer(qa["question"], qa["answer"]),
                    negative_passage=negative_passage,
                    negative_answer=neg_item["answer"] if neg_item else None,
                    negative_full_answer=neg_item["full_answer"] if neg_item else None,
                    qa_type=qa_type,
                )
            )

        if len(group_qas) >= 2 or (simple_multifact_row and group_qas):
            groups.append(
                MemoryGroup(
                    source_id=source_id,
                    passage=passage,
                    qas=group_qas,
                    negative_passage=negative_passage,
                )
            )
    print(f"[PRAG:data] loaded {len(groups)} memory QA groups from {path}")
    return groups
