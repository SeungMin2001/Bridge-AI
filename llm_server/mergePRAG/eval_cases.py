"""Shared diagnostic cases for service-style MergePRAG training/evaluation.

Keep these examples synchronized across:
- prepare_service_hardpairs.py  (training data generation)
- test_mergeprag.py            (fast sanity diagnostic)
- debug_mergeprag.py           (deep diagnostic)
"""
import os

SERVICE_DIAGNOSTIC_CASE = {
    "question": "When is the homework due?",
    "alt_question": "When is the project proposal due?",
    "passage": (
        "Professor Lee said the homework is due on Monday. "
        "The project proposal is due on Friday. "
        "Remember: homework Monday, proposal Friday."
    ),
    "compare_passage": (
        "Professor Lee said the homework is due on Friday. "
        "The project proposal is due on Monday. "
        "Remember: homework Friday, proposal Monday."
    ),
    "answer": "Monday",
    "compare_answer": "Friday",
    "alt_answer": "Friday",
    "generation_instruction": "Answer with only the due date:",
}


SYNTHETIC_CODE_DIAGNOSTIC_CASE = {
    "case_name": "synthetic_code",
    "question": "What code word is assigned to the daxmel marker?",
    "alt_question": "What code word is assigned to the norqu marker?",
    "passage": (
        "Private lecture ledger ZX-91 states that the daxmel marker is assigned "
        "the code word virel. The norqu marker is assigned the code word jandor. "
        "Remember exactly: daxmel -> virel; norqu -> jandor."
    ),
    "compare_passage": (
        "Private lecture ledger ZX-91 states that the daxmel marker is assigned "
        "the code word jandor. The norqu marker is assigned the code word virel. "
        "Remember exactly: daxmel -> jandor; norqu -> virel."
    ),
    "answer": "virel",
    "compare_answer": "jandor",
    "alt_answer": "jandor",
    "generation_instruction": "Answer with only the exact code word:",
}


DIAGNOSTIC_CASES = {
    "service": SERVICE_DIAGNOSTIC_CASE,
    "default": SERVICE_DIAGNOSTIC_CASE,
    "synthetic": SYNTHETIC_CODE_DIAGNOSTIC_CASE,
    "synthetic_code": SYNTHETIC_CODE_DIAGNOSTIC_CASE,
}


def get_diagnostic_case(name: str | None = None) -> dict:
    """Select a diagnostic case without changing the training data generator."""
    selected = (name or os.getenv("MERGEPRAG_DIAGNOSTIC_CASE", "service")).strip().lower()
    try:
        case = DIAGNOSTIC_CASES[selected]
    except KeyError as exc:
        valid = ", ".join(sorted(DIAGNOSTIC_CASES))
        raise ValueError(
            f"Unknown MERGEPRAG_DIAGNOSTIC_CASE={selected!r}. Valid: {valid}"
        ) from exc
    return {"case_name": selected, **case}
