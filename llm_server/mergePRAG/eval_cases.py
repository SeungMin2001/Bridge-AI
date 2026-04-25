"""Shared diagnostic cases for service-style MergePRAG training/evaluation.

Keep these examples synchronized across:
- prepare_service_hardpairs.py  (training data generation)
- test_mergeprag.py            (fast sanity diagnostic)
- debug_mergeprag.py           (deep diagnostic)
"""

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
