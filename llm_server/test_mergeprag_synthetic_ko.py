"""Run the MergePRAG diagnostic on a Korean synthetic no-prior code-word case.

Usage:
    cd llm_server
    python test_mergeprag_synthetic_ko.py
"""
import os

os.environ.setdefault("MERGEPRAG_DIAGNOSTIC_CASE", "synthetic_ko")

import test_mergeprag  # noqa: F401,E402
