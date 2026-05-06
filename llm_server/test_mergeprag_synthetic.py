"""Run the MergePRAG diagnostic on a synthetic no-prior code-word case.

Usage:
    cd llm_server
    python test_mergeprag_synthetic.py
"""
import os

os.environ.setdefault("MERGEPRAG_DIAGNOSTIC_CASE", "synthetic")

import test_mergeprag  # noqa: F401,E402
