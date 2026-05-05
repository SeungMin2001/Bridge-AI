"""Run deep MergePRAG diagnostics on a synthetic no-prior code-word case."""
import os

os.environ.setdefault("MERGEPRAG_DIAGNOSTIC_CASE", "synthetic")

import debug_mergeprag  # noqa: F401,E402
