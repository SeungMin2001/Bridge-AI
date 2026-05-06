"""Augment multi-fact service-memory sources with a stronger local LLM.

This module is intentionally separate from augment.py so existing single-fact
augmentation outputs are not overwritten by the multi-fact experiment.
"""

from __future__ import annotations

import os

from .augment import build_parser, run
from .config import MULTIFACT_AUGMENTED_TRAIN_PATH, MULTIFACT_AUGMENTED_VALID_PATH, MULTIFACT_SOURCE_PATH
DEFAULT_AUGMENT_MODEL = os.getenv("PRAG_MULTIFACT_AUGMENT_MODEL", "Qwen/Qwen3.5-4B")


def main() -> None:
    parser = build_parser()
    parser.description = "Generate augmented PRAG supervision from multi-fact source passages."
    parser.set_defaults(
        input=str(MULTIFACT_SOURCE_PATH),
        train_output=str(MULTIFACT_AUGMENTED_TRAIN_PATH),
        valid_output=str(MULTIFACT_AUGMENTED_VALID_PATH),
        model=DEFAULT_AUGMENT_MODEL,
        max_new_tokens=2048,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
