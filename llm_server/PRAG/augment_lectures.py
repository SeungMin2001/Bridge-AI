"""Augment local lecture transcript chunks into PRAG supervision."""

from __future__ import annotations

import os

from .augment import build_parser, run
from .config import LECTURE_AUGMENTED_TRAIN_PATH, LECTURE_AUGMENTED_VALID_PATH, LECTURE_SOURCE_PATH


DEFAULT_AUGMENT_MODEL = os.getenv("PRAG_LECTURE_AUGMENT_MODEL", "Qwen/Qwen2.5-3B-Instruct")


def main() -> None:
    parser = build_parser()
    parser.description = "Generate augmented PRAG supervision from local lecture transcript chunks."
    parser.set_defaults(
        input=str(LECTURE_SOURCE_PATH),
        train_output=str(LECTURE_AUGMENTED_TRAIN_PATH),
        valid_output=str(LECTURE_AUGMENTED_VALID_PATH),
        model=DEFAULT_AUGMENT_MODEL,
        max_new_tokens=2048,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
