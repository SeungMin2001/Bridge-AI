"""Augment AI Hub university lecture chunks into PRAG supervision."""

from __future__ import annotations

import os

from .augment import build_parser, run
from .config import AIHUB_LECTURE_AUGMENTED_TRAIN_PATH, AIHUB_LECTURE_AUGMENTED_VALID_PATH, AIHUB_LECTURE_TRAIN_SOURCE_PATH


DEFAULT_AUGMENT_MODEL = os.getenv("PRAG_AIHUB_LECTURE_AUGMENT_MODEL", "Qwen/Qwen3.5-4B")


def main() -> None:
    parser = build_parser()
    parser.description = "Generate augmented PRAG supervision from AI Hub university lecture chunks."
    parser.set_defaults(
        input=str(AIHUB_LECTURE_TRAIN_SOURCE_PATH),
        train_output=str(AIHUB_LECTURE_AUGMENTED_TRAIN_PATH),
        valid_output=str(AIHUB_LECTURE_AUGMENTED_VALID_PATH),
        model=DEFAULT_AUGMENT_MODEL,
        max_new_tokens=2048,
        valid_every=5,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
