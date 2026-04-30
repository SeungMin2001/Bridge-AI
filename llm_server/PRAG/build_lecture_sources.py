"""Build source passages from local lecture transcript markdown files.

The transcript files are noisy ASR outputs, so this stage only cleans obvious
formatting noise and chunks the text. Fact extraction is left to the augmentation
LLM, which is instructed to keep only facts that are clear from each chunk.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .config import LECTURE_SCRIPT_DIR, LECTURE_SOURCE_PATH


def clean_transcript(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    text = re.sub(r"(?m)^\s*>+\s?", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+|(?<=[가-힣](?:다|요|죠|까|네|음|함|됨|임))\.\s*", text)
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, max_chars: int, min_chars: int, overlap_chars: int) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            if len(current) >= min_chars:
                chunks.append(current.strip())
            tail = current[-overlap_chars:].strip() if overlap_chars > 0 else ""
            current = f"{tail} {sentence}".strip() if tail else sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence
    if len(current) >= min_chars:
        chunks.append(current.strip())
    elif current and chunks:
        chunks[-1] = f"{chunks[-1]} {current}".strip()[: max_chars + overlap_chars]
    elif current:
        chunks.append(current.strip())
    return chunks


def iter_script_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )


def build_rows(input_dir: Path, max_chars: int, min_chars: int, overlap_chars: int) -> list[dict]:
    rows = []
    for file_path in iter_script_files(input_dir):
        course = file_path.stem
        text = clean_transcript(file_path.read_text(encoding="utf-8", errors="ignore"))
        chunks = chunk_text(text, max_chars=max_chars, min_chars=min_chars, overlap_chars=overlap_chars)
        for idx, chunk in enumerate(chunks):
            rows.append({
                "source_id": f"lecture_{course}_{idx:04d}",
                "speaker": "교수님",
                "course": course,
                "passage": f"[수업명: {course}] {chunk}",
                "source_file": str(file_path),
                "chunk_index": idx,
            })
        print(f"[PRAG:lecture-source] {file_path.name}: chars={len(text)} chunks={len(chunks)}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(LECTURE_SCRIPT_DIR))
    parser.add_argument("--output", default=str(LECTURE_SOURCE_PATH))
    parser.add_argument("--max-chars", type=int, default=1100)
    parser.add_argument("--min-chars", type=int, default=350)
    parser.add_argument("--overlap-chars", type=int, default=120)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Lecture script directory not found: {input_dir}")
    rows = build_rows(input_dir, args.max_chars, args.min_chars, args.overlap_chars)
    write_jsonl(Path(args.output), rows)
    print(f"[PRAG:lecture-source] total_rows={len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
