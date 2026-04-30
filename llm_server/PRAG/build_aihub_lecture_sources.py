"""Build PRAG source chunks from AI Hub Korean university lecture labels."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .build_lecture_sources import chunk_text, clean_transcript
from .config import (
    AIHUB_LECTURE_DIR,
    AIHUB_LECTURE_TRAIN_SOURCE_PATH,
    AIHUB_LECTURE_VALID_SOURCE_PATH,
)


ENCODINGS = ("utf-8", "utf-8-sig", "cp949", "euc-kr")


def read_json(path: Path) -> dict | None:
    for encoding in ENCODINGS:
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


def get_nested(data: dict, *keys: str) -> str:
    value = data
    for key in keys:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    return str(value or "").strip()


def is_useful_utterance(text: str) -> bool:
    text = clean_transcript(text)
    if len(text) < 8:
        return False
    filler = re.sub(r"[ .,!?\-~…]+", "", text)
    if filler in {"네", "예", "음", "어", "자", "아", "그죠", "그렇죠"}:
        return False
    return any("\uac00" <= ch <= "\ud7a3" for ch in text)


def find_json_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.json"))


def infer_split_root(base_dir: Path, split: str) -> Path:
    return base_dir / split / "02.라벨링데이터"


def collect_lectures(label_root: Path, max_files: int = 0) -> dict[str, dict]:
    lectures: dict[str, dict] = {}
    files = find_json_files(label_root)
    if max_files:
        files = files[:max_files]
    for idx, path in enumerate(files, start=1):
        data = read_json(path)
        if not data:
            continue
        text = get_nested(data, "06_transcription", "1_text")
        if not is_useful_utterance(text):
            continue
        lecture_id = get_nested(data, "02_srcinfo", "1_id") or path.parent.name
        category = get_nested(data, "01_dataset", "5_category") or path.parts[-4] if len(path.parts) >= 4 else ""
        major = get_nested(data, "03_lectureinfo", "3_major_category")
        collection_type = get_nested(data, "03_lectureinfo", "5_collection_type")
        role = get_nested(data, "05_speakerinfo", "4_role")
        key = f"{category}/{major}/{lecture_id}"
        item = lectures.setdefault(
            key,
            {
                "lecture_id": lecture_id,
                "category": category,
                "major": major,
                "collection_type": collection_type,
                "role": role,
                "texts": [],
                "files": [],
            },
        )
        item["texts"].append(clean_transcript(text))
        item["files"].append(str(path))
        if idx % 10000 == 0:
            print(f"[PRAG:aihub-source] scanned={idx} useful_lectures={len(lectures)}")
    return lectures


def rows_from_lectures(lectures: dict[str, dict], split: str, max_chars: int, min_chars: int, overlap_chars: int, max_lectures: int = 0) -> list[dict]:
    rows = []
    lecture_items = sorted(lectures.items())
    if max_lectures:
        lecture_items = lecture_items[:max_lectures]
    for _key, lecture in lecture_items:
        merged = clean_transcript(" ".join(lecture["texts"]))
        chunks = chunk_text(merged, max_chars=max_chars, min_chars=min_chars, overlap_chars=overlap_chars)
        course = lecture["major"] or lecture["category"] or lecture["lecture_id"]
        for idx, chunk in enumerate(chunks):
            rows.append({
                "source_id": f"aihub_{split}_{lecture['lecture_id']}_{idx:04d}",
                "speaker": lecture["role"] or "강사",
                "course": course,
                "category": lecture["category"],
                "major": lecture["major"],
                "collection_type": lecture["collection_type"],
                "passage": f"[AIHub 대학강의/{course}/강의ID:{lecture['lecture_id']}] {chunk}",
                "lecture_id": lecture["lecture_id"],
                "chunk_index": idx,
                "source_files": lecture["files"][:3],
                "utterance_count": len(lecture["texts"]),
            })
        print(
            f"[PRAG:aihub-source] {split} {lecture['lecture_id']}: "
            f"utterances={len(lecture['texts'])} chars={len(merged)} chunks={len(chunks)}"
        )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(AIHUB_LECTURE_DIR))
    parser.add_argument("--train-root", default="")
    parser.add_argument("--valid-root", default="")
    parser.add_argument("--train-output", default=str(AIHUB_LECTURE_TRAIN_SOURCE_PATH))
    parser.add_argument("--valid-output", default=str(AIHUB_LECTURE_VALID_SOURCE_PATH))
    parser.add_argument("--max-chars", type=int, default=1100)
    parser.add_argument("--min-chars", type=int, default=350)
    parser.add_argument("--overlap-chars", type=int, default=120)
    parser.add_argument("--max-train-files", type=int, default=0)
    parser.add_argument("--max-valid-files", type=int, default=0)
    parser.add_argument("--max-train-lectures", type=int, default=0)
    parser.add_argument("--max-valid-lectures", type=int, default=0)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    train_root = Path(args.train_root) if args.train_root else infer_split_root(base_dir, "Training")
    valid_root = Path(args.valid_root) if args.valid_root else infer_split_root(base_dir, "Validation")
    train_lectures = collect_lectures(train_root, max_files=args.max_train_files)
    valid_lectures = collect_lectures(valid_root, max_files=args.max_valid_files)
    train_rows = rows_from_lectures(
        train_lectures,
        split="train",
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        overlap_chars=args.overlap_chars,
        max_lectures=args.max_train_lectures,
    )
    valid_rows = rows_from_lectures(
        valid_lectures,
        split="valid",
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        overlap_chars=args.overlap_chars,
        max_lectures=args.max_valid_lectures,
    )
    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.valid_output), valid_rows)
    print(f"[PRAG:aihub-source] train_rows={len(train_rows)} -> {args.train_output}")
    print(f"[PRAG:aihub-source] valid_rows={len(valid_rows)} -> {args.valid_output}")


if __name__ == "__main__":
    main()
