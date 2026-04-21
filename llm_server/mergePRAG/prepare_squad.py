"""
Download SQuAD v1.1 from Stanford official source and save processed JSONL for MergePRAG training.

왜 SQuAD인가:
  논문 저자는 HotpotQA를 decompose해서 (질문, 단일 passage, 답) 삼중항으로 학습함 (passage→answer 결정성 확보).
  HotpotQA를 그대로 쓰면 multi-hop 질문에 single-hop passage를 주어 task CE가 passage별 K/V를 학습시키지 못함.
  SQuAD는 원래부터 (question, passage, answer) 삼중항이라 decompose 없이 조건을 만족한다.

왜 HF datasets 대신 직접 다운로드:
  일부 환경의 `datasets` 버전이 최신 HF script의 `List` feature type을 지원하지 않아
  ValueError: Feature type 'List' not found 가 발생. 공식 JSON은 버전 의존성이 없다.

Output rows (MergePRAGDataset 포맷과 호환):
  {"source_id": ..., "task": "final_qa", "question": ..., "answer": ..., "passage": ...}

Usage:
  python -m llm_server.mergePRAG.prepare_squad
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path


TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
VALID_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

OUT_TRAIN_PATH = Path(r"C:\Users\user\Documents\last_project\data\SQuAD_train_processed.jsonl")
OUT_VALID_PATH = Path(r"C:\Users\user\Documents\last_project\data\SQuAD_valid_processed.jsonl")


def _download_json(url: str) -> dict:
    print(f"[prepare_squad] downloading {url}")
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _iter_rows(raw: dict):
    """SQuAD v1.1 구조: data → [{title, paragraphs → [{context, qas → [{id, question, answers}]}]}]"""
    for article in raw.get("data", []):
        title = article.get("title")
        for paragraph in article.get("paragraphs", []):
            context = str(paragraph.get("context", "")).strip()
            if not context:
                continue
            for qa in paragraph.get("qas", []):
                question = str(qa.get("question", "")).strip()
                qa_id = qa.get("id")
                answers = qa.get("answers") or []
                answer = ""
                for ans in answers:
                    text = ans.get("text") if isinstance(ans, dict) else None
                    if isinstance(text, str) and text.strip():
                        answer = text.strip()
                        break
                if not question or not answer:
                    continue
                yield {
                    "source_id": qa_id,
                    "task": "final_qa",
                    "question": question,
                    "answer": answer,
                    "passage": context,
                    "title": title,
                }


def _dump_split(raw: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in _iter_rows(raw):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[prepare_squad] saved {kept} rows -> {output_path}")


def main() -> None:
    train_raw = _download_json(TRAIN_URL)
    valid_raw = _download_json(VALID_URL)
    _dump_split(train_raw, OUT_TRAIN_PATH)
    _dump_split(valid_raw, OUT_VALID_PATH)
    print("[prepare_squad] done")


if __name__ == "__main__":
    main()
