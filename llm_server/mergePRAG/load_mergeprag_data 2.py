from datasets import load_dataset
import json


# ── NarrativeQA 변환 ──
def extract_narrativeqa(sample):
    # Wikipedia 요약을 passage로 사용 (full text는 수만 단어라 너무 김)
    passage = sample["document"]["summary"]["text"].strip()
    question = sample["question"]["text"].strip()

    # 답변이 여러 개일 수 있으므로 첫 번째 사용
    answers = sample["answers"]
    if not answers:
        return None
    answer = answers[0]["text"].strip()

    if not passage or not question or not answer:
        return None

    return {"question": question, "answer": answer, "facts": [passage]}


# ── 저장 ──
def save_to_jsonl(samples, output_path, extract_fn):
    count = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            converted = extract_fn(sample)
            if converted is None or not converted["facts"]:
                skipped += 1
                continue
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
    print(f"saved {count} samples (skipped {skipped}) → {output_path}")


# ── 실행 ──
if __name__ == "__main__":
    TRAIN_OUT = r"C:\Users\user\Documents\last_project\data\NarrativeQA_train.jsonl"
    VALID_OUT  = r"C:\Users\user\Documents\last_project\data\NarrativeQA_valid.jsonl"

    print("NarrativeQA 다운로드 중...")
    ds = load_dataset("deepmind/narrativeqa")

    print(f"train: {len(ds['train'])}개, validation: {len(ds['validation'])}개")

    save_to_jsonl(ds["train"], TRAIN_OUT, extract_narrativeqa)
    save_to_jsonl(ds["validation"], VALID_OUT, extract_narrativeqa)

    print("완료!")
