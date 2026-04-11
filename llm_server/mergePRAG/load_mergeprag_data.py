from datasets import load_dataset
import json


# ── HotPotQA 변환 ──
def extract_hotpot(sample):
    sf_titles = sample["supporting_facts"]["title"]
    sf_sent_ids = sample["supporting_facts"]["sent_id"]
    ctx_titles = sample["context"]["title"]
    ctx_sentences = sample["context"]["sentences"]

    facts = []
    for sf_title, sf_sent_id in zip(sf_titles, sf_sent_ids):
        for ctx_title, sentences in zip(ctx_titles, ctx_sentences):
            if ctx_title == sf_title:
                if 0 <= sf_sent_id < len(sentences):
                    facts.append(sentences[sf_sent_id].strip())
                break

    return {"question": sample["question"], "answer": sample["answer"], "facts": facts}


# ── MuSiQue 변환 ──
def extract_musique(sample):
    if not sample.get("answerable", True):
        return None

    # is_supporting=True인 paragraph만 facts로 사용
    facts = [
        p["paragraph_text"].strip()
        for p in sample["paragraphs"]
        if p.get("is_supporting", False)
    ]

    if not facts:
        return None

    return {"question": sample["question"], "answer": sample["answer"], "facts": facts}


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
    TRAIN_OUT = r"C:\Users\user\Documents\last_project\data\MuSiQue_train.jsonl"
    VALID_OUT  = r"C:\Users\user\Documents\last_project\data\MuSiQue_valid.jsonl"

    print("MuSiQue 다운로드 중...")
    ds = load_dataset("dgslibisey/MuSiQue")

    print(f"train: {len(ds['train'])}개, validation: {len(ds['validation'])}개")

    save_to_jsonl(ds["train"], TRAIN_OUT, extract_musique)
    save_to_jsonl(ds["validation"], VALID_OUT, extract_musique)

    print("완료!")
