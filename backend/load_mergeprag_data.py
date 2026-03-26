from datasets import load_from_disk
import json

# 1. 샘플 1개에서 fact 추출
def extract_facts_from_sample(sample):
    question = sample["question"]
    answer = sample["answer"]

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

    return {
        "question": question,
        "answer": answer,
        "facts": facts
    }


# 2. 전체 split 저장
def save_dataset_to_jsonl(dataset_path, output_path):
    dataset = load_from_disk(dataset_path)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            converted = extract_facts_from_sample(sample)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1

    print(f"saved {count} samples to {output_path}")


# 3. 경로 설정
train_path = r"C:\Users\user\Documents\last_project\data\hotpotqa\train"
valid_path = r"C:\Users\user\Documents\last_project\data\hotpotqa\validation"

train_output = r"C:\Users\user\Documents\last_project\data\HotPot_train_min.jsonl"
valid_output = r"C:\Users\user\Documents\last_project\data\HotPot_valid_min.jsonl"


# 4. 실행
save_dataset_to_jsonl(train_path, train_output)
save_dataset_to_jsonl(valid_path, valid_output)