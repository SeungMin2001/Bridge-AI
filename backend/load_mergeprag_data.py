from datasets import load_from_disk
import json

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
                    facts.append(sentences[sf_sent_id])
                break

    return {
        "question": question,
        "answer": answer,
        "facts": facts
    }
    
dataset = load_from_disk(r"C:\Users\user\Documents\last_project\data\hotpotqa\train")
sample = dataset[0]

converted = extract_facts_from_sample(sample)
print(json.dumps(converted, ensure_ascii=False, indent=2))