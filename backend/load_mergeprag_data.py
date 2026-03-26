from datasets import load_from_disk
import json

dataset = load_from_disk(r"C:\Users\user\Documents\last_project\data\hotpotqa\train")
sample = dataset[0]

print("question:", sample["question"])
print("answer:", sample["answer"])
print("supporting_facts:", sample["supporting_facts"])
print("context keys:", sample["context"].keys())
print("context titles example:", sample["context"]["title"][:2])
print("context sentences example:", sample["context"]["sentences"][:2])