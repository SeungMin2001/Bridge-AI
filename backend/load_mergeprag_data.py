from datasets import load_dataset
import os

save_dir = r"C:\Users\user\Documents\last_project\data\hotpotqa"

os.makedirs(save_dir, exist_ok=True)

dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")
dataset.save_to_disk(save_dir)

print(dataset)
print(f"saved to: {save_dir}")