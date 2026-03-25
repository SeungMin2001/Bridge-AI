from datasets import load_dataset
import os

# save_dir = r"C:\Users\user\Documents\last_project\data\hotpotqa"

# os.makedirs(save_dir, exist_ok=True)

# dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")
# dataset.save_to_disk(save_dir)

# print(dataset)
# print(f"saved to: {save_dir}")

from datasets import load_from_disk

train_ds = load_from_disk(r"C:\Users\user\Documents\last_project\data\hotpotqa\train")
print(train_ds)
print("length:", len(train_ds))
print(train_ds[0].keys())
print(train_ds[0])