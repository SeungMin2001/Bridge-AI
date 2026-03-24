from datasets import load_dataset

ds = load_dataset(
    "DragonLine/ksponspeech",
    split="train[:100]",
    cache_dir=r"C:\Users\user\Documents\last_project\data"
)

print(ds.features)
print(ds[0])