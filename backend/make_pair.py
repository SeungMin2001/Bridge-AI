from datasets import load_dataset

DATA_DIR = r"C:\Users\user\Documents\ksponspeech_data"

ds = load_dataset(
    "cheulyop/ksponspeech",
    data_dir=DATA_DIR,
    split="train[:3]"
)

print(ds.column_names)
print(ds[0])