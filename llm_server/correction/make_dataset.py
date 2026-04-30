from datasets import load_dataset

data_files = {
    "train": r"C:\Users\user\Documents\last_project\data\pair_dataset\train_pairs.csv",
    "validation": r"C:\Users\user\Documents\last_project\data\pair_dataset\valid_pairs.csv",
    "test": r"C:\Users\user\Documents\last_project\data\pair_dataset\test_pairs.csv"
}

dataset = load_dataset("csv", data_files=data_files)

