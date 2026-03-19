import pandas as pd

train_path = "C:/Users/user/Documents/ksponspeech_data/train_pairs.csv"
valid_path = "C:/Users/user/Documents/ksponspeech_data/valid_pairs.csv"

train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)

# 필요한 컬럼만 남기고 결측 제거
train_df = train_df[["input_text", "target_text"]].dropna()
valid_df = valid_df[["input_text", "target_text"]].dropna()

# 문자열로 강제 변환
train_df["input_text"] = train_df["input_text"].astype(str)
train_df["target_text"] = train_df["target_text"].astype(str)
valid_df["input_text"] = valid_df["input_text"].astype(str)
valid_df["target_text"] = valid_df["target_text"].astype(str)

# 저장
train_df.to_csv("C:/Users/user/Documents/ksponspeech_data/train_pairs_clean.csv", index=False, encoding="utf-8-sig")
valid_df.to_csv("C:/Users/user/Documents/ksponspeech_data/valid_pairs_clean.csv", index=False, encoding="utf-8-sig")