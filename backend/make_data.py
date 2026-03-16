import pandas as pd
import whisper
from datasets import load_dataset
import torch

device="mps" if torch.backends.mps.is_available() else "cuda"
model=whisper.load_model("turbo",device=device) #모델설정(transcript할 모델)

rows = []

# 예시: 공개 데이터셋 한 부분을 불러온 뒤 일부만 사용
# 실제 컬럼명은 데이터셋마다 다를 수 있으니 print(sample)로 먼저 확인
ds = load_dataset("cheulyop/ksponspeech", split="validation")

for i, sample in enumerate(ds.select(range(100))):
    audio_path = sample["audio"]["path"]
    target_text = sample["sentence"].strip()   # 컬럼명은 확인 필요

    result = model.transcribe(audio_path, language="ko")
    input_text = result["text"].strip()

    rows.append({
        "input_text": input_text,
        "target_text": target_text
    })

df = pd.DataFrame(rows)
df.to_csv("train.csv", index=False, encoding="utf-8-sig")
print(df.head())