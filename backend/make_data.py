import re
import torch
import whisper
import numpy as np
import pandas as pd
from datasets import load_dataset

name = ["train", "validation", "test"]   # 필요하면 vaild로 다시 수정
cache_path = r"C:\Users\user\Documents\ksponspeech_data"

device = "mps" if torch.backends.mps.is_available() else "cuda"
print("device:", device)

model = whisper.load_model("large-v3", device=device)

def extract_spelling_transcript(text: str) -> str:
    text = text.strip()
    text = text.replace("\n", " ")
    text = re.sub(r"[a-zA-Z]/\s*", "", text)
    text = text.replace("*", "")
    text = re.sub(r"\(([^()]*)\)/\(([^()]*)\)", r"\1", text)
    text = text.replace("(", "").replace(")", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

for k in name:
    data = load_dataset(
        "DragonLine/ksponspeech",
        split=f"{k}[:1000]",   # 각 split에서 1000개만
        cache_dir=cache_path
    )

    rows = []

    for i, sample in enumerate(data):
        audio_array = sample["audio"]["array"].astype(np.float32)
        raw_target = sample["transcripts"]
        target_text = extract_spelling_transcript(raw_target)

        result = model.transcribe(
            audio_array,
            language="ko",
            task="transcribe",
            fp16=False,
            temperature=0.0,
            condition_on_previous_text=False,
            verbose=False,
        )

        input_text = result["text"].strip()

        rows.append({
            "input_text": input_text,
            "target_text": target_text,
            "raw_target": raw_target
        })

        print(f"{i+1}/{len(data)}")
        print("[INPUT     ]", input_text)
        print("[RAW TARGET ]", raw_target)
        print("[TARGET     ]", target_text)
        print("-" * 60)

    df = pd.DataFrame(rows)
    df.to_csv(
        fr"C:\Users\user\Documents\ksponspeech_data\{k}_pairs_1000.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(f"saved: {k}_pairs_1000.csv")