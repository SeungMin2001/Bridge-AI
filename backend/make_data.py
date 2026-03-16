import re
import pandas as pd
import whisper
from datasets import load_dataset
import torch
import numpy as np

name=["train","vaild","test"]
cache_path = r"C:\Users\user\Documents\ksponspeech_data"

for k in name:
    data = load_dataset(
        "DragonLine/ksponspeech",
        split=k,
        cache_dir=cache_path
    )

    device="mps" if torch.backends.mps.is_available() else "cuda"

    print("device: ",device)

    model = whisper.load_model("large-v3",device=device)

    def clean_transcript(text: str) -> str:
        text = text.strip()
        text = text.replace("\n", " ")
        text = re.sub(r"[a-zA-Z]/\s*", "", text)   # b/ 같은 표기 제거
        text = text.replace("*", "")               # * 제거
        text = re.sub(r"\s+", " ", text).strip()   # 공백 정리
        return text

    rows = []

    for i, sample in enumerate(data):
        audio_array = sample["audio"]["array"].astype(np.float32)
        target_text = clean_transcript(sample["transcripts"])

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
            "target_text": target_text
        })

        print(f"{i+1}/{len(data)}")
        print("[INPUT ]", input_text)
        print("[TARGET]", target_text)
        print("-" * 50)

    df = pd.DataFrame(rows)
    df.to_csv(
        fr"C:\Users\user\Documents\ksponspeech_data\{k}_pairs.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(f"saved: {k}_pairs.csv")