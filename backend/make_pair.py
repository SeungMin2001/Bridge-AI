from datasets import load_dataset
from faster_whisper import WhisperModel
import pandas as pd
import re, os
import soundfile as sf

ds = load_dataset(
    "DragonLine/ksponspeech",
    split="train[:100]",
    cache_dir=r"C:\Users\user\Documents\last_project\data"
)

def clean_text(text):
    text = text.replace("b/", "")
    text = text.replace("*", "")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

sample_ds = ds.select(range(20))
model = WhisperModel("small", device="cuda", compute_type="float16")

os.makedirs("temp_audio", exist_ok=True)
rows = []

for i, item in enumerate(sample_ds):
    gold_text = clean_text(item["transcripts"])

    wav_path = f"temp_audio/sample_{i}.wav"
    sf.write(wav_path, item["audio"]["array"], item["audio"]["sampling_rate"])

    segments, _ = model.transcribe(wav_path, language="ko")
    whisper_text = " ".join(seg.text.strip() for seg in segments).strip()

    rows.append({
        "input_text": whisper_text,
        "target_text": gold_text
    })

pd.DataFrame(rows).to_csv("sample_pairs.csv", index=False, encoding="utf-8-sig")
print("완료")