import os
import re
import pandas as pd
from datasets import load_dataset, Audio
from transformers import pipeline
from tqdm import tqdm

# =========================
# 설정
# =========================
DATASET_NAME = "cheulyop/ksponspeech"   # 필요시 다른 KsponSpeech 미러로 변경
TRAIN_SPLIT = "train"
VALID_SPLIT = "validation"              # 없으면 train 일부를 나눠도 됨

WHISPER_MODEL = "openai/whisper-large-v3"
DEVICE = 0  # GPU: 0, CPU: -1

TRAIN_MAX_SAMPLES = 10   # 처음엔 작게
VALID_MAX_SAMPLES = 10

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# 텍스트 정리
# =========================
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()

    # KsponSpeech류에 자주 있는 표기 노이즈 최소 정리
    # 예: "아/ 우리랑 l/" 같은 표식, 여러 공백 등
    text = re.sub(r"[+/l]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 괄호 내 대안표기 처리:
    # "(1년)/(일 년)" -> "1년"
    # "(그러니까)/(그까)" -> "그러니까"
    def pick_first_option(match):
        content = match.group(0)
        parts = content.split("/")
        if parts:
            return parts[0].replace("(", "").replace(")", "").strip()
        return content

    text = re.sub(r"\([^)]*\)(/\([^)]*\))+", pick_first_option, text)

    return text


# =========================
# Whisper 파이프라인
# =========================
asr = pipeline(
    "automatic-speech-recognition",
    model=WHISPER_MODEL,
    device=DEVICE,
)

def transcribe_audio(audio_item):
    """
    audio_item 예시:
    {"array": np.ndarray, "sampling_rate": 16000, "path": "..."}
    """
    result = asr(
        {"array": audio_item["array"], "sampling_rate": audio_item["sampling_rate"]},
        generate_kwargs={"language": "ko", "task": "transcribe"},
        return_timestamps=False,
    )
    return result["text"].strip()


def detect_text_column(ds):
    # 후보 컬럼명 탐색
    candidates = ["sentence", "transcript", "text", "label"]
    for c in candidates:
        if c in ds.column_names:
            return c
    raise ValueError(f"텍스트 컬럼을 찾지 못함. columns={ds.column_names}")

def build_pairs(split_name, max_samples, out_csv):
    ds = load_dataset(DATASET_NAME, split=split_name)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    text_col = detect_text_column(ds)

    records = []
    total = min(len(ds), max_samples)

    for ex in tqdm(ds.select(range(total)), total=total, desc=f"building {split_name}"):
        gold_text = normalize_text(ex[text_col])
        if not gold_text:
            continue

        try:
            whisper_text = transcribe_audio(ex["audio"])
            whisper_text = normalize_text(whisper_text)

            if not whisper_text:
                continue

            records.append({
                "input_text": whisper_text,
                "target_text": gold_text
            })
        except Exception as e:
            print(f"skip due to error: {e}")
            continue

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"saved: {out_csv}, rows={len(df)}")


if __name__ == "__main__":
    build_pairs(TRAIN_SPLIT, TRAIN_MAX_SAMPLES, os.path.join(OUT_DIR, "pairs_train.csv"))
    build_pairs(VALID_SPLIT, VALID_MAX_SAMPLES, os.path.join(OUT_DIR, "pairs_valid.csv"))