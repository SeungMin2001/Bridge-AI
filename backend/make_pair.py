import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset
from faster_whisper import WhisperModel

# =========================
# 경로 설정
# =========================
CACHE_DIR = r"C:\Users\user\Documents\last_project\data\hf_cache"
TEMP_WAV_DIR = r"C:\Users\user\Documents\last_project\data\temp_audio"
OUTPUT_DIR = r"C:\Users\user\Documents\last_project\data\pair_dataset"

os.makedirs(TEMP_WAV_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Whisper 설정
# =========================
WHISPER_SIZE = "large-v3"      # 테스트 후 "large-v3"로 변경 가능
DEVICE = "cuda"             # GPU 없으면 "cpu"
COMPUTE_TYPE = "float16"    # CPU면 보통 "int8"
SAVE_INTERVAL = 100         # 중간 저장 주기

# =========================
# KsponSpeech 전처리
# =========================
def clean_kspon_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ")

    # b/, o/, l/, n/ 제거
    text = re.sub(r"\b[boln]/\s*", "", text)

    # (철자)/(발음) -> 철자만 사용
    text = re.sub(r"\(([^()]+)\)/\(([^()]+)\)", r"\1", text)

    # 별표 제거
    text = text.replace("*", "")

    # 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# Whisper 전사
# =========================
def transcribe_with_whisper(model: WhisperModel, wav_path: str) -> str:
    segments, _ = model.transcribe(wav_path, language="ko")
    text = " ".join(seg.text.strip() for seg in segments).strip()
    text = re.sub(r"\s+", " ", text)
    return text

# =========================
# split별 pair 생성
# =========================
def make_pair_csv(split_name: str, output_csv: str, model: WhisperModel):
    print(f"\n[{split_name}] 데이터 로드 시작")
    ds = load_dataset(
        "DragonLine/ksponspeech",
        split=split_name,
        cache_dir=CACHE_DIR
    )

    rows = []

    print(f"[{split_name}] pair 생성 시작")
    for i, item in enumerate(tqdm(ds, desc=split_name)):
        try:
            target_text = clean_kspon_text(item["transcripts"])
            if not target_text:
                continue

            wav_path = os.path.join(TEMP_WAV_DIR, f"{split_name}_{i}.wav")
            sf.write(
                wav_path,
                item["audio"]["array"],
                item["audio"]["sampling_rate"]
            )

            input_text = transcribe_with_whisper(model, wav_path)
            if not input_text:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                continue

            rows.append({
                "input_text": input_text,
                "target_text": target_text
            })

            if len(rows) % SAVE_INTERVAL == 0:
                pd.DataFrame(rows).to_csv(
                    output_csv,
                    index=False,
                    encoding="utf-8-sig"
                )

            if os.path.exists(wav_path):
                os.remove(wav_path)

        except Exception as e:
            print(f"[에러] split={split_name}, index={i}, error={e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"[{split_name}] 완료")
    print(f"저장 경로: {output_csv}")
    print(f"총 pair 수: {len(df)}")

# =========================
# 메인 실행
# =========================
def main():
    print("Whisper 모델 로드 시작")
    model = WhisperModel(
        WHISPER_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
    print("Whisper 모델 로드 완료")

    make_pair_csv(
        split_name="train",
        output_csv=os.path.join(OUTPUT_DIR, "train_pairs.csv"),
        model=model
    )

    make_pair_csv(
        split_name="valid",
        output_csv=os.path.join(OUTPUT_DIR, "valid_pairs.csv"),
        model=model
    )

    make_pair_csv(
        split_name="test",
        output_csv=os.path.join(OUTPUT_DIR, "test_pairs.csv"),
        model=model
    )

    print("\n모든 pair 생성 완료")

if __name__ == "__main__":
    main()
>>>>>>> shin
