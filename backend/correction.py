"""
KoBART 기반 전사 텍스트 교정 모듈
Whisper STT 출력을 fine-tuned KoBART 모델로 교정한다.
"""
import os
import re
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

# 모델 경로: 환경변수 또는 기본 경로
MODEL_PATH = os.environ.get(
    "KOBART_MODEL_PATH",
    r"C:\Users\user\Documents\last_project\models\kobart_correction"
)

_tokenizer = None
_model = None
_device = None


def load_correction_model():
    """교정 모델을 로드한다. 서버 시작 시 1회 호출."""
    global _tokenizer, _model, _device

    if not os.path.isdir(MODEL_PATH):
        print(f"[교정] 모델 경로 없음: {MODEL_PATH}")
        print("[교정] 교정 모델 없이 실행합니다. (raw_text 그대로 반환)")
        return False

    _device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[교정] 모델 로딩: {MODEL_PATH} (device={_device})")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    _model.to(_device)
    _model.eval()
    print("[교정] 모델 로드 완료")
    return True


def _clean_fillers(text: str) -> str:
    """필러 단어 및 불필요한 기호를 제거한다."""
    # "어.", "응.", "음.", "어/", "어+", "어," 등 필러 패턴 제거
    text = re.sub(r'[어응음으으음아에]+[./?+,!]*\s*', '', text)
    # "u/" 같은 비한글 필러 제거
    text = re.sub(r'\bu/\b', '', text)
    # 연속 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def correct_text(text: str) -> str:
    """Whisper 전사 텍스트를 교정하여 반환한다."""
    # 1단계: 필러 제거
    cleaned = _clean_fillers(text)
    if not cleaned:
        return text

    # 모델이 없으면 필러 제거만 적용
    if _model is None or _tokenizer is None:
        return cleaned

    # 2단계: KoBART 교정
    inputs = _tokenizer(
        cleaned,
        return_tensors="pt",
        max_length=128,
        truncation=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
        )

    corrected = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 3단계: 품질 체크 - 교정 결과가 원본보다 짧거나 이상하면 필러제거본 반환
    if len(corrected) < len(cleaned) * 0.3 or not corrected:
        return cleaned

    return corrected
