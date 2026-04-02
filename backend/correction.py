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


def _trim_repetition(text: str) -> str:
    """반복 패턴이 시작되는 지점을 찾아 잘라낸다."""
    tokens = text.split()
    for i in range(len(tokens)):
        # 같은 토큰이 3번 연속 반복되면 그 지점에서 자름
        if i + 2 < len(tokens) and tokens[i] == tokens[i + 1] == tokens[i + 2]:
            return ' '.join(tokens[:i]).strip()
        # 2-gram 반복 감지 (A B A B)
        if i + 3 < len(tokens) and tokens[i] == tokens[i + 2] and tokens[i + 1] == tokens[i + 3]:
            return ' '.join(tokens[:i]).strip()
    return text


def correct_text(text: str) -> str:
    """Whisper 전사 텍스트를 교정하여 반환한다."""
    if _model is None or _tokenizer is None:
        return text

    if not text or not text.strip():
        return text

    inputs = _tokenizer(
        text,
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
        )

    corrected = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 반복 패딩 제거
    corrected = _trim_repetition(corrected)

    if not corrected:
        return text

    return corrected
