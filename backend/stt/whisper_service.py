"""Whisper STT runtime shared by websocket recording and uploaded audio transcription.

This module owns model loading, resampling, segment filtering, and optional text
correction so feature code can call a small API instead of duplicating STT setup.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchaudio

from correction import correct_text, load_correction_model


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

STT_BACKEND = os.getenv("STT_BACKEND", "faster-whisper").strip().lower().replace("_", "-")
requested_stt_device = os.getenv("STT_DEVICE", "").strip().lower()
requested_stt_model = os.getenv("STT_MODEL", "").strip()


def _mps_available() -> bool:
    """Return whether the current PyTorch runtime can use Apple MPS."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _load_stt_model():
    """Load the configured Whisper backend once for this process."""
    if STT_BACKEND in {"faster", "faster-whisper"}:
        from faster_whisper import WhisperModel

        if requested_stt_device in {"cuda", "cpu"}:
            device = requested_stt_device
        elif requested_stt_device == "mps":
            print("[STT] faster-whisper는 mps를 지원하지 않아 cpu로 실행합니다.")
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = requested_stt_model or "large-v3-turbo"
        return (
            WhisperModel(
                model_name,
                device=device,
                compute_type="float16" if device == "cuda" else "float32",
            ),
            device,
            model_name,
        )

    if STT_BACKEND in {"openai", "openai-whisper", "whisper"}:
        import whisper as openai_whisper

        if not hasattr(openai_whisper, "load_model"):
            raise RuntimeError(
                "openai-whisper 패키지가 아니라 다른 whisper 패키지가 import되었습니다. "
                "requirements의 whisper==1.1.10을 제거한 뒤 다시 설치해주세요."
            )

        if requested_stt_device == "mps":
            if _mps_available():
                device = "mps"
            else:
                print("[STT] 요청한 mps를 사용할 수 없어 cpu로 실행합니다.")
                device = "cpu"
        elif requested_stt_device == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_stt_device == "cpu":
            device = "cpu"
        else:
            device = "mps" if _mps_available() else ("cuda" if torch.cuda.is_available() else "cpu")

        model_name = requested_stt_model or "turbo"
        if device == "mps":
            stt_model = openai_whisper.load_model(model_name, device="cpu")
            alignment_heads = getattr(stt_model, "alignment_heads", None)
            if alignment_heads is not None and getattr(alignment_heads, "is_sparse", False):
                stt_model.register_buffer("alignment_heads", alignment_heads.to_dense(), persistent=False)
            return stt_model.to("mps"), device, model_name

        return openai_whisper.load_model(model_name, device=device), device, model_name

    raise ValueError(
        "지원하지 않는 STT_BACKEND입니다. "
        "faster-whisper 또는 openai-whisper 중 하나를 사용하세요."
    )


model, stt_device, stt_model_name = _load_stt_model()
audio_device = "cuda" if stt_device == "cuda" else "cpu"
print(f"[STT] backend={STT_BACKEND}, model={stt_model_name}, device={stt_device}")

correction_enabled = load_correction_model()

transcribe_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
correction_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="correction")

# STT가 mps여도 torchaudio resample은 CPU/CUDA 쪽이 안정적이라 별도 장치로 처리합니다.
_pcm48_resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000).to(audio_device)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _filter_whisper_segment_dicts(segments: list[dict]) -> list[dict]:
    """Filter low-confidence Whisper segments while preserving timestamps."""
    filtered = []
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue

        no_speech_prob = _safe_float(seg.get("no_speech_prob"), 0.0)
        avg_logprob = _safe_float(seg.get("avg_logprob"), 0.0)
        if no_speech_prob > 0.6:
            print(f"[필터] no_speech_prob={no_speech_prob:.2f} → 제거: {text!r}")
            continue
        if avg_logprob < -1.0:
            print(f"[필터] avg_logprob={avg_logprob:.2f} → 제거: {text!r}")
            continue

        filtered.append({
            "start": _safe_float(seg.get("start"), 0.0),
            "end": _safe_float(seg.get("end"), 0.0),
            "text": text,
            "raw_text": text,
        })
    return filtered


def _join_segment_texts(segments: list[dict]) -> str:
    """Join filtered segment texts into one chunk string."""
    return " ".join(seg["text"] for seg in segments if seg.get("text")).strip()


def resample_pcm48_to_16k(audio_float: np.ndarray) -> np.ndarray:
    """Convert browser 48kHz float PCM to 16kHz float PCM for STT/diarization."""
    audio_tensor = torch.from_numpy(audio_float.astype(np.float32, copy=False)).to(audio_device)
    return _pcm48_resampler(audio_tensor).cpu().numpy()


def _transcribe_faster_whisper_audio(audio_source) -> list[dict]:
    """Transcribe a 16kHz numpy array or media file path with faster-whisper."""
    segments, _ = model.transcribe(
        audio_source,
        language="ko",
        task="transcribe",
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    return _filter_whisper_segment_dicts([
        {
            "start": getattr(seg, "start", 0.0),
            "end": getattr(seg, "end", 0.0),
            "text": getattr(seg, "text", ""),
            "no_speech_prob": getattr(seg, "no_speech_prob", 0.0),
            "avg_logprob": getattr(seg, "avg_logprob", 0.0),
        }
        for seg in segments
    ])


def _transcribe_openai_whisper_audio(audio_source) -> list[dict]:
    """Transcribe a 16kHz numpy array or media file path with openai-whisper."""
    if isinstance(audio_source, np.ndarray):
        audio_source = audio_source.astype(np.float32, copy=False)

    result = model.transcribe(
        audio_source,
        language="ko",
        task="transcribe",
        temperature=0.0,
        condition_on_previous_text=False,
        fp16=stt_device == "cuda",
        verbose=None,
    )
    segments = _filter_whisper_segment_dicts(result.get("segments") or [])
    if segments:
        return segments

    fallback_text = str(result.get("text") or "").strip()
    return [{"start": 0.0, "end": 0.0, "text": fallback_text, "raw_text": fallback_text}] if fallback_text else []


def transcribe_16k_chunk(audio_16k: np.ndarray) -> str:
    """Transcribe one realtime 16kHz audio chunk into plain text."""
    if STT_BACKEND in {"faster", "faster-whisper"}:
        return _join_segment_texts(_transcribe_faster_whisper_audio(audio_16k))
    if STT_BACKEND in {"openai", "openai-whisper", "whisper"}:
        return _join_segment_texts(_transcribe_openai_whisper_audio(audio_16k))
    return ""


def transcribe_audio_file(audio_path: str | Path) -> list[dict]:
    """Transcribe an uploaded audio file and return timestamped text segments."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    source = str(path)
    if STT_BACKEND in {"faster", "faster-whisper"}:
        return _transcribe_faster_whisper_audio(source)
    if STT_BACKEND in {"openai", "openai-whisper", "whisper"}:
        return _transcribe_openai_whisper_audio(source)
    return []


def correct_transcript_text(text: str) -> str:
    """Run optional STT correction and fall back to the original text on failure."""
    try:
        return correct_text(text)
    except Exception as exc:
        print(f"[교정] 교정 실패, raw_text 사용: {exc}")
        return text

