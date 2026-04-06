"""
diart 화자 분리 마이크로서비스 (포트 8003)

잡음 제거된 오디오를 받아 화자 분리(Speaker Diarization)를 수행합니다.
diart + pyannote.audio 기반 / 별도 Docker 컨테이너에서 실행.

파이프라인 순서: 프론트 녹음 → 잡음 제거(:8002) → 화자 분리(:8003) → STT(:8000)

API:
  POST /diart      - WAV 파일 → JSON (화자별 세그먼트)
  POST /diart/raw  - float32 PCM bytes → JSON (화자별 세그먼트)
  GET  /health       - 서버 상태 확인

[필수] HF_TOKEN 환경변수:
  pyannote 모델 사용을 위해 HuggingFace 토큰이 필요합니다.
"""

import io
import logging
import os
import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample
from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="diart Server")

# 글로벌 파이프라인 변수
pipeline = None
PIPELINE_SAMPLE_RATE = 16000


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 화자 분리 파이프라인 로드"""
    global pipeline, PIPELINE_SAMPLE_RATE

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        logger.warning("HF_TOKEN이 설정되지 않았습니다! pyannote 모델 다운로드가 실패할 수 있습니다.")

    logger.info("화자 분리 파이프라인 로딩 중...")

    # 1차: pyannote.audio Pipeline 직접 사용 (더 안정적)
    try:
        from pyannote.audio import Pipeline as PyannotePipeline

        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token if hf_token else True
        )
        PIPELINE_SAMPLE_RATE = 16000
        logger.info("pyannote.audio 파이프라인 로딩 완료.")
        return

    except Exception as e:
        logger.warning(f"pyannote.audio 로딩 실패: {e}")

    # 2차 폴백: diart의 SpeakerDiarization
    try:
        from diart import SpeakerDiarization
        from diart.models import SegmentationModel, EmbeddingModel

        segmentation = SegmentationModel.from_pretrained(
            "pyannote/segmentation-3.0",
            use_auth_token=hf_token if hf_token else True
        )
        embedding = EmbeddingModel.from_pretrained(
            "pyannote/embedding",
            use_auth_token=hf_token if hf_token else True
        )

        pipeline = SpeakerDiarization(
            segmentation=segmentation,
            embedding=embedding,
        )
        PIPELINE_SAMPLE_RATE = pipeline.config.sample_rate
        logger.info(f"diart 파이프라인 로딩 완료. (SR: {PIPELINE_SAMPLE_RATE}Hz)")

    except Exception as e2:
        logger.error(f"diart 파이프라인 로딩도 실패: {e2}")
        pipeline = None


@app.get("/health")
async def health():
    """헬스체크"""
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "sample_rate": PIPELINE_SAMPLE_RATE,
        "pipeline_type": type(pipeline).__name__ if pipeline else None
    }


def run_diarization(audio_np: np.ndarray, sample_rate: int) -> list:
    """
    오디오 numpy 배열에 대해 화자 분리를 수행합니다.

    Returns:
        list[dict]: [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.5, "duration": 1.5}, ...]
    """
    # SR 맞추기
    if sample_rate != PIPELINE_SAMPLE_RATE:
        num_samples = int(len(audio_np) * PIPELINE_SAMPLE_RATE / sample_rate)
        audio_np = resample(audio_np, num_samples).astype(np.float32)
        sample_rate = PIPELINE_SAMPLE_RATE

    # pyannote 형식: {"waveform": [1, T] tensor, "sample_rate": int}
    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    # 화자 분리 실행
    diarization = pipeline(audio_input)

    # 결과 변환
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(turn.end - turn.start, 3)
        })

    return segments


@app.post("/diart")
async def diarize_wav(file: UploadFile = File(...)):
    """WAV 파일 → 화자 분리 JSON"""
    if pipeline is None:
        return JSONResponse(content={"error": "Pipeline not loaded"}, status_code=503)

    contents = await file.read()
    audio_np, original_sr = sf.read(io.BytesIO(contents), dtype="float32")

    logger.info(f"[diart] sr={original_sr}Hz, duration={len(audio_np)/original_sr:.2f}s")

    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)

    segments = run_diarization(audio_np, int(original_sr))
    unique_speakers = set(s["speaker"] for s in segments)

    return JSONResponse(content={
        "segments": segments,
        "num_speakers": len(unique_speakers),
        "duration": round(len(audio_np) / original_sr, 3)
    })


@app.post("/diart/raw")
async def diarize_raw(
    request: Request,
    sample_rate: int = Query(default=16000, description="입력 오디오 sample rate"),
):
    """float32 PCM raw bytes → 화자 분리 JSON"""
    if pipeline is None:
        return JSONResponse(content={"error": "Pipeline not loaded"}, status_code=503)

    request_body = await request.body()
    audio_np = np.frombuffer(request_body, dtype=np.float32)

    if len(audio_np) == 0:
        return JSONResponse(content={"segments": [], "num_speakers": 0, "duration": 0})

    logger.info(f"[diart/raw] sr={sample_rate}Hz, samples={len(audio_np)}, "
                f"duration={len(audio_np)/sample_rate:.2f}s")

    segments = run_diarization(audio_np, sample_rate)
    unique_speakers = set(s["speaker"] for s in segments)

    return JSONResponse(content={
        "segments": segments,
        "num_speakers": len(unique_speakers),
        "duration": round(len(audio_np) / sample_rate, 3)
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
