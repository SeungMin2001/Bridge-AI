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
PIPELINE_DEVICE = "cpu"


def resolve_diarize_device() -> str:
    """DIARIZE_DEVICE 환경변수를 실제로 사용할 torch device 이름으로 정리합니다."""
    requested = os.getenv("DIARIZE_DEVICE", "cpu").strip().lower()

    if requested in {"", "cpu"}:
        return "cpu"
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        logger.warning("DIARIZE_DEVICE=mps를 요청했지만 MPS를 사용할 수 없어 CPU로 실행합니다.")
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("DIARIZE_DEVICE=cuda를 요청했지만 CUDA를 사용할 수 없어 CPU로 실행합니다.")
        return "cpu"

    logger.warning(f"알 수 없는 DIARIZE_DEVICE={requested!r} 값입니다. CPU로 실행합니다.")
    return "cpu"


def move_pipeline_to_device(loaded_pipeline):
    """로드된 화자분리 파이프라인을 요청한 장치로 이동합니다."""
    global PIPELINE_DEVICE

    device_name = resolve_diarize_device()
    if device_name == "cpu":
        PIPELINE_DEVICE = "cpu"
        logger.info("화자분리 파이프라인 장치: cpu")
        return loaded_pipeline

    if not hasattr(loaded_pipeline, "to"):
        logger.warning(
            f"현재 파이프라인은 .to(...)를 지원하지 않아 DIARIZE_DEVICE={device_name} 적용을 건너뜁니다."
        )
        PIPELINE_DEVICE = "cpu"
        return loaded_pipeline

    try:
        # 신창영: 수정 이유 - 녹음 종료 후 전체 오디오 화자분리처럼 긴 작업에서 CPU 대신 MPS/CUDA를 실험할 수 있게 합니다.
        loaded_pipeline.to(torch.device(device_name))
        PIPELINE_DEVICE = device_name
        logger.info(f"화자분리 파이프라인 장치: {device_name}")
        return loaded_pipeline
    except Exception as e:
        logger.warning(f"DIARIZE_DEVICE={device_name} 적용 실패, CPU로 계속 실행합니다: {e}")
        try:
            loaded_pipeline.to(torch.device("cpu"))
        except Exception:
            pass
        PIPELINE_DEVICE = "cpu"
        return loaded_pipeline


def patch_huggingface_hub_auth_arg():
    """pyannote.audio 3.1.x의 use_auth_token 호출을 최신 huggingface_hub에 맞춥니다."""
    try:
        import inspect
        import huggingface_hub
    except Exception:
        return

    original_download = getattr(huggingface_hub, "hf_hub_download", None)
    if original_download is None or getattr(original_download, "_diart_auth_arg_patch", False):
        return

    try:
        parameters = inspect.signature(original_download).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "use_auth_token" in parameters:
        return

    def hf_hub_download_compat(*args, use_auth_token=None, token=None, **kwargs):
        if token is None and use_auth_token is not None:
            token = use_auth_token
        return original_download(*args, token=token, **kwargs)

    hf_hub_download_compat._diart_auth_arg_patch = True
    huggingface_hub.hf_hub_download = hf_hub_download_compat


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 화자 분리 파이프라인 로드"""
    global pipeline, PIPELINE_SAMPLE_RATE

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        logger.warning("HF_TOKEN이 설정되지 않았습니다! pyannote 모델 다운로드가 실패할 수 있습니다.")

    logger.info("화자 분리 파이프라인 로딩 중...")
    patch_huggingface_hub_auth_arg()

    # 1차: pyannote.audio Pipeline 직접 사용 (더 안정적)
    try:
        from pyannote.audio import Pipeline as PyannotePipeline

        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token if hf_token else True
        )
        pipeline = move_pipeline_to_device(pipeline)
        
        # 튜닝: 여기서부터 민감도(Threshold) 조절 코드를 추가합니다.
        try:
            diarize_threshold = float(os.getenv("DIARIZE_THRESHOLD", "0.5"))
            logger.info(f"임계값(threshold) 설정 중: {diarize_threshold}")
            # 1. 모델이 현재 가지고 있는 기본 파라미터 값들을 가져옵니다.
            params = pipeline.parameters(instantiated=True)
            
            # 2. 클러스터링 임계값을 원하는 수치로 변경합니다. (기본값은 대략 0.7 근처입니다)
            # 쪼개짐이 심하면 올리고 합쳐짐이 심하면 낮추세요.
            params["clustering"]["threshold"] = diarize_threshold  
            
            # 3. 변경된 파라미터를 파이프라인에 다시 주입(적용)합니다.
            pipeline.instantiate(params)
            logger.info(f"임계값(threshold) {diarize_threshold} 적용 완료.")
        except Exception as pe:
            logger.warning(f"임계값 튜닝 실패: {pe}")

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
            use_hf_token=hf_token if hf_token else True
        )
        embedding = EmbeddingModel.from_pretrained(
            "pyannote/embedding",
            use_hf_token=hf_token if hf_token else True
        )

        pipeline = SpeakerDiarization(
            segmentation=segmentation,
            embedding=embedding,
        )
        pipeline = move_pipeline_to_device(pipeline)
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
        "device": PIPELINE_DEVICE,
        "pipeline_type": type(pipeline).__name__ if pipeline else None
    }


def run_diarization(audio_np: np.ndarray, sample_rate: int,
                    min_speakers: int = None, max_speakers: int = None) -> list:
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
    audio_np_copy = audio_np.copy()  # writable 복사 (PyTorch 경고 방지)
    waveform = torch.from_numpy(audio_np_copy).unsqueeze(0)
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    # 화자 분리 실행 (min/max speakers 힌트 전달)
    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    logger.info(f"[diarize] pipeline 호출 (kwargs={kwargs})")
    diarization = pipeline(audio_input, **kwargs)

    # 결과 변환
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(turn.end - turn.start, 3)
        })

    # 상세 로깅
    unique_speakers = set(s["speaker"] for s in segments)
    logger.info(f"[diarize] 결과: {len(segments)} segments, 화자: {unique_speakers}")
    for seg in segments:
        logger.info(f"  → {seg['speaker']}: {seg['start']:.1f}s ~ {seg['end']:.1f}s ({seg['duration']:.1f}s)")

    return segments


@app.post("/diart")
async def diarize_wav(
    file: UploadFile = File(...),
    min_speakers: int = Query(default=None, description="최소 화자 수"),
    max_speakers: int = Query(default=None, description="최대 화자 수"),
):
    """WAV 파일 → 화자 분리 JSON"""
    if pipeline is None:
        return JSONResponse(content={"error": "Pipeline not loaded"}, status_code=503)

    contents = await file.read()
    audio_np, original_sr = sf.read(io.BytesIO(contents), dtype="float32")

    logger.info(f"[diart] sr={original_sr}Hz, duration={len(audio_np)/original_sr:.2f}s")

    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)

    segments = run_diarization(audio_np, int(original_sr), min_speakers, max_speakers)
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
    min_speakers: int = Query(default=None, description="최소 화자 수"),
    max_speakers: int = Query(default=None, description="최대 화자 수"),
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

    segments = run_diarization(audio_np, sample_rate, min_speakers, max_speakers)
    unique_speakers = set(s["speaker"] for s in segments)

    return JSONResponse(content={
        "segments": segments,
        "num_speakers": len(unique_speakers),
        "duration": round(len(audio_np) / sample_rate, 3)
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
