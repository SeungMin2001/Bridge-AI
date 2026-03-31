"""
DeepFilterNet 잡음 제거 마이크로서비스

메인 백엔드(numpy 2.4.3)와 numpy 버전이 충돌하므로
별도 Docker 컨테이너에서 실행됩니다.

API:
  POST /denoise
    - Content-Type: multipart/form-data
    - file: WAV 파일 (어떤 sample rate든 가능, 내부에서 48kHz로 리샘플링)
    - 응답: 잡음 제거된 WAV 파일 (원본 sample rate로 복원)

  POST /denoise/raw
    - Content-Type: application/octet-stream
    - Body: float32 PCM raw bytes
    - Query: sample_rate (기본 16000)
    - 응답: 잡음 제거된 float32 PCM raw bytes

  GET /health
    - 서버 상태 확인
"""

import io
import logging
import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample
from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.responses import Response, StreamingResponse
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepFilterNet Denoise Server")

# 글로벌 모델 변수
model = None
df_state = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 DeepFilterNet 모델 로드"""
    global model, df_state
    logger.info("DeepFilterNet 모델 로딩 중...")

    from df.enhance import init_df
    model, df_state, _ = init_df()

    logger.info(f"DeepFilterNet 모델 로딩 완료. (모델 sample rate: {df_state.sr()}Hz)")


@app.get("/health")
async def health():
    """헬스체크 엔드포인트"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_sr": df_state.sr() if df_state else None
    }


@app.post("/denoise")
async def denoise_wav(file: UploadFile = File(...)):
    """
    WAV 파일을 받아서 잡음 제거 후 WAV 파일로 반환합니다.

    - 입력: 어떤 sample rate의 WAV 파일이든 가능
    - 내부 처리: 48kHz로 리샘플링 → DeepFilterNet 처리 → 원본 SR로 복원
    - 출력: 잡음 제거된 WAV 파일
    """
    from df.enhance import enhance

    if model is None:
        return Response(content="Model not loaded", status_code=503)

    # 1. 업로드된 WAV 파일 읽기
    contents = await file.read()
    audio_np, original_sr = sf.read(io.BytesIO(contents), dtype="float32")

    logger.info(f"입력: sr={original_sr}Hz, shape={audio_np.shape}, duration={len(audio_np)/original_sr:.2f}s")

    # 2. 모노로 변환 (스테레오인 경우)
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)

    # 3. DeepFilterNet은 48kHz 필요 → 리샘플링
    model_sr = df_state.sr()  # 48000
    if original_sr != model_sr:
        num_samples_48k = int(len(audio_np) * model_sr / original_sr)
        audio_48k = resample(audio_np, num_samples_48k).astype(np.float32)
    else:
        audio_48k = audio_np

    # 4. torch 텐서 변환 후 잡음 제거
    audio_tensor = torch.from_numpy(audio_48k).unsqueeze(0)  # [1, samples]
    enhanced_tensor = enhance(model, df_state, audio_tensor)
    enhanced_np = enhanced_tensor.squeeze().cpu().numpy()

    # 5. 원본 sample rate로 복원
    if original_sr != model_sr:
        num_samples_orig = int(len(enhanced_np) * original_sr / model_sr)
        enhanced_np = resample(enhanced_np, num_samples_orig).astype(np.float32)

    # 6. WAV 파일로 응답
    output_buffer = io.BytesIO()
    sf.write(output_buffer, enhanced_np, original_sr, format="WAV", subtype="FLOAT")
    output_buffer.seek(0)

    logger.info(f"출력: sr={original_sr}Hz, duration={len(enhanced_np)/original_sr:.2f}s")

    return StreamingResponse(
        output_buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=denoised.wav"}
    )


@app.post("/denoise/raw")
async def denoise_raw(
    request: Request,
    sample_rate: int = Query(default=16000, description="입력 오디오의 sample rate"),
):
    """
    float32 PCM raw bytes를 받아서 잡음 제거 후 raw bytes로 반환합니다.
    메인 백엔드의 WebSocket 핸들러에서 직접 호출용.

    - 입력: float32 PCM raw bytes + sample_rate 쿼리 파라미터
    - 출력: float32 PCM raw bytes (잡음 제거 완료)
    """
    from df.enhance import enhance

    if model is None:
        return Response(content="Model not loaded", status_code=503)

    # Request body에서 raw bytes 읽기
    request_body = await request.body()

    # 1. raw bytes → numpy float32 배열
    audio_np = np.frombuffer(request_body, dtype=np.float32)

    if len(audio_np) == 0:
        return Response(content=b"", media_type="application/octet-stream")

    logger.info(f"[raw] 입력: sr={sample_rate}Hz, samples={len(audio_np)}, duration={len(audio_np)/sample_rate:.2f}s")

    # 2. 48kHz로 리샘플링
    model_sr = df_state.sr()  # 48000
    if sample_rate != model_sr:
        num_samples_48k = int(len(audio_np) * model_sr / sample_rate)
        audio_48k = resample(audio_np, num_samples_48k).astype(np.float32)
    else:
        audio_48k = audio_np

    # 3. 잡음 제거
    audio_tensor = torch.from_numpy(audio_48k).unsqueeze(0)
    enhanced_tensor = enhance(model, df_state, audio_tensor)
    enhanced_np = enhanced_tensor.squeeze().cpu().numpy()

    # 4. 원본 sample rate로 복원
    if sample_rate != model_sr:
        num_samples_orig = int(len(enhanced_np) * sample_rate / model_sr)
        enhanced_np = resample(enhanced_np, num_samples_orig).astype(np.float32)

    # 5. raw bytes로 반환
    return Response(
        content=enhanced_np.tobytes(),
        media_type="application/octet-stream"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
