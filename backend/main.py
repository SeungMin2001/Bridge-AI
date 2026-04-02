from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from faster_whisper import WhisperModel
from starlette.websockets import WebSocketDisconnect
import torchaudio
from data.save_transcript import save_transcript
from db import create_session
from correction import load_correction_model, correct_text
import torch
import uuid
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# faster-whisper: CTranslate2 기반, 같은 정확도에 2~4배 빠름
model = WhisperModel(
    "large-v3-turbo",
    device=device,
    compute_type="float16" if device == "cuda" else "float32",
)
correction_enabled = load_correction_model()

transcribe_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
correction_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="correction")

# GPU resampler (48kHz → 16kHz)
resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000).to(device)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_server_url = "https://dialysable-kyson-microelectrophoretic.ngrok-free.dev"


class ChatRequest(BaseModel):
    question: str


class RegisterRequest(BaseModel):
    url: str


@app.post("/register-llm")
async def register_llm(req: RegisterRequest):
    global llm_server_url
    llm_server_url = req.url.rstrip("/")
    print(f"[LLM] URL updated: {llm_server_url}")
    return {"status": "ok", "url": llm_server_url}


def remove_thinking(text: str) -> str:
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    if '<think>' in text:
        text = text[:text.index('<think>')]
    if 'assistant\n' in text:
        text = text.split('assistant\n')[-1]
    return text.strip()


@app.post("/chat")
async def chat(req: ChatRequest):
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=300.0)) as client:
        res = await client.post(
            f"{llm_server_url}/generate",
            json={"prompt": req.question},
            headers={"ngrok-skip-browser-warning": "true"},
        )
    data = res.json()
    answer = data.get("answer") or data.get("response") or ""
    return {"answer": remove_thinking(answer)}


def _transcribe_chunk(audio_16k: np.ndarray) -> str:
    """faster-whisper 전사 (스레드풀에서 실행)

    환각(Hallucination) 필터링 기준:
    - no_speech_prob > 0.6 : Whisper가 "무음/노이즈"라고 판단 → 제거
    - avg_logprob < -1.0   : 전사 신뢰도가 낮음 → 제거
    두 필터 모두 확률 기반이므로 실제 발화("감사합니다" 등)는 통과됨.
    """
    segments, _ = model.transcribe(
        audio_16k,
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

    result_parts = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        # 1. no_speech_prob 기반 필터 (실제 발화 시엔 낮은 값 → 통과)
        if seg.no_speech_prob > 0.6:
            print(f"[필터] no_speech_prob={seg.no_speech_prob:.2f} → 제거: {text!r}")
            continue
        # 2. avg_logprob 기반 필터 (실제 발화 시엔 높은 값 → 통과)
        if seg.avg_logprob < -1.0:
            print(f"[필터] avg_logprob={seg.avg_logprob:.2f} → 제거: {text!r}")
            continue
        result_parts.append(text)

    return " ".join(result_parts).strip()


def _correct_chunk(text: str) -> str:
    """KoBART 교정 (스레드풀에서 실행)"""
    try:
        return correct_text(text)
    except Exception as e:
        print(f"[교정] 교정 실패, raw_text 사용: {e}")
        return text


CHUNK_SIZE = 240000  # ~2.5초 (체감 응답 빠르게)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    audio_buffer = bytearray()
    session_id = str(uuid.uuid4())
    processed_seconds = 0.0
    try:
        await create_session(session_id)
    except Exception as e:
        print(f"[DB] create_session 실패 (전사는 계속 진행): {e}")

    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await ws.receive_bytes()
            audio_buffer.extend(data)

            if len(audio_buffer) >= CHUNK_SIZE:
                pcm_chunk = bytes(audio_buffer[:CHUNK_SIZE])
                del audio_buffer[:CHUNK_SIZE]

                chunk_duration = (len(pcm_chunk) / 2) / CHUNK_SIZE
                start_time = processed_seconds
                end_time = processed_seconds + chunk_duration

                audio_np = np.frombuffer(pcm_chunk, dtype=np.int16)
                audio_float = audio_np.astype(np.float32) / 32768.0

                rms = np.sqrt(np.mean(audio_float ** 2))
                if rms < 0.01:
                    continue

                # GPU resampling
                audio_tensor = torch.from_numpy(audio_float).to(device)
                audio_16k = resampler(audio_tensor).cpu().numpy()

                # 전사
                raw_text = await loop.run_in_executor(transcribe_pool, _transcribe_chunk, audio_16k)

                # 1단계: raw_text 즉시 전송 (빠른 체감)
                await ws.send_json({
                    "type": "raw",
                    "raw_text": raw_text,
                    "text": raw_text,
                })

                # 2단계: 교정 후 업데이트 전송
                if correction_enabled and raw_text:
                    corrected_text = await loop.run_in_executor(correction_pool, _correct_chunk, raw_text)
                    if corrected_text != raw_text:
                        await ws.send_json({
                            "type": "corrected",
                            "raw_text": raw_text,
                            "text": corrected_text,
                        })
                else:
                    corrected_text = raw_text

                transcript_data = {
                    "session_id": session_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "raw_text": raw_text,
                    "text": corrected_text,
                }
                try:
                    await save_transcript(transcript_data)
                except Exception as e:
                    print(f"[DB] save_transcript 실패: {e}")

                processed_seconds = end_time

    except (WebSocketDisconnect, ConnectionResetError):
        print(f"[WS] 클라이언트 연결 종료: session_id={session_id}")
