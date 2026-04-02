from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import whisper
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

model = whisper.load_model("large-v3-turbo", device=device)
correction_enabled = load_correction_model()

# 전사용, 교정용 스레드풀 분리 → 파이프라인 병렬 실행
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
    """Whisper 전사 (스레드풀에서 실행)"""
    res = model.transcribe(
        audio_16k,
        language="ko",
        task="transcribe",
        fp16=(device == "cuda"),
        temperature=0.0,
        condition_on_previous_text=False,
        verbose=False,
    )
    return res["text"].strip()


def _correct_chunk(text: str) -> str:
    """KoBART 교정 (스레드풀에서 실행)"""
    try:
        return correct_text(text)
    except Exception as e:
        print(f"[교정] 교정 실패, raw_text 사용: {e}")
        return text


CHUNK_SIZE = 360000

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
    # 이전 청크의 교정 task (파이프라인용)
    pending_correction = None
    pending_meta = None

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

                # PCM → float → GPU resampling
                audio_np = np.frombuffer(pcm_chunk, dtype=np.int16)
                audio_float = audio_np.astype(np.float32) / 32768.0

                rms = np.sqrt(np.mean(audio_float ** 2))
                if rms < 0.01:
                    continue

                audio_tensor = torch.from_numpy(audio_float).to(device)
                audio_16k = resampler(audio_tensor).cpu().numpy()

                # 이전 청크 교정이 완료됐으면 결과 전송
                if pending_correction is not None:
                    corrected_text = await pending_correction
                    meta = pending_meta
                    await ws.send_json({
                        "raw_text": meta["raw_text"],
                        "text": corrected_text,
                    })
                    meta["text"] = corrected_text
                    try:
                        await save_transcript(meta)
                    except Exception as e:
                        print(f"[DB] save_transcript 실패: {e}")

                # 현재 청크 전사 (전사 스레드풀)
                raw_text = await loop.run_in_executor(transcribe_pool, _transcribe_chunk, audio_16k)

                # 교정을 백그라운드로 시작하고, 다음 청크 수신과 겹침
                meta = {
                    "session_id": session_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "raw_text": raw_text,
                }
                if correction_enabled:
                    pending_correction = loop.run_in_executor(correction_pool, _correct_chunk, raw_text)
                    pending_meta = meta
                else:
                    # 교정 없으면 바로 전송
                    await ws.send_json({"raw_text": raw_text, "text": raw_text})
                    meta["text"] = raw_text
                    try:
                        await save_transcript(meta)
                    except Exception as e:
                        print(f"[DB] save_transcript 실패: {e}")
                    pending_correction = None
                    pending_meta = None

                processed_seconds = end_time

        # 루프 종료 시 마지막 교정 결과 전송
    except (WebSocketDisconnect, ConnectionResetError):
        if pending_correction is not None:
            try:
                corrected_text = await pending_correction
                meta = pending_meta
                meta["text"] = corrected_text
                await save_transcript(meta)
            except Exception:
                pass
        print(f"[WS] 클라이언트 연결 종료: session_id={session_id}")
        
