import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.save_transcript import save_transcript
from db import create_session
from db_api.workspace.router import router as workspace_router


app = FastAPI()
CHUNK_SIZE = 240000

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    # testmain.py가 정상 실행 중인지 확인하는 테스트 엔드포인트입니다.
    return {"ok": True, "service": "workspace-fastapi-test"}


@app.post("/api/mock/transcripts")
async def save_mock_transcript(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    text = data.get("text", "")
    speaker_id = data.get("speakerId") or data.get("speaker_id")
    speaker_name = data.get("speaker") or data.get("speaker_name")
    
    if not session_id:
        return {"ok": False, "error": "session_id is required"}

    try:
        await create_session(session_id, title="테스트 녹음")
    except Exception as error:
        print(f"[test-api] create_session failed: {error}")

    transcript_data = {
        "session_id": session_id,
        "start_time": 0.0,
        "end_time": 0.0,
        "speaker_id": speaker_id,
        "speaker_name": speaker_name,
        "raw_text": text,
        "text": text,
    }

    try:
        await save_transcript(transcript_data)
        return {"ok": True}
    except Exception as error:
        print(f"[test-api] save_transcript failed: {error}")
        return {"ok": False, "error": str(error)}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    audio_buffer = bytearray()
    session_id = ws.query_params.get("session_id") or str(uuid.uuid4())
    processed_seconds = 0.0
    chunk_index = 0

    try:
        await create_session(session_id, title="테스트 녹음")
    except Exception as error:
        print(f"[test-ws] create_session failed: {error}")

    try:
        while True:
            data = await ws.receive_bytes()
            audio_buffer.extend(data)

            if len(audio_buffer) < CHUNK_SIZE:
                continue

            pcm_chunk = bytes(audio_buffer[:CHUNK_SIZE])
            del audio_buffer[:CHUNK_SIZE]

            chunk_duration = (len(pcm_chunk) / 2) / CHUNK_SIZE
            start_time = processed_seconds
            end_time = processed_seconds + chunk_duration
            chunk_index += 1

            speaker_number = ((chunk_index - 1) // 5) % 3 + 1
            speaker_id = f"test-speaker-{speaker_number}"
            speaker_name = f"화자 {speaker_number}"
            raw_text = f"테스트 전사 데이터 {chunk_index}번째 조각입니다."
            corrected_text = raw_text

            await ws.send_json({
                "type": "raw",
                "raw_text": raw_text,
                "text": raw_text,
                "session_id": session_id,
                "speaker_id": speaker_id,
                "speaker": speaker_name,
                "start_time": start_time,
                "end_time": end_time,
            })

            transcript_data = {
                "session_id": session_id,
                "start_time": start_time,
                "end_time": end_time,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "raw_text": raw_text,
                "text": corrected_text,
            }

            try:
                await save_transcript(transcript_data)
            except Exception as error:
                print(f"[test-ws] save_transcript failed: {error}")

            processed_seconds = end_time

    except (WebSocketDisconnect, ConnectionResetError):
        print(f"[test-ws] client disconnected: session_id={session_id}")


# 나중에 main.py로 옮길 때도 아래 두 줄 구조만 동일하게 사용하면 됩니다.
app.include_router(workspace_router)
