from fastapi import FastAPI, WebSocket, HTTPException
import numpy as np
import whisper
from starlette.websockets import WebSocketDisconnect
from scipy.signal import resample
from data.save_transcript import save_transcript
from data.embeded_test import get_embedding
import torch
import uuid
import logging
from pydantic import BaseModel

from rag_service import retrieve_similar_chunks, generate_llm_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#from corrector import correct_text

device="mps" if torch.backends.mps.is_available() else "cuda"

model=whisper.load_model("large-v3",device=device) #모델설정(transcript할 모델)
app=FastAPI()

CHUNK_SIZE=288000 #1초

# REST API 데이터 검증을 위한 Pydantic 스키마
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    contexts: list[str]

# RAG를 위한 REST API 엔드포인트
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    사용자의 질문을 받아 벡터 검색을 수행하고, LLM이 생성한 답변을 반환합니다.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="메시지가 비어있을 수 없습니다.")

    logger.info(f"채팅 요청 처리 중: {request.message}")

    # 1. PostgreSQL(pgvector)에서 관련된 문맥(청크) 검색
    contexts = retrieve_similar_chunks(request.message)

    # 2. 로컬 Qwen LLM에 질문 전달 및 답변 생성
    answer = await generate_llm_response(request.message, contexts)

    if answer is None:
        raise HTTPException(status_code=500, detail="LLM에서 답변을 생성하는 데 실패했습니다.")

    return ChatResponse(answer=answer, contexts=contexts)

@app.websocket("/ws")
async def websocket_endpoint(ws:WebSocket):
    await ws.accept()
    audio_buffer=bytearray()
    session_id=str(uuid.uuid4())
    processed_seconds=0.0

    try:
        while True:
            data=await ws.receive_bytes()
            audio_buffer.extend(data)
            # 만약 버퍼가 충분히 쌓이면 전사시켜줘야함.
            #print("chunk:", len(data), "buffer:", len(audio_buffer)) # 계속 음성 INT16 data받아서 buffer에 쌓아놓기.

            is_transcribing=False

            if len(audio_buffer)>=CHUNK_SIZE and not is_transcribing: #버퍼에 쌓인게 청크기준보다 커지면 그만큼의 청크 빼내야함.
                is_transcribing=True

                pcm_chunk=bytes(audio_buffer[:CHUNK_SIZE])
                del audio_buffer[:CHUNK_SIZE]

                chunk_duration=(len(pcm_chunk)/2)/CHUNK_SIZE
                start_time=processed_seconds
                end_time=processed_seconds+chunk_duration

                audio_np=np.frombuffer(pcm_chunk,dtype=np.int16) #꺼낸 청크 읽고(int)로
                audio_float = audio_np.astype(np.float32) / 32768.0 #그걸 float로 다시 변환(모델이 float로 읽어야함)

                audio_16k=resample(audio_float,len(audio_float)*16000//48000)

                rms=np.sqrt(np.mean(audio_float**2))
                if rms<0.01:
                    continue
                res=model.transcribe( #모델 돌려서 전사하기.
                    audio_16k,
                    language="ko",
                    task="transcribe",
                    fp16=False,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    verbose=False,
                    )

                text=res["text"].strip()
                is_transcribing=False

                transcript_data={
                    "session_id":session_id,
                    "start_time":start_time,
                    "end_time":end_time,
                    "raw_text":text,
                    "text":text
                }
                await save_transcript(transcript_data)

                await ws.send_json({
                    "raw_text": text,
                    "text": text,
                })
                processed_seconds=end_time
    except WebSocketDisconnect:
        print("error")

