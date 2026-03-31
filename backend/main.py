from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import whisper
from starlette.websockets import WebSocketDisconnect
from scipy.signal import resample
from data.save_transcript import save_transcript
import torch
import uuid
import httpx
from pydantic import BaseModel

device = "mps" if torch.backends.mps.is_available() else "cuda"

model = whisper.load_model("large-v3", device=device)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LLM_SERVER_URL = "http://localhost:8001"


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(req: ChatRequest):
    from fastapi.responses import StreamingResponse

    async def stream():
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=300.0)) as client:
            async with client.stream("POST", f"{LLM_SERVER_URL}/generate", json={"prompt": req.question}) as res:
                async for chunk in res.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream(), media_type="text/plain")

CHUNK_SIZE=360000 

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
        
