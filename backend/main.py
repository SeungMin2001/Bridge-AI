from fastapi import FastAPI,WebSocket
import numpy as np
from faster_whisper import WhisperModel
from starlette.websockets import WebSocketDisconnect
from scipy.signal import resample

model=WhisperModel("tiny",device="cpu") #모델설정(transcript할 모델)
app=FastAPI()

CHUNK_SIZE=144000 #1초

@app.websocket("/ws")
async def websocket_endpoint(ws:WebSocket):
    await ws.accept()
    audio_buffer=bytearray()

    try:
        while True:
            data=await ws.receive_bytes()
            audio_buffer.extend(data)
            # 만약 버퍼가 충분히 쌓이면 전사시켜줘야함.
            print("chunk:", len(data), "buffer:", len(audio_buffer)) # 계속 음성 INT16 data받아서 buffer에 쌓아놓기.
            
            is_transcribing=False
            
            if len(audio_buffer)>=CHUNK_SIZE and not is_transcribing: #버퍼에 쌓인게 청크기준보다 커지면 그만큼의 청크 빼내야함.
                is_transcribing=True
                
                pcm_chunk=bytes(audio_buffer[:CHUNK_SIZE])
                del audio_buffer[:CHUNK_SIZE]
                audio_np=np.frombuffer(pcm_chunk,dtype=np.int16) #꺼낸 청크 읽고(int)로
                audio_float = audio_np.astype(np.float32) / 32768.0 #그걸 float로 다시 변환(모델이 float로 읽어야함)
                
                audio_16k=resample(audio_float,len(audio_float)*16000//48000)
                
                rms=np.sqrt(np.mean(audio_float**2))
                if rms<0.01:
                    continue
                segments, info=model.transcribe( #모델 돌려서 전사하기.
                    audio_16k,
                    language="ko",
                    vad_filter=True,
                    condition_on_previous_text=False
                    ) 
                
                is_transcribing=False
                
                text=""
                for segment in segments: #세그먼트 단위의 텍스트 다 합쳐주기.
                    text+=segment.text
                
                await ws.send_text(text)
                
                print(audio_float[:10])
    except WebSocketDisconnect:
        print("error")
        
