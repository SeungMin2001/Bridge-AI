from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import numpy as np
from faster_whisper import WhisperModel
from starlette.websockets import WebSocketDisconnect
import torchaudio
from data.save_transcript import save_transcript
# 워크스페이스 DB API 라우터를 main 서버에 연결할 때 사용 <-  신창영
from db_api.workspace.router import router as workspace_router
from db import create_session
from correction import load_correction_model, correct_text
from rag_search import search as rag_search, init as rag_init, add_document as rag_add_document
import torch
import uuid
import httpx
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
# 신창영이 임시 지움
# from quiz.quiz import router as quiz_router
# from summary.summary import router as summary_router
# from schedule.schedule import router as schedule_router

# device = "cuda" if torch.cuda.is_available() else (
#     "mps" if torch.backends.mps.is_available() else "cpu"
# )

device = "cpu"

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

@app.on_event("startup")
async def startup():
    rag_init()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 워크스페이스 DB API 엔드포인트를 main 앱에 등록 <- 신창영
app.include_router(workspace_router)
# ── 라우터 등록 ──
# 신창영이 임시 지움
# app.include_router(quiz_router)
# app.include_router(summary_router)
# app.include_router(schedule_router)

#python -c "from huggingface_hub import login; login(token='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')"


# 코랩 모델
#llm_server_url = "https://dialysable-kyson-microelectrophoretic.ngrok-free.dev"

# 윈도우 모델
#llm_server_url = "http://localhost:8001"

# 신창영이 임시 지움
# 도커+vllm (OpenAI 호환 API)
# llm_server_url = "http://localhost:8001"
# llm_model_name="Qwen/Qwen2.5-1.5B"

# Ollama OpenAI 호환 API
llm_server_url = "http://localhost:11434"
llm_model_name="qwen2.5:1.5b"
llm_api_key = "test-key"

class ChatRequest(BaseModel):
    question: str
    is_thinking: bool = True


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


def _build_prompt_and_citations(question: str):
    """RAG 검색 후 prompt와 citations 반환"""
    rag_result = rag_search(question, top_k=5)
    context = rag_result["context"]
    citations = rag_result["citations"]

    if context:
        prompt = (
            f"다음은 강의 내용에서 검색된 참고자료입니다:\n\n{context}\n\n"
            f"위 참고자료를 바탕으로 답변하고, 답변 마지막에 참고한 출처를 '[출처]' 형식으로 표시해주세요.\n\n"
            f"질문: {question}"
        )
    else:
        prompt = question

    return prompt, citations


@app.post("/chat")
async def chat(req: ChatRequest):
    """기존 비스트리밍 엔드포인트 (호환용)"""
    print(f"[CHAT] 요청 수신: {req.question}")
    try:
        prompt, citations = _build_prompt_and_citations(req.question)
        messages = [
            {"role": "system", "content": "You are a helpful lecture assistant. Answer in Korean. 반드시 3문장 이내로 핵심만 답변해. 불필요한 부연설명 하지 마."},
            {"role": "user", "content": prompt},
        ]
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=300.0)) as client:
            res = await client.post(

                f"{llm_server_url}/v1/chat/completions",
                json={
                    "model": llm_model_name,
                    "messages": messages,
                    "max_tokens": 128,
                    "temperature": 0.7,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                headers={"Authorization": f"Bearer {llm_api_key}"},

            )
        data = res.json()
        raw_answer = data["choices"][0]["message"]["content"]
        answer = remove_thinking(raw_answer)
        return {"thinking": "", "answer": answer, "citations": citations}
    except Exception as e:
        print(f"[CHAT] 에러: {e}")
        return {"thinking": "", "answer": f"오류: {e}", "citations": []}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 스트리밍 엔드포인트"""
    import time
    t0 = time.perf_counter()
    print(f"[CHAT STREAM] 요청 수신: {req.question}")

    prompt, citations = _build_prompt_and_citations(req.question)
    t_rag = time.perf_counter()
    print(f"⏱️ [RAG 검색] {(t_rag - t0)*1000:.0f}ms")

    messages = [
        {"role": "system", "content": "You are a helpful lecture assistant. Answer in Korean. 반드시 3문장 이내로 핵심만 답변해. 불필요한 부연설명 하지 마."},
        {"role": "user", "content": prompt},
    ]

    import json

    async def generate():
        nonlocal t0
        ttft_logged = False
        token_count = 0

        # 먼저 citations 전송
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations}, ensure_ascii=False)}\n\n"

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=300.0)) as client:
                async with client.stream(
                    "POST",
                    f"{llm_server_url}/v1/chat/completions",
                    json={
                        "model": llm_model_name,
                        "messages": messages,
                        "max_tokens": 128,
                        "temperature": 0.7,
                        "stream": True,
                        "chat_template_kwargs": {"enable_thinking": req.is_thinking},
                    },
                    headers={"Authorization": f"Bearer {llm_api_key}"},
                ) as stream:
                    async for line in stream.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if not ttft_logged:
                                print(f"⏱️ [TTFT] 첫 토큰까지: {(time.perf_counter() - t0)*1000:.0f}ms")
                                ttft_logged = True
                            token_count += 1
                            yield f"data: {json.dumps({'type': 'token', 'token': content}, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"[CHAT STREAM] 에러: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            total_elapsed = time.perf_counter() - request_started_at
            first_token_text = f"{first_token_elapsed:.3f}s" if first_token_elapsed is not None else "N/A"
            print(f"[CHAT STREAM] 응답 종료: first_token={first_token_text}, total={total_elapsed:.3f}s")

        total_ms = (time.perf_counter() - t0) * 1000
        tps = token_count / (total_ms / 1000) if total_ms > 0 else 0
        print(f"⏱️ [응답완료] 총: {total_ms:.0f}ms | 토큰: {token_count}개 | {tps:.1f} tok/s")

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


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

                # RAG vector store에 임베딩 추가
                if corrected_text:
                    try:
                        rag_add_document(corrected_text, {
                            "session_id": session_id,
                            "session_title": "실시간 녹음",
                            "course_title": "실시간 강의",
                            "session_date": str(__import__('datetime').date.today()),
                            "start_time": start_time,
                            "end_time": end_time,
                        })
                    except Exception as e:
                        print(f"[RAG] 임베딩 추가 실패 (전사는 정상): {e}")

                processed_seconds = end_time

    except (WebSocketDisconnect, ConnectionResetError):
        print(f"[WS] 클라이언트 연결 종료: session_id={session_id}")
