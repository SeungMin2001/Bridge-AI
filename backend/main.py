from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import numpy as np
import os
import json

# 신창영: 수정 이유 - MPS에서 아직 지원하지 않는 일부 PyTorch 연산은 CPU fallback으로 처리해 서버가 바로 죽지 않게 합니다.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from starlette.websockets import WebSocketDisconnect
import torchaudio
from data.save_transcript import save_transcript
# 신창영 : 워크스페이스 DB API 라우터를 main 서버에 연결
from db_api.workspace.router import router as workspace_router
from db import create_session, get_session_title, update_transcript_speakers
from correction import load_correction_model, correct_text
from rag_search import search as rag_search, init as rag_init, add_document as rag_add_document
import torch
import uuid
import httpx
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
# 신창영 : 현재 main 서버에서는 워크스페이스 기능 확인을 우선하여 quiz 라우터를 임시 제외
from quiz.quiz import router as quiz_router
from summary.summary import router as summary_router
from summary.test import router as summary_test_router
from schedule.schedule import router as schedule_router
from db_api.workspace.router import router as workspace_router
import logging

logger = logging.getLogger(__name__)

STT_BACKEND = os.getenv("STT_BACKEND", "faster-whisper").strip().lower().replace("_", "-")
requested_stt_device = os.getenv("STT_DEVICE", "").strip().lower()
requested_stt_model = os.getenv("STT_MODEL", "").strip()


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _load_stt_model():
    """STT_BACKEND 환경변수에 맞는 Whisper 백엔드를 로드합니다."""
    if STT_BACKEND in {"faster", "faster-whisper"}:
        from faster_whisper import WhisperModel

        if requested_stt_device in {"cuda", "cpu"}:
            device = requested_stt_device
        elif requested_stt_device == "mps":
            # 신창영: 수정 이유 - faster-whisper는 CTranslate2 기반이라 Apple MPS를 직접 사용할 수 없습니다.
            print("[STT] faster-whisper는 mps를 지원하지 않아 cpu로 실행합니다.")
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = requested_stt_model or "large-v3-turbo"
        return (
            WhisperModel(
                model_name,
                device=device,
                compute_type="float16" if device == "cuda" else "float32",
            ),
            device,
            model_name,
        )

    if STT_BACKEND in {"openai", "openai-whisper", "whisper"}:
        import whisper as openai_whisper

        if not hasattr(openai_whisper, "load_model"):
            raise RuntimeError(
                "openai-whisper 패키지가 아니라 다른 whisper 패키지가 import되었습니다. "
                "requirements의 whisper==1.1.10을 제거한 뒤 다시 설치해주세요."
            )

        if requested_stt_device == "mps":
            if _mps_available():
                device = "mps"
            else:
                print("[STT] 요청한 mps를 사용할 수 없어 cpu로 실행합니다.")
                device = "cpu"
        elif requested_stt_device == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_stt_device == "cpu":
            device = "cpu"
        else:
            # 신창영: 수정 이유 - Mac에서는 openai-whisper가 MPS를 쓸 수 있으므로 자동 선택 우선순위에 넣습니다.
            device = "mps" if _mps_available() else ("cuda" if torch.cuda.is_available() else "cpu")

        model_name = requested_stt_model or "turbo"
        if device == "mps":
            # 신창영: 수정 이유 - openai-whisper의 alignment_heads sparse buffer는 MPS 이동 시 SparseMPS 에러가 날 수 있어 CPU에서 dense로 바꾼 뒤 이동합니다.
            stt_model = openai_whisper.load_model(model_name, device="cpu")
            alignment_heads = getattr(stt_model, "alignment_heads", None)
            if alignment_heads is not None and getattr(alignment_heads, "is_sparse", False):
                stt_model.register_buffer("alignment_heads", alignment_heads.to_dense(), persistent=False)
            return stt_model.to("mps"), device, model_name

        return openai_whisper.load_model(model_name, device=device), device, model_name

    raise ValueError(
        "지원하지 않는 STT_BACKEND입니다. "
        "faster-whisper 또는 openai-whisper 중 하나를 사용하세요."
    )


model, stt_device, stt_model_name = _load_stt_model()
audio_device = "cuda" if stt_device == "cuda" else "cpu"
print(f"[STT] backend={STT_BACKEND}, model={stt_model_name}, device={stt_device}")

correction_enabled = load_correction_model()

transcribe_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
correction_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="correction")

# 신창영: 수정 이유 - STT가 mps여도 torchaudio resample은 CPU/CUDA 쪽이 안정적이라 별도 장치로 처리합니다.
resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000).to(audio_device)

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

# 신창영 : 워크스페이스 DB API 엔드포인트를 main 앱에 등록
app.include_router(workspace_router)
# ── 라우터 등록 ──
# 신창영 : quiz 라우터는 현재 테스트 범위에서 제외
# app.include_router(quiz_router)
# 신창영 : 녹음 종료 후 키워드/화자/세션 요약 API를 프론트에서 사용할 수 있도록 summary 라우터 등록
# ── 라우터 등록 ──
app.include_router(quiz_router)
app.include_router(summary_router)
app.include_router(summary_test_router)
app.include_router(schedule_router)
app.include_router(workspace_router)

#python -c "from huggingface_hub import login; login(token='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')"


# 코랩 모델
#llm_server_url = "https://dialysable-kyson-microelectrophoretic.ngrok-free.dev"

# 윈도우 모델
#llm_server_url = "http://localhost:8001"

# 신창영 : 기존 도커+vllm 설정 기록, 현재 실행 환경은 아래 환경변수 기반 LLM 설정을 사용
# 도커+vllm (OpenAI 호환 API)
# llm_server_url = "http://localhost:8001"
# llm_model_name="QuantTrio/Qwen3.5-4B-AWQ"

# 신창영 : LLM 설정을 하드코딩하지 않고 LLM_URL, LLM_MODEL, LLM_API_KEY 환경변수로 주입
llm_server_url = os.getenv("LLM_URL", "http://localhost:8001")
llm_model_name = os.getenv("LLM_MODEL", "QuantTrio/Qwen3.5-4B-AWQ")
llm_api_key = os.getenv("LLM_API_KEY", "test-key")

# 화자분리 서버
DIARIZE_URL = os.getenv("DIARIZE_URL", "http://localhost:8003/diart/raw")
DIARIZE_ENABLED = os.getenv("DIARIZE_ENABLED", "true").lower() == "true"



class ChatRequest(BaseModel):
    question: str
    is_thinking: bool = True
    # 신창영 : 선택 파일 기준 RAG 검색을 위해 session_id를 추가
    # 신창영 : 기존 구조는 session_id 없이 전체 전사문을 검색
    session_id: str | None = None


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


def _build_prompt_and_citations(question: str, session_id: str | None = None):
    """RAG 검색 후 prompt와 citations 반환"""
    # 신창영 : 선택 파일 기준 검색을 위해 session_id를 RAG 검색 함수에 전달
    # rag_result = rag_search(question, top_k=5)
    rag_result = rag_search(question, top_k=5, session_id=session_id)
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
        # 신창영 : 기존에는 req.session_id 없이 전체 전사문을 대상으로 RAG 검색
        # prompt, citations = _build_prompt_and_citations(req.question)
        prompt, citations = _build_prompt_and_citations(req.question, req.session_id)
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
    request_started_at = time.perf_counter()
    print(f"[CHAT STREAM] 요청 수신: {req.question}")

    # 신창영 : 스트리밍 채팅도 session_id를 전달하여 선택 파일 검색을 지원
    # prompt, citations = _build_prompt_and_citations(req.question)
    prompt, citations = _build_prompt_and_citations(req.question, req.session_id)
    t_rag = time.perf_counter()
    print(f"⏱️ [RAG 검색] {(t_rag - request_started_at)*1000:.0f}ms")

    messages = [
        {"role": "system", "content": "You are a helpful lecture assistant. Answer in Korean. 반드시 3문장 이내로 핵심만 답변해. 불필요한 부연설명 하지 마."},
        {"role": "user", "content": prompt},
    ]

    import json

    async def generate():
        first_token_logged = False
        first_token_elapsed = None

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
                            if not first_token_logged:
                                first_token_logged = True
                                first_token_elapsed = time.perf_counter() - request_started_at
                                print(f"[CHAT STREAM] 첫 토큰 도착: {first_token_elapsed:.3f}s")
                            yield f"data: {json.dumps({'type': 'token', 'token': content}, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"[CHAT STREAM] 에러: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            total_elapsed = time.perf_counter() - request_started_at
            first_token_text = f"{first_token_elapsed:.3f}s" if first_token_elapsed is not None else "N/A"
            print(f"[CHAT STREAM] 응답 종료: first_token={first_token_text}, total={total_elapsed:.3f}s")

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _filter_whisper_segments(segments: list[dict]) -> str:
    """Whisper segment 신뢰도 값으로 무음/환각 전사를 줄입니다."""
    result_parts = []
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        no_speech_prob = float(seg.get("no_speech_prob") or 0.0)
        avg_logprob = float(seg.get("avg_logprob") or 0.0)
        if no_speech_prob > 0.6:
            print(f"[필터] no_speech_prob={no_speech_prob:.2f} → 제거: {text!r}")
            continue
        if avg_logprob < -1.0:
            print(f"[필터] avg_logprob={avg_logprob:.2f} → 제거: {text!r}")
            continue
        result_parts.append(text)
    return " ".join(result_parts).strip()


def _transcribe_with_faster_whisper(audio_16k: np.ndarray) -> str:
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

    return _filter_whisper_segments([
        {
            "text": seg.text,
            "no_speech_prob": seg.no_speech_prob,
            "avg_logprob": seg.avg_logprob,
        }
        for seg in segments
    ])


def _transcribe_with_openai_whisper(audio_16k: np.ndarray) -> str:
    # 신창영: 수정 이유 - openai-whisper는 torch 기반이라 Mac에서 STT_DEVICE=mps로 Apple GPU를 사용할 수 있습니다.
    result = model.transcribe(
        audio_16k.astype(np.float32),
        language="ko",
        task="transcribe",
        temperature=0.0,
        condition_on_previous_text=False,
        fp16=stt_device == "cuda",
        # 신창영: 수정 이유 - openai-whisper에서 verbose=False는 tqdm 진행률을 켜므로, 실시간 청크마다 100% 로그가 반복되지 않게 None을 사용합니다.
        verbose=None,
    )
    filtered = _filter_whisper_segments(result.get("segments") or [])
    return filtered or str(result.get("text") or "").strip()


def _transcribe_chunk(audio_16k: np.ndarray) -> str:
    """선택된 STT_BACKEND로 한 오디오 청크를 전사합니다."""
    if STT_BACKEND in {"faster", "faster-whisper"}:
        return _transcribe_with_faster_whisper(audio_16k)
    if STT_BACKEND in {"openai", "openai-whisper", "whisper"}:
        return _transcribe_with_openai_whisper(audio_16k)
    return ""


def _correct_chunk(text: str) -> str:
    """KoBART 교정 (스레드풀에서 실행)"""
    try:
        return correct_text(text)
    except Exception as e:
        print(f"[교정] 교정 실패, raw_text 사용: {e}")
        return text


CHUNK_SIZE = 240000  # ~2.5초 (체감 응답 빠르게)
DIARIZE_SAMPLE_RATE = 16000
DIARIZE_BOOTSTRAP_SECONDS = 8
# 신창영: 수정 이유 - 전사는 2.5초 단위로 즉시 보내고, 화자분리는 별도 창 길이로 모아 분석하기 위해 분리 가능한 설정으로 둡니다.
DIARIZE_WINDOW_SECONDS = float(os.getenv("DIARIZE_WINDOW_SECONDS", DIARIZE_BOOTSTRAP_SECONDS))
DIARIZE_BUFFER_SIZE = int(DIARIZE_SAMPLE_RATE * 4 * DIARIZE_WINDOW_SECONDS)  # float32 16kHz 기준 N초
DIARIZE_TIMEOUT_SECONDS = float(os.getenv("DIARIZE_TIMEOUT_SECONDS", "60"))
# 신창영: 수정 이유 - 녹음 종료 후 전체 오디오는 8초 창보다 분석 시간이 길 수 있어 별도 timeout을 둡니다.
DIARIZE_FINAL_TIMEOUT_SECONDS = float(os.getenv("DIARIZE_FINAL_TIMEOUT_SECONDS", "180"))


async def _call_diarize(audio_float32: np.ndarray, sample_rate: int = 16000,
                        min_speakers: int = 2, source: str = "window",
                        timeout_seconds: float = DIARIZE_TIMEOUT_SECONDS) -> list:
    """diart 서버에 오디오를 보내 화자 세그먼트를 받아옵니다."""
    try:
        audio_payload = audio_float32.astype(np.float32, copy=False)
        sample_count = len(audio_payload)
        payload_bytes = audio_payload.nbytes
        duration = sample_count / sample_rate if sample_rate else 0.0
        rms = float(np.sqrt(np.mean(audio_payload ** 2))) if sample_count else 0.0
        logger.info(
            f"[DIARIZE:{source}] 요청 준비: url={DIARIZE_URL}, sr={sample_rate}Hz, "
            f"samples={sample_count}, bytes={payload_bytes}, duration={duration:.2f}s, "
            f"rms={rms:.4f}, min_speakers={min_speakers}, timeout={timeout_seconds:.1f}s"
        )
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
            resp = await client.post(
                DIARIZE_URL,
                content=audio_payload.tobytes(),
                headers={"Content-Type": "application/octet-stream"},
                params={"sample_rate": sample_rate, "min_speakers": min_speakers},
            )
            resp.raise_for_status()
            data = resp.json()
            segments = data.get("segments", [])

            # 상세 로깅
            speakers = set(s["speaker"] for s in segments)
            logger.info(f"[DIARIZE:{source}] 응답: {len(segments)} segments, 화자: {speakers}, "
                        f"num_speakers: {data.get('num_speakers')}")
            for seg in segments:
                logger.info(f"[DIARIZE:{source}]   → {seg['speaker']}: "
                            f"{seg['start']:.1f}s ~ {seg['end']:.1f}s")

            return segments
    except Exception as e:
        logger.exception(f"[DIARIZE:{source}] 화자분리 호출 실패 ({type(e).__name__}): {e!r}")
        return []


def _dominant_speaker(segments: list, start: float, end: float) -> str | None:
    """주어진 시간 구간에서 가장 많이 말한 화자를 반환합니다."""
    if not segments:
        return None
    speaker_durations: dict[str, float] = {}
    for seg in segments:
        # 구간 겹침 계산
        overlap_start = max(seg["start"], start)
        overlap_end = min(seg["end"], end)
        if overlap_end > overlap_start:
            spk = seg["speaker"]
            speaker_durations[spk] = speaker_durations.get(spk, 0) + (overlap_end - overlap_start)
    if not speaker_durations:
        return None
    return max(speaker_durations, key=speaker_durations.get)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    audio_buffer = bytearray()
    diarize_buffer = bytearray()  # 화자분리용 (STT와 별도로 더 긴 오디오 수집)
    full_diarize_buffer = bytearray()  # 녹음 종료 후 전체 오디오 기준으로 화자를 다시 보정하기 위한 RAM 버퍼
    requested_session_id = ws.query_params.get("session_id")
    requested_recording_id = (ws.query_params.get("recording_id") or "").strip()
    requested_diarize = (ws.query_params.get("diarize") or "true").strip().lower()
    effective_diarize = DIARIZE_ENABLED and requested_diarize not in {"0", "false", "no", "off"}
    send_lock = asyncio.Lock()
    try:
        session_id = str(uuid.UUID(requested_session_id)) if requested_session_id else str(uuid.uuid4())
    except (TypeError, ValueError):
        session_id = str(uuid.uuid4())
    recording_id = requested_recording_id or f"recording-{uuid.uuid4()}"
    processed_seconds = 0.0
    diarize_segments: list = []  # 최근 화자분리 결과 캐시
    diarize_buffer_start_time = 0.0  # 현재 버퍼의 시작 절대시간
    transcript_chunks: list[dict] = []
    diarize_status_active_sent = not effective_diarize
    diarize_tasks: set[asyncio.Task] = set()
    diarize_call_lock = asyncio.Lock()

    # 신창영: 수정 이유 - STT 처리와 백그라운드 화자분리 task가 동시에 WebSocket을 보낼 수 있어 전송 순서를 보호합니다.
    async def send_ws_json(payload: dict):
        async with send_lock:
            await ws.send_json(payload)

    try:
        await create_session(session_id)
    except Exception as e:
        print(f"[DB] create_session 실패 (전사는 계속 진행): {e}")

    session_title = "실시간 녹음"
    course_title = "실시간 강의"
    session_date = str(__import__("datetime").date.today())
    try:
        from db import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT title, course_id, session_date FROM sessions WHERE session_id = $1::uuid",
                session_id,
            )
            if row:
                session_title = row["title"] or session_title
                if row["session_date"]:
                    session_date = str(row["session_date"])
                if row["course_id"]:
                    course_row = await conn.fetchrow(
                        "SELECT title FROM courses WHERE course_id = $1::uuid",
                        row["course_id"],
                    )
                    if course_row:
                        course_title = course_row["title"] or course_title
    except Exception as e:
        print(f"[WS] DB 정보 조회 실패 (기본값 사용): {e}")

    # 세션 시작 알림
    await send_ws_json({
        "type": "session_start",
        "session_id": session_id,
        "recording_id": recording_id,
        "diarization_enabled": effective_diarize,
    })
    await send_ws_json({
        "type": "diarization_status",
        "status": "bootstrapping" if effective_diarize else "disabled",
        "diarization_enabled": effective_diarize,
    })

    loop = asyncio.get_event_loop()

    async def process_transcript_chunk(audio_16k: np.ndarray, start_time: float, end_time: float):
        """STT, 교정, 저장, RAG 추가를 한 청크 단위로 처리합니다."""
        try:
            raw_text = await loop.run_in_executor(transcribe_pool, _transcribe_chunk, audio_16k)
        except Exception as e:
            # 신창영: 수정 이유 - MPS/STT 런타임 에러가 나도 WebSocket 전체가 500으로 죽지 않게 프론트에 에러만 전달합니다.
            logger.exception("[STT] 전사 실패")
            await send_ws_json({
                "type": "error",
                "scope": "stt",
                "message": str(e),
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
            })
            return

        speaker_id = _dominant_speaker(diarize_segments, start_time, end_time) if effective_diarize else None
        chunk_id = f"{recording_id}:{start_time:.3f}:{end_time:.3f}"
        logger.info(f"[SPEAKER] {start_time:.1f}~{end_time:.1f}s → {speaker_id} "
                    f"(segments: {len(diarize_segments)}개, "
                    f"buffer_start: {diarize_buffer_start_time:.1f}s)")

        await send_ws_json({
            "type": "raw",
            "chunk_id": chunk_id,
            "recording_id": recording_id,
            "raw_text": raw_text,
            "text": raw_text,
            "speaker_id": speaker_id,
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
        })

        if correction_enabled and raw_text:
            corrected_text = await loop.run_in_executor(correction_pool, _correct_chunk, raw_text)
            if corrected_text != raw_text:
                await send_ws_json({
                    "type": "corrected",
                    "chunk_id": chunk_id,
                    "recording_id": recording_id,
                    "raw_text": raw_text,
                    "text": corrected_text,
                    "speaker_id": speaker_id,
                    "start_time": round(start_time, 3),
                    "end_time": round(end_time, 3),
                })
        else:
            corrected_text = raw_text

        transcript_data = {
            "session_id": session_id,
            "recording_id": recording_id,
            "start_time": start_time,
            "end_time": end_time,
            "raw_text": raw_text,
            "text": corrected_text,
            "speaker_id": speaker_id,
        }
        chunk_record = None
        if corrected_text:
            chunk_record = {
                "chunk_id": chunk_id,
                "recording_id": recording_id,
                "start_time": start_time,
                "end_time": end_time,
                "raw_text": raw_text,
                "text": corrected_text,
                "speaker_id": speaker_id,
            }
            transcript_chunks.append(chunk_record)
        saved_transcript = {}
        try:
            saved_transcript = await save_transcript(transcript_data) or {}
            if chunk_record is not None:
                chunk_record["transcript_id"] = saved_transcript.get("transcript_id")
        except Exception as e:
            print(f"[DB] save_transcript 실패: {e}")

        if saved_transcript.get("transcript_id"):
            await send_ws_json({
                "type": "saved",
                "chunk_id": chunk_id,
                "recording_id": recording_id,
                "raw_text": raw_text,
                "text": corrected_text,
                "transcript_id": saved_transcript.get("transcript_id"),
                "chunk_index": saved_transcript.get("chunk_index"),
                "start_time": start_time,
                "end_time": end_time,
            })

        if corrected_text:
            try:
                rag_add_document(corrected_text, {
                    "session_id": session_id,
                    "recording_id": recording_id,
                    "session_title": session_title,
                    "course_title": course_title,
                    "session_date": session_date,
                    "transcript_id": saved_transcript.get("transcript_id", ""),
                    "chunk_index": saved_transcript.get("chunk_index"),
                    "created_at": saved_transcript.get("created_at", ""),
                    "start_time": start_time,
                    "end_time": end_time,
                    "speaker_id": speaker_id or "UNKNOWN",
                })
            except Exception as e:
                print(f"[RAG] 임베딩 추가 실패 (전사는 정상): {e}")

    def build_speaker_updates(segments: list, force: bool = False) -> list[dict]:
        """전사 청크 시간과 화자 구간을 비교해서 speaker_id 보정 목록을 만듭니다."""
        speaker_updates = []
        for chunk in transcript_chunks:
            new_speaker = _dominant_speaker(segments, chunk["start_time"], chunk["end_time"])
            if new_speaker and (force or new_speaker != chunk.get("speaker_id")):
                chunk["speaker_id"] = new_speaker
                speaker_updates.append({
                    "chunk_id": chunk["chunk_id"],
                    "transcript_id": chunk.get("transcript_id"),
                    "recording_id": chunk["recording_id"],
                    "speaker_id": new_speaker,
                    "start_time": round(chunk["start_time"], 3),
                    "end_time": round(chunk["end_time"], 3),
                })
        return speaker_updates

    async def handle_diarize_window(audio_bytes: bytes, window_start_time: float):
        """STT 흐름을 막지 않고 별도 창 단위로 화자분리를 수행합니다."""
        nonlocal diarize_segments, diarize_status_active_sent
        try:
            diarize_audio = np.frombuffer(audio_bytes, dtype=np.float32)
            if len(diarize_audio) == 0:
                return

            # 신창영: 수정 이유 - pyannote 호출이 길어져도 STT 전사 전송이 멈추지 않도록 별도 task에서 실행합니다.
            async with diarize_call_lock:
                new_segments = await _call_diarize(
                    diarize_audio,
                    DIARIZE_SAMPLE_RATE,
                    source=f"window@{window_start_time:.1f}s",
                )

            absolute_segments = []
            for segment in new_segments:
                absolute_segments.append({
                    **segment,
                    "start": round(float(segment["start"]) + window_start_time, 3),
                    "end": round(float(segment["end"]) + window_start_time, 3),
                })

            if not absolute_segments:
                logger.info("[DIARIZE] 화자분리 결과 없음")
                return

            diarize_segments.extend(absolute_segments)
            speaker_updates = build_speaker_updates(diarize_segments)

            await send_ws_json({
                "type": "diarization",
                "segments": absolute_segments,
                "num_speakers": len(set(s["speaker"] for s in diarize_segments)),
            })

            if not diarize_status_active_sent:
                diarize_status_active_sent = True
                await send_ws_json({
                    "type": "diarization_status",
                    "status": "active",
                    "diarization_enabled": True,
                })

            if speaker_updates:
                await send_ws_json({
                    "type": "speaker_update",
                    "items": speaker_updates,
                })

            logger.info(f"[DIARIZE] background {len(absolute_segments)} segments, "
                        f"updates: {len(speaker_updates)}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[DIARIZE] 백그라운드 화자분리 실패: {e}")

    def schedule_diarize_window(audio_bytes: bytes, window_start_time: float):
        # 신창영: 수정 이유 - 화자분리 task를 따로 예약해서 /ws 수신 루프와 STT 전사를 계속 진행합니다.
        sample_count = len(audio_bytes) // 4
        duration = sample_count / DIARIZE_SAMPLE_RATE if DIARIZE_SAMPLE_RATE else 0.0
        logger.info(
            f"[DIARIZE:window] 예약: start={window_start_time:.2f}s, "
            f"bytes={len(audio_bytes)}, samples={sample_count}, duration={duration:.2f}s"
        )
        task = asyncio.create_task(handle_diarize_window(audio_bytes, window_start_time))
        diarize_tasks.add(task)
        task.add_done_callback(diarize_tasks.discard)

    async def process_pcm_chunk(pcm_chunk: bytes):
        """브라우저가 보낸 48kHz int16 PCM 청크를 STT/화자분리용 16kHz 오디오로 처리합니다."""
        nonlocal processed_seconds, diarize_buffer_start_time
        if not pcm_chunk:
            return

        chunk_duration = (len(pcm_chunk) / 2) / 48000  # int16 = 2 bytes, 48kHz
        start_time = processed_seconds
        end_time = processed_seconds + chunk_duration

        audio_np = np.frombuffer(pcm_chunk, dtype=np.int16)
        if len(audio_np) == 0:
            return

        audio_float = audio_np.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))

        audio_tensor = torch.from_numpy(audio_float).to(audio_device)
        audio_16k = resampler(audio_tensor).cpu().numpy()

        if effective_diarize:
            if len(diarize_buffer) == 0:
                diarize_buffer_start_time = start_time
            audio_16k_bytes = audio_16k.astype(np.float32).tobytes()
            diarize_buffer.extend(audio_16k_bytes)
            # 신창영: 수정 이유 - 파일 저장 없이 녹음 종료 후 전체 오디오 기준으로 speaker_id를 다시 계산하기 위해 RAM에만 누적합니다.
            full_diarize_buffer.extend(audio_16k_bytes)

            if len(diarize_buffer) >= DIARIZE_BUFFER_SIZE:
                schedule_diarize_window(bytes(diarize_buffer), diarize_buffer_start_time)
                diarize_buffer.clear()

        processed_seconds = end_time
        if rms < 0.01:
            return

        # 신창영: 수정 이유 - 화자분리를 기다리지 않고 STT 결과를 즉시 프론트로 보냅니다.
        await process_transcript_chunk(audio_16k, start_time, end_time)

    async def finalize_recording():
        """녹음 종료 시 RAM에 모아 둔 전체 오디오로 화자분리를 다시 수행합니다."""
        nonlocal diarize_segments
        if audio_buffer:
            await process_pcm_chunk(bytes(audio_buffer))
            audio_buffer.clear()

        if diarize_tasks:
            await asyncio.gather(*list(diarize_tasks), return_exceptions=True)

        if not effective_diarize:
            logger.info("[DIARIZE:final] 생략: 화자분리 비활성화")
            await send_ws_json({
                "type": "finalized",
                "status": "skipped",
                "reason": "diarization_disabled",
                "recording_id": recording_id,
            })
            return

        if not full_diarize_buffer or not transcript_chunks:
            logger.info(
                f"[DIARIZE:final] 생략: empty_audio_or_transcripts "
                f"(buffer_bytes={len(full_diarize_buffer)}, transcripts={len(transcript_chunks)})"
            )
            await send_ws_json({
                "type": "finalized",
                "status": "skipped",
                "reason": "empty_audio_or_transcripts",
                "recording_id": recording_id,
            })
            return

        await send_ws_json({
            "type": "diarization_status",
            "status": "finalizing",
            "diarization_enabled": True,
        })

        try:
            full_buffer_bytes = len(full_diarize_buffer)
            full_samples = full_buffer_bytes // 4
            full_duration = full_samples / DIARIZE_SAMPLE_RATE if DIARIZE_SAMPLE_RATE else 0.0
            # 신창영: 수정 이유 - 녹음 종료 후 전체 오디오 버퍼가 실제로 모델에 전달되는지 터미널 로그에서 확인하기 위한 진단 로그입니다.
            logger.info(
                f"[DIARIZE:final] 전체 버퍼 전달 시작: "
                f"bytes={full_buffer_bytes}, samples={full_samples}, "
                f"duration={full_duration:.2f}s, transcripts={len(transcript_chunks)}"
            )
            full_audio = np.frombuffer(bytes(full_diarize_buffer), dtype=np.float32)
            async with diarize_call_lock:
                final_segments = await _call_diarize(
                    full_audio,
                    DIARIZE_SAMPLE_RATE,
                    source="final_full_audio",
                    timeout_seconds=DIARIZE_FINAL_TIMEOUT_SECONDS,
                )

            diarize_segments = [
                {
                    **segment,
                    "start": round(float(segment["start"]), 3),
                    "end": round(float(segment["end"]), 3),
                }
                for segment in final_segments
            ]
            speaker_updates = build_speaker_updates(diarize_segments, force=True)
            db_updated_count = await update_transcript_speakers(session_id, recording_id, speaker_updates)
            logger.info(
                f"[DIARIZE:final] 전체 보정 완료: final_segments={len(diarize_segments)}, "
                f"speaker_updates={len(speaker_updates)}, db_updated={db_updated_count}"
            )

            await send_ws_json({
                "type": "diarization",
                "segments": diarize_segments,
                "num_speakers": len(set(s["speaker"] for s in diarize_segments)),
                "final": True,
            })
            if speaker_updates:
                await send_ws_json({
                    "type": "speaker_update",
                    "items": speaker_updates,
                    "final": True,
                })
            await send_ws_json({
                "type": "finalized",
                "status": "ok",
                "recording_id": recording_id,
                "updated_count": len(speaker_updates),
                "db_updated_count": db_updated_count,
            })
        except Exception as e:
            logger.exception("[DIARIZE] 전체 오디오 최종 화자분리 실패")
            await send_ws_json({
                "type": "finalized",
                "status": "error",
                "recording_id": recording_id,
                "message": str(e),
            })

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            if message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "finalize":
                    await finalize_recording()
                    await ws.close()
                    return
                continue

            data = message.get("bytes")
            if not data:
                continue
            audio_buffer.extend(data)

            while len(audio_buffer) >= CHUNK_SIZE:
                pcm_chunk = bytes(audio_buffer[:CHUNK_SIZE])
                del audio_buffer[:CHUNK_SIZE]
                await process_pcm_chunk(pcm_chunk)

    except (WebSocketDisconnect, ConnectionResetError):
        print(f"[WS] 클라이언트 연결 종료: session_id={session_id}")
        # 신창영: 수정 이유 - 클라이언트가 끊긴 뒤 남은 백그라운드 화자분리 task가 WebSocket으로 보내지 않도록 정리합니다.
        for task in list(diarize_tasks):
            task.cancel()
        if diarize_tasks:
            await asyncio.gather(*diarize_tasks, return_exceptions=True)
