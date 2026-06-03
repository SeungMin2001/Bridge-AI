"""FastAPI 서버 진입점.

앱 공통 설정, STT WebSocket, 화자분리 연동, 기능별 라우터 등록을 담당합니다.
AI 채팅의 RAG/LLM 세부 로직은 backend/chat 패키지로 분리되어 있습니다.
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
import json

# 신창영: 수정 이유 - MPS에서 아직 지원하지 않는 일부 PyTorch 연산은 CPU fallback으로 처리해 서버가 바로 죽지 않게 합니다.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from starlette.websockets import WebSocketDisconnect
from data.save_transcript import save_transcript
# 신창영 : 워크스페이스 DB API 라우터를 main 서버에 연결
from db_api.workspace.router import router as workspace_router
from db_api.workspace.files_api import save_workspace_realtime_recording_file
from db import create_session, ensure_runtime_schema, update_transcript_speakers
from rag_search import init as rag_init, add_document as rag_add_document
import uuid
import httpx
import asyncio
# 신창영 : 현재 main 서버에서는 워크스페이스 기능 확인을 우선하여 quiz 라우터를 임시 제외
from quiz.quiz import router as quiz_router
from summary.summary import router as summary_router
from summary.test import router as summary_test_router
from schedule.schedule import router as schedule_router
from chat.router import router as chat_router
from stt.whisper_service import (
    correction_enabled,
    correction_pool,
    correct_transcript_text,
    resample_pcm48_to_16k,
    transcribe_16k_chunk,
    transcribe_pool,
)
import logging

logger = logging.getLogger(__name__)
DEMO_PIPELINE_LOG = True
DIARIZE_VERBOSE_LOG = os.getenv("DIARIZE_VERBOSE_LOG", "0").strip().lower() in {"1", "true", "yes", "on"}


def _demo_log(stage: str, message: str) -> None:
    if DEMO_PIPELINE_LOG:
        print(f"[DEMO:{stage}] {message}", flush=True)


def _diarize_debug(message: str) -> None:
    if DIARIZE_VERBOSE_LOG:
        logger.info(message)

app = FastAPI()

@app.on_event("startup")
async def startup():
    """서버 시작 시 RAG 검색 인덱스와 필요한 전역 리소스를 초기화합니다."""
    await ensure_runtime_schema()
    rag_init()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
app.include_router(chat_router)

# 화자분리 서버
DIARIZE_URL = os.getenv("DIARIZE_URL", "http://localhost:8003/diart/raw")
DIARIZE_ENABLED = os.getenv("DIARIZE_ENABLED", "true").lower() == "true"


CHUNK_SIZE = 240000  # ~2.5초 (체감 응답 빠르게)
CLIENT_AUDIO_SAMPLE_RATE = 48000
CLIENT_AUDIO_SAMPLE_WIDTH = 2
CLIENT_AUDIO_CHANNELS = 1
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
        _diarize_debug(
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
            _diarize_debug(f"[DIARIZE:{source}] 응답: {len(segments)} segments, 화자: {speakers}, "
                           f"num_speakers: {data.get('num_speakers')}")
            for seg in segments:
                _diarize_debug(f"[DIARIZE:{source}]   → {seg['speaker']}: "
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
    """브라우저 실시간 녹음 스트림을 받아 전사/저장/화자분리를 수행하는 WebSocket 엔드포인트입니다."""
    await ws.accept()
    audio_buffer = bytearray()
    recording_pcm_buffer = bytearray()  # 재생 가능한 WAV 파일로 저장할 브라우저 원본 PCM
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
        """동시 백그라운드 작업이 WebSocket 메시지를 순서 있게 보내도록 잠금으로 보호합니다."""
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
            _demo_log(
                "STT",
                f"전사 시작: start={start_time:.3f}s, end={end_time:.3f}s, samples={len(audio_16k)}",
            )
            raw_text = await loop.run_in_executor(transcribe_pool, transcribe_16k_chunk, audio_16k)
            _demo_log("STT", f"STT 원문 전사 완료: raw='{str(raw_text or '').strip()}'")
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
        _diarize_debug(f"[SPEAKER] {start_time:.1f}~{end_time:.1f}s → {speaker_id} "
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
            _demo_log("STT", f"KoBART 교정 시작: raw='{str(raw_text or '').strip()}'")
            corrected_text = await loop.run_in_executor(correction_pool, correct_transcript_text, raw_text)
            _demo_log("STT", f"KoBART 교정 완료: corrected='{str(corrected_text or '').strip()}'")
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
            if raw_text:
                _demo_log("STT", "KoBART 교정 생략: correction_disabled 또는 raw_text 없음")

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
            _demo_log(
                "STT",
                f"DB 저장 완료: transcript_id={saved_transcript.get('transcript_id')}, "
                f"text='{str(corrected_text or '').strip()}'",
            )
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
                _demo_log("STT", f"RAG 색인 시작: chunk_id={chunk_id}")
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
                _demo_log("STT", f"RAG 색인 완료: chunk_id={chunk_id}")
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
                _diarize_debug("[DIARIZE] 화자분리 결과 없음")
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

            _diarize_debug(f"[DIARIZE] background {len(absolute_segments)} segments, "
                           f"updates: {len(speaker_updates)}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[DIARIZE] 백그라운드 화자분리 실패: {e}")

    def schedule_diarize_window(audio_bytes: bytes, window_start_time: float):
        """누적된 화자분리 오디오 창을 백그라운드 task로 예약합니다."""
        # 신창영: 수정 이유 - 화자분리 task를 따로 예약해서 /ws 수신 루프와 STT 전사를 계속 진행합니다.
        sample_count = len(audio_bytes) // 4
        duration = sample_count / DIARIZE_SAMPLE_RATE if DIARIZE_SAMPLE_RATE else 0.0
        _diarize_debug(
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

        chunk_duration = (len(pcm_chunk) / CLIENT_AUDIO_SAMPLE_WIDTH) / CLIENT_AUDIO_SAMPLE_RATE
        start_time = processed_seconds
        end_time = processed_seconds + chunk_duration

        audio_np = np.frombuffer(pcm_chunk, dtype=np.int16)
        if len(audio_np) == 0:
            return

        audio_float = audio_np.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))

        audio_16k = resample_pcm48_to_16k(audio_float)

        if effective_diarize:
            if len(diarize_buffer) == 0:
                diarize_buffer_start_time = start_time
            audio_16k_bytes = audio_16k.astype(np.float32).tobytes()
            diarize_buffer.extend(audio_16k_bytes)
            # 신창영: 수정 이유 - 녹음 종료 후 전체 오디오 기준으로 speaker_id를 다시 계산하기 위해 RAM에 누적합니다.
            full_diarize_buffer.extend(audio_16k_bytes)

            if len(diarize_buffer) >= DIARIZE_BUFFER_SIZE:
                schedule_diarize_window(bytes(diarize_buffer), diarize_buffer_start_time)
                diarize_buffer.clear()

        processed_seconds = end_time
        if rms < 0.01:
            return

        # 신창영: 수정 이유 - 화자분리를 기다리지 않고 STT 결과를 즉시 프론트로 보냅니다.
        await process_transcript_chunk(audio_16k, start_time, end_time)

    async def save_realtime_audio_file() -> dict:
        """WebSocket으로 받은 원본 PCM을 WAV 파일로 저장하고 프론트 메타데이터를 반환합니다."""
        if not recording_pcm_buffer:
            return {}

        try:
            saved = await save_workspace_realtime_recording_file(
                session_id,
                recording_id,
                bytes(recording_pcm_buffer),
                sample_rate=CLIENT_AUDIO_SAMPLE_RATE,
                sample_width=CLIENT_AUDIO_SAMPLE_WIDTH,
                channels=CLIENT_AUDIO_CHANNELS,
                duration_seconds=processed_seconds,
            )
            recording = saved.get("recording") or {}
            return {
                "audioUrl": recording.get("audioUrl"),
                "storedName": recording.get("storedName"),
                "audioSize": recording.get("size"),
                "audioType": recording.get("type"),
                "durationSeconds": recording.get("durationSeconds"),
                "durationText": recording.get("durationText"),
            }
        except Exception as e:
            logger.exception("[WS] 실시간 녹음 파일 저장 실패")
            return {
                "audioSaveError": str(e),
            }

    async def finalize_recording():
        """녹음 종료 시 RAM에 모아 둔 전체 오디오로 화자분리를 다시 수행합니다."""
        nonlocal diarize_segments
        if audio_buffer:
            await process_pcm_chunk(bytes(audio_buffer))
            audio_buffer.clear()

        audio_file_payload = await save_realtime_audio_file()

        if diarize_tasks:
            await asyncio.gather(*list(diarize_tasks), return_exceptions=True)

        if not effective_diarize:
            _diarize_debug("[DIARIZE:final] 생략: 화자분리 비활성화")
            await send_ws_json({
                "type": "finalized",
                "status": "skipped",
                "reason": "diarization_disabled",
                "recording_id": recording_id,
                **audio_file_payload,
            })
            return

        if not full_diarize_buffer or not transcript_chunks:
            _diarize_debug(
                f"[DIARIZE:final] 생략: empty_audio_or_transcripts "
                f"(buffer_bytes={len(full_diarize_buffer)}, transcripts={len(transcript_chunks)})"
            )
            await send_ws_json({
                "type": "finalized",
                "status": "skipped",
                "reason": "empty_audio_or_transcripts",
                "recording_id": recording_id,
                **audio_file_payload,
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
            _diarize_debug(
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
            _diarize_debug(
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
                **audio_file_payload,
            })
        except Exception as e:
            logger.exception("[DIARIZE] 전체 오디오 최종 화자분리 실패")
            await send_ws_json({
                "type": "finalized",
                "status": "error",
                "recording_id": recording_id,
                "message": str(e),
                **audio_file_payload,
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
            recording_pcm_buffer.extend(data)
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

import os
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
frontend_dist = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
if os.path.isdir(frontend_dist):
    app.mount('/assets', StaticFiles(directory=os.path.join(frontend_dist, 'assets')), name='assets')
    @app.get('/{full_path:path}')
    async def serve_frontend(full_path: str):
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, 'index.html'))
