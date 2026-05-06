"""
요약 API 라우터

엔드포인트:
  POST /summary/keywords/generate          - 세션 전사문 기반 키워드 추출
  GET  /summary/keywords/session/{session_id} - 세션별 키워드 목록 조회
  POST /summary/speaker/generate           - 화자별 요약 생성
  POST /summary/session/generate           - 세션 요약 생성
  POST /summary/course/generate            - 코스 요약 생성
  GET  /summary/{summary_id}               - 요약 조회
  GET  /summary/session/{session_id}       - 세션별 요약 목록 조회
  GET  /summary/course/{course_id}         - 코스 요약(최신) 조회
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import uuid
import logging

from db import (
    get_course_id_by_session,
    get_transcripts_by_session,
)
from summary.keyword_db import (
    delete_keywords_by_transcript_ids,
    get_keywords_by_course,
    get_keywords_by_session,
    save_keywords,
)
from summary.keyword_service import extract_keywords_from_transcripts
from summary.summary_db import (
    get_latest_course_summary,
    get_latest_session_summaries_by_course,
    get_latest_speaker_summaries_by_course,
    get_latest_speaker_summaries_by_session,
    get_summaries_by_session,
    get_summary,
    save_summary,
)
from summary.summary_service import (
    generate_course_summary,
    generate_session_summary,
    generate_speaker_summary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summary", tags=["summary"])


class KeywordGenerateRequest(BaseModel):
    """세션 전사문 기반 키워드 추출 요청"""
    session_id: str
    recording_id: str | None = None
    top_k: int = Field(default=20, ge=1, le=100, description="전사문별 추출 키워드 수")
    window_size: int = Field(default=4, ge=2, le=10, description="키워드 그래프 윈도우")


class SpeakerSummaryGenerateRequest(BaseModel):
    """화자 텍스트 기반 요약 요청"""
    session_id: str
    recording_id: str | None = None
    speaker_id: str | None = None
    speaker_text: str = Field(..., min_length=10, description="화자별 전사 텍스트")
    summary_sentences: int = Field(default=3, ge=1, le=10)
    course_id: str | None = None
    source_start_time: float | None = None
    source_end_time: float | None = None


class SessionSummaryGenerateRequest(BaseModel):
    session_id: str
    recording_id: str | None = None
    summary_sentences: int = Field(default=3, ge=1, le=10)
    course_id: str | None = None
    keyword_limit: int | None = Field(default=None, ge=1, le=200)


class CourseSummaryGenerateRequest(BaseModel):
    course_id: str
    summary_sentences: int = Field(default=3, ge=1, le=10)
    keyword_limit: int | None = Field(default=None, ge=1, le=200)


@router.post("/keywords/generate")
async def generate_keywords(req: KeywordGenerateRequest):
    """세션 전사문에서 키워드를 추출하여 저장합니다."""
    logger.info(
        "[KEYWORD] 키워드 생성 요청: session_id=%s, top_k=%s",
        req.session_id,
        req.top_k,
    )

    # 전사문은 speaker 구분 없이 세션 기준으로만 저장됩니다.
    transcripts = await get_transcripts_by_session(req.session_id, req.recording_id)

    if not transcripts:
        raise HTTPException(
            status_code=404,
            detail="전사문이 없습니다.",
        )

    keywords = extract_keywords_from_transcripts(
        transcripts,
        top_k=req.top_k,
        window_size=req.window_size,
    )
    if not keywords:
        raise HTTPException(status_code=400, detail="키워드를 추출할 수 없습니다.")

    transcript_ids = [t["transcript_id"] for t in transcripts if t.get("transcript_id")]
    await delete_keywords_by_transcript_ids(transcript_ids)
    await save_keywords(keywords)

    return {
        "session_id": req.session_id,
        "count": len(keywords),
        "keywords": keywords,
    }


@router.get("/keywords/session/{session_id}")
async def keywords_by_session(session_id: str):
    """세션별 키워드 목록을 반환합니다."""
    keywords = await get_keywords_by_session(session_id)
    return {
        "session_id": session_id,
        "count": len(keywords),
        "keywords": keywords,
    }


@router.post("/speaker/generate")
async def generate_speaker_summaries(req: SpeakerSummaryGenerateRequest):
    """화자 텍스트 기반 요약을 생성합니다."""
    try:
        uuid.UUID(req.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"session_id 형식이 올바르지 않습니다: {req.session_id}") from exc

    course_id = req.course_id
    try:
        if course_id is None:
            course_id = await get_course_id_by_session(req.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"session_id 형식이 올바르지 않습니다: {req.session_id}") from exc
    except Exception as exc:
        logger.warning("[SUMMARY] 세션 코스 조회 실패, course_id 없이 요약 저장 진행: %s", exc)

    speaker_id = req.speaker_id or "UNKNOWN"

    try:
        summary_text = await generate_speaker_summary(
            req.speaker_text,
            speaker_id,
            summary_sentences=req.summary_sentences,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        logger.exception("[SUMMARY] 화자 요약 생성 실패")
        raise HTTPException(status_code=503, detail=str(exc))

    summary_id = str(uuid.uuid4())
    try:
        await save_summary(
            summary_id=summary_id,
            session_id=req.session_id,
            recording_id=req.recording_id,
            course_id=course_id,
            speaker_id=speaker_id,
            speaker_summary=summary_text,
            source_start_time=req.source_start_time,
            source_end_time=req.source_end_time,
            source_text=req.speaker_text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"요약 저장 ID 형식이 올바르지 않습니다: {exc}") from exc
    except Exception as exc:
        logger.exception("[SUMMARY] 화자 요약 저장 실패")
        raise HTTPException(status_code=503, detail=f"화자 요약 저장 실패: {exc}") from exc

    return {
        "summary_id": summary_id,
        "session_id": req.session_id,
        "course_id": course_id,
        "speaker_id": speaker_id,
        "speaker_summary": summary_text,
    }


@router.post("/session/generate")
async def generate_session_summary_api(req: SessionSummaryGenerateRequest):
    """키워드 + 화자 요약을 참고해 세션 요약을 생성합니다."""
    course_id = req.course_id or await get_course_id_by_session(req.session_id)

    keywords = await get_keywords_by_session(req.session_id, limit=req.keyword_limit, recording_id=req.recording_id)
    if not keywords:
        raise HTTPException(status_code=404, detail="세션 키워드가 없습니다.")

    speaker_summaries = await get_latest_speaker_summaries_by_session(req.session_id, req.recording_id)
    if not speaker_summaries:
        raise HTTPException(status_code=404, detail="화자 요약이 없습니다.")

    try:
        summary_text = await generate_session_summary(
            keywords,
            speaker_summaries,
            summary_sentences=req.summary_sentences,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    source_text = "\n".join(
        [kw["keyword_text"] for kw in keywords] +
        [s["speaker_summary"] for s in speaker_summaries if s.get("speaker_summary")]
    )

    summary_id = str(uuid.uuid4())
    await save_summary(
        summary_id=summary_id,
        session_id=req.session_id,
        recording_id=req.recording_id,
        course_id=course_id,
        session_summary=summary_text,
        source_text=source_text,
    )

    return {
        "summary_id": summary_id,
        "session_id": req.session_id,
        "course_id": course_id,
        "session_summary": summary_text,
    }


@router.post("/course/generate")
async def generate_course_summary_api(req: CourseSummaryGenerateRequest):
    """키워드 + 세션 요약 + 화자 요약을 참고해 코스 요약을 생성합니다."""
    keywords = await get_keywords_by_course(req.course_id, limit=req.keyword_limit)
    if not keywords:
        raise HTTPException(status_code=404, detail="코스 키워드가 없습니다.")

    session_summaries = await get_latest_session_summaries_by_course(req.course_id)
    if not session_summaries:
        raise HTTPException(status_code=404, detail="세션 요약이 없습니다.")

    speaker_summaries = await get_latest_speaker_summaries_by_course(req.course_id)
    if not speaker_summaries:
        raise HTTPException(status_code=404, detail="화자 요약이 없습니다.")

    try:
        summary_text = await generate_course_summary(
            keywords,
            session_summaries,
            speaker_summaries,
            summary_sentences=req.summary_sentences,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    source_text = "\n".join(
        [kw["keyword_text"] for kw in keywords] +
        [s["session_summary"] for s in session_summaries if s.get("session_summary")] +
        [s["speaker_summary"] for s in speaker_summaries if s.get("speaker_summary")]
    )

    summary_id = str(uuid.uuid4())
    await save_summary(
        summary_id=summary_id,
        session_id=None,
        course_id=req.course_id,
        course_summary=summary_text,
        source_text=source_text,
    )

    return {
        "summary_id": summary_id,
        "course_id": req.course_id,
        "course_summary": summary_text,
    }


@router.get("/session/{session_id}")
async def summaries_by_session(session_id: str):
    """세션에 연결된 요약 목록을 반환합니다."""
    summaries = await get_summaries_by_session(session_id)
    return {
        "session_id": session_id,
        "count": len(summaries),
        "summaries": summaries,
    }


@router.get("/course/{course_id}")
async def summary_by_course(course_id: str):
    """코스 최신 요약을 반환합니다."""
    summary = await get_latest_course_summary(course_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="코스 요약이 없습니다.")
    return summary


@router.get("/{summary_id}")
async def summary_get(summary_id: str):
    """summary_id로 요약을 조회합니다."""
    summary = await get_summary(summary_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"요약을 찾을 수 없습니다: {summary_id}")
    return summary
