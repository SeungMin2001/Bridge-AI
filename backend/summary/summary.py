"""
요약 API 라우터

엔드포인트:
  POST /summary/keywords/generate          - 세션 전사문 기반 키워드 추출
  POST /summary/keywords/speaker/generate  - 화자별 전사문 기반 키워드 추출(저장 없음)
  GET  /summary/keywords/session/{session_id} - 세션별 키워드 목록 조회
  POST /summary/speaker/generate           - 화자별 요약 생성
  POST /summary/session/generate           - 세션 요약 생성
  POST /summary/course/generate            - 코스 요약 생성
  POST /summary/material/generate          - PDF 강의자료 요약 생성
  DELETE /summary/{summary_id}             - 요약 삭제
  GET  /summary/{summary_id}               - 요약 조회
  GET  /summary/session/{session_id}       - 세션별 요약 목록 조회
  GET  /summary/course/{course_id}         - 코스 요약(최신) 조회
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
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
from summary.material_summary_service import generate_material_summary_with_textrank
from summary.summary_db import (
    delete_summary,
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
    generate_session_summary_from_text,
    generate_speaker_summary,
)
from materials.material_text_service import MaterialTextError, build_material_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summary", tags=["summary"])


class KeywordGenerateRequest(BaseModel):
    """세션 전사문 기반 키워드 추출 요청"""
    session_id: str
    recording_id: str | None = None
    top_k: int = Field(default=20, ge=1, le=100, description="전사문별 추출 키워드 수")
    window_size: int = Field(default=4, ge=2, le=10, description="키워드 그래프 윈도우")


class SpeakerKeywordGenerateRequest(BaseModel):
    """화자별 전사문 기반 키워드 추출 요청"""
    session_id: str
    recording_id: str | None = None
    speaker_id: str | None = None
    speaker_text: str = Field(..., min_length=10, description="화자별 전사 텍스트")
    top_k: int = Field(default=8, ge=1, le=30, description="화자별 추출 키워드 수")
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


class SessionTextSummaryGenerateRequest(BaseModel):
    session_id: str
    recording_id: str | None = None
    session_text: str = Field(..., min_length=10, description="화자 구분 없는 전체 전사 텍스트")
    keywords: list[str] = Field(default_factory=list, description="전체 요약에 참고할 키워드")
    summary_sentences: int = Field(default=3, ge=1, le=10)
    course_id: str | None = None


class CourseSummaryGenerateRequest(BaseModel):
    course_id: str
    summary_sentences: int = Field(default=3, ge=1, le=10)
    keyword_limit: int | None = Field(default=None, ge=1, le=200)


class MaterialSummaryGenerateRequest(BaseModel):
    """세션에 업로드된 PDF 강의자료 기반 요약 요청"""
    session_id: str
    material_ids: list[str] = Field(default_factory=list, description="요약 대상 material id 목록")
    stored_names: list[str] = Field(default_factory=list, description="요약 대상 저장 파일명 목록")
    summary_sentences: int = Field(default=8, ge=1, le=20)
    summary_level: str = Field(default="standard", description="brief, standard, detailed, page")
    course_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=60, description="TextRank로 LLM에 전달할 문장 수")


@router.post("/keywords/generate")
async def generate_keywords(req: KeywordGenerateRequest):
    """세션 전사문에서 키워드를 추출하여 저장합니다."""
    logger.info(
        "[KEYWORD] 키워드 생성 요청: session_id=%s, top_k=%s",
        req.session_id,
        req.top_k,
    )

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
        "recording_id": req.recording_id,
        "count": len(keywords),
        "keywords": keywords,
    }


@router.post("/keywords/speaker/generate")
async def generate_speaker_keywords(req: SpeakerKeywordGenerateRequest):
    """화자별 전사문에서 키워드를 추출하여 저장 없이 반환합니다."""
    speaker_id = req.speaker_id or "UNKNOWN"
    keywords = extract_keywords_from_transcripts(
        [{"transcript_id": None, "text": req.speaker_text}],
        top_k=req.top_k,
        window_size=req.window_size,
    )

    return {
        "session_id": req.session_id,
        "recording_id": req.recording_id,
        "speaker_id": speaker_id,
        "count": len(keywords),
        "keywords": keywords,
    }


@router.get("/keywords/session/{session_id}")
async def keywords_by_session(session_id: str, recording_id: str | None = None):
    """세션별 키워드 목록을 반환합니다."""
    keywords = await get_keywords_by_session(session_id, recording_id=recording_id)
    return {
        "session_id": session_id,
        "recording_id": recording_id,
        "count": len(keywords),
        "keywords": keywords,
    }


@router.post("/speaker/generate")
async def generate_speaker_summaries(req: SpeakerSummaryGenerateRequest):
    """화자 텍스트 기반 요약을 생성합니다."""
    course_id = req.course_id or await get_course_id_by_session(req.session_id)
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
        raise HTTPException(status_code=503, detail=str(exc))

    summary_id = str(uuid.uuid4())
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

    return {
        "summary_id": summary_id,
        "session_id": req.session_id,
        "recording_id": req.recording_id,
        "course_id": course_id,
        "speaker_id": speaker_id,
        "speaker_summary": summary_text,
    }


@router.post("/session/generate")
async def generate_session_summary_api(req: SessionSummaryGenerateRequest):
    """키워드 + 화자 요약을 참고해 세션 요약을 생성합니다."""
    course_id = req.course_id or await get_course_id_by_session(req.session_id)

    keywords = await get_keywords_by_session(
        req.session_id,
        limit=req.keyword_limit,
        recording_id=req.recording_id,
    )
    if not keywords:
        raise HTTPException(status_code=404, detail="세션 키워드가 없습니다.")

    speaker_summaries = await get_latest_speaker_summaries_by_session(
        req.session_id,
        recording_id=req.recording_id,
    )
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
        "recording_id": req.recording_id,
        "course_id": course_id,
        "session_summary": summary_text,
    }


@router.post("/session/text/generate")
async def generate_session_text_summary_api(req: SessionTextSummaryGenerateRequest):
    """화자분리 없이 전체 전사문을 직접 요약합니다."""
    course_id = req.course_id or await get_course_id_by_session(req.session_id)
    keywords = [{"keyword_text": keyword} for keyword in req.keywords if keyword]

    try:
        summary_text = await generate_session_summary_from_text(
            req.session_text,
            keywords=keywords,
            summary_sentences=req.summary_sentences,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    summary_id = str(uuid.uuid4())
    await save_summary(
        summary_id=summary_id,
        session_id=req.session_id,
        recording_id=req.recording_id,
        course_id=course_id,
        session_summary=summary_text,
        source_text=req.session_text,
    )

    return {
        "summary_id": summary_id,
        "session_id": req.session_id,
        "recording_id": req.recording_id,
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


@router.post("/material/generate")
async def generate_material_summary_api(req: MaterialSummaryGenerateRequest):
    """
    PDF 강의자료를 TextRank로 먼저 압축한 뒤 LLM으로 Markdown 요약을 생성해 저장합니다.

    저장 방식:
        summaries.session_summary = PDF 자료 요약 본문
        summaries.speaker_id = "MATERIAL"
        summaries.source_text = PDF/텍스트랭크 메타데이터 JSON
    """
    try:
        material_text, materials = await build_material_text(
            session_id=req.session_id,
            material_ids=req.material_ids,
            stored_names=req.stored_names,
        )
    except MaterialTextError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    course_id = req.course_id
    if course_id is None:
        try:
            course_id = await get_course_id_by_session(req.session_id)
        except Exception as exc:
            logger.warning("[SUMMARY] 세션 코스 조회 실패, course_id 없이 자료 요약 저장 진행: %s", exc)

    try:
        summary_text, textrank_metadata = await generate_material_summary_with_textrank(
            material_text,
            summary_level=req.summary_level,
            top_k=req.top_k or req.summary_sentences * 3,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    source_materials = [
        {
            "id": material.get("id"),
            "name": material.get("name"),
            "storedName": material.get("storedName"),
        }
        for material in materials
    ]
    source_text = json.dumps(
        {
            "type": "material",
            "materials": source_materials,
            "textLength": len(material_text),
            "textPreview": material_text[:3000],
            **textrank_metadata,
        },
        ensure_ascii=False,
    )

    summary_id = str(uuid.uuid4())
    await save_summary(
        summary_id=summary_id,
        session_id=req.session_id,
        course_id=course_id,
        speaker_id="MATERIAL",
        session_summary=summary_text,
        source_text=source_text,
    )

    return {
        "summary_id": summary_id,
        "session_id": req.session_id,
        "course_id": course_id,
        "speaker_id": "MATERIAL",
        "session_summary": summary_text,
        "material_summary": summary_text,
        "summary_level": textrank_metadata["summaryLevel"],
        "source_materials": source_materials,
        "candidate_count": textrank_metadata["candidateCount"],
        "ranked_sentence_count": textrank_metadata["rankedSentenceCount"],
    }


@router.get("/session/{session_id}")
async def summaries_by_session(session_id: str, recording_id: str | None = None):
    """세션에 연결된 요약 목록을 반환합니다."""
    summaries = await get_summaries_by_session(session_id)
    if recording_id:
        summaries = [
            item for item in summaries
            if item.get("recording_id") == recording_id or item.get("speaker_id") == "MATERIAL"
        ]
    return {
        "session_id": session_id,
        "recording_id": recording_id,
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


@router.delete("/{summary_id}")
async def summary_delete(summary_id: str):
    """summary_id로 요약을 삭제합니다."""
    deleted = await delete_summary(summary_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"요약을 찾을 수 없습니다: {summary_id}")
    return {"ok": True, "summary_id": summary_id}
