"""
요약 API 라우터

엔드포인트:
  POST /summary/generate              - 세션 전사문 기반 요약 생성
  POST /summary/generate/text         - 직접 텍스트로 요약 생성 (테스트용)
  GET  /summary/session/{session_id}  - 세션별 요약 목록 조회
  GET  /summary/{summary_id}          - 요약 단건 조회
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import uuid
import json
import logging

from db import get_transcripts_by_session
from summary.summary_db import (
    save_summary,
    get_summary,
    get_summaries_by_session,
    save_key_sentences,
)
from summary.summary_service import generate_summary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summary", tags=["summary"])


# ── 요청 스키마 ──
class SummaryGenerateRequest(BaseModel):
    """세션 기반 요약 생성 요청"""
    session_id: str
    num_key_sentences: int = Field(default=10, ge=3, le=30, description="TextRank 추출 핵심 문장 수 (3~30)")
    course_id: str | None = None


class SummaryGenerateTextRequest(BaseModel):
    """직접 텍스트로 요약 생성 (테스트용)"""
    text: str = Field(..., min_length=20, description="요약 대상 텍스트")
    num_key_sentences: int = Field(default=10, ge=3, le=30)
    session_id: str | None = None
    course_id: str | None = None


# ══════════════════════════════════════
#  POST /summary/generate
#  세션 전사문 → 요약 생성
# ══════════════════════════════════════
@router.post("/generate")
async def summary_generate(req: SummaryGenerateRequest):
    """
    session_id의 전사문을 조회하여 요약을 생성하고 DB에 저장합니다.

    Response:
    {
        "summary_id": "uuid",
        "summary": {
            "core_concepts": [...],
            "key_sentences": [...],
            "review_points": [...],
            "detailed_explanation": "...",
            "exam_points": [...]
        },
        "key_sentences_extracted": [...],
        "nouns": [...]
    }
    """
    logger.info(f"[SUMMARY] 요약 생성 요청: session_id={req.session_id}")

    # 1. 세션 전사문 조회
    transcripts = await get_transcripts_by_session(req.session_id)
    if not transcripts:
        raise HTTPException(
            status_code=404,
            detail=f"세션 '{req.session_id}'에 해당하는 전사문이 없습니다."
        )

    # 2. 전사문 텍스트 합치기
    transcript_text = "\n".join(t["text"] for t in transcripts if t["text"])
    if len(transcript_text.strip()) < 30:
        raise HTTPException(
            status_code=400,
            detail="전사문 텍스트가 너무 짧아 요약을 생성할 수 없습니다."
        )

    # 3. 요약 파이프라인 실행 (형태소→TextRank→LLM)
    try:
        result = await generate_summary(transcript_text, req.num_key_sentences)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # 4. DB 저장 — SUMMARIES
    summary_id = str(uuid.uuid4())
    summary_text_json = json.dumps(result["summary"], ensure_ascii=False)

    # 전사문 시간 범위 계산
    start_time = transcripts[0]["start_time"] if transcripts[0]["start_time"] else None
    end_time = transcripts[-1]["end_time"] if transcripts[-1]["end_time"] else None

    await save_summary(
        summary_id=summary_id,
        session_id=req.session_id,
        summary_text=summary_text_json,
        course_id=req.course_id,
        source_start_time=start_time,
        source_end_time=end_time,
        source_text=transcript_text[:2000],  # 원본 텍스트 일부 저장
    )

    # 5. DB 저장 — KEY_SENTENCES
    await save_key_sentences(req.session_id, result["key_sentences"])

    logger.info(f"[SUMMARY] 요약 저장 완료: summary_id={summary_id}")

    return {
        "summary_id": summary_id,
        "summary": result["summary"],
        "key_sentences_extracted": result["key_sentences"],
        "nouns": result["nouns"],
    }


# ══════════════════════════════════════
#  POST /summary/generate/text
#  직접 텍스트 → 요약 생성 (테스트용)
# ══════════════════════════════════════
@router.post("/generate/text")
async def summary_generate_from_text(req: SummaryGenerateTextRequest):
    """
    텍스트를 직접 입력하여 요약을 생성합니다.
    LLM 연동 테스트 및 프론트엔드 개발용.
    """
    logger.info(f"[SUMMARY] 텍스트 기반 요약 생성 요청: {len(req.text)} chars")

    try:
        result = await generate_summary(req.text, req.num_key_sentences)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # session_id가 없으면 임시 생성
    session_id = req.session_id or str(uuid.uuid4())
    summary_id = str(uuid.uuid4())
    summary_text_json = json.dumps(result["summary"], ensure_ascii=False)

    await save_summary(
        summary_id=summary_id,
        session_id=session_id,
        summary_text=summary_text_json,
        course_id=req.course_id,
    )

    await save_key_sentences(session_id, result["key_sentences"])

    return {
        "summary_id": summary_id,
        "session_id": session_id,
        "summary": result["summary"],
        "key_sentences_extracted": result["key_sentences"],
        "nouns": result["nouns"],
    }


# ══════════════════════════════════════
#  GET /summary/session/{session_id}
#  세션별 요약 목록 (/{summary_id}보다 먼저 선언 — 경로 충돌 방지)
# ══════════════════════════════════════
@router.get("/session/{session_id}")
async def summary_list_by_session(session_id: str):
    """session_id에 해당하는 요약 목록을 최신순으로 반환합니다."""
    summaries = await get_summaries_by_session(session_id)
    return {
        "session_id": session_id,
        "count": len(summaries),
        "summaries": summaries,
    }


# ══════════════════════════════════════
#  GET /summary/{summary_id}
#  요약 단건 조회
# ══════════════════════════════════════
@router.get("/{summary_id}")
async def summary_get(summary_id: str):
    """summary_id로 요약 데이터를 조회합니다."""
    summary = await get_summary(summary_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"요약을 찾을 수 없습니다: {summary_id}")

    # summary_text가 JSON 문자열이면 파싱
    if isinstance(summary["summary_text"], str):
        try:
            summary["summary_text"] = json.loads(summary["summary_text"])
        except json.JSONDecodeError:
            pass  # 파싱 실패 시 문자열 그대로 반환

    return summary
