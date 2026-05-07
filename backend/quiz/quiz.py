"""
퀴즈 API 라우터

엔드포인트:
  POST /quiz/generate          - 세션 전사문 기반 퀴즈 생성
  POST /quiz/generate/transcripts - 선택한 transcript_id 묶음 기반 퀴즈 생성
  POST /quiz/generate/text     - 직접 텍스트로 퀴즈 생성 (테스트용)
  GET  /quiz/{quiz_id}         - 퀴즈 조회
  POST /quiz/{quiz_id}/submit  - 퀴즈 채점 (사용자 답안 제출)
  GET  /quiz/session/{session_id} - 세션별 퀴즈 목록 조회
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import uuid
import logging

from db import get_transcripts_by_ids, get_transcripts_by_session
from quiz.quiz_db import (
    save_quiz,
    get_quiz,
    get_quizzes_by_session,
    update_quiz_result,
)
from quiz.quiz_service import count_quiz_types, generate_quiz, grade_quiz

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quiz", tags=["quiz"])


# ── 요청/응답 스키마 ──
class QuizTypeCounts(BaseModel):
    """퀴즈 유형별 생성 개수"""
    MULTIPLE_CHOICE: int = Field(default=0, ge=0, le=20, description="객관식 문항 수")
    OX: int = Field(default=0, ge=0, le=20, description="O/X 문항 수")
    SHORT_ANSWER: int = Field(default=0, ge=0, le=20, description="단답형 문항 수")


class QuizGenerateRequest(BaseModel):
    """세션 기반 퀴즈 생성 요청"""
    session_id: str
    num_questions: int = Field(default=5, ge=1, le=20, description="생성할 문제 수 (1~20)")
    type_counts: QuizTypeCounts | None = Field(default=None, description="퀴즈 유형별 생성 개수")
    user_id: str | None = None
    course_id: str | None = None


class QuizGenerateTranscriptsRequest(BaseModel):
    """특정 전사 chunk 묶음 기반 퀴즈 생성 요청"""
    session_id: str
    transcript_ids: list[str] = Field(..., description="퀴즈 생성 대상 transcript_id 목록")
    num_questions: int = Field(default=5, ge=1, le=20, description="생성할 문제 수 (1~20)")
    type_counts: QuizTypeCounts | None = Field(default=None, description="퀴즈 유형별 생성 개수")
    user_id: str | None = None
    course_id: str | None = None


class QuizGenerateTextRequest(BaseModel):
    """직접 텍스트로 퀴즈 생성 (LLM 테스트/프론트 개발용)"""
    text: str = Field(..., min_length=10, description="퀴즈 생성 대상 텍스트")
    num_questions: int = Field(default=5, ge=1, le=20)
    type_counts: QuizTypeCounts | None = Field(default=None, description="퀴즈 유형별 생성 개수")
    session_id: str | None = None
    user_id: str | None = None
    course_id: str | None = None


class QuizSubmitRequest(BaseModel):
    """퀴즈 채점 요청"""
    answers: dict[str, str] = Field(
        ...,
        description='{ "1": "사용자답", "2": "사용자답", ... } (key=question_index)',
        examples=[{"1": "1. 데이터 중복 제거", "2": "X", "3": "제1정규형"}],
    )


def build_transcript_text(transcripts: list[dict]) -> str:
    lines = []
    for transcript in transcripts:
        text = (transcript.get("text") or "").strip()
        if not text:
            continue
        recording_id = transcript.get("recording_id")
        chunk_index = transcript.get("chunk_index")
        if recording_id:
            lines.append(f"[recording_id={recording_id}, chunk={chunk_index}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def normalize_transcript_ids(transcript_ids: list[str]) -> list[str]:
    normalized = []
    seen = set()
    for transcript_id in transcript_ids:
        value = str(transcript_id or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def dump_type_counts(type_counts: QuizTypeCounts | None) -> dict[str, int] | None:
    if type_counts is None:
        return None
    if hasattr(type_counts, "model_dump"):
        return type_counts.model_dump()
    return type_counts.dict()


async def create_and_save_quiz(
    session_id: str,
    transcript_text: str,
    num_questions: int,
    type_counts: QuizTypeCounts | None = None,
    user_id: str | None = None,
    course_id: str | None = None,
) -> dict:
    if len(transcript_text.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="전사문 텍스트가 너무 짧아 퀴즈를 생성할 수 없습니다."
        )

    try:
        quiz_data = await generate_quiz(
            transcript_text,
            num_questions,
            dump_type_counts(type_counts),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    quiz_id = str(uuid.uuid4())
    await save_quiz(
        quiz_id=quiz_id,
        session_id=session_id,
        quiz_data=quiz_data,
        total_questions=len(quiz_data),
        user_id=user_id,
        course_id=course_id,
    )

    logger.info(f"[QUIZ] 퀴즈 저장 완료: quiz_id={quiz_id}, {len(quiz_data)}문제")
    return {
        "quiz_id": quiz_id,
        "total_questions": len(quiz_data),
        "type_counts": count_quiz_types(quiz_data),
        "quiz_data": quiz_data,
    }

#  세션 전사문 → 퀴즈 생성
@router.post("/generate")
async def quiz_generate(req: QuizGenerateRequest):
    """
    session_id의 전사문을 조회하여 퀴즈를 생성하고 DB에 저장합니다.

    Response:
    {
        "quiz_id": "uuid",
        "total_questions": 5,
        "quiz_data": [ ... ]
    }
    """
    logger.info(f"[QUIZ] 퀴즈 생성 요청: session_id={req.session_id}, num={req.num_questions}")

    # 1. 세션 전사문 조회
    transcripts = await get_transcripts_by_session(req.session_id)
    if not transcripts:
        raise HTTPException(
            status_code=404,
            detail=f"세션 '{req.session_id}'에 해당하는 전사문이 없습니다."
        )

    transcript_text = build_transcript_text(transcripts)
    return await create_and_save_quiz(
        session_id=req.session_id,
        transcript_text=transcript_text,
        num_questions=req.num_questions,
        type_counts=req.type_counts,
        user_id=req.user_id,
        course_id=req.course_id,
    )


#  특정 transcript_id 묶음 → 퀴즈 생성
@router.post("/generate/transcripts")
async def quiz_generate_from_transcripts(req: QuizGenerateTranscriptsRequest):
    """
    선택한 녹음본/자료에 연결된 transcript_id 목록만 사용하여 퀴즈를 생성하고 DB에 저장합니다.
    """
    logger.info(
        "[QUIZ] transcript_id 기반 퀴즈 생성 요청: session_id=%s, transcript_ids=%s, num=%s",
        req.session_id,
        len(req.transcript_ids),
        req.num_questions,
    )

    transcript_ids = normalize_transcript_ids(req.transcript_ids)
    if not transcript_ids:
        raise HTTPException(status_code=422, detail="transcript_ids는 1개 이상 필요합니다.")

    try:
        transcripts = await get_transcripts_by_ids(req.session_id, transcript_ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"transcript_id 형식이 올바르지 않습니다: {exc}") from exc

    if not transcripts:
        raise HTTPException(
            status_code=404,
            detail="선택한 전사 chunk를 찾을 수 없습니다."
        )

    found_ids = {item["transcript_id"] for item in transcripts}
    missing_ids = [item for item in transcript_ids if item not in found_ids]
    result = await create_and_save_quiz(
        session_id=req.session_id,
        transcript_text=build_transcript_text(transcripts),
        num_questions=req.num_questions,
        type_counts=req.type_counts,
        user_id=req.user_id,
        course_id=req.course_id,
    )
    return {
        **result,
        "source_transcript_ids": [item["transcript_id"] for item in transcripts],
        "missing_transcript_ids": missing_ids,
    }

#  직접 텍스트 → 퀴즈 생성
@router.post("/generate/text")
async def quiz_generate_from_text(req: QuizGenerateTextRequest):
    """
    텍스트를 직접 입력하여 퀴즈를 생성합니다.
    LLM 연동 테스트 및 프론트엔드 개발용.
    session_id가 없으면 임시 ID를 생성합니다.
    """
    logger.info(f"[QUIZ] 텍스트 기반 퀴즈 생성 요청: {len(req.text)} chars, num={req.num_questions}")

    try:
        quiz_data = await generate_quiz(
            req.text,
            req.num_questions,
            dump_type_counts(req.type_counts),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # session_id가 없으면 임시 생성
    session_id = req.session_id or str(uuid.uuid4())

    quiz_id = str(uuid.uuid4())
    await save_quiz(
        quiz_id=quiz_id,
        session_id=session_id,
        quiz_data=quiz_data,
        total_questions=len(quiz_data),
        user_id=req.user_id,
        course_id=req.course_id,
    )

    return {
        "quiz_id": quiz_id,
        "session_id": session_id,
        "total_questions": len(quiz_data),
        "type_counts": count_quiz_types(quiz_data),
        "quiz_data": quiz_data,
    }

#  세션별 퀴즈 목록 (/{quiz_id}보다 먼저 선언해야 경로 충돌 방지)
@router.get("/session/{session_id}")
async def quiz_list_by_session(session_id: str):
    """session_id에 해당하는 퀴즈 목록을 최신순으로 반환합니다."""
    quizzes = await get_quizzes_by_session(session_id)
    return {
        "session_id": session_id,
        "count": len(quizzes),
        "quizzes": quizzes,
    }


#  퀴즈 조회
@router.get("/{quiz_id}")
async def quiz_get(quiz_id: str):
    """quiz_id로 퀴즈 데이터를 조회합니다."""
    quiz = await get_quiz(quiz_id)
    if quiz is None:
        raise HTTPException(status_code=404, detail=f"퀴즈를 찾을 수 없습니다: {quiz_id}")
    return quiz


#  퀴즈 채점
@router.post("/{quiz_id}/submit")
async def quiz_submit(quiz_id: str, req: QuizSubmitRequest):
    """
    사용자 답안을 제출하여 채점합니다.

    Request:
    { "answers": { "1": "1. 데이터 중복 제거", "2": "X", "3": "제1정규형" } }

    Response:
    {
        "quiz_id": "uuid",
        "total_questions": 5,
        "correct_count": 3,
        "score": 60.0,
        "quiz_data": [ ... (user_answer, is_correct 채워짐) ]
    }
    """
    logger.info(f"[QUIZ] 채점 요청: quiz_id={quiz_id}, answers={len(req.answers)}개")

    # 1. 퀴즈 조회
    quiz = await get_quiz(quiz_id)
    if quiz is None:
        raise HTTPException(status_code=404, detail=f"퀴즈를 찾을 수 없습니다: {quiz_id}")

    # 2. 채점
    graded_data, correct_count = grade_quiz(quiz["quiz_data"], req.answers)

    # 3. DB 업데이트
    await update_quiz_result(quiz_id, graded_data, correct_count)

    total = quiz["total_questions"]
    score = round((correct_count / total) * 100, 1) if total > 0 else 0.0

    logger.info(f"[QUIZ] 채점 완료: {correct_count}/{total} ({score}%)")

    return {
        "quiz_id": quiz_id,
        "total_questions": total,
        "correct_count": correct_count,
        "score": score,
        "quiz_data": graded_data,
    }
