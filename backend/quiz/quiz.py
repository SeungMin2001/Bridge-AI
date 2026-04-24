"""
퀴즈 API 라우터

엔드포인트:
  POST /quiz/generate          - 세션 전사문 기반 퀴즈 생성
  POST /quiz/generate/text     - 직접 텍스트로 퀴즈 생성 (테스트용)
  GET  /quiz/{quiz_id}         - 퀴즈 조회
  POST /quiz/{quiz_id}/submit  - 퀴즈 채점 (사용자 답안 제출)
  GET  /quiz/session/{session_id} - 세션별 퀴즈 목록 조회
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import uuid
import logging

from db import get_transcripts_by_session
from quiz.quiz_db import (
    save_quiz,
    get_quiz,
    get_quizzes_by_session,
    update_quiz_result,
)
from quiz.quiz_service import generate_quiz, grade_quiz

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quiz", tags=["quiz"])


# ── 요청/응답 스키마 ──
class QuizGenerateRequest(BaseModel):
    """세션 기반 퀴즈 생성 요청"""
    session_id: str
    num_questions: int = Field(default=5, ge=1, le=20, description="생성할 문제 수 (1~20)")
    user_id: str | None = None
    course_id: str | None = None


class QuizGenerateTextRequest(BaseModel):
    """직접 텍스트로 퀴즈 생성 (LLM 테스트/프론트 개발용)"""
    text: str = Field(..., min_length=10, description="퀴즈 생성 대상 텍스트")
    num_questions: int = Field(default=5, ge=1, le=20)
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

    # 2. 전사문 텍스트 합치기
    transcript_text = "\n".join(t["text"] for t in transcripts if t["text"])
    if len(transcript_text.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="전사문 텍스트가 너무 짧아 퀴즈를 생성할 수 없습니다."
        )

    # 3. 퀴즈 생성 (LLM 호출 또는 Mock)
    try:
        quiz_data = await generate_quiz(transcript_text, req.num_questions)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # 4. DB 저장
    quiz_id = str(uuid.uuid4())
    await save_quiz(
        quiz_id=quiz_id,
        session_id=req.session_id,
        quiz_data=quiz_data,
        total_questions=len(quiz_data),
        user_id=req.user_id,
        course_id=req.course_id,
    )

    logger.info(f"[QUIZ] 퀴즈 저장 완료: quiz_id={quiz_id}, {len(quiz_data)}문제")

    return {
        "quiz_id": quiz_id,
        "total_questions": len(quiz_data),
        "quiz_data": quiz_data,
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
        quiz_data = await generate_quiz(req.text, req.num_questions)
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
        "quiz_data": quiz_data,
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


#  세션별 퀴즈 목록
@router.get("/session/{session_id}")
async def quiz_list_by_session(session_id: str):
    """session_id에 해당하는 퀴즈 목록을 최신순으로 반환합니다."""
    quizzes = await get_quizzes_by_session(session_id)
    return {
        "session_id": session_id,
        "count": len(quizzes),
        "quizzes": quizzes,
    }
