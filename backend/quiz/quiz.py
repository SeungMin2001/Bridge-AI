"""
퀴즈 API 라우터

엔드포인트:
주석
  POST /quiz/generate          - 세션 전사문 기반 퀴즈 생성
  POST /quiz/generate/transcripts - 선택한 transcript_id 묶음 기반 퀴즈 생성
  POST /quiz/generate/materials - 선택한 PDF 강의자료 기반 퀴즈 생성
  POST /quiz/generate/sources  - 선택한 PDF 강의자료와 전사 chunk를 함께 사용해 퀴즈 생성
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
    delete_quiz,
)
from quiz.quiz_service import count_quiz_types, generate_quiz, grade_quiz
from quiz.material_service import MaterialQuizError, build_material_quiz_text

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
    source_title: str | None = Field(default=None, description="퀴즈 목록에 표시할 소스 제목")
    user_id: str | None = None
    course_id: str | None = None


class QuizGenerateTranscriptsRequest(BaseModel):
    """특정 전사 chunk 묶음 기반 퀴즈 생성 요청"""
    session_id: str
    transcript_ids: list[str] = Field(..., description="퀴즈 생성 대상 transcript_id 목록")
    num_questions: int = Field(default=5, ge=1, le=20, description="생성할 문제 수 (1~20)")
    type_counts: QuizTypeCounts | None = Field(default=None, description="퀴즈 유형별 생성 개수")
    source_title: str | None = Field(default=None, description="퀴즈 목록에 표시할 소스 제목")
    user_id: str | None = None
    course_id: str | None = None


class QuizGenerateMaterialsRequest(BaseModel):
    """세션에 업로드된 PDF 강의자료 기반 퀴즈 생성 요청"""
    session_id: str
    material_ids: list[str] = Field(default_factory=list, description="퀴즈 생성 대상 material id 목록")
    stored_names: list[str] = Field(default_factory=list, description="퀴즈 생성 대상 저장 파일명 목록")
    num_questions: int = Field(default=5, ge=1, le=20, description="생성할 문제 수 (1~20)")
    type_counts: QuizTypeCounts | None = Field(default=None, description="퀴즈 유형별 생성 개수")
    source_title: str | None = Field(default=None, description="퀴즈 목록에 표시할 소스 제목")
    user_id: str | None = None
    course_id: str | None = None


class QuizGenerateSourcesRequest(BaseModel):
    """선택한 PDF 강의자료와 전사 chunk를 함께 사용하는 퀴즈 생성 요청"""
    session_id: str
    material_ids: list[str] = Field(default_factory=list, description="퀴즈 생성 대상 material id 목록")
    stored_names: list[str] = Field(default_factory=list, description="퀴즈 생성 대상 저장 파일명 목록")
    transcript_ids: list[str] = Field(default_factory=list, description="퀴즈 생성 대상 transcript_id 목록")
    num_questions: int = Field(default=5, ge=1, le=20, description="생성할 문제 수 (1~20)")
    type_counts: QuizTypeCounts | None = Field(default=None, description="퀴즈 유형별 생성 개수")
    source_title: str | None = Field(default=None, description="퀴즈 목록에 표시할 소스 제목")
    user_id: str | None = None
    course_id: str | None = None


class QuizGenerateTextRequest(BaseModel):
    """직접 텍스트로 퀴즈 생성 (LLM 테스트/프론트 개발용)"""
    text: str = Field(..., min_length=10, description="퀴즈 생성 대상 텍스트")
    num_questions: int = Field(default=5, ge=1, le=20)
    type_counts: QuizTypeCounts | None = Field(default=None, description="퀴즈 유형별 생성 개수")
    session_id: str | None = None
    source_title: str | None = Field(default=None, description="퀴즈 목록에 표시할 소스 제목")
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


def build_source_title_from_materials(materials: list[dict]) -> str | None:
    titles = [
        str(material.get("name") or material.get("storedName") or "").strip()
        for material in materials
        if material.get("name") or material.get("storedName")
    ]
    if not titles:
        return None
    if len(titles) == 1:
        return titles[0]
    return f"{titles[0]} 외 +{len(titles) - 1}개 소스"


async def build_selected_transcript_text(
    session_id: str,
    transcript_ids: list[str],
) -> tuple[str, list[str], list[str]]:
    """선택한 transcript_id 목록을 조회해 퀴즈 입력 텍스트와 누락 id를 반환합니다."""
    normalized_ids = normalize_transcript_ids(transcript_ids)
    if not normalized_ids:
        return "", [], []

    try:
        transcripts = await get_transcripts_by_ids(session_id, normalized_ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"transcript_id 형식이 올바르지 않습니다: {exc}") from exc

    found_ids = {item["transcript_id"] for item in transcripts}
    missing_ids = [item for item in normalized_ids if item not in found_ids]
    return build_transcript_text(transcripts), [item["transcript_id"] for item in transcripts], missing_ids


async def create_and_save_quiz(
    session_id: str,
    transcript_text: str,
    num_questions: int,
    type_counts: QuizTypeCounts | None = None,
    user_id: str | None = None,
    course_id: str | None = None,
    source_title: str | None = None,
) -> dict:
    if len(transcript_text.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="퀴즈 생성 대상 텍스트가 너무 짧아 퀴즈를 생성할 수 없습니다."
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
        source_title=source_title,
    )

    logger.info(f"[QUIZ] 퀴즈 저장 완료: quiz_id={quiz_id}, {len(quiz_data)}문제")
    return {
        "quiz_id": quiz_id,
        "total_questions": len(quiz_data),
        "type_counts": count_quiz_types(quiz_data),
        "source_title": source_title,
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
        source_title=req.source_title,
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

    transcript_text, source_transcript_ids, missing_ids = await build_selected_transcript_text(
        req.session_id,
        req.transcript_ids,
    )
    if not source_transcript_ids:
        raise HTTPException(status_code=422, detail="transcript_ids는 1개 이상 필요합니다.")

    result = await create_and_save_quiz(
        session_id=req.session_id,
        transcript_text=transcript_text,
        num_questions=req.num_questions,
        type_counts=req.type_counts,
        user_id=req.user_id,
        course_id=req.course_id,
        source_title=req.source_title,
    )
    return {
        **result,
        "source_transcript_ids": source_transcript_ids,
        "missing_transcript_ids": missing_ids,
    }


#  PDF 강의자료 → 퀴즈 생성
@router.post("/generate/materials")
async def quiz_generate_from_materials(req: QuizGenerateMaterialsRequest):
    """
    세션에 업로드된 PDF 강의자료 파일을 백엔드에서 읽어 퀴즈를 생성하고 DB에 저장합니다.
    material_ids 또는 stored_names가 비어 있으면 세션의 모든 PDF 강의자료를 대상으로 합니다.
    """
    logger.info(
        "[QUIZ] PDF 강의자료 기반 퀴즈 생성 요청: session_id=%s, material_ids=%s, stored_names=%s, num=%s",
        req.session_id,
        len(req.material_ids),
        len(req.stored_names),
        req.num_questions,
    )

    try:
        material_text, materials = await build_material_quiz_text(
            session_id=req.session_id,
            material_ids=req.material_ids,
            stored_names=req.stored_names,
        )
    except MaterialQuizError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    result = await create_and_save_quiz(
        session_id=req.session_id,
        transcript_text=material_text,
        num_questions=req.num_questions,
        type_counts=req.type_counts,
        user_id=req.user_id,
        course_id=req.course_id,
        source_title=req.source_title or build_source_title_from_materials(materials),
    )

    return {
        **result,
        "source_materials": [
            {
                "id": material.get("id"),
                "name": material.get("name"),
                "storedName": material.get("storedName"),
            }
            for material in materials
        ],
    }


#  선택 소스 묶음 → 퀴즈 생성
@router.post("/generate/sources")
async def quiz_generate_from_sources(req: QuizGenerateSourcesRequest):
    """
    좌측 소스 사이드바에서 선택한 PDF 강의자료와 녹음 전사 chunk를 합쳐 퀴즈를 생성합니다.
    PDF와 전사를 함께 선택한 경우 한 요청에서 같은 문제 생성 범위로 사용합니다.
    """
    logger.info(
        "[QUIZ] 선택 소스 기반 퀴즈 생성 요청: session_id=%s, materials=%s/%s, transcripts=%s, num=%s",
        req.session_id,
        len(req.material_ids),
        len(req.stored_names),
        len(req.transcript_ids),
        req.num_questions,
    )

    text_parts = []
    source_materials = []
    source_transcript_ids = []
    missing_transcript_ids = []

    if req.material_ids or req.stored_names:
        try:
            material_text, source_materials = await build_material_quiz_text(
                session_id=req.session_id,
                material_ids=req.material_ids,
                stored_names=req.stored_names,
            )
        except MaterialQuizError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        if material_text.strip():
            text_parts.append(f"[강의자료]\n{material_text.strip()}")

    if req.transcript_ids:
        transcript_text, source_transcript_ids, missing_transcript_ids = await build_selected_transcript_text(
            req.session_id,
            req.transcript_ids,
        )
        if transcript_text.strip():
            text_parts.append(f"[전사]\n{transcript_text.strip()}")

    if not text_parts:
        raise HTTPException(status_code=422, detail="퀴즈 생성 대상 소스가 없습니다.")

    result = await create_and_save_quiz(
        session_id=req.session_id,
        transcript_text="\n\n".join(text_parts),
        num_questions=req.num_questions,
        type_counts=req.type_counts,
        user_id=req.user_id,
        course_id=req.course_id,
        source_title=req.source_title or build_source_title_from_materials(source_materials),
    )

    return {
        **result,
        "source_materials": [
            {
                "id": material.get("id"),
                "name": material.get("name"),
                "storedName": material.get("storedName"),
            }
            for material in source_materials
        ],
        "source_transcript_ids": source_transcript_ids,
        "missing_transcript_ids": missing_transcript_ids,
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
        source_title=req.source_title,
    )

    return {
        "quiz_id": quiz_id,
        "session_id": session_id,
        "total_questions": len(quiz_data),
        "type_counts": count_quiz_types(quiz_data),
        "source_title": req.source_title,
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


#  퀴즈 삭제
@router.delete("/{quiz_id}")
async def quiz_delete(quiz_id: str):
    """quiz_id로 저장된 퀴즈를 삭제합니다."""
    deleted = await delete_quiz(quiz_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"퀴즈를 찾을 수 없습니다: {quiz_id}")
    return {
        "quiz_id": quiz_id,
        "deleted": True,
    }


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
