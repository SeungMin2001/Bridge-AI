import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid as _uuid
from datetime import datetime
import json
from db import get_pool


def _parse_quiz_data(value) -> list:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return value if isinstance(value, list) else []


def _count_quiz_types(quiz_data: list) -> dict[str, int]:
    counts = {
        "MULTIPLE_CHOICE": 0,
        "OX": 0,
    }
    for question in quiz_data:
        quiz_type = question.get("type") if isinstance(question, dict) else None
        if quiz_type in counts:
            counts[quiz_type] += 1
    return counts


def _supported_quiz_data(quiz_data: list) -> list:
    return [
        question for question in quiz_data
        if isinstance(question, dict) and question.get("type") in {"MULTIPLE_CHOICE", "OX"}
    ]


def _safe_correct_count(value, total: int) -> int | None:
    if value is None:
        return None
    try:
        return min(max(0, int(value)), total)
    except (TypeError, ValueError):
        return None


#  퀴즈 CRUD
async def save_quiz(
    quiz_id: str,
    session_id: str,
    quiz_data: list,
    total_questions: int,
    user_id: str | None = None,
    course_id: str | None = None,
    request_id: str | None = None,
    source_title: str | None = None,
) -> dict:
    """퀴즈를 QUIZZES 테이블에 저장"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO quizzes
                (quiz_id, user_id, course_id, request_id, session_id,
                 quiz_data, total_questions, correct_count, source_title, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """,
            _uuid.UUID(quiz_id),
            _uuid.UUID(user_id) if user_id else None,
            _uuid.UUID(course_id) if course_id else None,
            _uuid.UUID(request_id) if request_id else None,
            _uuid.UUID(session_id),
            json.dumps(quiz_data, ensure_ascii=False),
            total_questions,
            None,  # correct_count는 채점 후 업데이트
            source_title,
            datetime.now(),
        )

    return {"quiz_id": quiz_id, "total_questions": total_questions}


async def get_quiz(quiz_id: str) -> dict | None:
    """quiz_id로 퀴즈 조회"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT quiz_id, user_id, course_id, request_id, session_id,
                   quiz_data, total_questions, correct_count, source_title, created_at
            FROM quizzes
            WHERE quiz_id = $1
        """, _uuid.UUID(quiz_id))

        if row is None:
            return None

        quiz_data_parsed = _supported_quiz_data(_parse_quiz_data(row["quiz_data"]))

        return {
            "quiz_id": str(row["quiz_id"]),
            "user_id": str(row["user_id"]) if row["user_id"] else None,
            "course_id": str(row["course_id"]) if row["course_id"] else None,
            "request_id": str(row["request_id"]) if row["request_id"] else None,
            "session_id": str(row["session_id"]) if row["session_id"] else None,
            "quiz_data": quiz_data_parsed,
            "total_questions": len(quiz_data_parsed),
            "correct_count": _safe_correct_count(row["correct_count"], len(quiz_data_parsed)),
            "source_title": row["source_title"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }


async def get_quizzes_by_session(session_id: str) -> list[dict]:
    """session_id에 해당하는 퀴즈 목록 조회"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT quiz_id, user_id, course_id, session_id,
                   quiz_data, total_questions, correct_count, source_title, created_at
            FROM quizzes
            WHERE session_id = $1
            ORDER BY created_at DESC
        """, _uuid.UUID(session_id))

        quizzes = []
        for r in rows:
            quiz_data_parsed = _supported_quiz_data(_parse_quiz_data(r["quiz_data"]))
            total_questions = len(quiz_data_parsed)
            quizzes.append({
                "quiz_id": str(r["quiz_id"]),
                "user_id": str(r["user_id"]) if r["user_id"] else None,
                "course_id": str(r["course_id"]) if r["course_id"] else None,
                "session_id": str(r["session_id"]) if r["session_id"] else None,
                "total_questions": total_questions,
                "correct_count": _safe_correct_count(r["correct_count"], total_questions),
                "source_title": r["source_title"],
                "type_counts": _count_quiz_types(quiz_data_parsed),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            })
        return quizzes


async def update_quiz_result(quiz_id: str, quiz_data: list, correct_count: int) -> dict:
    """퀴즈 채점 결과 업데이트 (user_answer, is_correct 포함된 quiz_data + correct_count)"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE quizzes
            SET quiz_data = $1, correct_count = $2
            WHERE quiz_id = $3
        """,
            json.dumps(quiz_data, ensure_ascii=False),
            correct_count,
            _uuid.UUID(quiz_id),
        )

    return {"quiz_id": quiz_id, "correct_count": correct_count}


async def delete_quiz(quiz_id: str) -> bool:
    """quiz_id로 저장된 퀴즈 삭제"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("""
            DELETE FROM quizzes
            WHERE quiz_id = $1
        """, _uuid.UUID(quiz_id))

    return result.endswith(" 1")
