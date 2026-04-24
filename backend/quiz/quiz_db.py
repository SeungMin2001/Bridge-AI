import uuid as _uuid
from datetime import datetime
import json
from db import get_pool

# ══════════════════════════════════════
#  퀴즈 CRUD
# ══════════════════════════════════════
async def save_quiz(
    quiz_id: str,
    session_id: str,
    quiz_data: list,
    total_questions: int,
    user_id: str | None = None,
    course_id: str | None = None,
    request_id: str | None = None,
) -> dict:
    """퀴즈를 QUIZZES 테이블에 저장"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO quizzes
                (quiz_id, user_id, course_id, request_id, session_id,
                 quiz_data, total_questions, correct_count, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
            _uuid.UUID(quiz_id),
            _uuid.UUID(user_id) if user_id else None,
            _uuid.UUID(course_id) if course_id else None,
            _uuid.UUID(request_id) if request_id else None,
            _uuid.UUID(session_id),
            json.dumps(quiz_data, ensure_ascii=False),
            total_questions,
            None,  # correct_count는 채점 후 업데이트
            datetime.now(),
        )

    return {"quiz_id": quiz_id, "total_questions": total_questions}


async def get_quiz(quiz_id: str) -> dict | None:
    """quiz_id로 퀴즈 조회"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT quiz_id, user_id, course_id, request_id, session_id,
                   quiz_data, total_questions, correct_count, created_at
            FROM quizzes
            WHERE quiz_id = $1
        """, _uuid.UUID(quiz_id))

        if row is None:
            return None

        quiz_data_raw = row["quiz_data"]
        if isinstance(quiz_data_raw, str):
            quiz_data_parsed = json.loads(quiz_data_raw)
        else:
            quiz_data_parsed = quiz_data_raw

        return {
            "quiz_id": str(row["quiz_id"]),
            "user_id": str(row["user_id"]) if row["user_id"] else None,
            "course_id": str(row["course_id"]) if row["course_id"] else None,
            "request_id": str(row["request_id"]) if row["request_id"] else None,
            "session_id": str(row["session_id"]) if row["session_id"] else None,
            "quiz_data": quiz_data_parsed,
            "total_questions": row["total_questions"],
            "correct_count": row["correct_count"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }


async def get_quizzes_by_session(session_id: str) -> list[dict]:
    """session_id에 해당하는 퀴즈 목록 조회"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT quiz_id, user_id, course_id, session_id,
                   total_questions, correct_count, created_at
            FROM quizzes
            WHERE session_id = $1
            ORDER BY created_at DESC
        """, _uuid.UUID(session_id))

        return [
            {
                "quiz_id": str(r["quiz_id"]),
                "user_id": str(r["user_id"]) if r["user_id"] else None,
                "course_id": str(r["course_id"]) if r["course_id"] else None,
                "session_id": str(r["session_id"]) if r["session_id"] else None,
                "total_questions": r["total_questions"],
                "correct_count": r["correct_count"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]


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
