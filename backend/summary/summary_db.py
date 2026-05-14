import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid as _uuid
from datetime import datetime

from db import get_pool


async def ensure_summaries_schema(conn) -> None:
    await conn.execute("ALTER TABLE summaries ADD COLUMN IF NOT EXISTS recording_id TEXT NULL")


def _row_to_summary(row: dict) -> dict:
    """DB row를 API 응답용 dict로 변환합니다."""
    return {
        "summary_id": str(row["summary_id"]),
        "session_id": str(row["session_id"]) if row["session_id"] else None,
        "recording_id": row["recording_id"],
        "course_id": str(row["course_id"]) if row["course_id"] else None,
        "transcript_id": str(row["transcript_id"]) if row["transcript_id"] else None,
        "speaker_id": row["speaker_id"],
        "speaker_summary": row["speaker_summary"],
        "session_summary": row["session_summary"],
        "course_summary": row["course_summary"],
        "source_start_time": row["source_start_time"],
        "source_end_time": row["source_end_time"],
        "source_text": row["source_text"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }


async def save_summary(
    summary_id: str,
    session_id: str | None,
    recording_id: str | None = None,
    course_id: str | None = None,
    transcript_id: str | None = None,
    speaker_id: str | None = None,
    speaker_summary: str | None = None,
    session_summary: str | None = None,
    course_summary: str | None = None,
    source_start_time: float | None = None,
    source_end_time: float | None = None,
    source_text: str | None = None,
) -> dict:
    """요약 레코드를 summaries 테이블에 저장합니다."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_summaries_schema(conn)
        await conn.execute(
            """
            INSERT INTO summaries
                (summary_id, session_id, recording_id, course_id, transcript_id, speaker_id,
                 speaker_summary, session_summary, course_summary,
                 source_start_time, source_end_time, source_text, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
            _uuid.UUID(summary_id),
            _uuid.UUID(session_id) if session_id else None,
            recording_id,
            _uuid.UUID(course_id) if course_id else None,
            _uuid.UUID(transcript_id) if transcript_id else None,
            speaker_id,
            speaker_summary,
            session_summary,
            course_summary,
            source_start_time,
            source_end_time,
            source_text,
            datetime.now(),
        )

    return {"summary_id": summary_id}


async def get_summary(summary_id: str) -> dict | None:
    """summary_id로 요약을 단건 조회합니다."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_summaries_schema(conn)
        row = await conn.fetchrow(
            """
            SELECT summary_id, session_id, recording_id, course_id, transcript_id, speaker_id,
                   speaker_summary, session_summary, course_summary,
                   source_start_time, source_end_time, source_text, created_at
            FROM summaries
            WHERE summary_id = $1
            """,
            _uuid.UUID(summary_id),
        )

        if row is None:
            return None

        return _row_to_summary(row)


async def delete_summary(summary_id: str) -> bool:
    """summary_id에 해당하는 요약 레코드를 삭제합니다."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM summaries
            WHERE summary_id = $1
            """,
            _uuid.UUID(summary_id),
        )

    return result.endswith("1")


async def get_summaries_by_session(session_id: str) -> list[dict]:
    """session_id에 연결된 요약 목록을 조회합니다."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_summaries_schema(conn)
        rows = await conn.fetch(
            """
            SELECT summary_id, session_id, recording_id, course_id, transcript_id, speaker_id,
                   speaker_summary, session_summary, course_summary,
                   source_start_time, source_end_time, source_text, created_at
            FROM summaries
            WHERE session_id = $1
            ORDER BY created_at DESC
            """,
            _uuid.UUID(session_id),
        )

        return [_row_to_summary(r) for r in rows]


async def get_latest_speaker_summaries_by_session(
    session_id: str,
    recording_id: str | None = None,
) -> list[dict]:
    """세션에서 최신 화자 요약을 speaker_id별로 조회합니다."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_summaries_schema(conn)
        params = [_uuid.UUID(session_id)]
        recording_filter = ""
        if recording_id:
            params.append(recording_id)
            recording_filter = " AND recording_id = $2"
        rows = await conn.fetch(
            f"""
            SELECT DISTINCT ON (speaker_id)
                   summary_id, session_id, recording_id, course_id, transcript_id, speaker_id,
                   speaker_summary, session_summary, course_summary,
                   source_start_time, source_end_time, source_text, created_at
            FROM summaries
            WHERE session_id = $1 AND speaker_summary IS NOT NULL
              {recording_filter}
            ORDER BY speaker_id, created_at DESC
            """,
            *params,
        )

        return [_row_to_summary(r) for r in rows]


async def get_latest_session_summary(session_id: str) -> dict | None:
    """세션 요약 중 최신 1건을 조회합니다."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_summaries_schema(conn)
        row = await conn.fetchrow(
            """
            SELECT summary_id, session_id, recording_id, course_id, transcript_id, speaker_id,
                   speaker_summary, session_summary, course_summary,
                   source_start_time, source_end_time, source_text, created_at
            FROM summaries
            WHERE session_id = $1
              AND session_summary IS NOT NULL
              AND (speaker_id IS NULL OR speaker_id <> 'MATERIAL')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            _uuid.UUID(session_id),
        )

        if row is None:
            return None

        return _row_to_summary(row)


async def get_latest_speaker_summaries_by_course(course_id: str) -> list[dict]:
    """코스에서 최신 화자 요약을 speaker_id별로 조회합니다."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_summaries_schema(conn)
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (speaker_id)
                   summary_id, session_id, recording_id, course_id, transcript_id, speaker_id,
                   speaker_summary, session_summary, course_summary,
                   source_start_time, source_end_time, source_text, created_at
            FROM summaries
            WHERE course_id = $1 AND speaker_summary IS NOT NULL
            ORDER BY speaker_id, created_at DESC
            """,
            _uuid.UUID(course_id),
        )

        return [_row_to_summary(r) for r in rows]


async def get_latest_session_summaries_by_course(course_id: str) -> list[dict]:
    """코스에서 최신 세션 요약을 session_id별로 조회합니다."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_summaries_schema(conn)
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (session_id)
                   summary_id, session_id, recording_id, course_id, transcript_id, speaker_id,
                   speaker_summary, session_summary, course_summary,
                   source_start_time, source_end_time, source_text, created_at
            FROM summaries
            WHERE course_id = $1
              AND session_summary IS NOT NULL
              AND (speaker_id IS NULL OR speaker_id <> 'MATERIAL')
            ORDER BY session_id, created_at DESC
            """,
            _uuid.UUID(course_id),
        )

        return [_row_to_summary(r) for r in rows]


async def get_latest_course_summary(course_id: str) -> dict | None:
    """코스 요약 중 최신 1건을 조회합니다."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_summaries_schema(conn)
        row = await conn.fetchrow(
            """
            SELECT summary_id, session_id, recording_id, course_id, transcript_id, speaker_id,
                   speaker_summary, session_summary, course_summary,
                   source_start_time, source_end_time, source_text, created_at
            FROM summaries
            WHERE course_id = $1 AND course_summary IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1
            """,
            _uuid.UUID(course_id),
        )

        if row is None:
            return None

        return _row_to_summary(row)
