"""
일정 DB CRUD

대응 테이블: SCHEDULES
  - 'pending'   : LLM이 추출한 직후, 사용자 확인 대기 상태
  - 'confirmed' : 사용자가 확정 → 달력에 표시됨
  - 'ignored'   : 사용자가 무시 → 달력에 표시 안 됨 (DB에는 남아 있음)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid as _uuid
from datetime import datetime
from db import get_pool

SAMPLE_SESSION_IDS = (
    _uuid.UUID("22222222-2222-2222-2222-222222222222"),
    _uuid.UUID("55555555-5555-5555-5555-555555555555"),
)


async def ensure_schedule_schema(conn) -> None:
    # 신창영 : 기존 DB를 유지한 채 녹음본 단위 연결 컬럼만 추가
    await conn.execute("ALTER TABLE schedules ADD COLUMN IF NOT EXISTS recording_id TEXT NULL")
    await conn.execute("ALTER TABLE schedules ADD COLUMN IF NOT EXISTS notion_page_id TEXT NULL")


async def save_schedule(
    schedule_id: str,
    session_id: str | None,
    title: str,
    recording_id: str | None = None,
    description: str | None = None,
    event_type: str | None = None,
    due_date: datetime | None = None,
    source_start_time: float | None = None,
    source_end_time: float | None = None,
    source_text: str | None = None,
    transcript_id: str | None = None,
    notion_page_id: str | None = None,
) -> dict:
    """추출된 일정을 SCHEDULES 테이블에 저장한다. 초기 status는 'pending'."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_schedule_schema(conn)
        await conn.execute("""
            INSERT INTO schedules
                (schedule_id, session_id, recording_id, transcript_id,
                 title, description, event_type, due_date,
                 status, calendar_flag,
                 source_start_time, source_end_time, source_text,
                 notion_page_id, created_at, updated_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
            """,
            _uuid.UUID(schedule_id),
            _uuid.UUID(session_id) if session_id else None,
            recording_id or None,
            _uuid.UUID(transcript_id) if transcript_id else None,
            title,
            description,
            event_type,
            due_date,
            "pending",        # 사용자 확인 대기
            False,            # 달력 미표시 (확정 전)
            source_start_time,
            source_end_time,
            source_text,
            notion_page_id,
            datetime.now(),
            datetime.now(),
        )
    return {"schedule_id": schedule_id, "status": "pending"}


async def get_all_schedules() -> list[dict]:
    """전체 일정을 반환한다. 프론트의 일정 관리 화면 초기 로딩용."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_schedule_schema(conn)
        rows = await conn.fetch("""
            SELECT s.schedule_id, s.session_id, s.recording_id, s.transcript_id,
                   s.title, s.description, s.event_type, s.due_date,
                   s.status, s.calendar_flag,
                   s.source_start_time, s.source_end_time, s.source_text,
                   s.notion_page_id, s.created_at, s.updated_at,
                   se.title AS session_title,
                   c.title AS course_title
            FROM schedules s
            LEFT JOIN sessions se ON s.session_id = se.session_id
            LEFT JOIN courses c ON se.course_id = c.course_id
            WHERE s.session_id IS NULL OR s.session_id <> ALL($1::uuid[])
            ORDER BY s.due_date ASC NULLS LAST, s.created_at DESC
        """, list(SAMPLE_SESSION_IDS))
        return [_row_to_dict(r) for r in rows]


async def get_schedule(schedule_id: str) -> dict | None:
    """schedule_id로 일정 단건 조회. 전사문 출처 정보도 함께 반환한다."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_schedule_schema(conn)
        row = await conn.fetchrow("""
            SELECT schedule_id, session_id, recording_id, transcript_id,
                   title, description, event_type, due_date,
                   status, calendar_flag,
                   source_start_time, source_end_time, source_text,
                   notion_page_id, created_at, updated_at
            FROM schedules
            WHERE schedule_id = $1
        """, _uuid.UUID(schedule_id))

        if row is None:
            return None

        return _row_to_dict(row)


async def get_schedules_by_session(session_id: str) -> list[dict]:
    """세션에서 추출된 모든 일정을 최신순으로 반환한다."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_schedule_schema(conn)
        rows = await conn.fetch("""
            SELECT schedule_id, session_id, recording_id, transcript_id,
                   title, description, event_type, due_date,
                   status, calendar_flag,
                   source_start_time, source_end_time, source_text,
                   notion_page_id, created_at, updated_at
            FROM schedules
            WHERE session_id = $1
            ORDER BY created_at DESC
        """, _uuid.UUID(session_id))
        return [_row_to_dict(r) for r in rows]


async def get_confirmed_schedules() -> list[dict]:
    """확정(confirmed)된 일정만 반환한다. 프론트의 달력 표시용."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_schedule_schema(conn)
        rows = await conn.fetch("""
            SELECT schedule_id, session_id, recording_id, transcript_id,
                   title, description, event_type, due_date,
                   status, calendar_flag,
                   source_start_time, source_end_time, source_text,
                   notion_page_id, created_at, updated_at
            FROM schedules
            WHERE status = 'confirmed' AND calendar_flag = true
              AND (session_id IS NULL OR session_id <> ALL($1::uuid[]))
            ORDER BY due_date ASC NULLS LAST
        """, list(SAMPLE_SESSION_IDS))
        return [_row_to_dict(r) for r in rows]


async def get_ignored_schedules_metadata() -> list[dict]:
    """시멘틱 중복 필터링용 ignored 일정의 제목과 날짜를 반환한다."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_schedule_schema(conn)
        rows = await conn.fetch("""
            SELECT title, due_date
            FROM schedules
            WHERE status = 'ignored'
        """)
        return [
            {
                "title": r["title"],
                "due_date": r["due_date"],
            }
            for r in rows
        ]


async def update_schedule_status(schedule_id: str, new_status: str) -> dict:
    """
    일정의 flag를 변경한다.
    - 'confirmed' → calendar_flag = true  (달력 표시)
    - 'ignored'   → calendar_flag = false (달력 미표시, DB에는 보존)
    """
    calendar_flag = (new_status == "confirmed")
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE schedules
            SET status = $1, calendar_flag = $2, updated_at = $3
            WHERE schedule_id = $4
        """,
            new_status,
            calendar_flag,
            datetime.now(),
            _uuid.UUID(schedule_id),
        )
    return {"schedule_id": schedule_id, "status": new_status, "calendar_flag": calendar_flag}


async def update_schedule_notion_id(schedule_id: str, notion_page_id: str | None) -> dict:
    """일정의 notion_page_id를 업데이트한다."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE schedules
            SET notion_page_id = $1, updated_at = $2
            WHERE schedule_id = $3
        """,
            notion_page_id,
            datetime.now(),
            _uuid.UUID(schedule_id),
        )
    return {"schedule_id": schedule_id, "notion_page_id": notion_page_id}


async def get_all_notion_page_ids() -> set[str]:
    """
    DB에 이미 저장된 모든 notion_page_id를 set으로 반환한다.
    노션 캘린더에서 일정을 가져올 때 중복을 필터링하는 데 사용한다.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_schedule_schema(conn)
        rows = await conn.fetch("""
            SELECT notion_page_id FROM schedules
            WHERE notion_page_id IS NOT NULL
        """)
        return {r["notion_page_id"] for r in rows}


async def find_ignored_titles(session_id: str | None = None) -> set[str]:
    """
    이전에 'ignored' 처리된 일정의 title 집합을 반환한다.
    같은 일정이 다시 추출되었을 때 알림을 보내지 않기 위해 사용한다.
    session_id가 주어지면 해당 세션 한정, 없으면 전체 범위.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        if session_id:
            rows = await conn.fetch("""
                SELECT DISTINCT title FROM schedules
                WHERE status = 'ignored' AND session_id = $1
            """, _uuid.UUID(session_id))
        else:
            rows = await conn.fetch("""
                SELECT DISTINCT title FROM schedules
                WHERE status = 'ignored'
            """)
        return {r["title"] for r in rows}


async def get_schedule_with_transcript(schedule_id: str) -> dict | None:
    """
    일정과 연결된 전사문 정보를 함께 반환한다.
    확정된 일정을 클릭했을 때, 원본 전사문의 위치를 찾아갈 수 있도록 한다.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_schedule_schema(conn)
        row = await conn.fetchrow("""
            SELECT s.schedule_id, s.session_id, s.recording_id, s.transcript_id,
                   s.title, s.description, s.event_type, s.due_date,
                   s.status, s.calendar_flag,
                   s.source_start_time, s.source_end_time, s.source_text,
                   s.notion_page_id, s.created_at, s.updated_at,
                   t.chunk_index, t.chunk_text, t.corrected_text
            FROM schedules s
            LEFT JOIN transcripts t ON s.transcript_id = t.transcript_id
            WHERE s.schedule_id = $1
        """, _uuid.UUID(schedule_id))

        if row is None:
            return None

        result = _row_to_dict(row)
        # 전사문 출처 정보 추가
        result["transcript_source"] = {
            "chunk_index": row["chunk_index"],
            "chunk_text": row["chunk_text"],
            "corrected_text": row["corrected_text"],
        } if row["chunk_index"] is not None else None
        return result


def _row_to_dict(row) -> dict:
    """DB Row를 딕셔너리로 변환하는 내부 헬퍼."""
    keys = set(row.keys())
    return {
        "schedule_id": str(row["schedule_id"]),
        "session_id": str(row["session_id"]) if row["session_id"] else None,
        "recording_id": row["recording_id"] if "recording_id" in keys else None,
        "transcript_id": str(row["transcript_id"]) if row["transcript_id"] else None,
        "title": row["title"],
        "description": row["description"],
        "event_type": row["event_type"],
        "due_date": row["due_date"].isoformat() if row["due_date"] else None,
        "status": row["status"],
        "calendar_flag": row["calendar_flag"],
        "source_start_time": row["source_start_time"],
        "source_end_time": row["source_end_time"],
        "source_text": row["source_text"],
        "notion_page_id": row["notion_page_id"] if "notion_page_id" in keys else None,
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        "session_title": row["session_title"] if "session_title" in keys else None,
        "course_title": row["course_title"] if "course_title" in keys else None,
    }
