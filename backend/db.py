import asyncpg
import os

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "rag"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "1234"),
}

_pool = None


async def get_pool():
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(**DB_CONFIG, max_inactive_connection_lifetime=300)
    return _pool


async def create_session(session_id: str, title: str = "강의 녹음"):
    pool = await get_pool()
    import uuid
    from datetime import datetime, date
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO sessions (session_id, course_id, session_date, title, status, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT DO NOTHING
        """,
            uuid.UUID(session_id),
            uuid.UUID("00000000-0000-0000-0000-000000000000"),  # 임시 course_id
            date.today(),
            title,
            "recording",
            datetime.now(),
        )


async def save_transcript_to_db(transcript_data: dict, segment_index: int):
    """전사 청크를 transcripts 테이블에 저장합니다."""
    pool = await get_pool()
    import uuid
    from datetime import datetime
    # 신창영 : save_transcript 단계에서 만든 transcript_id를 DB row에도 동일하게 사용
    transcript_id = uuid.UUID(transcript_data["transcript_id"]) if transcript_data.get("transcript_id") else uuid.uuid4()
    # 신창영 : RAG citation과 녹음본 저장 시각을 맞추기 위해 created_at을 외부에서 전달받을 수 있게 처리
    created_at = transcript_data.get("created_at") or datetime.now()
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    if getattr(created_at, "tzinfo", None) is not None:
        created_at = created_at.astimezone().replace(tzinfo=None)
    async with pool.acquire() as conn:
        # 신창영 : 녹음본 JSON의 id와 DB 전사 청크를 직접 연결하기 위한 recording_id 컬럼 보장
        await conn.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS recording_id TEXT NULL")
        # transcripts 저장
        await conn.execute("""
            INSERT INTO transcripts
                (transcript_id, session_id, recording_id, chunk_index, start_time, end_time, chunk_text, corrected_text, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
            transcript_id,
            uuid.UUID(transcript_data["session_id"]),
            transcript_data.get("recording_id") or None,
            segment_index,
            float(transcript_data["start_time"]),
            float(transcript_data["end_time"]),
            transcript_data.get("raw_text") or transcript_data.get("text"),
            transcript_data.get("text"),
            created_at,
        )
    return {
        "transcript_id": str(transcript_id),
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
    }


#  세션별 전사문 조회
async def get_transcripts_by_session(session_id: str, recording_id: str | None = None) -> list[dict]:
    """session_id에 해당하는 전사문을 시간순으로 조회"""
    import uuid as _uuid
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS recording_id TEXT NULL")
        if recording_id:
            rows = await conn.fetch("""
                SELECT transcript_id, recording_id, chunk_index, start_time, end_time,
                       chunk_text, corrected_text
                FROM transcripts
                WHERE session_id = $1 AND recording_id = $2
                ORDER BY chunk_index ASC
            """, _uuid.UUID(session_id), recording_id)
        else:
            rows = await conn.fetch("""
                SELECT transcript_id, recording_id, chunk_index, start_time, end_time,
                       chunk_text, corrected_text
                FROM transcripts
                WHERE session_id = $1
                ORDER BY chunk_index ASC
            """, _uuid.UUID(session_id))
        return [
            {
                "transcript_id": str(r["transcript_id"]),
                "recording_id": r["recording_id"],
                "chunk_index": r["chunk_index"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "chunk_text": r["chunk_text"],
                "corrected_text": r["corrected_text"],
                "text": r["corrected_text"] or r["chunk_text"],
            }
            for r in rows
        ]


async def get_course_id_by_session(session_id: str) -> str | None:
    """session_id로 course_id 조회"""
    import uuid as _uuid
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT course_id
            FROM sessions
            WHERE session_id = $1
        """, _uuid.UUID(session_id))
        if row is None or row["course_id"] is None:
            return None
        return str(row["course_id"])


async def get_session_title(session_id: str) -> str:
    """session_id로 세션 제목 조회"""
    import uuid as _uuid
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT title FROM sessions WHERE session_id = $1
        """, _uuid.UUID(session_id))
        return row["title"] if row and row["title"] else "알 수 없는 세션"
