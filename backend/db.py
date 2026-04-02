import asyncpg
import os

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "lecto"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
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
    pool = await get_pool()
    import uuid
    from datetime import datetime
    async with pool.acquire() as conn:
        # transcripts 저장
        await conn.execute("""
            INSERT INTO transcripts
                (transcript_id, session_id, segment_index, start_time, end_time, original_text, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
            uuid.uuid4(),
            uuid.UUID(transcript_data["session_id"]),
            segment_index,
            float(transcript_data["start_time"]),
            float(transcript_data["end_time"]),
            transcript_data["text"],
            datetime.now(),
        )

        # chunks 저장 (embedding은 RAG 단계에서 별도 처리)
        await conn.execute("""
            INSERT INTO chunks
                (chunk_id, session_id, chunk_index, start_time, end_time, chunk_text, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
            uuid.uuid4(),
            uuid.UUID(transcript_data["session_id"]),
            segment_index,
            float(transcript_data["start_time"]),
            float(transcript_data["end_time"]),
            transcript_data["text"],
            datetime.now(),
        )
