import asyncpg
from db_config import db_config

DB_CONFIG = db_config()

_pool = None


async def get_pool():
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(**DB_CONFIG, max_inactive_connection_lifetime=300)
    return _pool


async def ensure_transcripts_schema(conn) -> None:
    await conn.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS recording_id TEXT NULL")
    await conn.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS chunk_index INTEGER NULL")
    await conn.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS chunk_text TEXT NULL")
    await conn.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS speaker_id TEXT NULL")
    await conn.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS speaker_name TEXT NULL")
    await conn.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'transcripts' AND column_name = 'segment_index'
            ) THEN
                EXECUTE 'UPDATE transcripts SET chunk_index = COALESCE(chunk_index, segment_index) WHERE chunk_index IS NULL';
            END IF;

            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'transcripts' AND column_name = 'original_text'
            ) THEN
                EXECUTE 'UPDATE transcripts SET chunk_text = COALESCE(chunk_text, corrected_text, original_text) WHERE chunk_text IS NULL';
            END IF;
        END $$;
    """)
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_session_id ON transcripts(session_id)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_recording_id ON transcripts(recording_id)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_session_recording ON transcripts(session_id, recording_id)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_session_chunk ON transcripts(session_id, chunk_index)")
    await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_transcripts_text_trgm
        ON transcripts
        USING gin ((COALESCE(corrected_text, chunk_text, '')) gin_trgm_ops)
    """)


async def ensure_sessions_schema(conn) -> None:
    await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS file_kind VARCHAR(50) NULL")
    await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS tag VARCHAR(50) NULL")
    await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS icon VARCHAR(50) NULL")
    await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS color VARCHAR(50) NULL")
    await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS session_pdf JSONB NULL")
    await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS session_voicefile JSONB NULL")
    await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary_notes JSONB NULL")


async def ensure_courses_schema(conn) -> None:
    await conn.execute("ALTER TABLE courses ADD COLUMN IF NOT EXISTS parent_course_id UUID NULL")
    await conn.execute("ALTER TABLE courses ADD COLUMN IF NOT EXISTS color VARCHAR(50) NULL")
    await conn.execute("ALTER TABLE courses ADD COLUMN IF NOT EXISTS icon VARCHAR(50) NULL")


async def ensure_feature_tables_schema(conn) -> None:
    """Create/upgrade optional feature tables used by workspace, quiz, summary, and schedule routes."""
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS key_sentences (
            key_id UUID PRIMARY KEY,
            transcript_id UUID NULL,
            sentence_text TEXT NOT NULL,
            score REAL NULL,
            rank_order INTEGER NULL,
            created_at TIMESTAMP NOT NULL
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS quizzes (
            quiz_id UUID PRIMARY KEY,
            user_id UUID NULL,
            course_id UUID NULL,
            request_id UUID NULL,
            session_id UUID NULL,
            quiz_data JSONB NULL,
            total_questions INTEGER NULL,
            correct_count INTEGER NULL,
            created_at TIMESTAMP NULL
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS concept_requests (
            request_id UUID PRIMARY KEY,
            user_id UUID NULL,
            session_id UUID NULL,
            clicked_text VARCHAR(255) NULL,
            normalized_term VARCHAR(255) NULL,
            request_time REAL NULL,
            request_type VARCHAR(50) NULL,
            status VARCHAR(50) NULL,
            created_at TIMESTAMP NULL
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS explanations (
            explanation_id UUID PRIMARY KEY,
            request_id UUID NULL,
            answer_text TEXT NULL,
            source_links TEXT NULL,
            model_name VARCHAR(100) NULL,
            latency_ms INTEGER NULL,
            created_at TIMESTAMP NULL
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS explanation_chunks (
            explanation_chunk_id UUID PRIMARY KEY,
            explanation_id UUID NULL,
            transcript_id UUID NULL,
            similarity_score REAL NULL,
            rank_order INTEGER NULL,
            quoted_text TEXT NULL
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS schedules (
            schedule_id UUID PRIMARY KEY,
            session_id UUID NULL,
            recording_id TEXT NULL,
            transcript_id UUID NULL,
            title TEXT NULL,
            description TEXT NULL,
            event_type TEXT NULL,
            due_date TIMESTAMP NULL,
            status VARCHAR(30) NULL,
            calendar_flag BOOLEAN NULL,
            source_start_time REAL NULL,
            source_end_time REAL NULL,
            source_text TEXT NULL,
            notion_page_id TEXT NULL,
            created_at TIMESTAMP NULL,
            updated_at TIMESTAMP NULL
        )
    """)
    await conn.execute("ALTER TABLE schedules ADD COLUMN IF NOT EXISTS calendar_flag BOOLEAN NULL")
    await conn.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'schedules'
                  AND column_name = 'calendar_flag'
                  AND data_type <> 'boolean'
            ) THEN
                ALTER TABLE schedules
                ALTER COLUMN calendar_flag TYPE BOOLEAN
                USING CASE
                    WHEN lower(COALESCE(calendar_flag::text, '')) IN ('true', 't', '1', 'yes', 'y') THEN TRUE
                    ELSE FALSE
                END;
            END IF;
        END $$;
    """)
    await conn.execute("ALTER TABLE schedules ADD COLUMN IF NOT EXISTS recording_id TEXT NULL")
    await conn.execute("ALTER TABLE schedules ADD COLUMN IF NOT EXISTS transcript_id UUID NULL")
    await conn.execute("ALTER TABLE schedules ADD COLUMN IF NOT EXISTS status VARCHAR(30) NULL")
    await conn.execute("ALTER TABLE schedules ADD COLUMN IF NOT EXISTS source_start_time REAL NULL")
    await conn.execute("ALTER TABLE schedules ADD COLUMN IF NOT EXISTS source_end_time REAL NULL")
    await conn.execute("ALTER TABLE schedules ADD COLUMN IF NOT EXISTS source_text TEXT NULL")
    await conn.execute("ALTER TABLE schedules ADD COLUMN IF NOT EXISTS notion_page_id TEXT NULL")
    await conn.execute("ALTER TABLE schedules ADD COLUMN IF NOT EXISTS created_at TIMESTAMP NULL")
    await conn.execute("ALTER TABLE schedules ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP NULL")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            summary_id UUID PRIMARY KEY,
            session_id UUID NULL,
            recording_id TEXT NULL,
            course_id UUID NULL,
            transcript_id UUID NULL,
            speaker_id TEXT NULL,
            speaker_summary TEXT NULL,
            session_summary TEXT NULL,
            course_summary TEXT NULL,
            source_start_time REAL NULL,
            source_end_time REAL NULL,
            source_text TEXT NULL,
            created_at TIMESTAMP NULL
        )
    """)
    await conn.execute("ALTER TABLE summaries ADD COLUMN IF NOT EXISTS recording_id TEXT NULL")
    await conn.execute("ALTER TABLE summaries ADD COLUMN IF NOT EXISTS course_id UUID NULL")
    await conn.execute("ALTER TABLE summaries ADD COLUMN IF NOT EXISTS transcript_id UUID NULL")
    await conn.execute("ALTER TABLE summaries ADD COLUMN IF NOT EXISTS speaker_id TEXT NULL")
    await conn.execute("ALTER TABLE summaries ADD COLUMN IF NOT EXISTS speaker_summary TEXT NULL")
    await conn.execute("ALTER TABLE summaries ADD COLUMN IF NOT EXISTS session_summary TEXT NULL")
    await conn.execute("ALTER TABLE summaries ADD COLUMN IF NOT EXISTS course_summary TEXT NULL")
    await conn.execute("ALTER TABLE summaries ADD COLUMN IF NOT EXISTS source_start_time REAL NULL")
    await conn.execute("ALTER TABLE summaries ADD COLUMN IF NOT EXISTS source_end_time REAL NULL")
    await conn.execute("ALTER TABLE summaries ADD COLUMN IF NOT EXISTS source_text TEXT NULL")
    await conn.execute("ALTER TABLE summaries ADD COLUMN IF NOT EXISTS created_at TIMESTAMP NULL")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS course_memories (
            memory_id UUID PRIMARY KEY,
            course_id UUID UNIQUE,
            merged_k BYTEA,
            merged_v BYTEA,
            passage_count INTEGER DEFAULT 0,
            updated_at TIMESTAMP NOT NULL
        )
    """)


async def ensure_runtime_schema() -> None:
    """Apply lightweight local schema upgrades needed by current develop code."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_courses_schema(conn)
        await ensure_sessions_schema(conn)
        await ensure_transcripts_schema(conn)
        await ensure_feature_tables_schema(conn)


async def create_session(session_id: str, title: str = "강의 녹음"):
    pool = await get_pool()
    import uuid
    from datetime import datetime, date
    from db_api.workspace.default_folder import ensure_default_folder
    async with pool.acquire() as conn:
        default_course_id = await ensure_default_folder(conn)
        await conn.execute("""
            INSERT INTO sessions (session_id, course_id, session_date, title, status, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT DO NOTHING
        """,
            uuid.UUID(session_id),
            default_course_id,
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
        await ensure_transcripts_schema(conn)
        # transcripts 저장
        await conn.execute("""
            INSERT INTO transcripts
                (transcript_id, session_id, recording_id, chunk_index, start_time, end_time,
                 speaker_id, speaker_name, chunk_text, corrected_text, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """,
            transcript_id,
            uuid.UUID(transcript_data["session_id"]),
            transcript_data.get("recording_id"),
            segment_index,
            float(transcript_data["start_time"]),
            float(transcript_data["end_time"]),
            transcript_data.get("speaker_id"),
            transcript_data.get("speaker_name"),
            transcript_data.get("raw_text") or transcript_data.get("text"),
            transcript_data.get("text"),
            created_at,
        )
    return {
        "transcript_id": str(transcript_id),
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
    }


async def update_transcript_speakers(session_id: str, recording_id: str, speaker_updates: list[dict]) -> int:
    """전체 화자분리 결과로 기존 전사 청크의 speaker_id만 갱신합니다."""
    import uuid as _uuid
    if not speaker_updates:
        return 0

    pool = await get_pool()
    updated_count = 0
    async with pool.acquire() as conn:
        await ensure_transcripts_schema(conn)
        async with conn.transaction():
            for item in speaker_updates:
                speaker_id = item.get("speaker_id")
                if not speaker_id:
                    continue

                transcript_id = item.get("transcript_id") or item.get("transcriptId")
                if transcript_id:
                    result = await conn.execute("""
                        UPDATE transcripts
                        SET speaker_id = $1
                        WHERE session_id = $2
                          AND recording_id = $3
                          AND transcript_id = $4
                    """,
                        speaker_id,
                        _uuid.UUID(session_id),
                        recording_id,
                        _uuid.UUID(str(transcript_id)),
                    )
                else:
                    result = await conn.execute("""
                        UPDATE transcripts
                        SET speaker_id = $1
                        WHERE session_id = $2
                          AND recording_id = $3
                          AND start_time = $4
                          AND end_time = $5
                    """,
                        speaker_id,
                        _uuid.UUID(session_id),
                        recording_id,
                        float(item["start_time"]),
                        float(item["end_time"]),
                    )

                updated_count += int(result.rsplit(" ", 1)[-1])

    return updated_count


#  세션별 전사문 조회
async def get_transcripts_by_session(session_id: str, recording_id: str | None = None) -> list[dict]:
    """session_id에 해당하는 전사문을 시간순으로 조회"""
    import uuid as _uuid
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_transcripts_schema(conn)
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


async def get_transcripts_by_ids(session_id: str, transcript_ids: list[str]) -> list[dict]:
    """session_id에 속한 특정 transcript_id 전사문을 시간순으로 조회"""
    import uuid as _uuid
    if not transcript_ids:
        return []

    parsed_ids = [_uuid.UUID(item) for item in transcript_ids]
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_transcripts_schema(conn)
        rows = await conn.fetch("""
            SELECT transcript_id, recording_id, chunk_index, start_time, end_time,
                   chunk_text, corrected_text
            FROM transcripts
            WHERE session_id = $1 AND transcript_id = ANY($2::uuid[])
            ORDER BY chunk_index ASC
        """, _uuid.UUID(session_id), parsed_ids)
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
