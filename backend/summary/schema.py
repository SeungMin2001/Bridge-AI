async def ensure_summary_schema(conn) -> None:
    """요약 기능에 필요한 테이블과 컬럼을 보장합니다."""
    # 신창영 : 기존 DB를 초기화하지 않고 요약 기능에 필요한 스키마만 비파괴적으로 보강
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS summaries (
            summary_id UUID PRIMARY KEY,
            session_id UUID NULL,
            recording_id TEXT NULL,
            course_id UUID NULL,
            transcript_id UUID NULL,
            speak_id TEXT NULL,
            speaker_id TEXT NULL,
            speaker_summary TEXT NULL,
            session_summary TEXT NULL,
            course_summary TEXT NULL,
            source_start_time REAL NULL,
            source_end_time REAL NULL,
            source_text TEXT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS key_sentences (
            key_id UUID PRIMARY KEY,
            transcript_id UUID NULL,
            sentence_text TEXT NOT NULL,
            score REAL NULL,
            rank_order INT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );

        ALTER TABLE summaries ADD COLUMN IF NOT EXISTS summary_id UUID;
        ALTER TABLE summaries ADD COLUMN IF NOT EXISTS session_id UUID NULL;
        ALTER TABLE summaries ADD COLUMN IF NOT EXISTS recording_id TEXT NULL;
        ALTER TABLE summaries ADD COLUMN IF NOT EXISTS course_id UUID NULL;
        ALTER TABLE summaries ADD COLUMN IF NOT EXISTS transcript_id UUID NULL;
        ALTER TABLE summaries ADD COLUMN IF NOT EXISTS summary_text TEXT NULL;
        ALTER TABLE summaries ADD COLUMN IF NOT EXISTS speak_id TEXT NULL;
        ALTER TABLE summaries ADD COLUMN IF NOT EXISTS speaker_id TEXT NULL;
        ALTER TABLE summaries ADD COLUMN IF NOT EXISTS speaker_summary TEXT NULL;
        ALTER TABLE summaries ADD COLUMN IF NOT EXISTS session_summary TEXT NULL;
        ALTER TABLE summaries ADD COLUMN IF NOT EXISTS course_summary TEXT NULL;
        ALTER TABLE summaries ADD COLUMN IF NOT EXISTS source_start_time REAL NULL;
        ALTER TABLE summaries ADD COLUMN IF NOT EXISTS source_end_time REAL NULL;
        ALTER TABLE summaries ADD COLUMN IF NOT EXISTS source_text TEXT NULL;
        ALTER TABLE summaries ADD COLUMN IF NOT EXISTS created_at TIMESTAMP NOT NULL DEFAULT NOW();
        ALTER TABLE summaries ALTER COLUMN summary_text DROP NOT NULL;
        ALTER TABLE summaries ALTER COLUMN speak_id DROP NOT NULL;

        ALTER TABLE key_sentences ADD COLUMN IF NOT EXISTS key_id UUID;
        ALTER TABLE key_sentences ADD COLUMN IF NOT EXISTS transcript_id UUID NULL;
        ALTER TABLE key_sentences ADD COLUMN IF NOT EXISTS sentence_text TEXT;
        ALTER TABLE key_sentences ADD COLUMN IF NOT EXISTS score REAL NULL;
        ALTER TABLE key_sentences ADD COLUMN IF NOT EXISTS rank_order INT NULL;
        ALTER TABLE key_sentences ADD COLUMN IF NOT EXISTS created_at TIMESTAMP NOT NULL DEFAULT NOW();
        """
    )
