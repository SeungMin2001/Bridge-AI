from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _delete_count(status: str) -> int:
    try:
        return int(str(status).split()[-1])
    except (TypeError, ValueError):
        return 0


async def _table_exists(conn, table_name: str) -> bool:
    return bool(await conn.fetchval("SELECT to_regclass($1)::text", f"public.{table_name}"))


async def delete_session_related_rows(conn, session_ids: list) -> dict:
    """신창영 : 삭제된 파일의 전사/RAG/일정/요약 참조가 검색에 남지 않도록 관련 row를 함께 삭제"""
    session_ids = [session_id for session_id in session_ids if session_id]
    if not session_ids:
        return {
            "deletedTranscriptCount": 0,
            "deletedRagCount": 0,
            "deletedScheduleCount": 0,
            "deletedSummaryCount": 0,
        }

    session_id_texts = [str(session_id) for session_id in session_ids]
    transcript_rows = await conn.fetch(
        """
        SELECT transcript_id
        FROM transcripts
        WHERE session_id = ANY($1::uuid[])
        """,
        session_ids,
    )
    transcript_ids = [row["transcript_id"] for row in transcript_rows]

    deleted_schedule_count = 0
    deleted_summary_count = 0

    if transcript_ids:
        await conn.execute(
            """
            DELETE FROM key_sentences
            WHERE transcript_id = ANY($1::uuid[])
            """,
            transcript_ids,
        )
        await conn.execute(
            """
            DELETE FROM explanation_chunks
            WHERE transcript_id = ANY($1::uuid[])
            """,
            transcript_ids,
        )
        deleted_schedule_count += _delete_count(await conn.execute(
            """
            DELETE FROM schedules
            WHERE transcript_id = ANY($1::uuid[])
            """,
            transcript_ids,
        ))
        deleted_summary_count += _delete_count(await conn.execute(
            """
            DELETE FROM summaries
            WHERE transcript_id = ANY($1::uuid[])
            """,
            transcript_ids,
        ))

    deleted_schedule_count += _delete_count(await conn.execute(
        """
        DELETE FROM schedules
        WHERE session_id = ANY($1::uuid[])
        """,
        session_ids,
    ))
    deleted_summary_count += _delete_count(await conn.execute(
        """
        DELETE FROM summaries
        WHERE session_id = ANY($1::uuid[])
        """,
        session_ids,
    ))
    await conn.execute(
        """
        DELETE FROM quizzes
        WHERE session_id = ANY($1::uuid[])
        """,
        session_ids,
    )
    await conn.execute(
        """
        DELETE FROM concept_requests
        WHERE session_id = ANY($1::uuid[])
        """,
        session_ids,
    )

    deleted_rag_count = 0
    if await _table_exists(conn, "data_rag"):
        # 신창영 : 파일 삭제 후에도 data_rag 벡터가 남아 "실시간 녹음" 참조가 재노출되는 문제 방지
        deleted_rag_count = _delete_count(await conn.execute(
            """
            DELETE FROM data_rag
            WHERE metadata_->>'session_id' = ANY($1::text[])
            """,
            session_id_texts,
        ))

    deleted_transcript_count = _delete_count(await conn.execute(
        """
        DELETE FROM transcripts
        WHERE session_id = ANY($1::uuid[])
        """,
        session_ids,
    ))

    return {
        "deletedTranscriptCount": deleted_transcript_count,
        "deletedRagCount": deleted_rag_count,
        "deletedScheduleCount": deleted_schedule_count,
        "deletedSummaryCount": deleted_summary_count,
    }


async def delete_recording_related_rows(conn, session_id, recording_id: str) -> dict:
    """신창영 : 녹음본 하나를 삭제할 때 해당 recording_id의 전사/RAG/일정/요약만 정리"""
    if not session_id or not recording_id:
        return {
            "deletedTranscriptCount": 0,
            "deletedRagCount": 0,
            "deletedScheduleCount": 0,
            "deletedSummaryCount": 0,
        }

    await conn.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS recording_id TEXT NULL")
    await conn.execute("ALTER TABLE schedules ADD COLUMN IF NOT EXISTS recording_id TEXT NULL")
    await conn.execute("ALTER TABLE summaries ADD COLUMN IF NOT EXISTS recording_id TEXT NULL")

    transcript_rows = await conn.fetch(
        """
        SELECT transcript_id
        FROM transcripts
        WHERE session_id = $1 AND recording_id = $2
        """,
        session_id,
        recording_id,
    )
    transcript_ids = [row["transcript_id"] for row in transcript_rows]

    deleted_schedule_count = 0
    deleted_summary_count = 0
    if transcript_ids:
        await conn.execute(
            """
            DELETE FROM key_sentences
            WHERE transcript_id = ANY($1::uuid[])
            """,
            transcript_ids,
        )
        await conn.execute(
            """
            DELETE FROM explanation_chunks
            WHERE transcript_id = ANY($1::uuid[])
            """,
            transcript_ids,
        )
        deleted_schedule_count += _delete_count(await conn.execute(
            """
            DELETE FROM schedules
            WHERE transcript_id = ANY($1::uuid[])
            """,
            transcript_ids,
        ))
        deleted_summary_count += _delete_count(await conn.execute(
            """
            DELETE FROM summaries
            WHERE transcript_id = ANY($1::uuid[])
            """,
            transcript_ids,
        ))

    deleted_schedule_count += _delete_count(await conn.execute(
        """
        DELETE FROM schedules
        WHERE session_id = $1 AND recording_id = $2
        """,
        session_id,
        recording_id,
    ))
    deleted_summary_count += _delete_count(await conn.execute(
        """
        DELETE FROM summaries
        WHERE session_id = $1 AND recording_id = $2
        """,
        session_id,
        recording_id,
    ))

    deleted_rag_count = 0
    if await _table_exists(conn, "data_rag"):
        deleted_rag_count = _delete_count(await conn.execute(
            """
            DELETE FROM data_rag
            WHERE metadata_->>'session_id' = $1
              AND metadata_->>'recording_id' = $2
            """,
            str(session_id),
            recording_id,
        ))

    deleted_transcript_count = _delete_count(await conn.execute(
        """
        DELETE FROM transcripts
        WHERE session_id = $1 AND recording_id = $2
        """,
        session_id,
        recording_id,
    ))

    return {
        "deletedTranscriptCount": deleted_transcript_count,
        "deletedRagCount": deleted_rag_count,
        "deletedScheduleCount": deleted_schedule_count,
        "deletedSummaryCount": deleted_summary_count,
    }


def delete_transcript_json_files(session_ids: list) -> int:
    # 신창영 : DB row 삭제와 별개로 로컬 JSONL 전사 파일도 함께 삭제
    deleted_count = 0
    seen_paths = set()

    for session_id in {str(session_id) for session_id in session_ids if session_id}:
        for base_dir in (
            BACKEND_ROOT / "data" / "transcripts",
            Path.cwd() / "data" / "transcripts",
        ):
            target_path = base_dir / f"{session_id}.jsonl"
            if target_path in seen_paths:
                continue

            seen_paths.add(target_path)
            if target_path.exists() and target_path.is_file():
                target_path.unlink()
                deleted_count += 1

    return deleted_count
