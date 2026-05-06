import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid as _uuid
from datetime import datetime

from db import get_pool
from summary.schema import ensure_summary_schema


async def delete_keywords_by_transcript_ids(transcript_ids: list[str]) -> int:
    """특정 전사문에 연결된 키워드를 삭제합니다."""
    if not transcript_ids:
        return 0
    uuid_list = [_uuid.UUID(tid) for tid in transcript_ids]
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_summary_schema(conn)
        result = await conn.execute(
            """
            DELETE FROM key_sentences
            WHERE transcript_id = ANY($1::uuid[])
            """,
            uuid_list,
        )
    try:
        return int(result.split()[-1])
    except (AttributeError, ValueError, IndexError):
        return 0


async def save_keywords(keywords: list[dict]) -> int:
    """키워드 목록을 KEY_SENTENCES 테이블에 저장합니다."""
    if not keywords:
        return 0
    now = datetime.now()
    records = []
    for kw in keywords:
        records.append((
            _uuid.uuid4(),
            _uuid.UUID(kw["transcript_id"]) if kw.get("transcript_id") else None,
            kw["keyword_text"],
            kw.get("score"),
            kw.get("rank_order"),
            now,
        ))

    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_summary_schema(conn)
        await conn.executemany(
            """
            INSERT INTO key_sentences
                (key_id, transcript_id, sentence_text, score, rank_order, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            records,
        )

    return len(records)


async def get_keywords_by_session(session_id: str, limit: int | None = None, recording_id: str | None = None) -> list[dict]:
    """세션 기준 키워드 목록을 조회합니다."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_summary_schema(conn)
        params = [_uuid.UUID(session_id)]
        recording_filter = ""
        if recording_id:
            recording_filter = f" AND t.recording_id = ${len(params) + 1}"
            params.append(recording_id)
        base_query = """
            SELECT ks.key_id, ks.transcript_id, ks.sentence_text,
                   ks.score, ks.rank_order, ks.created_at
            FROM key_sentences ks
            JOIN transcripts t ON ks.transcript_id = t.transcript_id
            WHERE t.session_id = $1
        """ + recording_filter + """
            ORDER BY ks.rank_order ASC
        """
        if limit:
            params.append(limit)
            rows = await conn.fetch(base_query + f" LIMIT ${len(params)}", *params)
        else:
            rows = await conn.fetch(base_query, *params)

        return [
            {
                "key_id": str(r["key_id"]),
                "transcript_id": str(r["transcript_id"]) if r["transcript_id"] else None,
                "keyword_text": r["sentence_text"],
                "score": r["score"],
                "rank_order": r["rank_order"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]


async def get_keywords_by_course(course_id: str, limit: int | None = None) -> list[dict]:
    """코스 기준 키워드 목록을 조회합니다."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await ensure_summary_schema(conn)
        base_query = """
            SELECT ks.key_id, ks.transcript_id, ks.sentence_text,
                   ks.score, ks.rank_order, ks.created_at
            FROM key_sentences ks
            JOIN transcripts t ON ks.transcript_id = t.transcript_id
            JOIN sessions s ON t.session_id = s.session_id
            WHERE s.course_id = $1
            ORDER BY ks.rank_order ASC
        """
        if limit:
            rows = await conn.fetch(base_query + " LIMIT $2", _uuid.UUID(course_id), limit)
        else:
            rows = await conn.fetch(base_query, _uuid.UUID(course_id))

        return [
            {
                "key_id": str(r["key_id"]),
                "transcript_id": str(r["transcript_id"]) if r["transcript_id"] else None,
                "keyword_text": r["sentence_text"],
                "score": r["score"],
                "rank_order": r["rank_order"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]
