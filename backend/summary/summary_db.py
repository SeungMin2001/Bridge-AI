"""
요약 DB CRUD (summary_db.py)

대응 테이블:
  - SUMMARIES: 세션별 구조화된 요약 결과 저장
  - KEY_SENTENCES: TextRank 추출 핵심 문장 저장
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid as _uuid
from datetime import datetime
import json
from db import get_pool


# ══════════════════════════════════════
#  SUMMARIES CRUD
# ══════════════════════════════════════
async def save_summary(
    summary_id: str,
    session_id: str,
    summary_text: str,
    course_id: str | None = None,
    source_start_time: float | None = None,
    source_end_time: float | None = None,
    source_text: str | None = None,
) -> dict:
    """요약 결과를 SUMMARIES 테이블에 저장"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO summaries
                (summary_id, session_id, course_id, transcript_id,
                 summary_text, source_start_time, source_end_time,
                 source_text, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
            _uuid.UUID(summary_id),
            _uuid.UUID(session_id),
            _uuid.UUID(course_id) if course_id else None,
            None,  # transcript_id — 세션 전체 요약이므로 NULL
            summary_text,
            source_start_time,
            source_end_time,
            source_text,
            datetime.now(),
        )

    return {"summary_id": summary_id}


async def get_summary(summary_id: str) -> dict | None:
    """summary_id로 요약 단건 조회"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT summary_id, session_id, course_id, transcript_id,
                   summary_text, source_start_time, source_end_time,
                   source_text, created_at
            FROM summaries
            WHERE summary_id = $1
        """, _uuid.UUID(summary_id))

        if row is None:
            return None

        return {
            "summary_id": str(row["summary_id"]),
            "session_id": str(row["session_id"]) if row["session_id"] else None,
            "course_id": str(row["course_id"]) if row["course_id"] else None,
            "transcript_id": str(row["transcript_id"]) if row["transcript_id"] else None,
            "summary_text": row["summary_text"],
            "source_start_time": row["source_start_time"],
            "source_end_time": row["source_end_time"],
            "source_text": row["source_text"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }


async def get_summaries_by_session(session_id: str) -> list[dict]:
    """session_id에 해당하는 요약 목록 조회"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT summary_id, session_id, course_id,
                   summary_text, created_at
            FROM summaries
            WHERE session_id = $1
            ORDER BY created_at DESC
        """, _uuid.UUID(session_id))

        return [
            {
                "summary_id": str(r["summary_id"]),
                "session_id": str(r["session_id"]) if r["session_id"] else None,
                "course_id": str(r["course_id"]) if r["course_id"] else None,
                "summary_text": r["summary_text"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]


# ══════════════════════════════════════
#  KEY_SENTENCES CRUD
# ══════════════════════════════════════
async def save_key_sentences(session_id: str, sentences: list[dict]) -> int:
    """
    TextRank 추출 핵심 문장들을 KEY_SENTENCES 테이블에 배치 저장

    Args:
        session_id: 세션 ID (transcript_id 대신 사용, 전체 세션 대상이므로)
        sentences: [{"text": "문장", "score": 0.85, "rank": 1}, ...]

    Returns:
        저장된 문장 수
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        for s in sentences:
            await conn.execute("""
                INSERT INTO key_sentences
                    (key_id, transcript_id, sentence_text, score, rank_order, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """,
                _uuid.uuid4(),
                None,  # transcript_id — 세션 전체 대상이므로 NULL 허용
                s["text"],
                s.get("score"),
                s.get("rank"),
                datetime.now(),
            )

    return len(sentences)


async def get_key_sentences_by_session(session_id: str) -> list[dict]:
    """
    세션에 속한 전사문의 핵심 문장 조회.
    현재는 transcript_id 기반이 아니라 전체 조회 후 반환.
    향후 session_id FK가 KEY_SENTENCES에 추가되면 직접 필터 가능.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT key_id, transcript_id, sentence_text, score, rank_order, created_at
            FROM key_sentences
            ORDER BY rank_order ASC
        """)

        return [
            {
                "key_id": str(r["key_id"]),
                "transcript_id": str(r["transcript_id"]) if r["transcript_id"] else None,
                "sentence_text": r["sentence_text"],
                "score": r["score"],
                "rank_order": r["rank_order"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]
