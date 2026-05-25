#!/usr/bin/env python3
"""Audit whether chat RAG evidence is backed by current DB transcript rows."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import psycopg2  # noqa: E402

from db_config import psycopg2_config  # noqa: E402
from rag_search import search  # noqa: E402


def _rag_table_name() -> str:
    raw_name = os.getenv("RAG_TABLE_NAME", "rag")
    safe_name = re.sub(r"[^A-Za-z0-9_]", "", raw_name) or "rag"
    return f"data_{safe_name}"


def _metadata_column(cur, table_name: str) -> str | None:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
          AND column_name IN ('metadata_', 'metadata')
        ORDER BY CASE WHEN column_name = 'metadata_' THEN 0 ELSE 1 END
        LIMIT 1
        """,
        (table_name,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s)", (table_name,))
    return cur.fetchone()[0] is not None


def _fetch_db_text(cur, transcript_id: str) -> str:
    if not transcript_id:
        return ""
    cur.execute(
        """
        SELECT COALESCE(corrected_text, chunk_text, '')
        FROM transcripts
        WHERE transcript_id::text = %s
        """,
        (transcript_id,),
    )
    row = cur.fetchone()
    return str(row[0] or "") if row else ""


def _print_db_summary(cur) -> None:
    cur.execute("SELECT COUNT(*) FROM transcripts")
    transcript_count = cur.fetchone()[0]
    print(f"[audit] transcripts rows={transcript_count}")

    table_name = _rag_table_name()
    if not _table_exists(cur, table_name):
        print(f"[audit] vector table missing: {table_name}")
        return

    metadata_column = _metadata_column(cur, table_name)
    cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
    vector_count = cur.fetchone()[0]
    print(f"[audit] vector rows={vector_count} table={table_name}")

    if not metadata_column:
        print("[audit] vector metadata column missing")
        return

    cur.execute(
        f"""
        SELECT COUNT(*)
        FROM "{table_name}" v
        LEFT JOIN transcripts t
          ON t.transcript_id::text = v."{metadata_column}"->>'transcript_id'
        WHERE COALESCE(v."{metadata_column}"->>'transcript_id', '') <> ''
          AND t.transcript_id IS NULL
        """
    )
    stale_count = cur.fetchone()[0]
    print(f"[audit] stale vector rows without transcript DB row={stale_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RAG evidence/citation consistency.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--session-id")
    parser.add_argument("--recording-id")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    source_filter = None
    if args.recording_id:
        source_filter = {"recording_ids": [args.recording_id]}

    with psycopg2.connect(**psycopg2_config()) as conn:
        with conn.cursor() as cur:
            _print_db_summary(cur)

            result = search(
                args.question,
                top_k=args.top_k,
                session_id=args.session_id,
                source_filter=source_filter,
            )

            print(f"[audit] context_chars={len(result.get('context', ''))}")
            for index, cite in enumerate(result.get("citations", []), 1):
                transcript_id = cite.get("transcript_id") or ""
                db_text = _fetch_db_text(cur, transcript_id)
                cite_text = str(cite.get("text") or "")
                exact_match = bool(db_text) and db_text.strip() == cite_text.strip()
                contains_match = bool(db_text) and cite_text.strip() in db_text
                print(
                    f"[audit] #{index} citation={cite.get('citation')} "
                    f"source={cite.get('source_type')} tid={transcript_id} "
                    f"db_exact={exact_match} db_contains={contains_match}"
                )
                print(f"        cite_text={cite_text[:180]}")
                if db_text and not exact_match:
                    print(f"        db_text={db_text[:180]}")


if __name__ == "__main__":
    main()
