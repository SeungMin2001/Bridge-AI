#!/usr/bin/env python3
"""Create a fresh BridgePRAG demo DB schema and seed lecture samples.

This script is intended for the local Windows PostgreSQL runtime used by the
service demo. It creates a separate database (`bridgeprag_demo` by default), so
the existing `rag` database is not modified.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from db_config import DEFAULT_DB_NAME  # noqa: E402


def admin_config(database: str = "postgres") -> dict:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": database,
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "1234"),
    }


def create_database(db_name: str, *, reset: bool = False) -> None:
    import psycopg2

    conn = psycopg2.connect(**admin_config("postgres"))
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            if reset:
                cur.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s AND pid <> pg_backend_pid()
                    """,
                    (db_name,),
                )
                cur.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        conn.close()


def ensure_schema(db_name: str) -> None:
    import psycopg2

    with psycopg2.connect(**admin_config(db_name)) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS courses (
                    course_id UUID PRIMARY KEY,
                    user_id UUID NULL,
                    parent_course_id UUID NULL,
                    title TEXT NOT NULL,
                    type TEXT NULL,
                    description TEXT NULL,
                    color TEXT NULL,
                    icon TEXT NULL,
                    created_at TIMESTAMP NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id UUID PRIMARY KEY,
                    course_id UUID NULL REFERENCES courses(course_id) ON DELETE SET NULL,
                    session_date DATE NULL,
                    title TEXT NULL,
                    duration_sec INTEGER NULL DEFAULT 0,
                    status TEXT NULL,
                    created_at TIMESTAMP NULL DEFAULT NOW(),
                    file_kind VARCHAR(50) NULL,
                    tag VARCHAR(50) NULL,
                    icon VARCHAR(50) NULL,
                    color VARCHAR(50) NULL,
                    session_pdf JSONB NULL DEFAULT '[]'::jsonb,
                    session_voicefile JSONB NULL DEFAULT '[]'::jsonb,
                    summary_notes JSONB NULL DEFAULT '[]'::jsonb,
                    audio_path TEXT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS transcripts (
                    transcript_id UUID PRIMARY KEY,
                    session_id UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    recording_id TEXT NULL,
                    chunk_index INTEGER NULL,
                    start_time DOUBLE PRECISION NULL,
                    end_time DOUBLE PRECISION NULL,
                    original_text TEXT NULL,
                    chunk_text TEXT NULL,
                    corrected_text TEXT NULL,
                    confidence DOUBLE PRECISION NULL,
                    speaker_id TEXT NULL,
                    speaker_name TEXT NULL,
                    created_at TIMESTAMP NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    summary_id UUID PRIMARY KEY,
                    session_id UUID NULL,
                    recording_id TEXT NULL,
                    course_id UUID NULL,
                    transcript_id UUID NULL,
                    speaker_id TEXT NULL,
                    summary_type TEXT NULL,
                    summary_text TEXT NULL,
                    metadata JSONB NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS keywords (
                    keyword_id UUID PRIMARY KEY,
                    session_id UUID NULL,
                    recording_id TEXT NULL,
                    course_id UUID NULL,
                    transcript_id UUID NULL,
                    keyword TEXT NOT NULL,
                    score DOUBLE PRECISION NULL,
                    metadata JSONB NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schedules (
                    schedule_id UUID PRIMARY KEY,
                    session_id UUID NULL,
                    recording_id TEXT NULL,
                    title TEXT NULL,
                    description TEXT NULL,
                    event_type TEXT NULL,
                    due_date TIMESTAMP NULL,
                    source_text TEXT NULL,
                    calendar_flag TEXT NULL,
                    metadata JSONB NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_course_id ON sessions(course_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_session_id ON transcripts(session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_recording_id ON transcripts(recording_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_chunk_index ON transcripts(session_id, chunk_index)")
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a fresh BridgePRAG demo database.")
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate only the demo database named by --db-name.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Seed PostgreSQL rows but skip BGE/PGVector indexing.",
    )
    args = parser.parse_args()

    create_database(args.db_name, reset=args.reset)
    ensure_schema(args.db_name)

    from scripts.seed_bridgeprag_lecture_samples import main as seed_samples_main

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "seed_bridgeprag_lecture_samples.py",
            *(["--skip-index"] if args.skip_index else []),
        ]
        seed_samples_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
