"""Shared PostgreSQL connection settings for the local backend runtime."""

from __future__ import annotations

import os


DEFAULT_DB_NAME = "rag"


def db_config() -> dict:
    """Return one DB configuration used by both asyncpg and psycopg2 paths."""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("DB_NAME", DEFAULT_DB_NAME),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "1234"),
    }


def psycopg2_config() -> dict:
    """psycopg2 accepts the same keys, but keep a named helper for clarity."""
    return db_config()
