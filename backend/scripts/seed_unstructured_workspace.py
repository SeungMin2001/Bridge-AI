#!/usr/bin/env python3
"""Seed data/비정형데이터.md as a workspace recording transcript."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import psycopg2
from psycopg2.extras import Json, execute_batch, register_uuid


register_uuid()


FOLDER_TITLE = "비정형데이터"
SESSION_TITLE = "비정형데이터강의1"
RECORDING_TITLE = "비정형데이터강의1 녹음본"
RECORDING_ID = "recording-unstructured-data-lecture-1"
WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def deterministic_uuid(name: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"lectoai:{name}")


def connect():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "shin"),
        user=os.getenv("DB_USER") or os.getenv("USER") or "changyoung",
        password=os.getenv("DB_PASSWORD", ""),
    )


def clean_source_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\ufeff", " ")).strip()


def hard_wrap(sentence: str, max_chars: int) -> list[str]:
    words = sentence.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, max_chars: int = 760) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+", clean_source_text(text))
    chunks: list[str] = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        candidates = hard_wrap(part, max_chars) if len(part) > max_chars else [part]
        for candidate in candidates:
            next_text = f"{current} {candidate}".strip()
            if current and len(next_text) > max_chars:
                chunks.append(current)
                current = candidate
            else:
                current = next_text

    if current:
        chunks.append(current)

    return chunks


def format_duration(seconds: float) -> str:
    safe_seconds = max(0, int(round(seconds)))
    hours = safe_seconds // 3600
    minutes = (safe_seconds % 3600) // 60
    remain_seconds = safe_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remain_seconds:02d}"


def format_elapsed_time(seconds: float) -> str:
    safe_seconds = max(0, int(seconds))
    hours = safe_seconds // 3600
    minutes = (safe_seconds % 3600) // 60
    remain_seconds = safe_seconds % 60
    if hours:
        return f"{hours}:{minutes:02d}:{remain_seconds:02d}"
    return f"{minutes}:{remain_seconds:02d}"


def format_week_label(value: date) -> str:
    return f"{value.year}. {value.month}. {value.day}. {WEEKDAYS[value.weekday()]}"


def json_value(value, fallback):
    if value in (None, ""):
        return fallback
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def get_or_create_course(cur) -> uuid.UUID:
    cur.execute(
        """
        SELECT course_id
        FROM courses
        WHERE title = %s
          AND parent_course_id IS NULL
        ORDER BY created_at ASC NULLS LAST
        LIMIT 1
        """,
        (FOLDER_TITLE,),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    course_id = deterministic_uuid(FOLDER_TITLE)
    cur.execute(
        """
        INSERT INTO courses (
            course_id, user_id, parent_course_id, title, type, description, color, icon, created_at
        )
        VALUES (%s, NULL, NULL, %s, 'folder', NULL, '#3b82f6', 'folder', %s)
        ON CONFLICT (course_id) DO UPDATE
        SET title = EXCLUDED.title,
            type = EXCLUDED.type,
            icon = EXCLUDED.icon
        """,
        (course_id, FOLDER_TITLE, datetime.now()),
    )
    return course_id


def get_or_create_session(cur, course_id: uuid.UUID) -> uuid.UUID:
    cur.execute(
        """
        SELECT session_id
        FROM sessions
        WHERE course_id = %s
          AND title = %s
        ORDER BY created_at ASC NULLS LAST
        LIMIT 1
        """,
        (course_id, SESSION_TITLE),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    session_id = deterministic_uuid(f"{FOLDER_TITLE}/{SESSION_TITLE}")
    cur.execute(
        """
        INSERT INTO sessions (
            session_id, course_id, session_date, title, duration_sec, status, created_at,
            file_kind, tag, icon, color, session_pdf, session_voicefile, summary_notes
        )
        VALUES (%s, %s, %s, %s, 0, 'created', %s, 'lecture', '수업', 'article', '#3b82f6',
                '[]'::jsonb, '[]'::jsonb, '[]'::jsonb)
        ON CONFLICT (session_id) DO UPDATE
        SET course_id = EXCLUDED.course_id,
            title = EXCLUDED.title,
            file_kind = EXCLUDED.file_kind,
            tag = EXCLUDED.tag,
            icon = EXCLUDED.icon
        """,
        (session_id, course_id, date.today(), SESSION_TITLE, datetime.now()),
    )
    return session_id


def build_transcript_records(session_id: uuid.UUID, started_at: datetime, chunks: list[str]) -> tuple[list[dict], list[tuple], int]:
    transcriptions: list[dict] = []
    db_rows: list[tuple] = []
    cursor_seconds = 0.0

    for index, text in enumerate(chunks):
        duration = max(6.0, min(55.0, len(text) / 13.0))
        start_time = round(cursor_seconds, 2)
        end_time = round(cursor_seconds + duration, 2)
        cursor_seconds = end_time
        transcript_id = deterministic_uuid(f"{session_id}:{RECORDING_ID}:{index}")
        created_at = started_at + timedelta(seconds=start_time)

        transcriptions.append({
            "id": f"{RECORDING_ID}-{index}",
            "recordingId": RECORDING_ID,
            "time": format_elapsed_time(start_time),
            "speakerId": None,
            "speaker": None,
            "text": text,
            "segments": [{
                "id": str(transcript_id),
                "text": text,
                "status": "confirmed",
                "transcript_id": str(transcript_id),
                "transcriptId": str(transcript_id),
                "chunk_index": index,
                "start": start_time,
                "end": end_time,
                "start_time": start_time,
                "end_time": end_time,
            }],
        })

        db_rows.append((
            transcript_id,
            session_id,
            RECORDING_ID,
            index,
            start_time,
            end_time,
            None,
            None,
            text,
            text,
            created_at,
        ))

    return transcriptions, db_rows, int(round(cursor_seconds))


def remove_recording_from_weeks(weeks: list) -> list:
    cleaned = []
    for week in weeks:
        if not isinstance(week, dict):
            continue
        next_week = dict(week)
        recordings = next_week.get("recordings")
        if isinstance(recordings, list):
            next_week["recordings"] = [
                item for item in recordings
                if item.get("id") != RECORDING_ID and item.get("recordingId") != RECORDING_ID
            ]
        cleaned.append(next_week)
    return cleaned


def upsert_recording_resource(cur, session_id: uuid.UUID, transcriptions: list[dict], duration_seconds: int) -> None:
    now = datetime.now().replace(microsecond=0)
    ended_at = now + timedelta(seconds=duration_seconds)
    today = date.today()
    week_key = today.isoformat()
    week_id = f"week-{week_key}"

    recording = {
        "id": RECORDING_ID,
        "recordingId": RECORDING_ID,
        "title": RECORDING_TITLE,
        "startedAt": now.isoformat(),
        "endedAt": ended_at.isoformat(),
        "durationText": format_duration(duration_seconds),
        "recordingMode": "lecture",
        "diarizationEnabled": False,
        "materialIds": [],
        "materialNames": [],
        "audioUrl": None,
        "transcriptions": transcriptions,
    }

    cur.execute("SELECT session_voicefile FROM sessions WHERE session_id = %s", (session_id,))
    row = cur.fetchone()
    existing_weeks = remove_recording_from_weeks(json_value(row[0] if row else None, []))

    target_week = None
    for week in existing_weeks:
        if week.get("weekKey") == week_key or week.get("id") == week_id:
            target_week = week
            break

    if target_week is None:
        target_week = {
            "weekId": week_id,
            "id": week_id,
            "weekKey": week_key,
            "label": "1주차",
            "dateLabel": format_week_label(today),
            "expanded": True,
            "materialFolderExpanded": True,
            "recordingFolderExpanded": True,
            "recordings": [],
        }
        existing_weeks.insert(0, target_week)

    target_week["recordings"] = [recording, *(target_week.get("recordings") or [])]

    cur.execute(
        """
        UPDATE sessions
        SET session_voicefile = %s::jsonb,
            duration_sec = %s,
            status = 'created'
        WHERE session_id = %s
        """,
        (Json(existing_weeks), duration_seconds, session_id),
    )


def rewrite_transcript_jsonl(session_id: uuid.UUID, db_rows: list[tuple]) -> None:
    output_dir = repo_root() / "backend" / "data" / "transcripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{session_id}.jsonl"
    kept_lines: list[dict] = []

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("recording_id") != RECORDING_ID:
                kept_lines.append(item)

    new_lines = [
        {
            "transcript_id": str(row[0]),
            "session_id": str(row[1]),
            "recording_id": row[2],
            "start_time": row[4],
            "end_time": row[5],
            "speaker_id": row[6],
            "speaker_name": row[7],
            "raw_text": row[8],
            "text": row[9],
            "created_at": row[10].isoformat(),
        }
        for row in db_rows
    ]
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in [*kept_lines, *new_lines]) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    source_path = repo_root() / "data" / "비정형데이터.md"
    source_text = source_path.read_text(encoding="utf-8")
    chunks = chunk_text(source_text)
    if not chunks:
        raise RuntimeError(f"No transcript text found in {source_path}")

    started_at = datetime.now().replace(microsecond=0)

    with connect() as conn:
        with conn.cursor() as cur:
            course_id = get_or_create_course(cur)
            session_id = get_or_create_session(cur, course_id)
            transcriptions, db_rows, duration_seconds = build_transcript_records(session_id, started_at, chunks)

            cur.execute(
                "DELETE FROM transcripts WHERE session_id = %s AND recording_id = %s",
                (session_id, RECORDING_ID),
            )
            execute_batch(
                cur,
                """
                INSERT INTO transcripts (
                    transcript_id, session_id, recording_id, chunk_index, start_time, end_time,
                    speaker_id, speaker_name, chunk_text, corrected_text, created_at
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (transcript_id) DO UPDATE
                SET chunk_index = EXCLUDED.chunk_index,
                    start_time = EXCLUDED.start_time,
                    end_time = EXCLUDED.end_time,
                    speaker_id = EXCLUDED.speaker_id,
                    speaker_name = EXCLUDED.speaker_name,
                    chunk_text = EXCLUDED.chunk_text,
                    corrected_text = EXCLUDED.corrected_text,
                    created_at = EXCLUDED.created_at
                """,
                db_rows,
                page_size=100,
            )
            upsert_recording_resource(cur, session_id, transcriptions, duration_seconds)

        conn.commit()

    rewrite_transcript_jsonl(session_id, db_rows)
    print(
        json.dumps(
            {
                "folder": FOLDER_TITLE,
                "session": SESSION_TITLE,
                "recording": RECORDING_TITLE,
                "session_id": str(session_id),
                "recording_id": RECORDING_ID,
                "chunks": len(chunks),
                "duration": format_duration(duration_seconds),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
