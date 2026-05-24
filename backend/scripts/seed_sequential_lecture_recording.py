#!/usr/bin/env python3
"""Seed a generated lecture transcript into the 순차자료구조 workspace session."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import Json, execute_batch, register_uuid


register_uuid()


SESSION_TITLE = "순차자료구조"
RECORDING_ID = "recording-sequential-linear-list-lecture"
RECORDING_TITLE = "순차 자료구조와 선형 리스트 강의 스크립트"
SCRIPT_PATH = Path("backend/data/generated_lectures/sequential_linear_list_lecture_script.txt")


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


def json_value(value, fallback):
    if value in (None, ""):
        return fallback
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


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


def chunk_text(text: str, max_chars: int = 520) -> list[str]:
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


def week_resource_key(entry: dict) -> str:
    return str(entry.get("weekKey") or entry.get("weekId") or entry.get("id") or entry.get("label") or "")


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


def resource_shell_from_materials(session_pdf, session_voicefile) -> dict:
    materials = json_value(session_pdf, [])
    voices = json_value(session_voicefile, [])
    source = None
    for candidate in [*voices, *materials]:
        if isinstance(candidate, dict):
            source = candidate
            break

    if source:
        return {
            "weekId": source.get("weekId") or source.get("id") or source.get("weekKey") or "week-generated",
            "id": source.get("id") or source.get("weekId") or source.get("weekKey") or "week-generated",
            "weekKey": source.get("weekKey") or source.get("id") or source.get("weekId") or "generated",
            "label": source.get("label") or "1주차",
            "dateLabel": source.get("dateLabel") or "",
            "expanded": source.get("expanded", True),
            "materialFolderExpanded": source.get("materialFolderExpanded", True),
            "recordingFolderExpanded": True,
            "recordings": [],
        }

    now = datetime.now(timezone.utc)
    return {
        "weekId": f"week-{now.date().isoformat()}",
        "id": f"week-{now.date().isoformat()}",
        "weekKey": now.date().isoformat(),
        "label": "1주차",
        "dateLabel": "",
        "expanded": True,
        "materialFolderExpanded": True,
        "recordingFolderExpanded": True,
        "recordings": [],
    }


def build_transcript_records(session_id: uuid.UUID, started_at: datetime, chunks: list[str]) -> tuple[list[dict], list[tuple], int]:
    transcriptions: list[dict] = []
    db_rows: list[tuple] = []
    cursor_seconds = 0.0

    for index, text in enumerate(chunks):
        duration = max(9.0, min(42.0, len(text) / 12.5))
        start_time = round(cursor_seconds, 2)
        end_time = round(cursor_seconds + duration, 2)
        cursor_seconds = end_time
        transcript_id = deterministic_uuid(f"{session_id}:{RECORDING_ID}:{index}")
        chunk_id = f"{RECORDING_ID}:{start_time:.3f}:{end_time:.3f}"
        created_at = started_at + timedelta(seconds=start_time)

        segment = {
            "id": str(transcript_id),
            "text": text,
            "status": "confirmed",
            "chunkId": chunk_id,
            "transcript_id": str(transcript_id),
            "transcriptId": str(transcript_id),
            "chunk_index": index,
            "start": start_time,
            "end": end_time,
            "start_time": start_time,
            "end_time": end_time,
            "speakerId": "speaker-professor",
            "speaker_id": "speaker-professor",
            "speaker": "교수",
        }
        transcriptions.append({
            "id": f"{RECORDING_ID}-{index}",
            "recordingId": RECORDING_ID,
            "time": format_elapsed_time(start_time),
            "start": start_time,
            "end": end_time,
            "speakerId": "speaker-professor",
            "speaker": "교수",
            "text": text,
            "segments": [segment],
        })

        db_rows.append((
            transcript_id,
            session_id,
            RECORDING_ID,
            index,
            start_time,
            end_time,
            "speaker-professor",
            "교수",
            text,
            text,
            created_at.replace(tzinfo=None),
        ))

    return transcriptions, db_rows, int(round(cursor_seconds))


def upsert_recording_resource(cur, session_id: uuid.UUID, transcriptions: list[dict], duration_seconds: int) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    ended_at = now + timedelta(seconds=duration_seconds)

    cur.execute(
        """
        SELECT session_pdf, session_voicefile
        FROM sessions
        WHERE session_id = %s
        """,
        (session_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"Session not found: {session_id}")

    session_pdf, session_voicefile = row
    existing_weeks = remove_recording_from_weeks(json_value(session_voicefile, []))
    materials = json_value(session_pdf, [])
    material_ids = []
    material_names = []
    for week in materials:
        for material in week.get("materials", []) if isinstance(week, dict) else []:
            if material.get("id"):
                material_ids.append(material.get("id"))
            if material.get("name"):
                material_names.append(material.get("name"))

    recording = {
        "id": RECORDING_ID,
        "recordingId": RECORDING_ID,
        "title": RECORDING_TITLE,
        "startedAt": now.isoformat(),
        "endedAt": ended_at.isoformat(),
        "durationText": format_duration(duration_seconds),
        "durationSeconds": duration_seconds,
        "recordingMode": "generated_script",
        "diarizationEnabled": False,
        "materialIds": material_ids,
        "materialNames": material_names,
        "audioUrl": None,
        "storedName": "",
        "originalName": "",
        "type": "text/generated-lecture",
        "uploadedAt": now.isoformat(),
        "transcribedAt": now.isoformat(),
        "transcriptionStatus": "completed",
        "transcriptCount": len(transcriptions),
        "transcriptions": transcriptions,
    }

    target_key = week_resource_key(materials[0]) if materials else ""
    next_weeks = []
    inserted = False
    for week in existing_weeks:
        next_week = dict(week)
        next_recordings = next_week.get("recordings") if isinstance(next_week.get("recordings"), list) else []
        if target_key and week_resource_key(next_week) == target_key:
            next_week["recordings"] = [recording, *next_recordings]
            inserted = True
        next_weeks.append(next_week)

    if not inserted:
        shell = resource_shell_from_materials(session_pdf, session_voicefile)
        shell["recordings"] = [recording]
        next_weeks = [shell, *next_weeks]

    cur.execute(
        """
        UPDATE sessions
        SET session_voicefile = %s::jsonb,
            duration_sec = %s,
            status = 'created'
        WHERE session_id = %s
        """,
        (Json(next_weeks), duration_seconds, session_id),
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
    source_path = repo_root() / SCRIPT_PATH
    source_text = source_path.read_text(encoding="utf-8")
    chunks = chunk_text(source_text)
    if not chunks:
        raise RuntimeError(f"No transcript text found in {source_path}")

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE title = %s
                ORDER BY created_at DESC NULLS LAST
                LIMIT 1
                """,
                (SESSION_TITLE,),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"Session title not found: {SESSION_TITLE}")

            session_id = row[0]
            started_at = datetime.now(timezone.utc).replace(microsecond=0)
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
    print(json.dumps({
        "session": SESSION_TITLE,
        "session_id": str(session_id),
        "recording": RECORDING_TITLE,
        "recording_id": RECORDING_ID,
        "chunks": len(chunks),
        "duration": format_duration(duration_seconds),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
