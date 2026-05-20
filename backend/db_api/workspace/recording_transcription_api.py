"""Workspace uploaded-recording transcription API helpers.

Uploaded audio files are stored in session_voicefile JSON, while transcript chunks
live in the transcripts table and RAG index. This module keeps that sync in one
place so upload/playback code and realtime websocket code stay separate.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from data.save_transcript import save_transcript
from db import get_pool
from db_api.workspace.common import WorkspaceApiError, uuid_or_none
from db_api.workspace.files_api import get_workspace_recording_path
from db_api.workspace.serializers import json_value, session_node
from db_api.workspace.session_cleanup import delete_recording_related_rows
from rag_search import add_document as rag_add_document


logger = logging.getLogger(__name__)


def _recording_key(recording: dict) -> str:
    """Return the stable id used to match a recording inside session_voicefile."""
    return str(recording.get("id") or recording.get("recordingId") or "")


def _find_recording(resources, recording_id: str) -> dict | None:
    """Find one recording in grouped or legacy voicefile resources."""
    for entry in json_value(resources, []):
        if not isinstance(entry, dict):
            continue

        recordings = entry.get("recordings")
        if isinstance(recordings, list):
            for recording in recordings:
                if isinstance(recording, dict) and _recording_key(recording) == recording_id:
                    return recording
            continue

        if _recording_key(entry) == recording_id:
            return entry

    return None


def _replace_recording(resources, recording_id: str, updater) -> tuple[list, dict | None]:
    """Replace one recording in session_voicefile and return the updated copy."""
    next_resources = []
    updated_recording = None

    for entry in json_value(resources, []):
        if not isinstance(entry, dict):
            continue

        if isinstance(entry.get("recordings"), list):
            next_entry = dict(entry)
            next_recordings = []
            for recording in entry["recordings"]:
                if isinstance(recording, dict) and _recording_key(recording) == recording_id:
                    updated_recording = updater(dict(recording))
                    next_recordings.append(updated_recording)
                else:
                    next_recordings.append(recording)
            next_entry["recordings"] = next_recordings
            next_resources.append(next_entry)
            continue

        if _recording_key(entry) == recording_id:
            updated_recording = updater(dict(entry))
            next_resources.append(updated_recording)
        else:
            next_resources.append(entry)

    return next_resources, updated_recording


def _format_elapsed_time(seconds: float | int | None) -> str:
    """Format elapsed seconds as m:ss or h:mm:ss for frontend script rows."""
    safe_seconds = max(0, int(float(seconds or 0)))
    hours = safe_seconds // 3600
    minutes = (safe_seconds % 3600) // 60
    remain_seconds = str(safe_seconds % 60).zfill(2)
    if hours:
        return f"{hours}:{str(minutes).zfill(2)}:{remain_seconds}"
    return f"{minutes}:{remain_seconds}"


def _format_duration_text(seconds: float | int | None) -> str:
    """Format elapsed seconds as hh:mm:ss for recording metadata."""
    safe_seconds = max(0, int(round(float(seconds or 0))))
    hours = safe_seconds // 3600
    minutes = (safe_seconds % 3600) // 60
    remain_seconds = safe_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remain_seconds:02d}"


async def _fetch_session_row(session_uuid):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            """
            SELECT s.session_id, s.course_id, s.session_date, s.title, s.status, s.created_at,
                   s.file_kind, s.tag, s.icon, s.color, s.session_pdf, s.session_voicefile,
                   s.summary_notes, c.title AS course_title
            FROM sessions s
            LEFT JOIN courses c ON c.course_id = s.course_id
            WHERE s.session_id = $1
            """,
            session_uuid,
        )


async def _update_recording_status(session_uuid, recording_id: str, patch: dict) -> tuple[dict, dict]:
    """Patch one recording JSON object and persist the latest session_voicefile."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            current = await conn.fetchrow(
                """
                SELECT session_voicefile
                FROM sessions
                WHERE session_id = $1
                FOR UPDATE
                """,
                session_uuid,
            )
            if current is None:
                raise WorkspaceApiError("Session file not found.", status_code=404)

            def updater(recording):
                return {**recording, **patch}

            next_voicefile, updated_recording = _replace_recording(
                current["session_voicefile"],
                recording_id,
                updater,
            )
            if updated_recording is None:
                raise WorkspaceApiError("Recording not found.", status_code=404)

            row = await conn.fetchrow(
                """
                UPDATE sessions
                SET session_voicefile = $2::jsonb
                WHERE session_id = $1
                RETURNING session_id, course_id, session_date, title, status, created_at,
                          file_kind, tag, icon, color, session_pdf, session_voicefile, summary_notes
                """,
                session_uuid,
                json.dumps(next_voicefile),
            )

    return session_node(row), updated_recording


async def _cleanup_previous_recording_transcripts(session_uuid, recording_id: str) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            return await delete_recording_related_rows(conn, session_uuid, recording_id)


async def _save_uploaded_transcripts(
    *,
    session_id: str,
    recording_id: str,
    segments: list[dict],
    session_title: str,
    course_title: str,
    session_date: str,
) -> list[dict]:
    """Persist timestamped file-transcription segments to DB, JSONL, and RAG."""
    from stt.whisper_service import correction_enabled, correction_pool, correct_transcript_text

    loop = asyncio.get_event_loop()
    saved_items = []
    for index, segment in enumerate(segments):
        raw_text = str(segment.get("text") or segment.get("raw_text") or "").strip()
        if not raw_text:
            continue

        corrected_text = raw_text
        if correction_enabled:
            corrected_text = await loop.run_in_executor(correction_pool, correct_transcript_text, raw_text)

        start_time = float(segment.get("start") or 0.0)
        end_time = float(segment.get("end") or start_time)
        transcript_data = {
            "session_id": session_id,
            "recording_id": recording_id,
            "start_time": start_time,
            "end_time": end_time,
            "raw_text": raw_text,
            "text": corrected_text,
            "speaker_id": segment.get("speaker_id"),
        }
        saved = await save_transcript(transcript_data) or {}
        transcript_id = saved.get("transcript_id")
        chunk_index = saved.get("chunk_index", index)
        created_at = saved.get("created_at", "")

        saved_item = {
            "transcript_id": transcript_id,
            "chunk_index": chunk_index,
            "created_at": created_at,
            "start_time": start_time,
            "end_time": end_time,
            "raw_text": raw_text,
            "text": corrected_text,
            "speaker_id": segment.get("speaker_id"),
        }
        saved_items.append(saved_item)

        try:
            rag_add_document(corrected_text, {
                "session_id": session_id,
                "recording_id": recording_id,
                "session_title": session_title,
                "course_title": course_title,
                "session_date": session_date,
                "transcript_id": transcript_id or "",
                "chunk_index": chunk_index,
                "created_at": created_at,
                "start_time": start_time,
                "end_time": end_time,
                "speaker_id": segment.get("speaker_id") or "UNKNOWN",
            })
        except Exception as exc:
            logger.warning("[workspace] 업로드 전사 RAG 추가 실패: recording=%s error=%s", recording_id, exc)

    return saved_items


def _build_frontend_transcriptions(recording_id: str, transcript_items: list[dict]) -> list[dict]:
    """Convert DB transcript chunks into the script structure used by the UI."""
    transcriptions = []
    for item in transcript_items:
        start_time = float(item.get("start_time") or 0.0)
        end_time = float(item.get("end_time") or start_time)
        text = str(item.get("text") or "").strip()
        if not text:
            continue

        transcript_id = item.get("transcript_id") or str(uuid4())
        chunk_id = f"{recording_id}:{start_time:.3f}:{end_time:.3f}"
        segment = {
            "id": transcript_id,
            "chunkId": chunk_id,
            "transcriptId": transcript_id,
            "transcript_id": transcript_id,
            "chunk_index": item.get("chunk_index"),
            "text": text,
            "start": start_time,
            "end": end_time,
            "start_time": start_time,
            "end_time": end_time,
            "speakerId": item.get("speaker_id"),
            "speaker_id": item.get("speaker_id"),
            "status": "confirmed",
        }
        transcriptions.append({
            "recordingId": recording_id,
            "time": _format_elapsed_time(start_time),
            "text": text,
            "start": start_time,
            "end": end_time,
            "speakerId": item.get("speaker_id"),
            "segments": [segment],
        })
    return transcriptions


async def transcribe_session_recording(session_id: str, recording_id: str) -> dict:
    """Transcribe one uploaded recording and sync results back to the workspace file."""
    session_uuid = uuid_or_none(session_id, "session_id")
    if session_uuid is None:
        raise WorkspaceApiError("session_id is required.")
    if not recording_id:
        raise WorkspaceApiError("recording_id is required.")

    row = await _fetch_session_row(session_uuid)
    if row is None:
        raise WorkspaceApiError("Session file not found.", status_code=404)

    recording = _find_recording(row["session_voicefile"], recording_id)
    if recording is None:
        raise WorkspaceApiError("Recording not found.", status_code=404)

    stored_name = recording.get("storedName")
    if not stored_name:
        raise WorkspaceApiError("저장된 음성파일이 없어 전사할 수 없습니다.", status_code=400)

    audio_path = get_workspace_recording_path(str(stored_name))
    await _update_recording_status(session_uuid, recording_id, {
        "transcriptionStatus": "processing",
        "transcriptionError": None,
    })

    try:
        from stt.whisper_service import transcribe_audio_file, transcribe_pool

        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(transcribe_pool, transcribe_audio_file, audio_path)

        cleanup_result = await _cleanup_previous_recording_transcripts(session_uuid, recording_id)
        transcript_items = await _save_uploaded_transcripts(
            session_id=str(row["session_id"]),
            recording_id=recording_id,
            segments=segments,
            session_title=row["title"] or "",
            course_title=row["course_title"] or "",
            session_date=str(row["session_date"] or ""),
        )
        transcriptions = _build_frontend_transcriptions(recording_id, transcript_items)
        transcript_duration = max((item.get("end_time") or 0 for item in transcript_items), default=0)
        duration_seconds = max(float(recording.get("durationSeconds") or 0), float(transcript_duration or 0))
        completed_at = datetime.now(timezone.utc).isoformat()
        node, updated_recording = await _update_recording_status(session_uuid, recording_id, {
            "transcriptionStatus": "completed",
            "transcriptionError": None,
            "transcribedAt": completed_at,
            "transcriptCount": len(transcriptions),
            "durationSeconds": int(round(duration_seconds)),
            "durationText": _format_duration_text(duration_seconds),
            "transcriptions": transcriptions,
        })

        return {
            "ok": True,
            "sessionId": str(row["session_id"]),
            "recordingId": recording_id,
            "recording": updated_recording,
            "node": node,
            "transcriptCount": len(transcriptions),
            **cleanup_result,
        }
    except Exception as exc:
        logger.exception("[workspace] uploaded recording transcription failed")
        try:
            await _update_recording_status(session_uuid, recording_id, {
                "transcriptionStatus": "failed",
                "transcriptionError": str(exc),
            })
        except Exception:
            logger.warning("[workspace] failed to mark recording transcription as failed", exc_info=True)
        raise
