import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from fastapi.responses import FileResponse

from db import get_pool
from db_api.workspace.common import WorkspaceApiError, uuid_or_none


BACKEND_ROOT = Path(__file__).resolve().parents[2]
MATERIAL_UPLOAD_DIR = Path(os.getenv("WORKSPACE_MATERIAL_UPLOAD_DIR", BACKEND_ROOT / "uploads" / "workspace" / "materials"))
RECORDING_UPLOAD_DIR = Path(os.getenv("WORKSPACE_RECORDING_UPLOAD_DIR", BACKEND_ROOT / "uploads" / "workspace" / "recordings"))
ALLOWED_RECORDING_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}


def _safe_suffix(filename: str = "") -> str:
    suffix = Path(filename or "").suffix
    if len(suffix) > 16:
        return ""
    return suffix


def _safe_audio_suffix(filename: str = "") -> str:
    suffix = _safe_suffix(filename).lower()
    if suffix not in ALLOWED_RECORDING_SUFFIXES:
        raise WorkspaceApiError("지원하지 않는 음성파일 형식입니다.", status_code=400)
    return suffix


def _format_duration_text(duration_seconds: float | int | None) -> str:
    total_seconds = _duration_seconds_int(duration_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _duration_seconds_int(duration_seconds: float | int | None) -> int:
    try:
        return max(0, int(round(float(duration_seconds))))
    except (TypeError, ValueError):
        return 0


def _normalize_upload_title(filename: str = "", title: str | None = None) -> str:
    next_title = str(title or "").strip()
    if next_title:
        return next_title

    stem = Path(filename or "").stem.strip()
    return stem or "업로드한 녹음본"


async def save_workspace_material(session_id: str, upload: UploadFile) -> dict:
    session_uuid = uuid_or_none(session_id, "session_id")
    if session_uuid is None:
        raise WorkspaceApiError("session_id is required.")

    pool = await get_pool()
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            """
            SELECT EXISTS(
                SELECT 1
                FROM sessions
                WHERE session_id = $1
            )
            """,
            session_uuid,
        )

    if not exists:
        raise WorkspaceApiError("Session file not found.", status_code=404)

    MATERIAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    material_id = f"material-{uuid4()}"
    stored_name = f"{material_id}{_safe_suffix(upload.filename)}"
    target_path = MATERIAL_UPLOAD_DIR / stored_name
    size = 0

    with target_path.open("wb") as target:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            target.write(chunk)

    await upload.close()

    return {
        "ok": True,
        "material": {
            "id": material_id,
            "name": upload.filename or stored_name,
            "size": size,
            "type": upload.content_type or "application/octet-stream",
            "uploadedAt": datetime.now(timezone.utc).isoformat(),
            "url": f"/workspace/uploads/materials/{stored_name}",
            "storedName": stored_name,
        },
    }


async def save_workspace_recording_file(
    session_id: str,
    upload: UploadFile,
    *,
    title: str | None = None,
    duration_seconds: float | int | None = None,
) -> dict:
    session_uuid = uuid_or_none(session_id, "session_id")
    if session_uuid is None:
        raise WorkspaceApiError("session_id is required.")

    safe_suffix = _safe_audio_suffix(upload.filename or "")
    content_type = upload.content_type or ""
    if content_type and not (content_type.startswith("audio/") or content_type == "application/octet-stream"):
        raise WorkspaceApiError("음성파일만 업로드할 수 있습니다.", status_code=400)

    pool = await get_pool()
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            """
            SELECT EXISTS(
                SELECT 1
                FROM sessions
                WHERE session_id = $1
            )
            """,
            session_uuid,
        )

    if not exists:
        raise WorkspaceApiError("Session file not found.", status_code=404)

    RECORDING_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    recording_id = f"recording-{uuid4()}"
    stored_name = f"{recording_id}{safe_suffix}"
    target_path = RECORDING_UPLOAD_DIR / stored_name
    size = 0

    with target_path.open("wb") as target:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            target.write(chunk)

    await upload.close()

    uploaded_at = datetime.now(timezone.utc).isoformat()
    return {
        "ok": True,
        "recording": {
            "id": recording_id,
            "recordingId": recording_id,
            "title": _normalize_upload_title(upload.filename or stored_name, title),
            "startedAt": uploaded_at,
            "endedAt": uploaded_at,
            "durationText": _format_duration_text(duration_seconds),
            "durationSeconds": _duration_seconds_int(duration_seconds),
            "recordingMode": "uploaded",
            "diarizationEnabled": False,
            "materialIds": [],
            "materialNames": [],
            "audioUrl": f"/workspace/uploads/recordings/{stored_name}",
            "storedName": stored_name,
            "originalName": upload.filename or stored_name,
            "size": size,
            "type": upload.content_type or "audio/*",
            "uploadedAt": uploaded_at,
            "transcriptionStatus": "not_started",
            "transcriptions": [],
        },
    }


def get_workspace_material_file(stored_name: str) -> FileResponse:
    safe_name = Path(stored_name).name
    if safe_name != stored_name:
        raise WorkspaceApiError("Invalid file name.", status_code=400)

    target_path = MATERIAL_UPLOAD_DIR / safe_name
    if not target_path.is_file():
        raise WorkspaceApiError("Material file not found.", status_code=404)

    return FileResponse(target_path)


def get_workspace_recording_file(stored_name: str) -> FileResponse:
    safe_name = Path(stored_name).name
    if safe_name != stored_name:
        raise WorkspaceApiError("Invalid file name.", status_code=400)

    target_path = RECORDING_UPLOAD_DIR / safe_name
    if not target_path.is_file():
        raise WorkspaceApiError("Recording file not found.", status_code=404)

    return FileResponse(target_path)


def _iter_resource_items(value, resource_key: str):
    if not isinstance(value, list):
        return

    for entry in value:
        if not isinstance(entry, dict):
            continue

        items = entry.get(resource_key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    yield item
        else:
            yield entry


def _stored_name_from_material(material: dict) -> str | None:
    stored_name = material.get("storedName")
    if stored_name:
        return Path(str(stored_name)).name

    url = material.get("url")
    if isinstance(url, str) and "/workspace/uploads/materials/" in url:
        return Path(url.rsplit("/", 1)[-1]).name

    return None


def _stored_name_from_recording(recording: dict) -> str | None:
    stored_name = recording.get("storedName")
    if stored_name:
        return Path(str(stored_name)).name

    url = recording.get("audioUrl") or recording.get("url")
    if isinstance(url, str) and "/workspace/uploads/recordings/" in url:
        return Path(url.rsplit("/", 1)[-1]).name

    return None


def delete_workspace_material_files(session_pdf) -> int:
    deleted_count = 0

    for material in _iter_resource_items(session_pdf, "materials"):
        stored_name = _stored_name_from_material(material)
        if not stored_name:
            continue

        target_path = MATERIAL_UPLOAD_DIR / stored_name
        try:
            target_path.unlink()
            deleted_count += 1
        except FileNotFoundError:
            continue

    return deleted_count


def delete_workspace_recording_files(session_voicefile, recording_id: str | None = None) -> int:
    deleted_count = 0

    for recording in _iter_resource_items(session_voicefile, "recordings"):
        current_id = str(recording.get("id") or recording.get("recordingId") or "")
        if recording_id and current_id != recording_id:
            continue

        stored_name = _stored_name_from_recording(recording)
        if not stored_name:
            continue

        target_path = RECORDING_UPLOAD_DIR / stored_name
        try:
            target_path.unlink()
            deleted_count += 1
        except FileNotFoundError:
            continue

    return deleted_count


def delete_workspace_transcript_file(session_id: str):
    session_uuid = uuid_or_none(session_id, "session_id")
    if not session_uuid:
        return

    target_path = BACKEND_ROOT / "data" / "transcripts" / f"{session_uuid}.jsonl"
    try:
        target_path.unlink()
    except FileNotFoundError:
        pass
