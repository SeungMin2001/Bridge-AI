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


def _safe_suffix(filename: str = "") -> str:
    suffix = Path(filename or "").suffix
    if len(suffix) > 16:
        return ""
    return suffix


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


def get_workspace_material_file(stored_name: str) -> FileResponse:
    safe_name = Path(stored_name).name
    if safe_name != stored_name:
        raise WorkspaceApiError("Invalid file name.", status_code=400)

    target_path = MATERIAL_UPLOAD_DIR / safe_name
    if not target_path.is_file():
        raise WorkspaceApiError("Material file not found.", status_code=404)

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
