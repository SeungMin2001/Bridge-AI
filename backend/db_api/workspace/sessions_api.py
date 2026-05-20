import json
import logging
import os

from datetime import date, datetime
from uuid import uuid4

from fastapi import UploadFile

from db import get_pool
from db_api.workspace.common import WorkspaceApiError, required_text, uuid_or_none
from db_api.workspace.default_folder import ensure_default_folder
from db_api.workspace.files_api import delete_workspace_material_files, delete_workspace_recording_files, save_workspace_recording_file
from db_api.workspace.session_cleanup import delete_recording_related_rows, delete_session_related_rows, delete_transcript_json_files
from db_api.workspace.serializers import json_value, session_node, split_week_resources, week_resource_key


logger = logging.getLogger(__name__)


async def create_session_file(payload: dict) -> dict:
    # 새 파일 생성 요청을 SESSIONS 테이블에 저장
    session_id = uuid4()
    course_id = uuid_or_none(payload.get("course_id"), "course_id")
    title = required_text(payload, "title")
    file_kind = payload.get("file_kind") or "lecture"
    created_at = datetime.now()

    pool = await get_pool()
    async with pool.acquire() as conn:
        if course_id is None:
            course_id = await ensure_default_folder(conn)

        row = await conn.fetchrow(
            """
            INSERT INTO sessions (
                session_id,
                course_id,
                session_date,
                title,
                status,
                created_at,
                file_kind,
                tag,
                icon,
                color,
                session_pdf,
                session_voicefile,
                summary_notes
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb,$12::jsonb,$13::jsonb)
            RETURNING session_id, course_id, session_date, title, status, created_at,
                      file_kind, tag, icon, color, session_pdf, session_voicefile, summary_notes
            """,
            session_id,
            course_id,
            date.today(),
            title,
            payload.get("status") or "created",
            created_at,
            file_kind,
            payload.get("tag") or ("회의" if file_kind == "meeting" else "수업"),
            payload.get("icon") or ("groups_2" if file_kind == "meeting" else "article"),
            payload.get("color") or "#3b82f6",
            json.dumps([]),
            json.dumps([]),
            json.dumps([]),
        )

    return {
        "ok": True,
        "sessionId": str(row["session_id"]),
        "node": session_node(row),
    }


async def update_session_file(session_id: str, payload: dict) -> dict:
    session_uuid = uuid_or_none(session_id, "session_id")
    if session_uuid is None:
        raise WorkspaceApiError("session_id is required.")

    title = required_text(payload, "title")

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE sessions
            SET title = $2
            WHERE session_id = $1
            RETURNING session_id, course_id, session_date, title, status, created_at,
                      file_kind, tag, icon, color, session_pdf, session_voicefile, summary_notes
            """,
            session_uuid,
            title,
        )

    if row is None:
        raise WorkspaceApiError("Session file not found.", status_code=404)

    return {
        "ok": True,
        "sessionId": str(row["session_id"]),
        "node": session_node(row),
    }


async def delete_session_file(session_id: str) -> dict:
    # 신창영 : 파일 삭제 요청 시 sessions row만 지우지 않고 전사/RAG 관련 데이터까지 함께 정리
    session_uuid = uuid_or_none(session_id, "session_id")
    if session_uuid is None:
        raise WorkspaceApiError("session_id is required.")

    pool = await get_pool()
    cleanup_result = {}
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT session_id, session_pdf, session_voicefile
                FROM sessions
                WHERE session_id = $1
                """,
                session_uuid,
            )

            if row is None:
                raise WorkspaceApiError("Session file not found.", status_code=404)

            # 신창영 : 삭제된 파일의 전사문이 AI 채팅 참조로 다시 노출되지 않도록 선행 정리
            cleanup_result = await delete_session_related_rows(conn, [session_uuid])
            await conn.execute(
                """
                DELETE FROM sessions
                WHERE session_id = $1
                """,
                session_uuid,
            )
            rag_table_name = f"data_{os.getenv('RAG_TABLE_NAME', 'rag')}"
            try:
                await conn.execute(
                    f"""
                    DELETE FROM {rag_table_name}
                    WHERE metadata_->>'session_id' = $1
                    """,
                    str(session_uuid),
                )
            except Exception as e:
                # RAG 테이블이 아직 생성되지 않은 경우 등 오류 무시
                pass

    deleted_material_count = delete_workspace_material_files(row["session_pdf"])
    deleted_recording_file_count = delete_workspace_recording_files(row["session_voicefile"])
    deleted_transcript_file_count = delete_transcript_json_files([row["session_id"]])

    return {
        "ok": True,
        "sessionId": str(row["session_id"]),
        "deletedMaterialCount": deleted_material_count,
        "deletedRecordingFileCount": deleted_recording_file_count,
        "deletedTranscriptFileCount": deleted_transcript_file_count,
        **cleanup_result,
    }


async def update_session_resources(session_id: str, payload: dict) -> dict:
    # 현재 파일 내부 구조를 강의자료/녹음본 컬럼으로 분리 저장
    session_uuid = uuid_or_none(session_id, "session_id")
    if session_uuid is None:
        raise WorkspaceApiError("session_id is required.")

    weeks = payload.get("weeks")
    if not isinstance(weeks, list):
        raise WorkspaceApiError("weeks must be a list.")

    session_pdf = split_week_resources(weeks, "materials")
    session_voicefile = split_week_resources(weeks, "recordings")

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE sessions
            SET session_pdf = $2::jsonb,
                session_voicefile = $3::jsonb
            WHERE session_id = $1
            RETURNING session_id, course_id, session_date, title, status, created_at,
                      file_kind, tag, icon, color, session_pdf, session_voicefile, summary_notes
            """,
            session_uuid,
            json.dumps(session_pdf),
            json.dumps(session_voicefile),
        )

    if row is None:
        raise WorkspaceApiError("Session file not found.", status_code=404)

    try:
        from materials.material_rag_service import sync_session_materials_to_rag
        await sync_session_materials_to_rag(str(session_uuid))
    except Exception as exc:
        logger.warning("[workspace] PDF RAG 동기화 실패: session=%s, error=%s", session_uuid, exc)

    return {
        "ok": True,
        "sessionId": str(row["session_id"]),
        "node": session_node(row),
    }


def _current_week_resource_shell() -> dict:
    today = date.today()
    week_start = date.fromordinal(today.toordinal() - today.weekday())
    week_key = week_start.isoformat()
    weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    return {
        "weekId": f"week-{week_key}",
        "weekKey": week_key,
        "label": "1주차",
        "dateLabel": f"{today.year}. {today.month}. {today.day}. {weekdays[today.weekday()]}",
        "recordings": [],
    }


def _resource_shell_from_existing(*resource_groups: list) -> dict:
    for resources in resource_groups:
        if not isinstance(resources, list):
            continue
        for entry in resources:
            if not isinstance(entry, dict):
                continue
            return {
                "weekId": entry.get("weekId") or entry.get("id") or entry.get("weekKey"),
                "weekKey": entry.get("weekKey"),
                "label": entry.get("label") or entry.get("weekLabel") or "1주차",
                "dateLabel": entry.get("dateLabel"),
                "recordings": [],
            }
    return _current_week_resource_shell()


def _append_recording_resource(session_pdf, session_voicefile, recording: dict) -> list:
    resources = json_value(session_voicefile, [])
    materials = json_value(session_pdf, [])
    recording_id = str(recording.get("id") or recording.get("recordingId") or "")
    target_key = week_resource_key(resources[0]) if resources else ""
    if not target_key and materials:
        target_key = week_resource_key(materials[0])

    next_resources = []
    inserted = False
    for entry in resources:
        if not isinstance(entry, dict):
            continue

        next_entry = dict(entry)
        next_recordings = [
            item for item in (next_entry.get("recordings") or [])
            if isinstance(item, dict) and str(item.get("id") or item.get("recordingId") or "") != recording_id
        ]

        if (target_key and week_resource_key(next_entry) == target_key) or (not target_key and not inserted):
            next_entry["recordings"] = [recording, *next_recordings]
            inserted = True
        else:
            next_entry["recordings"] = next_recordings
        next_resources.append(next_entry)

    if not inserted:
        shell = _resource_shell_from_existing(materials, resources)
        shell["recordings"] = [recording]
        next_resources = [shell, *next_resources]

    return next_resources


async def upload_session_recording(
    session_id: str,
    upload: UploadFile,
    *,
    title: str | None = None,
    duration_seconds: float | int | None = None,
) -> dict:
    session_uuid = uuid_or_none(session_id, "session_id")
    if session_uuid is None:
        raise WorkspaceApiError("session_id is required.")

    saved = await save_workspace_recording_file(
        session_id,
        upload,
        title=title,
        duration_seconds=duration_seconds,
    )
    recording = saved["recording"]

    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            current_row = await conn.fetchrow(
                """
                SELECT session_pdf, session_voicefile
                FROM sessions
                WHERE session_id = $1
                """,
                session_uuid,
            )
            if current_row is None:
                raise WorkspaceApiError("Session file not found.", status_code=404)

            session_voicefile = _append_recording_resource(
                current_row["session_pdf"],
                current_row["session_voicefile"],
                recording,
            )
            row = await conn.fetchrow(
                """
                UPDATE sessions
                SET session_voicefile = $2::jsonb
                WHERE session_id = $1
                RETURNING session_id, course_id, session_date, title, status, created_at,
                          file_kind, tag, icon, color, session_pdf, session_voicefile, summary_notes
                """,
                session_uuid,
                json.dumps(session_voicefile),
            )

    return {
        "ok": True,
        "sessionId": str(row["session_id"]),
        "recording": recording,
        "node": session_node(row),
    }


def _remove_recording_from_resources(resources, recording_id: str):
    if not isinstance(resources, list):
        return []

    cleaned = []
    for item in resources:
        if isinstance(item, dict) and isinstance(item.get("recordings"), list):
            recordings = item.get("recordings")
            next_item = dict(item)
            next_item["recordings"] = [recording for recording in recordings if recording.get("id") != recording_id]
            cleaned.append(next_item)
            continue
        if isinstance(item, dict) and item.get("id") == recording_id:
            continue
        cleaned.append(item)
    return cleaned


async def delete_session_recording(session_id: str, recording_id: str) -> dict:
    session_uuid = uuid_or_none(session_id, "session_id")
    if session_uuid is None:
        raise WorkspaceApiError("session_id is required.")
    if not recording_id:
        raise WorkspaceApiError("recording_id is required.")

    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT session_id, course_id, session_date, title, status, created_at,
                       file_kind, tag, icon, color, session_pdf, session_voicefile, summary_notes
                FROM sessions
                WHERE session_id = $1
                """,
                session_uuid,
            )
            if row is None:
                raise WorkspaceApiError("Session file not found.", status_code=404)

            cleanup_result = await delete_recording_related_rows(conn, session_uuid, recording_id)
            original_voicefile = row["session_voicefile"]
            session_voicefile = _remove_recording_from_resources(row["session_voicefile"], recording_id)
            row = await conn.fetchrow(
                """
                UPDATE sessions
                SET session_voicefile = $2::jsonb
                WHERE session_id = $1
                RETURNING session_id, course_id, session_date, title, status, created_at,
                          file_kind, tag, icon, color, session_pdf, session_voicefile, summary_notes
                """,
                session_uuid,
                json.dumps(session_voicefile),
            )

    deleted_recording_file_count = delete_workspace_recording_files(original_voicefile, recording_id)

    return {
        "ok": True,
        "sessionId": str(row["session_id"]),
        "recordingId": recording_id,
        "node": session_node(row),
        "deletedRecordingFileCount": deleted_recording_file_count,
        **cleanup_result,
    }
