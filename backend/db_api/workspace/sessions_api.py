import json
from datetime import date, datetime
from uuid import uuid4

from db import get_pool
from db_api.workspace.common import WorkspaceApiError, required_text, uuid_or_none
from db_api.workspace.files_api import delete_workspace_material_files
from db_api.workspace.serializers import session_node, split_week_resources


async def create_session_file(payload: dict) -> dict:
    # 새 파일 생성 요청을 SESSIONS 테이블에 저장
    session_id = uuid4()
    course_id = uuid_or_none(payload.get("course_id"), "course_id")
    title = required_text(payload, "title")
    file_kind = payload.get("file_kind") or "lecture"
    created_at = datetime.now()

    pool = await get_pool()
    async with pool.acquire() as conn:
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


async def delete_session_file(session_id: str) -> dict:
    # 파일 삭제 요청을 받아 SESSIONS 테이블에서 해당 파일 row 제거
    session_uuid = uuid_or_none(session_id, "session_id")
    if session_uuid is None:
        raise WorkspaceApiError("session_id is required.")

    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT session_id, session_pdf
                FROM sessions
                WHERE session_id = $1
                """,
                session_uuid,
            )

            if row is None:
                raise WorkspaceApiError("Session file not found.", status_code=404)

            await conn.execute(
                """
                DELETE FROM transcripts
                WHERE session_id = $1
                """,
                session_uuid,
            )
            await conn.execute(
                """
                DELETE FROM sessions
                WHERE session_id = $1
                """,
                session_uuid,
            )

    deleted_material_count = delete_workspace_material_files(row["session_pdf"])

    return {
        "ok": True,
        "sessionId": str(row["session_id"]),
        "deletedMaterialCount": deleted_material_count,
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

    return {
        "ok": True,
        "sessionId": str(row["session_id"]),
        "node": session_node(row),
    }
