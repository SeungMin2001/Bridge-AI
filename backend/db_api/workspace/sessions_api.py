import json
from datetime import date, datetime
from uuid import uuid4

from db import get_pool
from db_api.workspace.common import WorkspaceApiError, required_text, uuid_or_none
from db_api.workspace.serializers import session_node


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
                summary_notes,
                resource_tree
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb,$12::jsonb,$13::jsonb)
            RETURNING session_id, course_id, session_date, title, status, created_at,
                      file_kind, tag, icon, color, session_pdf, summary_notes, resource_tree
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
        row = await conn.fetchrow(
            """
            DELETE FROM sessions
            WHERE session_id = $1
            RETURNING session_id
            """,
            session_uuid,
        )

    if row is None:
        raise WorkspaceApiError("Session file not found.", status_code=404)

    return {
        "ok": True,
        "sessionId": str(row["session_id"]),
    }


async def update_session_resource_tree(session_id: str, payload: dict) -> dict:
    # 현재 파일 내부 주차/자료/녹음본 구조를 SESSIONS 테이블에 저장
    session_uuid = uuid_or_none(session_id, "session_id")
    if session_uuid is None:
        raise WorkspaceApiError("session_id is required.")

    resource_tree = payload.get("resource_tree")
    if not isinstance(resource_tree, list):
        raise WorkspaceApiError("resource_tree must be a list.")

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE sessions
            SET resource_tree = $2::jsonb
            WHERE session_id = $1
            RETURNING session_id, course_id, session_date, title, status, created_at,
                      file_kind, tag, icon, color, session_pdf, summary_notes, resource_tree
            """,
            session_uuid,
            json.dumps(resource_tree),
        )

    if row is None:
        raise WorkspaceApiError("Session file not found.", status_code=404)

    return {
        "ok": True,
        "sessionId": str(row["session_id"]),
        "node": session_node(row),
    }
