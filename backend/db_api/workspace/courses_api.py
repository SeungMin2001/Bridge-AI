from datetime import datetime
from uuid import uuid4

from db import get_pool
from db_api.workspace.common import WorkspaceApiError, required_text, uuid_or_none
from db_api.workspace.files_api import delete_workspace_material_files
from db_api.workspace.session_cleanup import delete_session_related_rows, delete_transcript_json_files
from db_api.workspace.serializers import course_node


async def create_course(payload: dict) -> dict:
    # 새 폴더 생성 요청을 COURSES 테이블에 저장.
    course_id = uuid4()
    user_id = uuid_or_none(payload.get("user_id"), "user_id")
    parent_course_id = uuid_or_none(payload.get("parent_course_id"), "parent_course_id")
    title = required_text(payload, "title")
    created_at = datetime.now()

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO courses (
                course_id,
                user_id,
                parent_course_id,
                title,
                type,
                description,
                color,
                icon,
                created_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            RETURNING course_id, user_id, parent_course_id, title, type, description, color, icon, created_at
            """,
            course_id,
            user_id,
            parent_course_id,
            title,
            payload.get("type") or "folder",
            payload.get("description"),
            payload.get("color") or "#3b82f6",
            payload.get("icon") or "folder",
            created_at,
        )

    return {
        "ok": True,
        "courseId": str(row["course_id"]),
        "node": course_node(row),
    }


async def update_course(course_id: str, payload: dict) -> dict:
    # 폴더 수정 요청을 COURSES 테이블에 반영합니다.
    course_uuid = uuid_or_none(course_id, "course_id")
    if course_uuid is None:
        raise WorkspaceApiError("course_id is required.")

    title = required_text(payload, "title")

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE courses
            SET
                title = $2,
                color = $3,
                icon = $4
            WHERE course_id = $1
            RETURNING course_id, user_id, parent_course_id, title, type, description, color, icon, created_at
            """,
            course_uuid,
            title,
            payload.get("color") or "#3b82f6",
            payload.get("icon") or "folder",
        )

    if row is None:
        raise WorkspaceApiError("Course folder not found.", status_code=404)

    return {
        "ok": True,
        "courseId": str(row["course_id"]),
        "node": course_node(row),
    }


async def delete_course(course_id: str) -> dict:
    # 신창영 : 폴더 삭제 시 하위 파일의 전사/RAG 데이터까지 함께 삭제
    course_uuid = uuid_or_none(course_id, "course_id")
    if course_uuid is None:
        raise WorkspaceApiError("course_id is required.")

    pool = await get_pool()
    session_pdf_values = []
    session_ids = []
    cleanup_result = {}
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                """
                WITH RECURSIVE course_tree AS (
                    SELECT course_id
                    FROM courses
                    WHERE course_id = $1

                    UNION ALL

                    SELECT child.course_id
                    FROM courses child
                    JOIN course_tree parent ON child.parent_course_id = parent.course_id
                )
                SELECT course_id
                FROM course_tree
                """,
                course_uuid,
            )

            if not rows:
                raise WorkspaceApiError("Course folder not found.", status_code=404)

            course_ids = [row["course_id"] for row in rows]

            session_rows = await conn.fetch(
                """
                SELECT session_id, session_pdf
                FROM sessions
                WHERE course_id = ANY($1::uuid[])
                """,
                course_ids,
            )
            session_ids = [row["session_id"] for row in session_rows]
            session_pdf_values = [row["session_pdf"] for row in session_rows]

            if session_ids:
                # 신창영 : 폴더 하위 세션의 오래된 전사/RAG 참조를 course 삭제 전에 정리
                cleanup_result = await delete_session_related_rows(conn, session_ids)

            await conn.execute(
                """
                DELETE FROM sessions
                WHERE course_id = ANY($1::uuid[])
                """,
                course_ids,
            )
            await conn.execute(
                """
                DELETE FROM courses
                WHERE course_id = ANY($1::uuid[])
                """,
                course_ids,
            )

    deleted_material_count = 0
    for session_pdf in session_pdf_values:
        deleted_material_count += delete_workspace_material_files(session_pdf)
    deleted_transcript_file_count = delete_transcript_json_files(session_ids)

    return {
        "ok": True,
        "courseId": str(course_uuid),
        "deletedCourseCount": len(course_ids),
        "deletedSessionCount": len(session_pdf_values),
        "deletedMaterialCount": deleted_material_count,
        "deletedTranscriptFileCount": deleted_transcript_file_count,
        **cleanup_result,
    }
