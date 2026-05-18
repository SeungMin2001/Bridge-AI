from db import get_pool
from db_api.workspace.default_folder import attach_orphan_sessions_to_default_folder, ensure_default_folder
from db_api.workspace.serializers import course_node, session_node


async def get_workspace_tree() -> dict:
    # COURSES와 SESSIONS를 조회해서 프론트 fileTree 형태로 조립합니다.
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            default_course_id = await ensure_default_folder(conn)
            await attach_orphan_sessions_to_default_folder(conn, default_course_id)

            course_rows = await conn.fetch(
                """
                SELECT course_id, user_id, parent_course_id, title, type, description, color, icon, created_at
                FROM courses
                ORDER BY created_at DESC NULLS LAST
                """
            )
            session_rows = await conn.fetch(
                """
                SELECT session_id, course_id, session_date, title, status, created_at,
                       file_kind, tag, icon, color, session_pdf, session_voicefile, summary_notes
                FROM sessions
                ORDER BY created_at DESC NULLS LAST
                """
            )

    course_nodes = {str(row["course_id"]): course_node(row) for row in course_rows}
    root_nodes = []

    for row in course_rows:
        node = course_nodes[str(row["course_id"])]
        parent_id = str(row["parent_course_id"]) if row["parent_course_id"] else None
        parent = course_nodes.get(parent_id)
        if parent:
            parent["children"].append(node)
        else:
            root_nodes.append(node)

    for row in session_rows:
        node = session_node(row)
        course_id = str(row["course_id"]) if row["course_id"] else None
        parent = course_nodes.get(course_id)
        if parent:
            parent["children"].append(node)
        else:
            root_nodes.append(node)

    root_nodes.sort(key=lambda node: 0 if node.get("isDefaultFolder") else 1)

    return {
        "ok": True,
        "tree": root_nodes,
        "coursesCount": len(course_rows),
        "sessionsCount": len(session_rows),
    }
