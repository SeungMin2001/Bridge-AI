from datetime import datetime
from uuid import uuid4

from db import get_pool
from db_api.workspace.common import required_text, uuid_or_none
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
