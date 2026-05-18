from datetime import datetime
from uuid import uuid4

from db_api.workspace.common import DEFAULT_FOLDER_DESCRIPTION, DEFAULT_FOLDER_LEGACY_TITLE, DEFAULT_FOLDER_TITLE


async def ensure_default_folder(conn):
    row = await conn.fetchrow(
        """
        SELECT course_id
        FROM courses
        WHERE parent_course_id IS NULL
          AND (description = $1 OR title = $2 OR title = $3)
        ORDER BY
          CASE WHEN description = $1 THEN 0 ELSE 1 END,
          created_at ASC NULLS LAST
        LIMIT 1
        """,
        DEFAULT_FOLDER_DESCRIPTION,
        DEFAULT_FOLDER_TITLE,
        DEFAULT_FOLDER_LEGACY_TITLE,
    )

    if row:
        await conn.execute(
            """
            UPDATE courses
            SET
                title = $2,
                type = COALESCE(type, 'folder'),
                description = $1,
                color = COALESCE(color, '#3b82f6'),
                icon = COALESCE(icon, 'folder')
            WHERE course_id = $3
            """,
            DEFAULT_FOLDER_DESCRIPTION,
            DEFAULT_FOLDER_TITLE,
            row["course_id"],
        )
        return row["course_id"]

    course_id = uuid4()
    inserted = await conn.fetchrow(
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
        RETURNING course_id
        """,
        course_id,
        None,
        None,
        DEFAULT_FOLDER_TITLE,
        "folder",
        DEFAULT_FOLDER_DESCRIPTION,
        "#3b82f6",
        "folder",
        datetime.now(),
    )
    return inserted["course_id"]


async def attach_orphan_sessions_to_default_folder(conn, default_course_id) -> None:
    await conn.execute(
        """
        UPDATE sessions
        SET course_id = $1
        WHERE course_id IS NULL
           OR NOT EXISTS (
                SELECT 1
                FROM courses
                WHERE courses.course_id = sessions.course_id
           )
        """,
        default_course_id,
    )
