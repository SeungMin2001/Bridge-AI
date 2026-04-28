import asyncio
import os

import asyncpg


async def main():
    config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("DB_NAME", "shin"),
        "user": os.getenv("DB_USER", "changyoung"),
        "password": os.getenv("DB_PASSWORD") or None,
    }

    safe_config = {**config, "password": "***" if config["password"] else None}
    print("[DB] config:", safe_config)

    conn = await asyncpg.connect(**config)
    try:
        database_info = await conn.fetchrow(
            """
            SELECT
                current_database() AS database_name,
                current_user AS user_name
            """
        )
        tables = await conn.fetch(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
            """
        )

        print("[DB] connected")
        print("[DB] database:", database_info["database_name"])
        print("[DB] user:", database_info["user_name"])
        print("[DB] tables:", ", ".join(row["table_name"] for row in tables) or "(none)")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
