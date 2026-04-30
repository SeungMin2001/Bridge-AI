import psycopg
from pgvector.psycopg import register_vector
import datetime
import os

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "db"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "rag"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "1234"),
    "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", 5)),
}

def get_db_conn():
    conn = psycopg.connect(**DB_CONFIG)
    register_vector(conn)
    return conn

def init_test_session(session_id: str):
    """WebSocket이 연결될 때 가짜 유저와 코스를 만들고 세션을 DB에 등록합니다."""
    conn = get_db_conn()
    cur = conn.cursor()

    try:
        # 1. 임시 테스트 유저 생성 (있으면 무시)
        cur.execute("""
            INSERT INTO users (user_id, email, name)
            VALUES ('00000000-0000-0000-0000-000000000001', 'test@test.com', 'Test User')
            ON CONFLICT (user_id) DO NOTHING;
        """)

        # 2. 임시 테스트 코스 생성 (있으면 무시)
        cur.execute("""
            INSERT INTO courses (course_id, user_id, title)
            VALUES ('00000000-0000-0000-0000-000000000002', '00000000-0000-0000-0000-000000000001', 'Test Course')
            ON CONFLICT (course_id) DO NOTHING;
        """)

        # 3. 현재 세션 ID를 DB에 등록 (이게 있어야 transcripts 저장이 가능)
        cur.execute("""
            INSERT INTO sessions (session_id, course_id, session_date, title)
            VALUES (%s, '00000000-0000-0000-0000-000000000002', %s, 'Realtime STT Session')
            ON CONFLICT (session_id) DO NOTHING;
        """, (session_id, datetime.date.today()))

        conn.commit()
        print(f"✅ DB에 세션({session_id}) 초기화 완료")
    except Exception as e:
        print("❌ 세션 초기화 에러:", e)
        conn.rollback()
    finally:
        cur.close()
        conn.close()