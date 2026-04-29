import os
import psycopg
from pgvector.psycopg import register_vector
import logging

# 로거 설정
# logger = logging.getLogger(__name__)
# def get_db_connection():
#     """
#     동기식 데이터베이스 연결을 생성하고 반환합니다.
#     Docker 호환성을 위해 환경 변수를 사용합니다.
#     """
#     # Docker 환경이 아닐 경우 기존 IP를 기본값으로 사용합니다.
#     db_host = os.getenv("DB_HOST", "100.104.164.84")
#     db_port = os.getenv("DB_PORT", "5432")
#     db_user = os.getenv("DB_USER", "postgres")
#     db_password = os.getenv("DB_PASSWORD", "1234")
#     db_name = os.getenv("DB_NAME", "rag")
#     conn_str = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password} connect_timeout=5"
#     try:
#         conn = psycopg.connect(conn_str)
#         register_vector(conn)
#         return conn
#     except Exception as e:
#         logger.error(f"데이터베이스 연결 실패: {e}")
#         return None

def get_conn():
    conn = psycopg.connect(
        #host=100.104.164.84
        "host=db port=5432 dbname=rag user=postgres password=1234 connect_timeout=5"
    )
    register_vector(conn)
    return conn

def save_chunk_to_db(chunk):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO chunks (
            chunk_id,
            session_id,
            start_time,
            end_time,
            chunk_text,
            embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (chunk_id) DO NOTHING
        """,
        (
            chunk["chunk_id"],
            chunk["session_id"],
            chunk["start_time"],
            chunk["end_time"],
            chunk["chunk_text"],
            chunk["embedding"],
        )
    )

    conn.commit()
    cur.close()
    conn.close()