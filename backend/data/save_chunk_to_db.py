import os
import psycopg
from pgvector.psycopg import register_vector
import logging

# 로거 설정
logger = logging.getLogger(__name__)

def get_db_connection():
    """
    동기식 데이터베이스 연결을 생성하고 반환합니다.
    Docker 호환성을 위해 환경 변수를 사용합니다.
    """
    # Docker 환경이 아닐 경우 기존 IP를 기본값으로 사용합니다.
    db_host = os.getenv("DB_HOST", "100.104.164.84")
    db_port = os.getenv("DB_PORT", "5432")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "1234")
    db_name = os.getenv("DB_NAME", "rag")

    conn_str = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password} connect_timeout=5"

    try:
        conn = psycopg.connect(conn_str)
        register_vector(conn)
        return conn
    except Exception as e:
        logger.error(f"데이터베이스 연결 실패: {e}")
        return None

def save_chunk_to_db(chunk):
    """
    청크(Chunk) 데이터를 DB에 저장합니다.
    """
    conn = get_db_connection()

    # DB 연결에 실패했을 경우 함수 종료
    if conn is None:
        logger.error("DB 연결에 실패하여 chunk를 저장할 수 없습니다.")
        return

    try:
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
    except Exception as e:
        logger.error(f"데이터 저장 실패: {e}")
        conn.rollback()  # 에러 발생 시 롤백
    finally:
        # 정상 처리되든 에러가 나든 커넥션 항상 닫기
        if 'cur' in locals():
            cur.close()
        if conn is not None:
            conn.close()