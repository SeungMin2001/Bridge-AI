import psycopg
from pgvector.psycopg import register_vector
from database import get_db_conn

# def get_conn():
#     conn = psycopg.connect(
#         "host=100.104.164.84 port=5432 dbname=rag user=postgres password=1234 connect_timeout=5"
#     )
#     register_vector(conn)
#     return conn

def save_chunk_to_db(chunk):
    conn = get_db_conn()
    cur = conn.cursor()

    try:
        # init.sql의 chunks 테이블 컬럼명에 맞게 쿼리 수정
        cur.execute(
            """
            INSERT INTO chunks (
                session_id,
                chunk_index,
                start_time,
                end_time,
                chunk_text,
                embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                chunk["session_id"],
                chunk["chunk_index"],
                chunk["start_time"],
                chunk["end_time"],
                chunk["chunk_text"],
                chunk["embedding"],
            )
        )
        conn.commit()
    except Exception as e:
        print("save_chunk_to_db 에러:", e)
        conn.rollback()
    finally:
        cur.close()
        conn.close()