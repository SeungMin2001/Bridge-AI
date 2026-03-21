import psycopg
from pgvector.psycopg import register_vector

def get_conn():
    conn = psycopg.connect(
        "host=100.93.71.4 port=5432 dbname=shin user=shinseungmin"
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