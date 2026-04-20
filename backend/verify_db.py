from database import get_db_conn

def verify():
    conn = get_db_conn()
    cur = conn.cursor()

    print("========================================")
    print(" Transcripts (STT 세그먼트) 테이블 확인")
    print("========================================")
    cur.execute("SELECT segment_index, start_time, original_text FROM transcripts ORDER BY start_time LIMIT 5;")
    rows = cur.fetchall()
    if not rows:
        print("데이터가 없습니다.")
    for row in rows:
        print(f"인덱스 {row[0]} | {row[1]}초 | 텍스트: {row[2]}")

    print("\n========================================")
    print(" Chunks (임베딩 덩어리) 테이블 확인")
    print("========================================")
    cur.execute("SELECT chunk_index, start_time, chunk_text FROM chunks ORDER BY start_time LIMIT 5;")
    rows = cur.fetchall()
    if not rows:
        print("데이터가 없습니다.")
    for row in rows:
        print(f"청크 {row[0]} | {row[1]}초 | 텍스트: {row[2][:30]}...")

    cur.close()
    conn.close()

if __name__ == "__main__":
    verify()