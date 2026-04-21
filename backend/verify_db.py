from database import get_db_conn

def verify():
    conn = get_db_conn()
    cur = conn.cursor()

    print("========================================")
    print(" Transcripts 테이블 확인")
    print("========================================")
    cur.execute("""
        SELECT chunk_index, start_time, chunk_text, corrected_text
        FROM transcripts
        ORDER BY start_time
        LIMIT 5;
    """)
    rows = cur.fetchall()
    if not rows:
        print("데이터가 없습니다.")
    for row in rows:
        corrected_preview = (row[3] or "")[:30]
        print(f"인덱스 {row[0]} | {row[1]}초 | 원문: {row[2][:30]}... | 교정: {corrected_preview}...")

    cur.close()
    conn.close()

if __name__ == "__main__":
    verify()