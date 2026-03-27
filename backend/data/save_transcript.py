import os
import json
from database import get_db_conn

async def save_transcript(transcript_data):
    try:
        print("save_transcript 호출됨")
        print("받은 데이터:", transcript_data)

        os.makedirs("data/transcripts", exist_ok=True)

        session_id = transcript_data["session_id"]
        file_path = f"data/transcripts/{session_id}.jsonl"

        print("저장 경로:", os.path.abspath(file_path))

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(transcript_data, ensure_ascii=False) + "\n")

        # DB 저장 로직
        conn = get_db_conn()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO transcripts (
                session_id, segment_index, start_time, end_time, original_text
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                transcript_data["session_id"],
                transcript_data["segment_index"],
                transcript_data["start_time"],
                transcript_data["end_time"],
                transcript_data["raw_text"]
            )
        )
        conn.commit()
        cur.close()
        conn.close()

        print(f"save_transcripts 저장 완료 (index: {transcript_data['segment_index']})")
    except Exception as e:
        print("save_transcript 에러:", e)