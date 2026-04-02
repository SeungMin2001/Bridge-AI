import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import save_transcript_to_db

_segment_counters = {}

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

        print("저장 완료")

        segment_index = _segment_counters.get(session_id, 0)
        _segment_counters[session_id] = segment_index + 1
        await save_transcript_to_db(transcript_data, segment_index)
        print("DB 저장 완료")

    except Exception as e:
        print("save_transcript 에러:", e)