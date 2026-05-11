import os
import json
import sys
import uuid
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import save_transcript_to_db

_segment_counters = {}

async def save_transcript(transcript_data):
    try:
        # print("save_transcript 호출됨")
        # print("받은 데이터:", transcript_data)

        os.makedirs("data/transcripts", exist_ok=True)

        session_id = transcript_data["session_id"]
        recording_id = transcript_data.get("recording_id") or ""
        # 신창영 : JSONL, transcripts 테이블, RAG metadata가 같은 전사 chunk를 가리키도록 transcript_id를 선생성
        transcript_data["transcript_id"] = transcript_data.get("transcript_id") or str(uuid.uuid4())
        # 신창영 : 녹음본 endedAt과 매칭하기 위해 전사 저장 시각을 함께 기록
        transcript_data["created_at"] = transcript_data.get("created_at") or datetime.now()
        file_path = f"data/transcripts/{session_id}.jsonl"
        transcript_data["recording_id"] = recording_id

        # print("저장 경로:", os.path.abspath(file_path))

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(transcript_data, ensure_ascii=False, default=str) + "\n")

        # print("저장 완료")

        segment_index = _segment_counters.get(session_id, 0)
        _segment_counters[session_id] = segment_index + 1
        saved_transcript = await save_transcript_to_db(transcript_data, segment_index)
        # print("DB 저장 완료")
        # 신창영 : 호출부가 RAG metadata에 transcript_id와 chunk_index를 넣을 수 있도록 저장 결과를 반환
        return {
            **saved_transcript,
            "chunk_index": segment_index,
        }

    except Exception as e:
        print("save_transcript 에러:", e)
