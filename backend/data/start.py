import os, json, torch
from save_chunk_to_db import save_chunk_to_db
from embeded_test import get_embedding
from make_chunks import load_transcripts, make_chunks

if __name__ == "__main__":
    transcripts_dir = "transcripts"

    for file_name in os.listdir(transcripts_dir):
        if not file_name.endswith(".jsonl"):
            continue

        file_path = os.path.join(transcripts_dir, file_name)
        items = load_transcripts(file_path)
        chunks = make_chunks(items, group_size=3)

        print(f"\n파일: {file_name}")
        for chunk in chunks:
            chunk["embedding"] = get_embedding(chunk["chunk_text"])
            save_chunk_to_db(chunk)
            print("저장 완료:", chunk["chunk_id"])
            
        break