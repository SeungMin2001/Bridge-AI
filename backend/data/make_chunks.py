import json, os

def load_transcripts(file_path):
    items = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    return items

def make_chunks(items, group_size=3):
    chunks = []

    for i in range(0, len(items), group_size):
        group = items[i:i + group_size]

        if not group:
            continue

        session_id = group[0]["session_id"]
        start_time = group[0]["start_time"]
        end_time = group[-1]["end_time"]
        chunk_text = " ".join(item["text"] for item in group)

        chunk_index = i // group_size #정수형 인덱스

        chunk = {
            "chunk_index": chunk_index,
            "session_id": session_id,
            "start_time": start_time,
            "end_time": end_time,
            "chunk_text": chunk_text
        }

        chunks.append(chunk)

    return chunks

