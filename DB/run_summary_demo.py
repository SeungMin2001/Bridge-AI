"""
DB/요약_샘플데이터.json을 사용해 요약 저장 흐름을 확인하는 테스트 스크립트.

기존 서비스 코드는 수정하지 않는다. 이 스크립트는 데모 파일과 전사 청크를
DB에 넣고, 현재 백엔드의 /summary API를 호출해 summaries 테이블 저장까지 검증한다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import asyncpg


SAMPLE_PATH = Path(__file__).with_name("요약_샘플데이터.json")


def db_config() -> dict:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "shin"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "1234"),
    }


def load_sample() -> dict:
    with SAMPLE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def display_time(value: datetime) -> str:
    period = "오후" if value.hour >= 12 else "오전"
    hour = value.hour % 12 or 12
    return f"{period} {hour}:{value.minute:02d}"


def flatten_segments(sample: dict) -> list[dict]:
    chunks = []
    for transcription in sample["transcriptions"]:
        for segment in transcription.get("segments", []):
            chunks.append({
                "speakerId": transcription.get("speakerId"),
                "speaker": transcription.get("speaker"),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": segment["text"],
            })
    return sorted(chunks, key=lambda item: item["start"])


def build_voicefile(sample: dict, recording_id: str, started_at: datetime, ended_at: datetime) -> list[dict]:
    transcriptions = []
    for index, item in enumerate(sample["transcriptions"], start=1):
        segments = item.get("segments", [])
        text = "\n".join(segment["text"] for segment in segments)
        transcriptions.append({
            "id": f"demo-transcription-{index}",
            "recordingId": recording_id,
            "speakerId": item.get("speakerId"),
            "speaker": item.get("speaker"),
            "time": display_time(started_at),
            "text": text,
            "segments": [
                {
                    "id": f"demo-segment-{index}-{segment_index}",
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "status": "confirmed",
                }
                for segment_index, segment in enumerate(segments, start=1)
            ],
        })

    return [
        {
            "weekId": "summary-demo-week",
            "weekKey": "summary-demo-week",
            "label": "요약 데모",
            "dateLabel": f"{started_at.year}년 {started_at.month}월 {started_at.day}일",
            "recordings": [
                {
                    "id": recording_id,
                    "recordingId": recording_id,
                    "title": f"{sample['recordingTitlePrefix']} {display_time(started_at)}",
                    "startedAt": started_at.isoformat(),
                    "endedAt": ended_at.isoformat(),
                    "durationText": "00:01:16",
                    "recordingMode": sample.get("recordingMode", "lecture"),
                    "materialIds": [],
                    "materialNames": [],
                    "audioUrl": None,
                    "transcriptions": transcriptions,
                }
            ],
        }
    ]


async def ensure_columns(conn: asyncpg.Connection) -> None:
    await conn.execute("ALTER TABLE courses ADD COLUMN IF NOT EXISTS parent_course_id UUID NULL")
    await conn.execute("ALTER TABLE courses ADD COLUMN IF NOT EXISTS color VARCHAR(50) NULL")
    await conn.execute("ALTER TABLE courses ADD COLUMN IF NOT EXISTS icon VARCHAR(50) NULL")
    await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS file_kind VARCHAR(50) NULL")
    await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS tag VARCHAR(50) NULL")
    await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS icon VARCHAR(50) NULL")
    await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS color VARCHAR(50) NULL")
    await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS session_pdf JSONB NULL")
    await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS session_voicefile JSONB NULL")
    await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary_notes JSONB NULL")
    await conn.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS recording_id TEXT NULL")


async def seed_demo_data(sample: dict) -> dict:
    config = db_config()
    conn = await asyncpg.connect(**config)
    try:
        await ensure_columns(conn)

        course_id = uuid.uuid4()
        session_id = uuid.uuid4()
        recording_id = f"summary-demo-{uuid.uuid4()}"
        started_at = datetime.now().replace(microsecond=0)
        chunks = flatten_segments(sample)
        ended_at = started_at + timedelta(seconds=max(chunk["end"] for chunk in chunks))
        voicefile = build_voicefile(sample, recording_id, started_at, ended_at)

        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO courses
                    (course_id, user_id, parent_course_id, title, type, description, color, icon, created_at)
                VALUES
                    ($1, NULL, NULL, $2, 'folder', $3, '#2563eb', 'folder', $4)
                """,
                course_id,
                sample["courseTitle"],
                "요약 기능 DB 저장 확인용 데모 과목",
                started_at,
            )
            await conn.execute(
                """
                INSERT INTO sessions
                    (session_id, course_id, session_date, title, audio_path, duration_sec, status, created_at,
                     file_kind, tag, icon, color, session_pdf, session_voicefile, summary_notes)
                VALUES
                    ($1, $2, $3, $4, NULL, $5, 'completed', $6,
                     'lecture', '수업', 'article', '#2563eb', '[]'::jsonb, $7::jsonb, '[]'::jsonb)
                """,
                session_id,
                course_id,
                date.today(),
                f"{sample['sessionTitlePrefix']} {display_time(started_at)}",
                int(max(chunk["end"] for chunk in chunks)),
                started_at,
                json.dumps(voicefile, ensure_ascii=False),
            )
            for index, chunk in enumerate(chunks):
                await conn.execute(
                    """
                    INSERT INTO transcripts
                        (transcript_id, session_id, recording_id, chunk_index, start_time, end_time,
                         chunk_text, corrected_text, created_at)
                    VALUES
                        ($1, $2, $3, $4, $5, $6, $7, $7, $8)
                    """,
                    uuid.uuid4(),
                    session_id,
                    recording_id,
                    index,
                    chunk["start"],
                    chunk["end"],
                    chunk["text"],
                    started_at + timedelta(seconds=chunk["end"]),
                )

        return {
            "course_id": str(course_id),
            "session_id": str(session_id),
            "recording_id": recording_id,
            "chunks": chunks,
            "transcriptions": voicefile[0]["recordings"][0]["transcriptions"],
        }
    finally:
        await conn.close()


def post_json(base_url: str, endpoint: str, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{endpoint} 호출 실패: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{base_url} 백엔드에 연결할 수 없습니다: {exc}") from exc


def build_speaker_payloads(seed: dict) -> list[dict]:
    payloads = []
    for transcription in seed["transcriptions"]:
        text = transcription.get("text", "").strip()
        if len(text) < 10:
            continue
        segments = transcription.get("segments", [])
        payloads.append({
            "session_id": seed["session_id"],
            "recording_id": seed["recording_id"],
            "speaker_id": transcription.get("speaker") or transcription.get("speakerId") or "UNKNOWN",
            "speaker_text": text,
            "summary_sentences": 3,
            "source_start_time": segments[0]["start"] if segments else None,
            "source_end_time": segments[-1]["end"] if segments else None,
        })
    return payloads


async def verify_saved_rows(session_id: str) -> dict:
    conn = await asyncpg.connect(**db_config())
    try:
        summary_rows = await conn.fetch(
            """
            SELECT summary_id, recording_id, speaker_id, speaker_summary, session_summary, created_at
            FROM summaries
            WHERE session_id = $1
            ORDER BY created_at DESC
            """,
            uuid.UUID(session_id),
        )
        keyword_count = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM key_sentences ks
            JOIN transcripts t ON t.transcript_id = ks.transcript_id
            WHERE t.session_id = $1
            """,
            uuid.UUID(session_id),
        )
        return {
            "keyword_count": int(keyword_count or 0),
            "summaries": [
                {
                    "summary_id": str(row["summary_id"]),
                    "recording_id": row["recording_id"],
                    "speaker_id": row["speaker_id"],
                    "speaker_summary": row["speaker_summary"],
                    "session_summary": row["session_summary"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                }
                for row in summary_rows
            ],
        }
    finally:
        await conn.close()


async def run(base_url: str, skip_api: bool) -> None:
    sample = load_sample()
    seed = await seed_demo_data(sample)

    print("\n[요약 데모] 샘플 전사 데이터 저장 완료")
    print(f"- course_id: {seed['course_id']}")
    print(f"- session_id: {seed['session_id']}")
    print(f"- recording_id: {seed['recording_id']}")

    if skip_api:
        print("\n--skip-api 옵션으로 요약 API 호출은 건너뜀")
        return

    print("\n[요약 데모] 기존 /summary API 호출 시작")
    keyword_result = post_json(base_url, "/summary/keywords/generate", {
        "session_id": seed["session_id"],
        "recording_id": seed["recording_id"],
        "top_k": 12,
        "window_size": 4,
    })
    print(f"- 키워드 저장: {keyword_result.get('count', 0)}개")

    for payload in build_speaker_payloads(seed):
        result = post_json(base_url, "/summary/speaker/generate", payload)
        print(f"- 화자 요약 저장: {payload['speaker_id']} / {result.get('summary_id')}")

    session_result = post_json(base_url, "/summary/session/generate", {
        "session_id": seed["session_id"],
        "recording_id": seed["recording_id"],
        "summary_sentences": 3,
    })
    print(f"- 세션 요약 저장: {session_result.get('summary_id')}")

    saved = await verify_saved_rows(seed["session_id"])
    print("\n[요약 데모] DB 저장 확인")
    print(f"- key_sentences: {saved['keyword_count']}개")
    print(f"- summaries: {len(saved['summaries'])}개")
    for item in saved["summaries"]:
        title = "세션 요약" if item["session_summary"] else f"화자 요약({item['speaker_id']})"
        summary = item["session_summary"] or item["speaker_summary"] or ""
        print(f"  - {title}: {summary[:120]}")

    print("\n프론트에서 확인할 때는 워크스페이스를 새로고침한 뒤 '요약 데모 파일'을 열고 요약 탭의 'AI 요약'을 보면 됨")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="요약 DB 저장 테스트용 데모 데이터 실행")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="백엔드 API 주소")
    parser.add_argument("--skip-api", action="store_true", help="DB 샘플 데이터만 만들고 요약 API 호출은 생략")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args.base_url.rstrip("/"), args.skip_api))
    except Exception as exc:
        print(f"\n[요약 데모] 실패: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
