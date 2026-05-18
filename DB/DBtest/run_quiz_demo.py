"""
DB/DBtest/퀴즈_샘플데이터.json을 사용해 transcript_id 기반 퀴즈 소스 데이터를 넣는 테스트 스크립트.

기본 실행은 샘플 세션에 녹음본 2개와 강의자료 2개만 저장한다.
--generate-api 옵션을 주면 각 녹음본의 transcript_id 묶음으로 퀴즈 생성 API까지 확인한다.
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

SAMPLE_PATH = Path(__file__).with_name("퀴즈_샘플데이터.json")


def load_asyncpg():
    try:
        import asyncpg
    except ModuleNotFoundError as exc:
        raise RuntimeError("asyncpg가 필요합니다. 백엔드 가상환경에서 실행하거나 asyncpg를 설치해주세요.") from exc
    return asyncpg


def db_config() -> dict:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME") or os.getenv("PGDATABASE", "shin"),
        "user": os.getenv("DB_USER") or os.getenv("PGUSER") or os.getenv("USER", "postgres"),
        "password": os.getenv("DB_PASSWORD") or os.getenv("PGPASSWORD", ""),
    }


def load_sample() -> dict:
    with SAMPLE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def display_time(value: datetime) -> str:
    period = "오후" if value.hour >= 12 else "오전"
    hour = value.hour % 12 or 12
    return f"{period} {hour}:{value.minute:02d}"


def display_date(value: datetime) -> str:
    return f"{value.year}년 {value.month}월 {value.day}일"


def recording_duration(recording: dict) -> int:
    ends = [
        float(segment["end"])
        for transcription in recording.get("transcriptions", [])
        for segment in transcription.get("segments", [])
    ]
    return int(max(ends) if ends else 0)


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


def material_resources(sample: dict, week_id: str, started_at: datetime) -> list[dict]:
    return [
        {
            "weekId": week_id,
            "weekKey": week_id,
            "label": "퀴즈 데모",
            "dateLabel": display_date(started_at),
            "materials": [
                {
                    "id": material["id"],
                    "name": material["name"],
                    "type": material.get("type", "application/pdf"),
                    "size": material.get("size", 102400),
                    "uploadedAt": started_at.isoformat(),
                }
                for material in sample.get("materials", [])
            ],
        }
    ]


def build_recording_resources(sample: dict, week_id: str, started_at: datetime) -> tuple[list[dict], dict]:
    recording_lookup = {}
    recordings = []

    for recording_index, recording in enumerate(sample.get("recordings", []), start=1):
        key = recording["key"]
        recording_started_at = started_at + timedelta(minutes=(recording_index - 1) * 10)
        recording_ended_at = recording_started_at + timedelta(seconds=recording_duration(recording))
        recording_id = f"quiz-demo-{key}-{uuid.uuid4()}"
        transcriptions = []
        transcript_ids = []

        for transcription_index, transcription in enumerate(recording.get("transcriptions", []), start=1):
            segments = []
            for segment_index, segment in enumerate(transcription.get("segments", []), start=1):
                transcript_id = str(uuid.uuid4())
                transcript_ids.append(transcript_id)
                segments.append({
                    "id": f"{key}-segment-{transcription_index}-{segment_index}",
                    "transcript_id": transcript_id,
                    "transcriptId": transcript_id,
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "status": "confirmed",
                })

            transcriptions.append({
                "id": f"{key}-transcription-{transcription_index}",
                "recordingId": recording_id,
                "speakerId": transcription.get("speakerId"),
                "speaker": transcription.get("speaker"),
                "time": display_time(recording_started_at),
                "text": "\n".join(segment["text"] for segment in segments),
                "segments": segments,
            })

        recording_item = {
            "id": recording_id,
            "recordingId": recording_id,
            "title": recording["title"],
            "startedAt": recording_started_at.isoformat(),
            "endedAt": recording_ended_at.isoformat(),
            "durationText": recording.get("durationText", "00:01:00"),
            "recordingMode": sample.get("recordingMode", "lecture"),
            "materialIds": recording.get("materialIds", []),
            "materialNames": recording.get("materialNames", []),
            "audioUrl": None,
            "transcriptions": transcriptions,
        }
        recordings.append(recording_item)
        recording_lookup[key] = {
            "recording_id": recording_id,
            "title": recording["title"],
            "transcript_ids": transcript_ids,
            "answers": recording.get("quiz", {}).get("submitAnswers", {}),
            "num_questions": int(recording.get("quiz", {}).get("numQuestions", 5)),
            "type_counts": recording.get("quiz", {}).get("typeCounts"),
            "resource": recording_item,
        }

    return [
        {
            "weekId": week_id,
            "weekKey": week_id,
            "label": "퀴즈 데모",
            "dateLabel": display_date(started_at),
            "recordings": recordings,
        }
    ], recording_lookup


def transcript_rows(recording_lookup: dict) -> list[dict]:
    rows = []
    chunk_index = 0
    for recording in recording_lookup.values():
        recording_id = recording["recording_id"]
        for transcription in recording["resource"]["transcriptions"]:
            for segment in transcription.get("segments", []):
                rows.append({
                    "transcript_id": segment["transcript_id"],
                    "recording_id": recording_id,
                    "chunk_index": chunk_index,
                    "start_time": float(segment["start"]),
                    "end_time": float(segment["end"]),
                    "text": segment["text"],
                })
                chunk_index += 1
    return rows


async def seed_demo_data(sample: dict) -> dict:
    asyncpg = load_asyncpg()
    conn = await asyncpg.connect(**db_config())
    try:
        await ensure_columns(conn)

        course_id = uuid.uuid4()
        session_id = uuid.uuid4()
        started_at = datetime.now().replace(microsecond=0)
        week_id = "quiz-demo-week"
        session_pdf = material_resources(sample, week_id, started_at)
        session_voicefile, recording_lookup = build_recording_resources(sample, week_id, started_at)
        rows = transcript_rows(recording_lookup)

        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO courses
                    (course_id, user_id, parent_course_id, title, type, description, color, icon, created_at)
                VALUES
                    ($1, NULL, NULL, $2, 'folder', $3, '#7c3aed', 'folder', $4)
                """,
                course_id,
                sample["courseTitle"],
                "퀴즈 기능 transcript_id 기반 생성 확인용 데모 과목",
                started_at,
            )
            await conn.execute(
                """
                INSERT INTO sessions
                    (session_id, course_id, session_date, title, audio_path, duration_sec, status, created_at,
                     file_kind, tag, icon, color, session_pdf, session_voicefile, summary_notes)
                VALUES
                    ($1, $2, $3, $4, NULL, $5, 'completed', $6,
                     'lecture', '수업', 'article', '#7c3aed', $7::jsonb, $8::jsonb, '[]'::jsonb)
                """,
                session_id,
                course_id,
                date.today(),
                f"{sample['sessionTitlePrefix']} {display_time(started_at)}",
                max((row["end_time"] for row in rows), default=0),
                started_at,
                json.dumps(session_pdf, ensure_ascii=False),
                json.dumps(session_voicefile, ensure_ascii=False),
            )
            for row in rows:
                await conn.execute(
                    """
                    INSERT INTO transcripts
                        (transcript_id, session_id, recording_id, chunk_index, start_time, end_time,
                         chunk_text, corrected_text, created_at)
                    VALUES
                        ($1, $2, $3, $4, $5, $6, $7, $7, $8)
                    """,
                    uuid.UUID(row["transcript_id"]),
                    session_id,
                    row["recording_id"],
                    row["chunk_index"],
                    row["start_time"],
                    row["end_time"],
                    row["text"],
                    started_at + timedelta(seconds=row["end_time"]),
                )

        return {
            "course_id": str(course_id),
            "session_id": str(session_id),
            "recordings": recording_lookup,
            "quiz_files": sample.get("quizFiles", []),
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


async def verify_quizzes(session_id: str) -> list[dict]:
    asyncpg = load_asyncpg()
    conn = await asyncpg.connect(**db_config())
    try:
        rows = await conn.fetch(
            """
            SELECT quiz_id, total_questions, correct_count, created_at
            FROM quizzes
            WHERE session_id = $1
            ORDER BY created_at DESC
            """,
            uuid.UUID(session_id),
        )
        return [
            {
                "quiz_id": str(row["quiz_id"]),
                "total_questions": row["total_questions"],
                "correct_count": row["correct_count"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
            for row in rows
        ]
    finally:
        await conn.close()


async def run(base_url: str, generate_api: bool, submit_answers: bool) -> None:
    sample = load_sample()
    seed = await seed_demo_data(sample)

    print("\n[퀴즈 데모] 샘플 세션/강의자료/녹음본 저장 완료")
    print(f"- course_id: {seed['course_id']}")
    print(f"- session_id: {seed['session_id']}")
    for key, recording in seed["recordings"].items():
        print(f"- {key}: {recording['title']} / transcript_id {len(recording['transcript_ids'])}개")

    if not generate_api:
        print("\n퀴즈 API 호출은 실행하지 않음")
        print("프론트에서 좌측 폴더 탭의 샘플 강의자료/녹음본을 선택한 뒤 퀴즈 탭에서 직접 생성하면 됨")
        return

    print("\n[퀴즈 데모] /quiz/generate/transcripts API 호출 시작")
    for quiz_file in seed["quiz_files"]:
        recording_key = quiz_file["recordingKey"]
        recording = seed["recordings"][recording_key]
        result = post_json(base_url, "/quiz/generate/transcripts", {
            "session_id": seed["session_id"],
            "transcript_ids": recording["transcript_ids"],
            "num_questions": recording["num_questions"],
            "type_counts": recording["type_counts"],
        })
        quiz_id = result.get("quiz_id")
        print(f"- {quiz_file['title']} 생성: quiz_id={quiz_id}, 문항={result.get('total_questions')}")

        if not submit_answers:
            continue

        submit_result = post_json(base_url, f"/quiz/{quiz_id}/submit", {
            "answers": recording["answers"]
        })
        print(
            f"  채점: {submit_result.get('correct_count')}/"
            f"{submit_result.get('total_questions')} ({submit_result.get('score')}점)"
        )

    saved = await verify_quizzes(seed["session_id"])
    print("\n[퀴즈 데모] DB 저장 확인")
    print(f"- quizzes: {len(saved)}개")
    for item in saved:
        print(
            f"  - {item['quiz_id']}: {item['correct_count']}/"
            f"{item['total_questions']} created_at={item['created_at']}"
        )

    print("\n프론트에서 확인할 때는 워크스페이스를 새로고침한 뒤 '퀴즈 데모 파일'을 열고 폴더 탭에서 샘플 녹음본/강의자료를 클릭하면 됨")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="transcript_id 기반 퀴즈 소스 샘플 데이터 실행")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="백엔드 API 주소")
    parser.add_argument("--generate-api", action="store_true", help="샘플 저장 후 퀴즈 생성 API까지 호출")
    parser.add_argument("--submit", action="store_true", help="퀴즈 생성 후 샘플 답안 제출/채점까지 실행")
    parser.add_argument("--skip-api", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-submit", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        generate_api = args.generate_api and not args.skip_api
        submit_answers = args.submit and not args.skip_submit
        asyncio.run(run(args.base_url.rstrip("/"), generate_api, submit_answers))
    except Exception as exc:
        print(f"\n[퀴즈 데모] 실패: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
