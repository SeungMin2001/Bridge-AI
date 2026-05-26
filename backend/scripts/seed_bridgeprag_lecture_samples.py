#!/usr/bin/env python3
"""Seed BridgePRAG-style lecture transcript samples into the local workspace DB.

The service does not store QA pairs directly. Instead, transcript chunks act as
retrieved passages and user questions act as QA queries. This script inserts
lecture-like passages with clear multi-fact content and prints suggested test
questions that should retrieve those passages.
"""

from __future__ import annotations

import json
import sys
import uuid
import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import psycopg2
from psycopg2.extras import Json, execute_batch, register_uuid


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from db_config import psycopg2_config  # noqa: E402


register_uuid()

FOLDER_TITLE = "BridgePRAG 샘플 강의"
FOLDER_DESCRIPTION = "BridgePRAG 서비스 테스트용 강의형 샘플 데이터"
SAMPLE_PREFIX = "bridgeprag-lecture-sample"
WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]


@dataclass(frozen=True)
class LectureSample:
    slug: str
    session_title: str
    recording_title: str
    course_title: str
    chunks: tuple[str, ...]
    questions: tuple[str, ...]


SAMPLES: tuple[LectureSample, ...] = (
    LectureSample(
        slug="os-process-thread",
        course_title="운영체제",
        session_title="운영체제 3주차 - 프로세스와 스레드",
        recording_title="운영체제 3주차 프로세스와 스레드 녹음본",
        chunks=(
            "오늘은 프로세스와 스레드의 차이를 정리하겠습니다. 프로세스는 실행 중인 프로그램의 단위이고, 독립적인 주소 공간과 자원을 가집니다. 반면 스레드는 하나의 프로세스 안에서 실행되는 흐름이기 때문에 코드 영역과 힙 영역을 공유하지만, 각 스레드는 자기만의 스택과 레지스터 상태를 가집니다.",
            "문맥 교환 비용은 프로세스보다 스레드가 일반적으로 더 작습니다. 프로세스 문맥 교환은 주소 공간과 페이지 테이블 전환까지 포함될 수 있지만, 스레드 문맥 교환은 같은 주소 공간 안에서 실행 흐름만 바뀌는 경우가 많기 때문입니다. 그래서 웹 서버처럼 동시에 많은 요청을 처리하는 경우 스레드를 활용하면 응답성이 좋아질 수 있습니다.",
            "다만 스레드가 메모리를 공유한다는 장점은 동시에 위험이 됩니다. 여러 스레드가 같은 변수에 동시에 접근하면 실행 순서에 따라 결과가 달라지는 경쟁 상태가 발생할 수 있습니다. 이 문제를 줄이기 위해 임계 구역에는 뮤텍스나 세마포어 같은 동기화 기법을 적용합니다.",
            "임계 구역 문제를 해결할 때는 세 가지 조건을 확인해야 합니다. 첫째, 한 번에 하나의 스레드만 들어가는 상호 배제 조건이 필요합니다. 둘째, 들어갈 수 있는 스레드가 있다면 선택이 무한히 지연되지 않는 진행 조건이 필요합니다. 셋째, 특정 스레드가 계속 밀리지 않도록 대기 횟수를 제한하는 한정 대기 조건이 필요합니다.",
        ),
        questions=(
            "프로세스와 스레드의 차이를 강의에서는 어떻게 설명했어?",
            "스레드에서 경쟁 상태가 생기는 이유와 해결 방법은 뭐야?",
            "임계 구역 문제의 세 가지 조건은 무엇이야?",
        ),
    ),
    LectureSample(
        slug="db-index-transaction",
        course_title="데이터베이스",
        session_title="데이터베이스 5주차 - 인덱스와 트랜잭션",
        recording_title="데이터베이스 5주차 인덱스와 트랜잭션 녹음본",
        chunks=(
            "이번 시간에는 인덱스와 트랜잭션을 함께 보겠습니다. 인덱스는 책의 색인처럼 원하는 행을 빨리 찾기 위한 보조 자료구조입니다. 데이터베이스에서는 B+트리 인덱스가 자주 사용되며, 루트에서 리프 노드까지 이동하면서 검색 범위를 점점 좁혀 갑니다.",
            "복합 인덱스에서는 컬럼 순서가 중요합니다. 예를 들어 학생 테이블에 학과와 학번 순서로 인덱스를 만들면, 학과 조건으로 먼저 좁힌 뒤 학번을 찾는 질의에는 효과적입니다. 하지만 학번만 단독으로 찾는 질의에서는 인덱스의 앞쪽 컬럼을 건너뛰기 어렵기 때문에 효과가 줄어들 수 있습니다.",
            "트랜잭션은 데이터베이스에서 하나의 논리적 작업 단위입니다. 원자성은 작업이 모두 성공하거나 모두 취소되어야 한다는 의미이고, 일관성은 트랜잭션 전후에 제약 조건이 깨지지 않아야 한다는 의미입니다. 고립성은 동시에 실행되는 트랜잭션이 서로 간섭하지 않도록 보장하는 성질이고, 지속성은 커밋된 결과가 장애 이후에도 남아야 한다는 성질입니다.",
            "장애 복구를 위해 로그 선행 기록, 즉 Write-Ahead Logging을 사용합니다. 데이터 페이지를 디스크에 쓰기 전에 변경 내용에 대한 로그를 먼저 안정적인 저장소에 기록합니다. 이렇게 하면 시스템이 중간에 멈춰도 로그를 기준으로 redo와 undo를 수행해 커밋된 결과는 살리고 미완료 작업은 되돌릴 수 있습니다.",
        ),
        questions=(
            "B+트리 인덱스는 강의에서 어떤 방식으로 설명됐어?",
            "복합 인덱스에서 컬럼 순서가 중요한 이유는 뭐야?",
            "트랜잭션의 ACID 속성을 설명해줘.",
        ),
    ),
    LectureSample(
        slug="rag-memory-bridge",
        course_title="인공지능 응용",
        session_title="RAG 7주차 - 검색 증강 생성과 메모리",
        recording_title="RAG 7주차 검색 증강 생성과 메모리 녹음본",
        chunks=(
            "검색 증강 생성은 대규모 언어모델이 모든 지식을 파라미터 안에 외우고 있다고 가정하지 않습니다. 사용자의 질문이 들어오면 관련 문서를 먼저 검색하고, 검색된 문맥을 모델에게 함께 제공해 답변을 생성합니다. 이 방식은 최신 정보나 수업 자료처럼 모델이 사전학습에서 보지 못한 지식을 다룰 때 유용합니다.",
            "검색된 passage의 수를 늘리면 답을 찾을 가능성은 높아질 수 있지만, 동시에 모델이 읽어야 하는 문맥 길이가 길어지고 노이즈 passage가 섞일 수 있습니다. 따라서 top-k를 무조건 크게 잡는 것보다 질문과 직접 관련된 passage를 고르고, 필요하면 재랭킹을 통해 근거 품질을 높이는 것이 중요합니다.",
            "Parametric RAG 계열 접근은 외부 지식을 긴 텍스트로만 넣지 않고, 모델 내부에서 사용할 수 있는 메모리 표현으로 바꾸려는 방향입니다. 예를 들어 passage를 K/V 메모리로 변환하면 어텐션 계산 과정에서 해당 정보를 참조하게 만들 수 있습니다. 이때 여러 passage의 정보가 충돌하지 않도록 병합 방식도 함께 고려해야 합니다.",
            "BridgePRAG의 핵심 아이디어는 passage만 따로 인코딩하지 않고 질문과 passage를 함께 묶어 메모리를 만드는 것입니다. 같은 passage라도 질문이 요구하는 정보가 다르면 중요하게 봐야 할 부분이 달라질 수 있습니다. 그래서 질문-문맥 pair를 기반으로 K/V 메모리를 만들면, 현재 질문 의도에 맞는 정보가 더 잘 반영될 수 있습니다.",
        ),
        questions=(
            "RAG에서 passage 수를 늘릴 때 생기는 장점과 단점은 뭐야?",
            "Parametric RAG는 외부 지식을 어떻게 활용하려는 접근이야?",
            "BridgePRAG가 passage만 인코딩하지 않고 질문과 passage를 함께 쓰는 이유는 뭐야?",
        ),
    ),
)


def deterministic_uuid(name: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{SAMPLE_PREFIX}:{name}")


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes = total // 60
    remain = total % 60
    return f"{minutes}:{remain:02d}"


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    remain = total % 60
    return f"{hours:02d}:{minutes:02d}:{remain:02d}"


def week_label(value: date) -> str:
    return f"{value.year}. {value.month}. {value.day}. {WEEKDAYS[value.weekday()]}"


def ensure_schema(cur) -> None:
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS file_kind VARCHAR(50) NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS tag VARCHAR(50) NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS icon VARCHAR(50) NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS color VARCHAR(50) NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS session_pdf JSONB NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS session_voicefile JSONB NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary_notes JSONB NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS recording_id TEXT NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS chunk_index INTEGER NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS chunk_text TEXT NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS speaker_id TEXT NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS speaker_name TEXT NULL")


def upsert_course(cur, title: str, *, parent_id: uuid.UUID | None = None) -> uuid.UUID:
    course_id = deterministic_uuid(f"course:{title}")
    cur.execute(
        """
        INSERT INTO courses (
            course_id, user_id, parent_course_id, title, type, description, color, icon, created_at
        )
        VALUES (%s, NULL, %s, %s, %s, %s, '#2563eb', %s, %s)
        ON CONFLICT (course_id) DO UPDATE
        SET parent_course_id = EXCLUDED.parent_course_id,
            title = EXCLUDED.title,
            type = EXCLUDED.type,
            description = EXCLUDED.description,
            color = EXCLUDED.color,
            icon = EXCLUDED.icon
        """,
        (
            course_id,
            parent_id,
            title,
            "folder" if parent_id is None else "course",
            FOLDER_DESCRIPTION if parent_id is None else f"{title} 샘플 강의",
            "folder" if parent_id is None else "school",
            datetime.now(),
        ),
    )
    return course_id


def build_rows(sample: LectureSample, session_id: uuid.UUID, recording_id: str, started_at: datetime):
    transcriptions: list[dict] = []
    db_rows: list[tuple] = []
    cursor = 0.0

    for index, text in enumerate(sample.chunks):
        duration = max(18.0, min(70.0, len(text) / 7.0))
        start_time = round(cursor, 2)
        end_time = round(cursor + duration, 2)
        cursor = end_time + 2.0
        transcript_id = deterministic_uuid(f"transcript:{sample.slug}:{index}")
        created_at = started_at + timedelta(seconds=start_time)
        speaker_id = "professor"
        speaker_name = "교수자"

        transcriptions.append({
            "id": f"{recording_id}-{index}",
            "recordingId": recording_id,
            "time": format_time(start_time),
            "speakerId": speaker_id,
            "speaker": speaker_name,
            "text": text,
            "segments": [{
                "id": str(transcript_id),
                "text": text,
                "status": "confirmed",
                "transcript_id": str(transcript_id),
                "transcriptId": str(transcript_id),
                "chunk_index": index,
                "start": start_time,
                "end": end_time,
                "start_time": start_time,
                "end_time": end_time,
            }],
        })
        db_rows.append((
            transcript_id,
            session_id,
            recording_id,
            index,
            start_time,
            end_time,
            speaker_id,
            speaker_name,
            text,
            text,
            created_at,
        ))

    return transcriptions, db_rows, int(round(max(cursor - 2.0, 0)))


def session_voicefile(sample: LectureSample, recording_id: str, transcriptions: list[dict], duration_sec: int):
    today = date.today()
    now = datetime.now().replace(microsecond=0)
    week_id = f"week-{today.isoformat()}"
    return [{
        "weekId": week_id,
        "id": week_id,
        "weekKey": today.isoformat(),
        "label": "샘플",
        "dateLabel": week_label(today),
        "expanded": True,
        "materialFolderExpanded": True,
        "recordingFolderExpanded": True,
        "recordings": [{
            "id": recording_id,
            "recordingId": recording_id,
            "title": sample.recording_title,
            "startedAt": now.isoformat(),
            "endedAt": (now + timedelta(seconds=duration_sec)).isoformat(),
            "durationText": format_duration(duration_sec),
            "recordingMode": "lecture",
            "diarizationEnabled": False,
            "materialIds": [],
            "materialNames": [],
            "audioUrl": None,
            "transcriptions": transcriptions,
        }],
    }]


def write_transcript_jsonl(session_id: uuid.UUID, rows: list[tuple]) -> None:
    output_dir = BACKEND_ROOT / "data" / "transcripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{session_id}.jsonl"
    lines = []
    for row in rows:
        lines.append({
            "transcript_id": str(row[0]),
            "session_id": str(row[1]),
            "recording_id": row[2],
            "start_time": row[4],
            "end_time": row[5],
            "speaker_id": row[6],
            "speaker_name": row[7],
            "raw_text": row[8],
            "text": row[9],
            "created_at": row[10].isoformat(),
        })
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in lines) + "\n", encoding="utf-8")


def index_transcripts(rows: list[tuple], *, course_title: str, session_title: str, session_date: date) -> int:
    """Insert seeded chunks into the same vector store used by service RAG."""
    from rag_search import add_document

    indexed = 0
    for row in rows:
        (
            transcript_id,
            session_id,
            recording_id,
            chunk_index,
            start_time,
            end_time,
            _speaker_id,
            _speaker_name,
            _raw_text,
            text,
            created_at,
        ) = row
        add_document(
            text,
            {
                "source_type": "transcript",
                "transcript_id": str(transcript_id),
                "session_id": str(session_id),
                "recording_id": str(recording_id),
                "course_title": course_title,
                "session_title": session_title,
                "session_date": str(session_date),
                "chunk_index": int(chunk_index),
                "start_time": float(start_time),
                "end_time": float(end_time),
                "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
            },
        )
        indexed += 1
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed BridgePRAG lecture-like workspace samples.")
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Only write PostgreSQL rows. Skip BGE/PGVector indexing.",
    )
    args = parser.parse_args()

    inserted = []
    pending_index_rows: list[tuple[LectureSample, list[tuple]]] = []
    config = psycopg2_config()
    with psycopg2.connect(**config) as conn:
        with conn.cursor() as cur:
            ensure_schema(cur)
            root_course_id = upsert_course(cur, FOLDER_TITLE)
            for sample in SAMPLES:
                course_id = upsert_course(cur, sample.course_title, parent_id=root_course_id)
                session_id = deterministic_uuid(f"session:{sample.slug}")
                recording_id = f"{SAMPLE_PREFIX}-{sample.slug}"
                started_at = datetime.now().replace(microsecond=0)
                transcriptions, transcript_rows, duration_sec = build_rows(sample, session_id, recording_id, started_at)

                cur.execute(
                    """
                    INSERT INTO sessions (
                        session_id, course_id, session_date, title, duration_sec, status, created_at,
                        file_kind, tag, icon, color, session_pdf, session_voicefile, summary_notes
                    )
                    VALUES (%s, %s, %s, %s, %s, 'created', %s,
                            'lecture', '수업', 'article', '#2563eb', '[]'::jsonb, %s::jsonb, '[]'::jsonb)
                    ON CONFLICT (session_id) DO UPDATE
                    SET course_id = EXCLUDED.course_id,
                        session_date = EXCLUDED.session_date,
                        title = EXCLUDED.title,
                        duration_sec = EXCLUDED.duration_sec,
                        status = EXCLUDED.status,
                        file_kind = EXCLUDED.file_kind,
                        tag = EXCLUDED.tag,
                        icon = EXCLUDED.icon,
                        color = EXCLUDED.color,
                        session_voicefile = EXCLUDED.session_voicefile
                    """,
                    (
                        session_id,
                        course_id,
                        date.today(),
                        sample.session_title,
                        duration_sec,
                        datetime.now(),
                        Json(session_voicefile(sample, recording_id, transcriptions, duration_sec)),
                    ),
                )
                cur.execute(
                    "DELETE FROM transcripts WHERE session_id = %s AND recording_id = %s",
                    (session_id, recording_id),
                )
                execute_batch(
                    cur,
                    """
                    INSERT INTO transcripts (
                        transcript_id, session_id, recording_id, chunk_index, start_time, end_time,
                        speaker_id, speaker_name, chunk_text, corrected_text, created_at
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (transcript_id) DO UPDATE
                    SET recording_id = EXCLUDED.recording_id,
                        chunk_index = EXCLUDED.chunk_index,
                        start_time = EXCLUDED.start_time,
                        end_time = EXCLUDED.end_time,
                        speaker_id = EXCLUDED.speaker_id,
                        speaker_name = EXCLUDED.speaker_name,
                        chunk_text = EXCLUDED.chunk_text,
                        corrected_text = EXCLUDED.corrected_text,
                        created_at = EXCLUDED.created_at
                    """,
                    transcript_rows,
                    page_size=50,
                )
                write_transcript_jsonl(session_id, transcript_rows)
                pending_index_rows.append((sample, transcript_rows))
                inserted.append({
                    "course": sample.course_title,
                    "session": sample.session_title,
                    "session_id": str(session_id),
                    "recording": sample.recording_title,
                    "recording_id": recording_id,
                    "chunks": len(sample.chunks),
                    "sample_questions": list(sample.questions),
                })
        conn.commit()

    indexed_chunks = 0
    if not args.skip_index:
        for sample, rows in pending_index_rows:
            indexed_chunks += index_transcripts(
                rows,
                course_title=sample.course_title,
                session_title=sample.session_title,
                session_date=date.today(),
            )

    print(json.dumps({
        "ok": True,
        "database": config.get("database"),
        "folder": FOLDER_TITLE,
        "indexed_chunks": indexed_chunks,
        "index_skipped": bool(args.skip_index),
        "inserted": inserted,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
