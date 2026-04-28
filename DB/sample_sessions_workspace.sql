-- SESSIONS 테이블 워크스페이스 파일 테스트 데이터
-- 실행 예시:
-- psql -d shin -f DB/sample_sessions_workspace.sql

INSERT INTO sessions (
    session_id,
    course_id,
    session_date,
    title,
    audio_path,
    duration_sec,
    status,
    created_at,
    file_kind,
    tag,
    icon,
    color,
    session_pdf,
    summary_notes
)
VALUES
(
    '11111111-1111-4111-8111-111111111111',
    NULL,
    CURRENT_DATE,
    '운영체제 강의 파일',
    NULL,
    3600,
    'created',
    NOW(),
    'lecture',
    '수업',
    'article',
    '#3b82f6',
    '[
      {
        "id": "material-os-01",
        "name": "운영체제_4주차.pdf",
        "size": 245760,
        "type": "application/pdf",
        "uploadedAt": "2026-04-28T21:20:00+09:00",
        "url": "/uploads/sample/os-week4.pdf"
      }
    ]'::jsonb,
    '[
      {
        "id": "note-os-01",
        "text": "프로세스는 실행 중인 프로그램이며, 스레드는 프로세스 내부의 실행 흐름이다.",
        "source": "운영체제 전사 00:12:30",
        "time": "오후 09:20"
      }
    ]'::jsonb
),
(
    '22222222-2222-4222-8222-222222222222',
    NULL,
    CURRENT_DATE,
    '팀 프로젝트 회의 파일',
    NULL,
    1800,
    'created',
    NOW(),
    'meeting',
    '회의',
    'groups_2',
    '#ec4899',
    '[]'::jsonb,
    '[
      {
        "id": "note-meeting-01",
        "text": "다음 회의 전까지 발표자료 초안을 준비하고 역할 분담을 확정한다.",
        "source": "회의 전사 00:08:15",
        "time": "오후 09:25"
      }
    ]'::jsonb
)
ON CONFLICT (session_id) DO UPDATE SET
    session_date = EXCLUDED.session_date,
    title = EXCLUDED.title,
    audio_path = EXCLUDED.audio_path,
    duration_sec = EXCLUDED.duration_sec,
    status = EXCLUDED.status,
    created_at = EXCLUDED.created_at,
    file_kind = EXCLUDED.file_kind,
    tag = EXCLUDED.tag,
    icon = EXCLUDED.icon,
    color = EXCLUDED.color,
    session_pdf = EXCLUDED.session_pdf,
    summary_notes = EXCLUDED.summary_notes;
