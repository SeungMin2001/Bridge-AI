-- 우측 AI 바 RAG/일정 확인용 통합 샘플 데이터
--
-- 실행:
--   psql -h localhost -p 5432 -d shin -f DB/sample_ai_sidebar_combined_test.sql

-- 일정 테이블이 예전 스키마로 생성된 DB를 위한 보정
ALTER TABLE schedules
ADD COLUMN IF NOT EXISTS calendar_flag BOOLEAN DEFAULT FALSE;

-- 기존 샘플 데이터 삭제
DELETE FROM schedules
WHERE session_id IN (
    '22222222-2222-2222-2222-222222222222',
    '55555555-5555-5555-5555-555555555555'
);

DELETE FROM transcripts
WHERE session_id IN (
    '22222222-2222-2222-2222-222222222222',
    '55555555-5555-5555-5555-555555555555'
);

DELETE FROM sessions
WHERE session_id IN (
    '22222222-2222-2222-2222-222222222222',
    '55555555-5555-5555-5555-555555555555'
);

DELETE FROM courses
WHERE course_id IN (
    '11111111-1111-1111-1111-111111111111',
    '44444444-4444-4444-4444-444444444444'
);

-- 샘플 데이터 추가

-- ---------------------------------------------------------------------
-- 운영체제 샘플
-- ---------------------------------------------------------------------

INSERT INTO courses (
    course_id,
    title,
    type,
    description,
    color,
    icon,
    created_at
)
VALUES (
    '11111111-1111-1111-1111-111111111111',
    'AI 테스트 강의',
    'folder',
    '우측 AI 바 RAG 확인용 샘플 데이터',
    '#3b82f6',
    'folder',
    NOW()
)
ON CONFLICT (course_id) DO UPDATE
SET title = EXCLUDED.title,
    description = EXCLUDED.description,
    color = EXCLUDED.color,
    icon = EXCLUDED.icon;

INSERT INTO sessions (
    session_id,
    course_id,
    session_date,
    title,
    status,
    created_at,
    file_kind,
    tag,
    icon,
    color,
    session_pdf,
    session_voicefile,
    summary_notes
)
VALUES (
    '22222222-2222-2222-2222-222222222222',
    '11111111-1111-1111-1111-111111111111',
    CURRENT_DATE,
    '운영체제 샘플 전사',
    'created',
    NOW(),
    'lecture',
    '수업',
    'article',
    '#3b82f6',
    '[]'::jsonb,
    jsonb_build_array(
        jsonb_build_object(
            'id', 'week-sample-os',
            'weekKey', 'sample-os',
            'label', '1주차',
            'dateLabel', '2026. 5. 2. 토요일',
            'recordings', jsonb_build_array(
                jsonb_build_object(
                    'id', 'recording-sample-os',
                    'title', '운영체제 샘플 녹음',
                    'endedAt', NOW(),
                    'durationText', '00:00:48',
                    'recordingMode', 'lecture',
                    'materialIds', '[]'::jsonb,
                    'materialNames', '[]'::jsonb,
                    'audioUrl', NULL,
                    'transcriptions', jsonb_build_array(
                        jsonb_build_object(
                            'time', '0:00~0:08',
                            'speakerId', NULL,
                            'speaker', NULL,
                            'text', '운영체제는 컴퓨터 하드웨어와 사용자 프로그램 사이에서 자원을 관리하는 핵심 소프트웨어입니다.',
                            'segments', jsonb_build_array(
                                jsonb_build_object(
                                    'id', 'sample-segment-os-1',
                                    'text', '운영체제는 컴퓨터 하드웨어와 사용자 프로그램 사이에서 자원을 관리하는 핵심 소프트웨어입니다.',
                                    'status', 'confirmed'
                                )
                            )
                        ),
                        jsonb_build_object(
                            'time', '0:08~0:16',
                            'speakerId', NULL,
                            'speaker', NULL,
                            'text', '프로세스는 실행 중인 프로그램을 의미하고, 운영체제는 각 프로세스에 CPU 시간과 메모리를 배분합니다.',
                            'segments', jsonb_build_array(
                                jsonb_build_object(
                                    'id', 'sample-segment-os-2',
                                    'text', '프로세스는 실행 중인 프로그램을 의미하고, 운영체제는 각 프로세스에 CPU 시간과 메모리를 배분합니다.',
                                    'status', 'confirmed'
                                )
                            )
                        ),
                        jsonb_build_object(
                            'time', '0:16~0:24',
                            'speakerId', NULL,
                            'speaker', NULL,
                            'text', '스레드는 하나의 프로세스 안에서 실행되는 작업 흐름이며 같은 프로세스의 메모리 공간을 공유합니다.',
                            'segments', jsonb_build_array(
                                jsonb_build_object(
                                    'id', 'sample-segment-os-3',
                                    'text', '스레드는 하나의 프로세스 안에서 실행되는 작업 흐름이며 같은 프로세스의 메모리 공간을 공유합니다.',
                                    'status', 'confirmed'
                                )
                            )
                        ),
                        jsonb_build_object(
                            'time', '0:24~0:32',
                            'speakerId', NULL,
                            'speaker', NULL,
                            'text', '가상 메모리는 실제 메모리보다 큰 공간을 쓰는 것처럼 보이게 하는 운영체제의 메모리 관리 기법입니다.',
                            'segments', jsonb_build_array(
                                jsonb_build_object(
                                    'id', 'sample-segment-os-4',
                                    'text', '가상 메모리는 실제 메모리보다 큰 공간을 쓰는 것처럼 보이게 하는 운영체제의 메모리 관리 기법입니다.',
                                    'status', 'confirmed'
                                )
                            )
                        ),
                        jsonb_build_object(
                            'time', '0:32~0:40',
                            'speakerId', NULL,
                            'speaker', NULL,
                            'text', '운영체제 보강 수업은 2026년 5월 6일 오전 10시에 진행하고, 발표자료 초안은 5월 8일 오후 11시 55분까지 제출하세요.',
                            'segments', jsonb_build_array(
                                jsonb_build_object(
                                    'id', 'sample-segment-os-5',
                                    'text', '운영체제 보강 수업은 2026년 5월 6일 오전 10시에 진행하고, 발표자료 초안은 5월 8일 오후 11시 55분까지 제출하세요.',
                                    'status', 'confirmed'
                                )
                            )
                        ),
                        jsonb_build_object(
                            'time', '0:40~0:48',
                            'speakerId', NULL,
                            'speaker', NULL,
                            'text', '운영체제 중간고사는 2026년 5월 15일 오전 9시에 진행하며 시험 범위는 프로세스와 메모리 관리입니다.',
                            'segments', jsonb_build_array(
                                jsonb_build_object(
                                    'id', 'sample-segment-os-6',
                                    'text', '운영체제 중간고사는 2026년 5월 15일 오전 9시에 진행하며 시험 범위는 프로세스와 메모리 관리입니다.',
                                    'status', 'confirmed'
                                )
                            )
                        )
                    )
                )
            )
        )
    ),
    '[]'::jsonb
)
ON CONFLICT (session_id) DO UPDATE
SET title = EXCLUDED.title,
    course_id = EXCLUDED.course_id,
    session_voicefile = EXCLUDED.session_voicefile,
    color = EXCLUDED.color,
    icon = EXCLUDED.icon;

DELETE FROM transcripts
WHERE session_id = '22222222-2222-2222-2222-222222222222';

INSERT INTO transcripts (
    transcript_id,
    session_id,
    chunk_index,
    start_time,
    end_time,
    chunk_text,
    corrected_text,
    created_at
)
VALUES
(
    '33333333-3333-3333-3333-333333333331',
    '22222222-2222-2222-2222-222222222222',
    0,
    0.0,
    8.0,
    '운영체제는 컴퓨터 하드웨어와 사용자 프로그램 사이에서 자원을 관리하는 핵심 소프트웨어입니다.',
    '운영체제는 컴퓨터 하드웨어와 사용자 프로그램 사이에서 자원을 관리하는 핵심 소프트웨어입니다.',
    NOW()
),
(
    '33333333-3333-3333-3333-333333333332',
    '22222222-2222-2222-2222-222222222222',
    1,
    8.0,
    16.0,
    '프로세스는 실행 중인 프로그램을 의미하고, 운영체제는 각 프로세스에 CPU 시간과 메모리를 배분합니다.',
    '프로세스는 실행 중인 프로그램을 의미하고, 운영체제는 각 프로세스에 CPU 시간과 메모리를 배분합니다.',
    NOW()
),
(
    '33333333-3333-3333-3333-333333333333',
    '22222222-2222-2222-2222-222222222222',
    2,
    16.0,
    24.0,
    '스레드는 하나의 프로세스 안에서 실행되는 작업 흐름이며 같은 프로세스의 메모리 공간을 공유합니다.',
    '스레드는 하나의 프로세스 안에서 실행되는 작업 흐름이며 같은 프로세스의 메모리 공간을 공유합니다.',
    NOW()
),
(
    '33333333-3333-3333-3333-333333333334',
    '22222222-2222-2222-2222-222222222222',
    3,
    24.0,
    32.0,
    '가상 메모리는 실제 메모리보다 큰 공간을 쓰는 것처럼 보이게 하는 운영체제의 메모리 관리 기법입니다.',
    '가상 메모리는 실제 메모리보다 큰 공간을 쓰는 것처럼 보이게 하는 운영체제의 메모리 관리 기법입니다.',
    NOW()
),
(
    '33333333-3333-3333-3333-333333333335',
    '22222222-2222-2222-2222-222222222222',
    4,
    32.0,
    40.0,
    '운영체제 보강 수업은 2026년 5월 6일 오전 10시에 진행하고, 발표자료 초안은 5월 8일 오후 11시 55분까지 제출하세요.',
    '운영체제 보강 수업은 2026년 5월 6일 오전 10시에 진행하고, 발표자료 초안은 5월 8일 오후 11시 55분까지 제출하세요.',
    NOW()
),
(
    '33333333-3333-3333-3333-333333333336',
    '22222222-2222-2222-2222-222222222222',
    5,
    40.0,
    48.0,
    '운영체제 중간고사는 2026년 5월 15일 오전 9시에 진행하며 시험 범위는 프로세스와 메모리 관리입니다.',
    '운영체제 중간고사는 2026년 5월 15일 오전 9시에 진행하며 시험 범위는 프로세스와 메모리 관리입니다.',
    NOW()
);

-- ---------------------------------------------------------------------
-- 네트워크 샘플
-- ---------------------------------------------------------------------

INSERT INTO courses (
    course_id,
    title,
    type,
    description,
    color,
    icon,
    created_at
)
VALUES (
    '44444444-4444-4444-4444-444444444444',
    '네트워크 테스트 강의',
    'folder',
    '우측 AI 바 RAG 확인용 네트워크 샘플 데이터',
    '#10b981',
    'folder',
    NOW()
)
ON CONFLICT (course_id) DO UPDATE
SET title = EXCLUDED.title,
    description = EXCLUDED.description,
    color = EXCLUDED.color,
    icon = EXCLUDED.icon;

INSERT INTO sessions (
    session_id,
    course_id,
    session_date,
    title,
    status,
    created_at,
    file_kind,
    tag,
    icon,
    color,
    session_pdf,
    session_voicefile,
    summary_notes
)
VALUES (
    '55555555-5555-5555-5555-555555555555',
    '44444444-4444-4444-4444-444444444444',
    CURRENT_DATE,
    '네트워크 샘플 전사',
    'created',
    NOW(),
    'lecture',
    '수업',
    'article',
    '#10b981',
    '[]'::jsonb,
    jsonb_build_array(
        jsonb_build_object(
            'id', 'week-sample-network',
            'weekKey', 'sample-network',
            'label', '2주차',
            'dateLabel', '2026. 5. 2. 토요일',
            'recordings', jsonb_build_array(
                jsonb_build_object(
                    'id', 'recording-sample-network',
                    'title', '네트워크 샘플 녹음',
                    'endedAt', NOW(),
                    'durationText', '00:00:52',
                    'recordingMode', 'lecture',
                    'materialIds', '[]'::jsonb,
                    'materialNames', '[]'::jsonb,
                    'audioUrl', NULL,
                    'transcriptions', jsonb_build_array(
                        jsonb_build_object(
                            'time', '0:00~0:09',
                            'speakerId', NULL,
                            'speaker', NULL,
                            'text', '네트워크는 여러 장치가 데이터를 주고받을 수 있도록 연결된 통신 구조입니다.',
                            'segments', jsonb_build_array(
                                jsonb_build_object(
                                    'id', 'sample-segment-network-1',
                                    'text', '네트워크는 여러 장치가 데이터를 주고받을 수 있도록 연결된 통신 구조입니다.',
                                    'status', 'confirmed'
                                )
                            )
                        ),
                        jsonb_build_object(
                            'time', '0:09~0:18',
                            'speakerId', NULL,
                            'speaker', NULL,
                            'text', 'IP 주소는 네트워크 안에서 장치를 식별하는 번호이고, 라우터는 목적지까지 패킷을 전달합니다.',
                            'segments', jsonb_build_array(
                                jsonb_build_object(
                                    'id', 'sample-segment-network-2',
                                    'text', 'IP 주소는 네트워크 안에서 장치를 식별하는 번호이고, 라우터는 목적지까지 패킷을 전달합니다.',
                                    'status', 'confirmed'
                                )
                            )
                        ),
                        jsonb_build_object(
                            'time', '0:18~0:27',
                            'speakerId', NULL,
                            'speaker', NULL,
                            'text', 'TCP는 데이터가 순서대로 정확히 도착하도록 확인하고 재전송을 수행하는 전송 계층 프로토콜입니다.',
                            'segments', jsonb_build_array(
                                jsonb_build_object(
                                    'id', 'sample-segment-network-3',
                                    'text', 'TCP는 데이터가 순서대로 정확히 도착하도록 확인하고 재전송을 수행하는 전송 계층 프로토콜입니다.',
                                    'status', 'confirmed'
                                )
                            )
                        ),
                        jsonb_build_object(
                            'time', '0:27~0:36',
                            'speakerId', NULL,
                            'speaker', NULL,
                            'text', 'DNS는 사람이 읽기 쉬운 도메인 이름을 실제 서버의 IP 주소로 바꿔주는 서비스입니다.',
                            'segments', jsonb_build_array(
                                jsonb_build_object(
                                    'id', 'sample-segment-network-4',
                                    'text', 'DNS는 사람이 읽기 쉬운 도메인 이름을 실제 서버의 IP 주소로 바꿔주는 서비스입니다.',
                                    'status', 'confirmed'
                                )
                            )
                        ),
                        jsonb_build_object(
                            'time', '0:36~0:44',
                            'speakerId', NULL,
                            'speaker', NULL,
                            'text', '네트워크 과제는 2026년 5월 7일 오후 11시 59분까지 제출하고 라우팅 실습은 5월 9일 오후 2시에 진행합니다.',
                            'segments', jsonb_build_array(
                                jsonb_build_object(
                                    'id', 'sample-segment-network-5',
                                    'text', '네트워크 과제는 2026년 5월 7일 오후 11시 59분까지 제출하고 라우팅 실습은 5월 9일 오후 2시에 진행합니다.',
                                    'status', 'confirmed'
                                )
                            )
                        ),
                        jsonb_build_object(
                            'time', '0:44~0:52',
                            'speakerId', NULL,
                            'speaker', NULL,
                            'text', '다음 주 네트워크 보강 안내는 취소되었으니 같은 알림은 무시해도 됩니다.',
                            'segments', jsonb_build_array(
                                jsonb_build_object(
                                    'id', 'sample-segment-network-6',
                                    'text', '다음 주 네트워크 보강 안내는 취소되었으니 같은 알림은 무시해도 됩니다.',
                                    'status', 'confirmed'
                                )
                            )
                        )
                    )
                )
            )
        )
    ),
    '[]'::jsonb
)
ON CONFLICT (session_id) DO UPDATE
SET title = EXCLUDED.title,
    course_id = EXCLUDED.course_id,
    session_voicefile = EXCLUDED.session_voicefile,
    color = EXCLUDED.color,
    icon = EXCLUDED.icon;

DELETE FROM transcripts
WHERE session_id = '55555555-5555-5555-5555-555555555555';

INSERT INTO transcripts (
    transcript_id,
    session_id,
    chunk_index,
    start_time,
    end_time,
    chunk_text,
    corrected_text,
    created_at
)
VALUES
(
    '66666666-6666-6666-6666-666666666661',
    '55555555-5555-5555-5555-555555555555',
    0,
    0.0,
    9.0,
    '네트워크는 여러 장치가 데이터를 주고받을 수 있도록 연결된 통신 구조입니다.',
    '네트워크는 여러 장치가 데이터를 주고받을 수 있도록 연결된 통신 구조입니다.',
    NOW()
),
(
    '66666666-6666-6666-6666-666666666662',
    '55555555-5555-5555-5555-555555555555',
    1,
    9.0,
    18.0,
    'IP 주소는 네트워크 안에서 장치를 식별하는 번호이고, 라우터는 목적지까지 패킷을 전달합니다.',
    'IP 주소는 네트워크 안에서 장치를 식별하는 번호이고, 라우터는 목적지까지 패킷을 전달합니다.',
    NOW()
),
(
    '66666666-6666-6666-6666-666666666663',
    '55555555-5555-5555-5555-555555555555',
    2,
    18.0,
    27.0,
    'TCP는 데이터가 순서대로 정확히 도착하도록 확인하고 재전송을 수행하는 전송 계층 프로토콜입니다.',
    'TCP는 데이터가 순서대로 정확히 도착하도록 확인하고 재전송을 수행하는 전송 계층 프로토콜입니다.',
    NOW()
),
(
    '66666666-6666-6666-6666-666666666664',
    '55555555-5555-5555-5555-555555555555',
    3,
    27.0,
    36.0,
    'DNS는 사람이 읽기 쉬운 도메인 이름을 실제 서버의 IP 주소로 바꿔주는 서비스입니다.',
    'DNS는 사람이 읽기 쉬운 도메인 이름을 실제 서버의 IP 주소로 바꿔주는 서비스입니다.',
    NOW()
),
(
    '66666666-6666-6666-6666-666666666665',
    '55555555-5555-5555-5555-555555555555',
    4,
    36.0,
    44.0,
    '네트워크 과제는 2026년 5월 7일 오후 11시 59분까지 제출하고 라우팅 실습은 5월 9일 오후 2시에 진행합니다.',
    '네트워크 과제는 2026년 5월 7일 오후 11시 59분까지 제출하고 라우팅 실습은 5월 9일 오후 2시에 진행합니다.',
    NOW()
),
(
    '66666666-6666-6666-6666-666666666666',
    '55555555-5555-5555-5555-555555555555',
    5,
    44.0,
    52.0,
    '다음 주 네트워크 보강 안내는 취소되었으니 같은 알림은 무시해도 됩니다.',
    '다음 주 네트워크 보강 안내는 취소되었으니 같은 알림은 무시해도 됩니다.',
    NOW()
);

-- ---------------------------------------------------------------------
-- 일정 샘플
-- ---------------------------------------------------------------------

INSERT INTO schedules (
    schedule_id,
    session_id,
    transcript_id,
    title,
    description,
    event_type,
    due_date,
    status,
    calendar_flag,
    source_start_time,
    source_end_time,
    source_text,
    created_at,
    updated_at
)
VALUES
(
    '77777777-7777-7777-7777-777777777771',
    '22222222-2222-2222-2222-222222222222',
    '33333333-3333-3333-3333-333333333335',
    '운영체제 보강 수업',
    '운영체제 보강 수업 일정입니다.',
    '수업',
    '2026-05-06 10:00:00',
    'confirmed',
    TRUE,
    32.0,
    40.0,
    '운영체제 보강 수업은 2026년 5월 6일 오전 10시에 진행하고, 발표자료 초안은 5월 8일 오후 11시 55분까지 제출하세요.',
    NOW(),
    NOW()
),
(
    '77777777-7777-7777-7777-777777777772',
    '22222222-2222-2222-2222-222222222222',
    '33333333-3333-3333-3333-333333333335',
    '발표자료 초안 제출',
    '발표자료 초안 제출 마감 일정입니다.',
    '과제',
    '2026-05-08 23:55:00',
    'pending',
    FALSE,
    32.0,
    40.0,
    '발표자료 초안은 5월 8일 오후 11시 55분까지 제출하세요.',
    NOW(),
    NOW()
),
(
    '77777777-7777-7777-7777-777777777773',
    '22222222-2222-2222-2222-222222222222',
    '33333333-3333-3333-3333-333333333336',
    '운영체제 중간고사',
    '프로세스와 메모리 관리 범위의 운영체제 중간고사입니다.',
    '시험',
    '2026-05-15 09:00:00',
    'confirmed',
    TRUE,
    40.0,
    48.0,
    '운영체제 중간고사는 2026년 5월 15일 오전 9시에 진행하며 시험 범위는 프로세스와 메모리 관리입니다.',
    NOW(),
    NOW()
),
(
    '88888888-8888-8888-8888-888888888881',
    '55555555-5555-5555-5555-555555555555',
    '66666666-6666-6666-6666-666666666665',
    '네트워크 과제 제출',
    '네트워크 과제 제출 마감 일정입니다.',
    '과제',
    '2026-05-07 23:59:00',
    'pending',
    FALSE,
    36.0,
    44.0,
    '네트워크 과제는 2026년 5월 7일 오후 11시 59분까지 제출하세요.',
    NOW(),
    NOW()
),
(
    '88888888-8888-8888-8888-888888888882',
    '55555555-5555-5555-5555-555555555555',
    '66666666-6666-6666-6666-666666666665',
    '라우팅 실습',
    '네트워크 라우팅 실습 일정입니다.',
    '프로젝트',
    '2026-05-09 14:00:00',
    'confirmed',
    TRUE,
    36.0,
    44.0,
    '라우팅 실습은 5월 9일 오후 2시에 진행합니다.',
    NOW(),
    NOW()
),
(
    '88888888-8888-8888-8888-888888888883',
    '55555555-5555-5555-5555-555555555555',
    '66666666-6666-6666-6666-666666666666',
    '네트워크 보강 안내',
    '취소된 보강 안내라서 ignored 상태로 남겨둔 샘플입니다.',
    '수업',
    '2026-05-10 10:00:00',
    'ignored',
    FALSE,
    44.0,
    52.0,
    '다음 주 네트워크 보강 안내는 취소되었으니 같은 알림은 무시해도 됩니다.',
    NOW(),
    NOW()
);
