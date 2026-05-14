-- init.sql 테이블/컬럼 상세 설명
-- 신창영 : 각 테이블과 컬럼의 역할을 CREATE TABLE 문에 인라인 주석으로 정리한 스크립트

-- pgvector 익스텐션 : transcripts.embedding 같은 벡터 컬럼을 만들고, 전사문 임베딩 기반 유사도 검색/RAG 검색을 수행하기 위해 사용한다.
CREATE EXTENSION IF NOT EXISTS vector;

-- 테이블명 : 일정 정보 테이블. 전사문에서 AI가 추출한 시험/과제/프로젝트 일정과 사용자가 직접 등록한 캘린더 일정을 저장한다.
CREATE TABLE SCHEDULES
(
    schedule_id       UUID PRIMARY KEY,      -- schedule_id : 일정 레코드의 고유 식별자. 각 일정마다 하나씩 발급되는 UUID 기본키이다.
    session_id        UUID NULL,             -- session_id : 일정이 추출되거나 연결된 워크스페이스 파일/녹음 세션 ID. sessions.session_id와 연결되는 값이다.
    recording_id      TEXT NULL,             -- recording_id : 일정이 추출된 개별 녹음본 ID. session_voicefile JSON 안의 녹음본 id와 같은 값이다.
    transcript_id     UUID NULL,             -- transcript_id : 일정 내용의 근거가 된 전사 청크 ID. 어떤 발화에서 일정이 나왔는지 추적할 때 사용한다.
    title             VARCHAR(255) NOT NULL, -- title : 캘린더에 표시할 일정 제목. 예: 중간고사, 과제 제출, 프로젝트 발표.
    description       TEXT NULL,             -- description : 일정의 상세 설명. 전사문에서 추출한 부가 내용이나 사용자가 입력한 설명을 저장한다.
    event_type        VARCHAR(50) NULL,      -- event_type : 일정 분류값. 시험, 과제, 프로젝트, 발표, 회의 같은 유형을 구분하는 데 사용한다.
    due_date          TIMESTAMP NULL,        -- due_date : 일정 날짜 또는 마감 시각. 캘린더에서 어느 날짜에 보여줄지 결정하는 핵심 값이다.
    status            VARCHAR(30) NULL,      -- status : 일정 처리 상태. 예정, 진행중, 완료, 취소 등 일정의 현재 상태를 표현한다.
    calendar_flag     BOOLEAN NULL,          -- calendar_flag : 캘린더 표시 여부. true이면 일정 페이지에 표시하고, false이면 후보/비표시 일정으로 남길 수 있다.
    source_start_time REAL NULL,             -- source_start_time : 일정 근거 발화의 녹음 시작 시간(초). 참조 링크에서 해당 구간으로 이동할 때 사용한다.
    source_end_time   REAL NULL,             -- source_end_time : 일정 근거 발화의 녹음 종료 시간(초). 참조 범위 표시와 하이라이트에 사용한다.
    source_text       TEXT NULL,             -- source_text : 일정 추출의 근거가 된 원문 전사 텍스트. 사용자가 일정 출처를 확인할 때 사용한다.
    created_at        TIMESTAMP NOT NULL,    -- created_at : 일정 레코드가 처음 생성된 시각. AI 추출 또는 사용자 등록 시점이다.
    updated_at        TIMESTAMP NULL         -- updated_at : 일정 레코드가 마지막으로 수정된 시각. 제목, 날짜, 상태 변경 시 갱신한다.
);

-- 테이블명 : AI 설명 답변 테이블. 사용자가 AI 채팅이나 개념 설명 기능에서 질문했을 때 생성된 답변 본문과 모델 정보를 저장한다.
CREATE TABLE EXPLANATIONS
(
    explanation_id UUID PRIMARY KEY,      -- explanation_id : AI 설명 답변의 고유 ID. explanation_chunks와 연결되는 기본 식별자이다.
    request_id     UUID NULL,             -- request_id : 답변을 발생시킨 개념/질문 요청 ID. concept_requests.request_id와 연결된다.
    answer_text    TEXT NULL,             -- answer_text : AI가 생성한 최종 답변 본문. 사용자가 채팅 화면에서 보는 설명 텍스트이다.
    source_links   TEXT NULL,             -- source_links : 답변에 사용된 출처 링크, 참조 전사 구간, 파일 참조 정보 등을 문자열로 저장하는 컬럼이다.
    model_name     VARCHAR(100) NULL,     -- model_name : 답변 생성에 사용한 LLM 모델명. 예: qwen2.5:1.5b.
    latency_ms     INT NULL,              -- latency_ms : 답변 생성에 걸린 시간(ms). 성능 확인이나 응답 지연 분석에 사용한다.
    created_at     TIMESTAMP NULL         -- created_at : AI 설명 답변이 생성된 시각.
);

-- 테이블명 : 요약 저장 테이블. 녹음 종료 후 생성되는 화자 요약, 세션 전체 요약, 과목 전체 요약을 한 테이블에 저장한다.
CREATE TABLE SUMMARIES
(
    summary_id        UUID PRIMARY KEY,   -- summary_id : 요약 레코드의 고유 ID. 각 화자 요약/세션 요약/과목 요약마다 하나씩 생성된다.
    session_id        UUID NULL,          -- session_id : 요약이 속한 세션/파일 ID. 녹음 파일 단위 요약을 불러올 때 sessions.session_id로 조회한다.
    recording_id      TEXT NULL,          -- recording_id : 요약이 생성된 개별 녹음본 ID. 파일 안의 특정 녹음본 요약을 구분한다.
    course_id         UUID NULL,          -- course_id : 요약이 속한 과목/폴더 ID. 과목 단위 요약이나 폴더 단위 집계에 사용한다.
    transcript_id     UUID NULL,          -- transcript_id : 요약 근거가 특정 전사 청크일 때 연결하는 ID. 현재는 주로 NULL일 수 있고, 세부 청크 요약 확장에 사용할 수 있다.
    speaker_id        TEXT NULL,          -- speaker_id : 화자 ID 또는 표시 라벨. 현재 실제 발화자 분리가 없으면 나, UNKNOWN 같은 값이 들어갈 수 있다.
    speaker_summary   TEXT NULL,          -- speaker_summary : 화자별 전사 텍스트를 요약한 문장. /summary/speaker/generate 결과가 저장된다.
    session_summary   TEXT NULL,          -- session_summary : 한 녹음 세션 전체를 요약한 문장. 키워드와 화자 요약을 바탕으로 생성한 결과가 저장된다.
    course_summary    TEXT NULL,          -- course_summary : 여러 세션을 묶은 과목/폴더 단위 요약 문장. 과목 전체 흐름을 요약할 때 사용한다.
    source_start_time REAL NULL,          -- source_start_time : 요약 근거가 되는 원문 구간의 시작 시간(초). 화자 요약 또는 구간 요약 참조에 사용한다.
    source_end_time   REAL NULL,          -- source_end_time : 요약 근거가 되는 원문 구간의 종료 시간(초). UI에서 참조 범위를 표시할 때 사용한다.
    source_text       TEXT NULL,          -- source_text : 요약에 사용된 원문 전사 또는 중간 요약 텍스트. 요약 근거 확인과 디버깅에 사용한다.
    created_at        TIMESTAMP NOT NULL  -- created_at : 요약이 생성되어 DB에 저장된 시각. 최신 요약을 선택할 때 정렬 기준으로 사용한다.
);

-- 테이블명 : 개념 설명 요청 테이블. 사용자가 전사문/자료에서 특정 단어를 클릭하거나 AI 채팅으로 설명을 요청한 기록을 저장한다.
CREATE TABLE CONCEPT_REQUESTS
(
    request_id      UUID PRIMARY KEY,     -- request_id : 개념 설명 요청의 고유 ID. explanations.request_id와 연결된다.
    user_id         UUID NULL,            -- user_id : 요청을 보낸 사용자 ID. 다중 사용자 기능에서 사용자별 요청을 구분한다.
    session_id      UUID NULL,            -- session_id : 요청이 발생한 세션/파일 ID. 어떤 강의나 녹음에서 나온 질문인지 연결한다.
    clicked_text    VARCHAR(255) NULL,    -- clicked_text : 사용자가 클릭하거나 선택한 원문 텍스트. 예: 미분계수, 중간고사 범위.
    normalized_term VARCHAR(255) NULL,    -- normalized_term : 검색/설명 품질을 위해 정규화한 용어. 띄어쓰기나 표현 차이를 정리한 대표 개념명이다.
    request_time    REAL NULL,            -- request_time : 녹음 또는 전사 기준 요청 발생 시간(초). 해당 발화 위치로 이동할 때 사용할 수 있다.
    request_type    VARCHAR(50) NULL,     -- request_type : 요청 유형. 개념 설명, 질문 답변, 요약, 퀴즈 생성 등 기능 종류를 구분한다.
    status          VARCHAR(50) NULL,     -- status : 요청 처리 상태. pending, processing, done, failed 같은 처리 흐름 추적에 사용한다.
    created_at      TIMESTAMP NULL        -- created_at : 요청 레코드가 생성된 시각.
);

-- 테이블명 : 핵심 문장/키워드 테이블. 전사문에서 TextRank 등으로 추출한 중요 키워드나 핵심 문장을 저장한다.
CREATE TABLE KEY_SENTENCES
(
    key_id        UUID PRIMARY KEY,       -- key_id : 핵심 문장 또는 키워드 레코드의 고유 ID.
    transcript_id UUID NULL,              -- transcript_id : 핵심 문장/키워드가 추출된 원본 전사 청크 ID. transcripts.transcript_id와 연결된다.
    sentence_text TEXT NOT NULL,          -- sentence_text : 추출된 핵심 문장 또는 키워드 텍스트. 현재 요약 기능에서는 키워드 문자열이 저장될 수 있다.
    score         REAL NULL,              -- score : 키워드/문장의 중요도 점수. 높을수록 더 중요하다고 판단된 항목이다.
    rank_order    INT NULL,               -- rank_order : 중요도 순위. 1에 가까울수록 우선순위가 높은 키워드/문장이다.
    created_at    TIMESTAMP NOT NULL      -- created_at : 핵심 문장/키워드가 추출되어 저장된 시각.
);

-- 테이블명 : 전사 청크 테이블. 녹음 중 생성된 실시간 전사 결과를 시간 구간별로 나누어 저장한다.
CREATE TABLE TRANSCRIPTS
(
    transcript_id  UUID PRIMARY KEY,      -- transcript_id : 전사 청크의 고유 ID. RAG 참조, 일정 추출, 키워드 추출의 기준 ID로 사용된다.
    session_id     UUID NULL,             -- session_id : 전사가 속한 세션/파일 ID. sessions.session_id와 연결되어 파일별 전사문을 조회한다.
    recording_id   TEXT NULL,             -- recording_id : 전사 청크가 속한 개별 녹음본 ID. session_voicefile[].id와 같은 값으로 녹음본 단위 삭제/참조에 사용한다.
    chunk_index    INT NULL,              -- chunk_index : 같은 세션 안에서 전사 청크의 순서. 전사문을 시간순으로 재구성할 때 사용한다.
    start_time     REAL NULL,             -- start_time : 해당 전사 청크의 시작 시간(초). 녹음본 참조 링크와 하이라이트에 사용한다.
    end_time       REAL NULL,             -- end_time : 해당 전사 청크의 종료 시간(초). 녹음본 구간 참조 범위를 만들 때 사용한다.
    speaker_id     TEXT NULL,             -- speaker_id : 전사 청크의 화자 ID. 화자 분리 기능이 연결될 경우 사용하며, 현재는 NULL일 수 있다.
    speaker_name   TEXT NULL,             -- speaker_name : 전사 청크의 화자 표시 이름. UI에서 사람 이름으로 보여줄 때 사용한다.
    chunk_text     TEXT NULL,             -- chunk_text : STT가 만든 원본 전사 텍스트. 보정 전 원문을 보존한다.
    corrected_text TEXT NULL,             -- corrected_text : 맞춤법/문맥 보정 후 전사 텍스트. RAG 검색과 화면 표시에서 우선 사용될 수 있다.
    embedding      VECTOR(1024) NULL,     -- embedding : 전사 청크의 벡터 임베딩. 의미 기반 검색과 RAG 유사도 검색에 사용한다.
    created_at     TIMESTAMP NULL         -- created_at : 전사 청크가 DB에 저장된 시각. 녹음본 JSON과 RAG 참조를 맞추는 데 사용할 수 있다.
);

-- 테이블명 : 퀴즈 테이블. 강의 전사문, 개념 요청, 과목 내용을 기반으로 생성된 퀴즈 데이터를 저장한다.
CREATE TABLE QUIZZES
(
    quiz_id         UUID PRIMARY KEY,     -- quiz_id : 퀴즈 결과의 고유 ID.
    user_id         UUID NULL,            -- user_id : 퀴즈를 생성하거나 풀이한 사용자 ID.
    course_id       UUID NULL,            -- course_id : 퀴즈가 연결된 과목/폴더 ID. 과목별 퀴즈 목록 조회에 사용한다.
    request_id      UUID NULL,            -- request_id : 퀴즈 생성을 유발한 요청 ID. concept_requests.request_id와 연결될 수 있다.
    session_id      UUID NULL,            -- session_id : 퀴즈가 연결된 세션/파일 ID. 특정 녹음 내용 기반 퀴즈를 구분한다.
    quiz_data       JSONB NULL,           -- quiz_data : 문항, 선택지, 정답, 해설 등을 담은 JSONB 데이터. 프론트에서 그대로 렌더링할 수 있다.
    total_questions INT NULL,             -- total_questions : 퀴즈 전체 문항 수.
    correct_count   INT NULL,             -- correct_count : 사용자가 맞힌 문항 수. 풀이 결과 저장 시 사용한다.
    created_at      TIMESTAMP NULL        -- created_at : 퀴즈가 생성되거나 결과가 저장된 시각.
);

-- 테이블명 : 과목/폴더 테이블. 워크스페이스 사이드바의 폴더, 과목, 상위/하위 구조를 저장한다.
CREATE TABLE COURSES
(
    course_id        UUID PRIMARY KEY,    -- course_id : 과목 또는 폴더의 고유 ID. sessions.course_id와 연결된다.
    user_id          UUID NULL,           -- user_id : 과목/폴더를 소유한 사용자 ID. 사용자별 워크스페이스 분리에 사용한다.
    parent_course_id UUID NULL,           -- parent_course_id : 상위 과목/폴더 ID. NULL이면 최상위 폴더이고, 값이 있으면 하위 폴더 구조를 만든다.
    title            VARCHAR(255) NULL,   -- title : 과목/폴더 이름. 워크스페이스 사이드바에 표시되는 제목이다.
    type             VARCHAR(50) NULL,    -- type : 과목/폴더의 유형. 폴더, 과목, 그룹 등으로 구분할 때 사용한다.
    description      TEXT NULL,           -- description : 과목/폴더 설명. 사용자가 적는 메모나 과목 소개를 저장할 수 있다.
    color            VARCHAR(50) NULL,    -- color : 프론트엔드에서 폴더/과목을 표시할 색상값. 예: #3b82f6.
    icon             VARCHAR(50) NULL,    -- icon : 프론트엔드에서 폴더/과목을 표시할 아이콘 이름.
    created_at       TIMESTAMP NULL       -- created_at : 과목/폴더가 생성된 시각.
);

-- 테이블명 : 과목별 MergePRAG 메모리 테이블. 과목 단위로 누적된 RAG 메모리 텐서를 저장한다.
CREATE TABLE course_memories
(
    memory_id     UUID PRIMARY KEY,       -- memory_id : 과목 메모리 레코드의 고유 ID.
    course_id     UUID NOT NULL REFERENCES courses (course_id) UNIQUE, -- course_id : 메모리가 연결된 과목/폴더 ID. courses.course_id와 1:1로 연결되도록 UNIQUE 제약이 있다.
    merged_k      BYTEA,                  -- merged_k : MergePRAG에서 사용하는 K 텐서를 직렬화한 BYTEA 데이터.
    merged_v      BYTEA,                  -- merged_v : MergePRAG에서 사용하는 V 텐서를 직렬화한 BYTEA 데이터.
    passage_count INT DEFAULT 0,          -- passage_count : 현재 메모리에 병합된 passage 또는 전사 조각의 개수.
    updated_at    TIMESTAMP NOT NULL      -- updated_at : 과목 메모리가 마지막으로 갱신된 시각.
);

-- 테이블명 : 사용자 테이블. 서비스 계정 또는 로컬 사용자 정보를 저장한다.
CREATE TABLE USERS
(
    user_id    UUID PRIMARY KEY,          -- user_id : 사용자의 고유 ID.
    email      VARCHAR(255) NULL,         -- email : 사용자 이메일 주소. 로그인/식별용으로 사용할 수 있다.
    name       VARCHAR(100) NULL,         -- name : 사용자 표시 이름.
    created_at TIMESTAMP NULL             -- created_at : 사용자 계정 또는 로컬 사용자 레코드가 생성된 시각.
);

-- 테이블명 : AI 설명 답변의 출처 청크 테이블. 답변이 어떤 전사 청크를 근거로 삼았는지 저장한다.
CREATE TABLE EXPLANATION_CHUNKS
(
    explanation_chunk_id UUID PRIMARY KEY, -- explanation_chunk_id : 설명 답변과 전사 청크 연결 레코드의 고유 ID.
    explanation_id       UUID NULL,        -- explanation_id : 연결된 AI 설명 답변 ID. explanations.explanation_id와 연결된다.
    transcript_id        UUID NULL,        -- transcript_id : 답변 근거로 사용된 전사 청크 ID. transcripts.transcript_id와 연결된다.
    similarity_score     REAL NULL,        -- similarity_score : 질문/요청과 전사 청크 사이의 유사도 점수.
    rank_order           INT NULL,         -- rank_order : 답변 생성 시 참조된 청크의 순위. 1에 가까울수록 더 중요한 출처이다.
    quoted_text          TEXT NULL         -- quoted_text : 답변 출처로 보여줄 전사 원문 일부. 참조 카드나 인용문 표시용이다.
);

-- 테이블명 : 세션/파일 테이블. 워크스페이스의 파일 하나와 그 안의 녹음본, 자료, 메모를 대표한다.
CREATE TABLE SESSIONS
(
    session_id        UUID PRIMARY KEY,   -- session_id : 세션/파일의 고유 ID. 전사, 일정, 요약, 녹음본 JSON이 이 ID를 기준으로 연결된다.
    course_id         UUID NULL,          -- course_id : 세션/파일이 속한 과목/폴더 ID. courses.course_id와 연결된다.
    session_date      DATE NULL,          -- session_date : 세션 날짜. 강의 날짜나 파일 생성 날짜를 저장한다.
    title             VARCHAR(255) NULL,  -- title : 세션/파일 제목. 예: 수학1 오후 7:23, 5월 2일 강의.
    audio_path        TEXT NULL,          -- audio_path : 단일 오디오 파일 경로를 저장하기 위한 컬럼. 현재는 session_voicefile JSONB를 더 주로 사용할 수 있다.
    duration_sec      INT NULL,           -- duration_sec : 세션 또는 대표 녹음의 길이(초). 녹음 길이 표시와 정렬에 사용한다.
    status            VARCHAR(50) NULL,   -- status : 세션 상태. recording, completed, archived 같은 처리 상태를 저장할 수 있다.
    created_at        TIMESTAMP NULL,     -- created_at : 세션/파일이 생성된 시각.
    file_kind         VARCHAR(50) NULL,   -- file_kind : 파일 종류. lecture, meeting 같은 값을 저장해 수업 파일과 회의 파일을 구분한다.
    tag               VARCHAR(50) NULL,   -- tag : 파일 태그. 수업, 회의, 프로젝트 등 UI에서 분류 라벨로 표시한다.
    icon              VARCHAR(50) NULL,   -- icon : 파일 표시 아이콘 이름. 예: article, groups_2.
    color             VARCHAR(50) NULL,   -- color : 파일 표시 색상. 프론트엔드 카드/아이콘 색상에 사용한다.
    session_pdf       JSONB NULL,         -- session_pdf : 세션에 첨부된 PDF/강의자료 목록을 담는 JSONB 컬럼. 파일명, 경로, 메타데이터를 저장한다.
    session_voicefile JSONB NULL,         -- session_voicefile : 세션에 저장된 녹음본 목록 JSONB. 각 녹음의 제목, 시작/종료 시각, 전사문, 오디오 경로 등을 저장한다.
    summary_notes     JSONB NULL          -- summary_notes : 사용자 메모나 요약 노트 JSONB. 파일 내부 노트, 수동 요약, 보조 메모를 저장할 수 있다.
);

-- ==========================================
-- Foreign Key 설정 (테스트를 위해 주석 처리)

-- ALTER TABLE COURSES ADD CONSTRAINT FK_COURSES_USERS FOREIGN KEY (user_id) REFERENCES USERS(user_id);
-- ALTER TABLE SESSIONS ADD CONSTRAINT FK_SESSIONS_COURSES FOREIGN KEY (course_id) REFERENCES COURSES(course_id);
-- ALTER TABLE TRANSCRIPTS ADD CONSTRAINT FK_TRANSCRIPTS_SESSIONS FOREIGN KEY (session_id) REFERENCES SESSIONS(session_id);

-- ALTER TABLE CONCEPT_REQUESTS ADD CONSTRAINT FK_CONCEPT_REQUESTS_USERS FOREIGN KEY (user_id) REFERENCES USERS(user_id);
-- ALTER TABLE CONCEPT_REQUESTS ADD CONSTRAINT FK_CONCEPT_REQUESTS_SESSIONS FOREIGN KEY (session_id) REFERENCES SESSIONS(session_id);

-- ALTER TABLE EXPLANATIONS ADD CONSTRAINT FK_EXPLANATIONS_CONCEPT_REQUESTS FOREIGN KEY (request_id) REFERENCES CONCEPT_REQUESTS(request_id);
-- ALTER TABLE EXPLANATION_CHUNKS ADD CONSTRAINT FK_EXPLANATION_CHUNKS_EXPLANATIONS FOREIGN KEY (explanation_id) REFERENCES EXPLANATIONS(explanation_id);
-- ALTER TABLE EXPLANATION_CHUNKS ADD CONSTRAINT FK_EXPLANATION_CHUNKS_TRANSCRIPTS FOREIGN KEY (transcript_id) REFERENCES TRANSCRIPTS(transcript_id);

-- ALTER TABLE QUIZZES ADD CONSTRAINT FK_QUIZZES_USERS FOREIGN KEY (user_id) REFERENCES USERS(user_id);
-- ALTER TABLE QUIZZES ADD CONSTRAINT FK_QUIZZES_COURSES FOREIGN KEY (course_id) REFERENCES COURSES(course_id);
-- ALTER TABLE QUIZZES ADD CONSTRAINT FK_QUIZZES_CONCEPT_REQUESTS FOREIGN KEY (request_id) REFERENCES CONCEPT_REQUESTS(request_id);
-- ALTER TABLE QUIZZES ADD CONSTRAINT FK_QUIZZES_SESSIONS FOREIGN KEY (session_id) REFERENCES SESSIONS(session_id);

-- ALTER TABLE KEY_SENTENCES ADD CONSTRAINT FK_KEY_SENTENCES_TRANSCRIPTS FOREIGN KEY (transcript_id) REFERENCES TRANSCRIPTS(transcript_id);

-- ALTER TABLE SCHEDULES ADD CONSTRAINT FK_SCHEDULES_SESSIONS FOREIGN KEY (session_id) REFERENCES SESSIONS(session_id);
-- ALTER TABLE SCHEDULES ADD CONSTRAINT FK_SCHEDULES_TRANSCRIPTS FOREIGN KEY (transcript_id) REFERENCES TRANSCRIPTS(transcript_id);

-- ALTER TABLE SUMMARIES ADD CONSTRAINT FK_SUMMARIES_SESSIONS FOREIGN KEY (session_id) REFERENCES SESSIONS(session_id);
-- ALTER TABLE SUMMARIES ADD CONSTRAINT FK_SUMMARIES_COURSES FOREIGN KEY (course_id) REFERENCES COURSES(course_id);
-- ALTER TABLE SUMMARIES ADD CONSTRAINT FK_SUMMARIES_TRANSCRIPTS FOREIGN KEY (transcript_id) REFERENCES TRANSCRIPTS(transcript_id);
