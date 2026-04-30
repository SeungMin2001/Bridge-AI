-- pgvector 익스텐션 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 과목별 MergePRAG 메모리 (K,V 텐서를 직렬화하여 저장)
CREATE TABLE course_memories
(
    memory_id     UUID PRIMARY KEY,
    course_id     UUID      NOT NULL REFERENCES courses (course_id) UNIQUE,
    merged_k      BYTEA,
    merged_v      BYTEA,
    passage_count INT DEFAULT 0,
    updated_at    TIMESTAMP NOT NULL
);

-- 테이블 생성
CREATE TABLE SCHEDULES
(
    schedule_id       UUID PRIMARY KEY,
    session_id        UUID NULL,
    transcript_id     UUID NULL,
    title             VARCHAR(255) NOT NULL, -- 제목
    description       TEXT NULL, -- 설명
    event_type        VARCHAR(50) NULL, -- 시험, 과제, 프로젝트
    due_date          TIMESTAMP NULL, -- 마감일
    status            VARCHAR(30) NULL, -- 예정, 진행중, 완료
    calendar_flag     BOOLEAN NULL, -- 캘린더 표시 여부
    source_start_time REAL NULL,
    source_end_time   REAL NULL,
    source_text       TEXT NULL,
    created_at        TIMESTAMP    NOT NULL,
    updated_at        TIMESTAMP NULL
);

CREATE TABLE EXPLANATIONS
(
    explanation_id UUID PRIMARY KEY,
    request_id     UUID NULL,
    answer_text    TEXT NULL,
    source_links   TEXT NULL,
    model_name     VARCHAR(100) NULL,
    latency_ms     INT NULL,
    created_at     TIMESTAMP NULL
);

CREATE TABLE SUMMARIES
(
    summary_id        UUID PRIMARY KEY,
    session_id        UUID NULL,
    course_id         UUID NULL,
    transcript_id     UUID NULL,
    speaker_id        TEXT NULL,
    speaker_summary   TEXT NULL,
    session_summary   TEXT NULL,
    course_summary    TEXT NULL,
    source_start_time REAL NULL,
    source_end_time   REAL NULL,
    source_text       TEXT NULL,
    created_at        TIMESTAMP NOT NULL
);

CREATE TABLE CONCEPT_REQUESTS
(
    request_id      UUID PRIMARY KEY,
    user_id         UUID NULL,
    session_id      UUID NULL,
    clicked_text    VARCHAR(255) NULL,
    normalized_term VARCHAR(255) NULL,
    request_time    REAL NULL,
    request_type    VARCHAR(50) NULL,
    status          VARCHAR(50) NULL,
    created_at      TIMESTAMP NULL
);

CREATE TABLE KEY_SENTENCES
(
    key_id        UUID PRIMARY KEY,
    transcript_id UUID NULL,
    sentence_text TEXT      NOT NULL,
    score         REAL NULL,
    rank_order    INT NULL,
    created_at    TIMESTAMP NOT NULL
);

CREATE TABLE TRANSCRIPTS
(
    transcript_id  UUID PRIMARY KEY,
    session_id     UUID NULL,
    chunk_index    INT NULL,
    start_time     REAL NULL,
    end_time       REAL NULL,
    speaker_id     TEXT NULL,
    speaker_name   TEXT NULL,
    chunk_text     TEXT NULL,
    corrected_text TEXT NULL,
    embedding      VECTOR(1024) NULL,
    created_at     TIMESTAMP NULL
);

CREATE TABLE QUIZZES
(
    quiz_id         UUID PRIMARY KEY,
    user_id         UUID NULL,
    course_id       UUID NULL,
    request_id      UUID NULL,
    session_id      UUID NULL,
    quiz_data       JSONB NULL,
    total_questions INT NULL,
    correct_count   INT NULL,
    created_at      TIMESTAMP NULL
);

CREATE TABLE COURSES
(
    course_id        UUID PRIMARY KEY,
    user_id          UUID NULL,
    parent_course_id UUID NULL,
    title            VARCHAR(255) NULL,
    type             VARCHAR(50) NULL,
    description      TEXT NULL,
    color            VARCHAR(50) NULL,
    icon             VARCHAR(50) NULL,
    created_at       TIMESTAMP NULL
);

CREATE TABLE USERS
(
    user_id    UUID PRIMARY KEY,
    email      VARCHAR(255) NULL,
    name       VARCHAR(100) NULL,
    created_at TIMESTAMP NULL
);

CREATE TABLE EXPLANATION_CHUNKS
(
    explanation_chunk_id UUID PRIMARY KEY,
    explanation_id       UUID NULL,
    transcript_id        UUID NULL,
    similarity_score     REAL NULL,
    rank_order           INT NULL,
    quoted_text          TEXT NULL
);

CREATE TABLE SESSIONS
(
    session_id   UUID PRIMARY KEY,
    course_id    UUID NULL,
    session_date DATE NULL,
    title        VARCHAR(255) NULL,
    audio_path   TEXT NULL,
    duration_sec INT NULL,
    status       VARCHAR(50) NULL,
    created_at   TIMESTAMP NULL,
    file_kind    VARCHAR(50) NULL,   -- lecture, meeting
    tag          VARCHAR(50) NULL,   -- 수업, 회의, 프로젝트
    icon         VARCHAR(50) NULL,   -- article, groups_2
    color        VARCHAR(50) NULL,   -- #3b82f6
    session_pdf  JSONB NULL,         -- 강의자료 목록
    session_voicefile  JSONB NULL,   -- 녹음본 목록
    summary_notes JSONB NULL
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







-- -- pgvector 확장 (chunks 테이블의 embedding 컬럼에 필요)
-- CREATE EXTENSION IF NOT EXISTS vector;
--
-- CREATE TABLE sessions (
--     session_id   UUID PRIMARY KEY,
--     course_id    UUID NOT NULL,
--     session_date DATE NOT NULL,
--     title        VARCHAR(255) NOT NULL,
--     audio_path   TEXT,
--     duration_sec INT,
--     status       VARCHAR(50),
--     created_at   TIMESTAMP NOT NULL
-- );
--
-- CREATE TABLE transcripts (
--     transcript_id  UUID PRIMARY KEY,
--     session_id     UUID NOT NULL REFERENCES sessions(session_id),
--     segment_index  INT NOT NULL,
--     start_time     REAL NOT NULL,
--     end_time       REAL NOT NULL,
--     original_text  TEXT NOT NULL,
--     corrected_text TEXT,
--     confidence     REAL,
--     created_at     TIMESTAMP NOT NULL
-- );
--
-- CREATE TABLE users (
--     user_id    UUID PRIMARY KEY,
--     email      VARCHAR(255) NOT NULL,
--     name       VARCHAR(100) NOT NULL,
--     created_at TIMESTAMP NOT NULL
-- );
--
-- CREATE TABLE courses (
--     course_id   UUID PRIMARY KEY,
--     user_id     UUID NOT NULL REFERENCES users(user_id),
--     title       VARCHAR(255) NOT NULL,
--     type        VARCHAR(50),
--     description TEXT,
--     created_at  TIMESTAMP NOT NULL
-- );
--
-- CREATE TABLE chunks (
--     chunk_id    UUID PRIMARY KEY,
--     session_id  UUID NOT NULL REFERENCES sessions(session_id),
--     chunk_index INT NOT NULL,
--     start_time  REAL NOT NULL,
--     end_time    REAL NOT NULL,
--     chunk_text  TEXT NOT NULL,
--     embedding   VECTOR(1024),
--     created_at  TIMESTAMP NOT NULL
-- );
--
-- CREATE TABLE concept_requests (
--     request_id      UUID PRIMARY KEY,
--     user_id         UUID NOT NULL REFERENCES users(user_id),
--     session_id      UUID NOT NULL REFERENCES sessions(session_id),
--     clicked_text    VARCHAR(255) NOT NULL,
--     normalized_term VARCHAR(255),
--     request_time    REAL NOT NULL,
--     request_type    VARCHAR(50),
--     status          VARCHAR(50),
--     created_at      TIMESTAMP NOT NULL
-- );
--
-- CREATE TABLE explanations (
--     explanation_id UUID PRIMARY KEY,
--     request_id     UUID NOT NULL REFERENCES concept_requests(request_id),
--     answer_text    TEXT NOT NULL,
--     source_links   TEXT,
--     model_name     VARCHAR(100),
--     latency_ms     INT,
--     created_at     TIMESTAMP NOT NULL
-- );
--
-- CREATE TABLE explanation_chunks (
--     explanation_chunk_id UUID PRIMARY KEY,
--     explanation_id       UUID NOT NULL REFERENCES explanations(explanation_id),
--     chunk_id             UUID NOT NULL REFERENCES chunks(chunk_id),
--     similarity_score     REAL,
--     rank_order           INT,
--     quoted_text          TEXT
-- );
