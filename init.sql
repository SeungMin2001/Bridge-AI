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
    title             VARCHAR(255) NOT NULL,
    description       TEXT NULL,
    event_type        VARCHAR(50) NULL,
    due_date          TIMESTAMP NULL,
    status            VARCHAR(30) NULL,
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
    summary_text      TEXT      NOT NULL,
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
    course_id   UUID PRIMARY KEY,
    user_id     UUID NULL,
    title       VARCHAR(255) NULL,
    type        VARCHAR(50) NULL,
    description TEXT NULL,
    created_at  TIMESTAMP NULL
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
    created_at   TIMESTAMP NULL
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
