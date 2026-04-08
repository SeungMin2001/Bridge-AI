-- pgvector 확장 (chunks 테이블의 embedding 컬럼에 필요)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE sessions (
    session_id   UUID PRIMARY KEY,
    course_id    UUID NOT NULL,
    session_date DATE NOT NULL,
    title        VARCHAR(255) NOT NULL,
    audio_path   TEXT,
    duration_sec INT,
    status       VARCHAR(50),
    created_at   TIMESTAMP NOT NULL
);

CREATE TABLE transcripts (
    transcript_id  UUID PRIMARY KEY,
    session_id     UUID NOT NULL REFERENCES sessions(session_id),
    segment_index  INT NOT NULL,
    start_time     REAL NOT NULL,
    end_time       REAL NOT NULL,
    original_text  TEXT NOT NULL,
    corrected_text TEXT,
    confidence     REAL,
    created_at     TIMESTAMP NOT NULL
);

CREATE TABLE users (
    user_id    UUID PRIMARY KEY,
    email      VARCHAR(255) NOT NULL,
    name       VARCHAR(100) NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE courses (
    course_id   UUID PRIMARY KEY,
    user_id     UUID NOT NULL REFERENCES users(user_id),
    title       VARCHAR(255) NOT NULL,
    type        VARCHAR(50),
    description TEXT,
    created_at  TIMESTAMP NOT NULL
);

CREATE TABLE chunks (
    chunk_id    UUID PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES sessions(session_id),
    chunk_index INT NOT NULL,
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL,
    chunk_text  TEXT NOT NULL,
    embedding   VECTOR(1024),
    created_at  TIMESTAMP NOT NULL
);

CREATE TABLE concept_requests (
    request_id      UUID PRIMARY KEY,
    user_id         UUID NOT NULL REFERENCES users(user_id),
    session_id      UUID NOT NULL REFERENCES sessions(session_id),
    clicked_text    VARCHAR(255) NOT NULL,
    normalized_term VARCHAR(255),
    request_time    REAL NOT NULL,
    request_type    VARCHAR(50),
    status          VARCHAR(50),
    created_at      TIMESTAMP NOT NULL
);

CREATE TABLE explanations (
    explanation_id UUID PRIMARY KEY,
    request_id     UUID NOT NULL REFERENCES concept_requests(request_id),
    answer_text    TEXT NOT NULL,
    source_links   TEXT,
    model_name     VARCHAR(100),
    latency_ms     INT,
    created_at     TIMESTAMP NOT NULL
);

CREATE TABLE explanation_chunks (
    explanation_chunk_id UUID PRIMARY KEY,
    explanation_id       UUID NOT NULL REFERENCES explanations(explanation_id),
    chunk_id             UUID NOT NULL REFERENCES chunks(chunk_id),
    similarity_score     REAL,
    rank_order           INT,
    quoted_text          TEXT
);

-- 과목별 MergePRAG 메모리 (K,V 텐서를 직렬화하여 저장)
CREATE TABLE course_memories (
    memory_id      UUID PRIMARY KEY,
    course_id      UUID NOT NULL REFERENCES courses(course_id) UNIQUE,
    merged_k       BYTEA,
    merged_v       BYTEA,
    passage_count  INT DEFAULT 0,
    updated_at     TIMESTAMP NOT NULL
);
