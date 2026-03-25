-- pgvector 익스텐션 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 테이블 생성
CREATE TABLE users (
                       user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                       email VARCHAR(255) NOT NULL,
                       name VARCHAR(100) NOT NULL,
                       created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE courses (
                         course_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                         user_id UUID NOT NULL,
                         title VARCHAR(255) NOT NULL,
                         type VARCHAR(50) NULL,
                         description TEXT NULL,
                         created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE sessions (
                          session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                          course_id UUID NOT NULL,
                          session_date DATE NOT NULL,
                          title VARCHAR(255) NOT NULL,
                          audio_path TEXT NULL,
                          duration_sec INT NULL,
                          status VARCHAR(50) NULL,
                          created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE transcripts (
                             transcript_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                             session_id UUID NOT NULL,
                             segment_index INT NOT NULL,
                             start_time REAL NOT NULL,
                             end_time REAL NOT NULL,
                             original_text TEXT NOT NULL,
                             corrected_text TEXT NULL,
                             confidence REAL NULL,
                             created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE chunks (
                        chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        session_id UUID NOT NULL,
                        chunk_index INT NOT NULL,
                        start_time REAL NOT NULL,
                        end_time REAL NOT NULL,
                        chunk_text TEXT NOT NULL,
                        embedding VECTOR(1024) NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE concept_requests (
                                  request_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                  user_id UUID NOT NULL,
                                  session_id UUID NOT NULL,
                                  clicked_text VARCHAR(255) NOT NULL,
                                  normalized_term VARCHAR(255) NULL,
                                  request_time REAL NOT NULL,
                                  request_type VARCHAR(50) NULL,
                                  status VARCHAR(50) NULL,
                                  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE explanations (
                              explanation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                              request_id UUID NOT NULL,
                              answer_text TEXT NOT NULL,
                              source_links TEXT NULL,
                              model_name VARCHAR(100) NULL,
                              latency_ms INT NULL,
                              created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE explanation_chunks (
                                    explanation_chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                    explanation_id UUID NOT NULL,
                                    chunk_id UUID NOT NULL,
                                    similarity_score REAL NULL,
                                    rank_order INT NULL,
                                    quoted_text TEXT NULL
);

-- 외래키 제약조건 추가
ALTER TABLE courses ADD CONSTRAINT fk_courses_users FOREIGN KEY (user_id) REFERENCES users (user_id);
ALTER TABLE sessions ADD CONSTRAINT fk_sessions_courses FOREIGN KEY (course_id) REFERENCES courses (course_id);
ALTER TABLE transcripts ADD CONSTRAINT fk_transcripts_sessions FOREIGN KEY (session_id) REFERENCES sessions (session_id);
ALTER TABLE chunks ADD CONSTRAINT fk_chunks_sessions FOREIGN KEY (session_id) REFERENCES sessions (session_id);
ALTER TABLE concept_requests ADD CONSTRAINT fk_concept_requests_users FOREIGN KEY (user_id) REFERENCES users (user_id);
ALTER TABLE concept_requests ADD CONSTRAINT fk_concept_requests_sessions FOREIGN KEY (session_id) REFERENCES sessions (session_id);
ALTER TABLE explanations ADD CONSTRAINT fk_explanations_concept_requests FOREIGN KEY (request_id) REFERENCES concept_requests (request_id);
ALTER TABLE explanation_chunks ADD CONSTRAINT fk_explanation_chunks_explanations FOREIGN KEY (explanation_id) REFERENCES explanations (explanation_id);
ALTER TABLE explanation_chunks ADD CONSTRAINT fk_explanation_chunks_chunks FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id);