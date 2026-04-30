-- 기존 데이터를 보존하지 않고 테이블을 다시 만들 때 사용
DROP TABLE IF EXISTS
    course_memories,
    explanation_chunks,
    explanations,
    summaries,
    concept_requests,
    key_sentences,
    transcripts,
    quizzes,
    schedules,
    sessions,
    courses,
    users
CASCADE;
