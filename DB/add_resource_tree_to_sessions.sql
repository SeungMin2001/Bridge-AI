-- 기존 sessions 테이블에 현재 파일 내부 폴더 구조 저장 컬럼 추가
ALTER TABLE sessions
ADD COLUMN IF NOT EXISTS resource_tree JSONB DEFAULT '[]'::jsonb;
