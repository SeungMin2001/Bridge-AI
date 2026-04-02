CREATE TABLE `sessions` (
	`session_id`	UUID	NULL,
	`course_id`	UUID	NOT NULL,
	`session_date`	DATE	NOT NULL,
	`title`	VARCHAR(255)	NOT NULL,
	`audio_path`	TEXT	NULL,
	`duration_sec`	INT	NULL,
	`status`	VARCHAR(50)	NULL,
	`created_at`	TIMESTAMP	NOT NULL
);

CREATE TABLE `transcripts` (
	`transcript_id`	UUID	NULL,
	`session_id`	UUID	NOT NULL,
	`segment_index`	INT	NOT NULL,
	`start_time`	REAL	NOT NULL,
	`end_time`	REAL	NOT NULL,
	`original_text`	TEXT	NOT NULL,
	`corrected_text`	TEXT	NULL,
	`confidence`	REAL	NULL,
	`created_at`	TIMESTAMP	NOT NULL
);

CREATE TABLE `users` (
	`user_id`	UUID	NULL,
	`email`	VARCHAR(255)	NOT NULL,
	`name`	VARCHAR(100)	NOT NULL,
	`created_at`	TIMESTAMP	NOT NULL
);

CREATE TABLE `courses` (
	`course_id`	UUID	NULL,
	`user_id`	UUID	NOT NULL,
	`title`	VARCHAR(255)	NOT NULL,
	`type`	VARCHAR(50)	NULL,
	`description`	TEXT	NULL,
	`created_at`	TIMESTAMP	NOT NULL
);

CREATE TABLE `chunks` (
	`chunk_id`	UUID	NULL,
	`session_id`	UUID	NOT NULL,
	`chunk_index`	INT	NOT NULL,
	`start_time`	REAL	NOT NULL,
	`end_time`	REAL	NOT NULL,
	`chunk_text`	TEXT	NOT NULL,
	`embedding`	VECTOR(1024)	NULL,
	`created_at`	TIMESTAMP	NOT NULL
);

CREATE TABLE `concept_requests` (
	`request_id`	UUID	NULL,
	`user_id`	UUID	NOT NULL,
	`session_id`	UUID	NOT NULL,
	`clicked_text`	VARCHAR(255)	NOT NULL,
	`normalized_term`	VARCHAR(255)	NULL,
	`request_time`	REAL	NOT NULL,
	`request_type`	VARCHAR(50)	NULL,
	`status`	VARCHAR(50)	NULL,
	`created_at`	TIMESTAMP	NOT NULL
);

CREATE TABLE `explanations` (
	`explanation_id`	UUID	NULL,
	`request_id`	UUID	NOT NULL,
	`answer_text`	TEXT	NOT NULL,
	`source_links`	TEXT	NULL,
	`model_name`	VARCHAR(100)	NULL,
	`latency_ms`	INT	NULL,
	`created_at`	TIMESTAMP	NOT NULL
);

CREATE TABLE `explanation_chunks` (
	`explanation_chunk_id`	UUID	NULL,
	`explanation_id`	UUID	NOT NULL,
	`chunk_id`	UUID	NOT NULL,
	`similarity_score`	REAL	NULL,
	`rank_order`	INT	NULL,
	`quoted_text`	TEXT	NULL
);

