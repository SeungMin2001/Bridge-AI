# 백엔드 서비스 구현 계획 (Implementation Plan)

> 최종 수정: 2026-04-25  
> 담당: 백엔드

---

## 현재 완료 상태

### ✅ 완료된 서비스

| # | 서비스 | 구현 위치 | API 엔드포인트 | 상태 |
|---|--------|----------|---------------|------|
| 1 | **실시간 STT** | `main.py` (WebSocket /ws) | `WS /ws` | ✅ 완료 |
| 2 | **PCM 리샘플링** | `main.py` (torchaudio 48→16kHz) | — | ✅ 완료 |
| 3 | **KoBART 교정** | `correction.py` | — | ✅ 완료 |
| 4 | **임베딩** | `data/embeded_test.py` + `rag_search.py` | — | ✅ 완료 |
| 5 | **Transcript 저장** | `db.py` + `data/save_transcript.py` | — | ✅ 완료 |
| 6 | **세션 메타데이터** | `db.py` (create_session) | — | ✅ 완료 |
| 7 | **RAG Hybrid Search** | `rag_search.py` (벡터+BM25+RRF) | — | ✅ 완료 |
| 8 | **AI 채팅 (비스트리밍)** | `main.py` | `POST /chat` | ✅ 완료 |
| 9 | **AI 채팅 (SSE 스트리밍)** | `main.py` | `POST /chat/stream` | ✅ 완료 |
| 10 | **채팅 근거링크 + 메타데이터** | `rag_search.py` (citation) | — | ✅ 완료 |
| 11 | **Thinking 모드** | `main.py` (enable_thinking) | — | ✅ 완료 |
| 12 | **MergePRAG** | `llm_server/api.py` + `llm_server/mergePRAG/` | — | ✅ 완료 |
| 13 | **과목 메모리 관리** | `llm_server/api.py` | `/memory/*` | ✅ 완료 |
| 14 | **퀴즈 생성/채점** | `quiz/` | `/quiz/*` (5개 엔드포인트) | ✅ 완료 |
| 15 | **요약 (TextRank+LLM)** | `summary/` | `/summary/*` (4개 엔드포인트) | ✅ 완료 |

### 퀴즈 서비스 상세 (완료)

```
backend/quiz/
├── quiz.py          # API 라우터 (5개 엔드포인트)
├── quiz_service.py  # 비즈니스 로직 (LLM 호출, 채점, Mock 모드)
└── quiz_db.py       # DB CRUD (asyncpg)
```

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /quiz/generate` | 세션 전사문 기반 퀴즈 생성 |
| `POST /quiz/generate/text` | 직접 텍스트 기반 퀴즈 생성 (테스트용) |
| `GET /quiz/session/{session_id}` | 세션별 퀴즈 목록 조회 |
| `GET /quiz/{quiz_id}` | 퀴즈 단건 조회 |
| `POST /quiz/{quiz_id}/submit` | 사용자 답안 채점 |

- JSONB 구조: `MULTIPLE_CHOICE`, `OX`, `SHORT_ANSWER` 3가지 유형
- `QUIZ_MOCK_MODE=true` 환경변수로 LLM 없이 테스트 가능
- 단답형은 퍼지 매칭 (공백/대소문자 무시 + 부분 포함)

---

## ❌ 미구현 서비스

### Phase 2: 요약 서비스 — ✅ 완료

> 대응 테이블: `SUMMARIES`, `KEY_SENTENCES`

**요구사항 정리**:
- kiwipiepy로 형태소 분리 (명사/동사 중심 전처리)
- TextRank(textrankr)로 핵심 문장 추출
- 요약 결과를 5가지 고정 포맷으로 출력 (핵심 개념, 주요 문장, 복습 포인트, 세부 설명, 시험 포인트)

**구현 방식 (quiz와 동일 패턴)**:

```
backend/summary/
├── summary.py          # API 라우터
├── summary_service.py  # TextRank + LLM 요약 파이프라인
└── summary_db.py       # SUMMARIES, KEY_SENTENCES CRUD
```

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /summary/generate` | session_id 전사문 기반 요약 생성 |
| `GET /summary/{summary_id}` | 요약 단건 조회 |
| `GET /summary/session/{session_id}` | 세션별 요약 목록 |

**핵심 파이프라인**:
```
① GET transcripts WHERE session_id = ?
② kiwipiepy.tokenize() → 명사/동사 추출
③ TextRank.summarize(sentences, k=10) → 핵심 문장 리스트
④ LLM에 핵심 문장 + 아래 JSON 템플릿 전달
⑤ SUMMARIES 테이블 INSERT + KEY_SENTENCES 테이블 INSERT
⑥ 결과 반환
```

**요약 응답 JSON 템플릿**:
```json
{
  "core_concepts": ["개념1", "개념2"],
  "key_sentences": ["핵심 문장1", "핵심 문장2"],
  "review_points": ["복습 포인트1", "복습 포인트2"],
  "detailed_explanation": "세부 설명 텍스트",
  "exam_points": ["시험 포인트1", "시험 포인트2"]
}
```

---

### Phase 3: 개념설명 DB 저장 — 🔴 다음 구현 대상

> 대응 테이블: `CONCEPT_REQUESTS`, `EXPLANATIONS`, `EXPLANATION_CHUNKS`

**현재 상태**: 채팅(`/chat`)에서 RAG 기반 개념설명은 동작하지만, 요청/응답을 DB에 저장하지 않음

```
backend/concept/
├── concept.py          # API 라우터
├── concept_service.py  # RAG 검색 + LLM 설명 생성
└── concept_db.py       # CONCEPT_REQUESTS, EXPLANATIONS, EXPLANATION_CHUNKS CRUD
```

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /concept/explain` | 키워드 클릭 → 개념설명 생성 + DB 저장 |
| `GET /concept/history` | 세션별 개념설명 히스토리 조회 |

**핵심 흐름**:
```
① CONCEPT_REQUESTS INSERT (clicked_text, normalized_term)
② rag_search(clicked_text) → context + citations
③ LLM 생성 → answer_text
④ EXPLANATIONS INSERT (request_id, answer_text, latency_ms)
⑤ EXPLANATION_CHUNKS INSERT (각 citation의 transcript_id, similarity_score)
```

---

### Phase 4: 학습자료 업로드 + 파일 첨부 채팅

> 대응 테이블: 없음 → `MATERIALS`, `MATERIAL_CHUNKS` 테이블 추가 필요

```
backend/material/
├── material.py          # API 라우터
├── material_service.py  # PDF/PPT 텍스트 추출 + 청크 분할
└── material_db.py       # MATERIALS, MATERIAL_CHUNKS CRUD
```

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /material/upload` | 파일 업로드 (multipart/form-data) |
| `GET /material/{material_id}` | 자료 정보 조회 |
| `GET /material/list` | 과목별 자료 목록 |
| `POST /chat/file` | 파일 첨부 채팅 (pdfplumber/python-pptx/OCR) |

---

### Phase 5: 일정 추출

> 대응 테이블: `SCHEDULES` (이미 ERD에 존재)

```
backend/schedule/
├── schedule.py          # API 라우터
├── schedule_service.py  # 전사문에서 일정 키워드 감지 + LLM 추출
└── schedule_db.py       # SCHEDULES CRUD
```

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /schedule/extract` | session_id 전사문에서 일정 자동 추출 |
| `GET /schedule/list` | 일정 목록 조회 |
| `PUT /schedule/{schedule_id}` | 일정 수정 |
| `DELETE /schedule/{schedule_id}` | 일정 삭제 |

---

## 현재 DB 연결 상태

| 파일 | 라이브러리 | 비고 |
|------|----------|------|
| `db.py` | asyncpg (비동기 풀) | **표준** — 새 서비스는 모두 이 방식 사용 |
| `quiz/quiz_db.py` | asyncpg (db.py의 get_pool 호출) | ✅ 통합됨 |
| `database.py` | psycopg (동기) | 레거시 — 추후 정리 |
| `data/save_chunk_to_db.py` | psycopg (동기, 하드코딩 IP) | 레거시 — 추후 정리 |
| `rag_search.py` | psycopg2 (동기, 하드코딩) | 레거시 — 추후 비동기 전환 |

---

## 디렉토리 구조 (현재 → 목표)

```
backend/
├── main.py               # FastAPI 앱 + STT + 채팅 (기존)
├── db.py                  # 공용 DB 풀 + 공용 함수 (get_transcripts_by_session 등)
├── correction.py          # KoBART 교정 (기존)
├── rag_search.py          # RAG Hybrid Search (기존)
│
├── quiz/                  # ✅ 완료
│   ├── quiz.py
│   ├── quiz_service.py
│   └── quiz_db.py
│
├── summary/               # ✅ 완료
│   ├── summary.py
│   ├── summary_service.py
│   └── summary_db.py
│
├── concept/               # 🟡 Phase 3
│   ├── concept.py
│   ├── concept_service.py
│   └── concept_db.py
│
├── material/              # 🟡 Phase 4
│   ├── material.py
│   ├── material_service.py
│   └── material_db.py
│
└── schedule/              # 🟢 Phase 5
    ├── schedule.py
    ├── schedule_service.py
    └── schedule_db.py
```

각 서비스 폴더는 `__init__.py` 없이 `sys.path` 방식으로 import 처리.
`main.py`에서 `from {service}.{module} import router` 형태로 등록.
