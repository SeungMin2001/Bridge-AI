# LectoAI - 시스템 모델 (System Model)

## 주제
**LectoAI: 실시간 강의 보조 AI 시스템**

## 필요성
- 강의 중 놓친 내용을 다시 확인할 수 없음
- 수업 중 모르는 개념을 즉시 질문하기 어려움
- 강의록 정리에 과도한 시간 소요
- 기존 AI 서비스(ChatGPT 등)는 강의 맥락을 모름

## 타당성
- 온디바이스 모델(Whisper, KoBART, Qwen)로 외부 API 없이 운영 → 법적 리스크 없음
- RTX 5070 Ti 16GB 단일 GPU로 전체 파이프라인 구동 가능
- MergePRAG 기술로 RAG 대비 더 정밀한 강의 맥락 주입 가능

## 목적
강의 음성을 실시간 전사하고, 강의 내용 기반으로 학생의 질문에 즉시 답변하는 자체 AI 시스템 구축

---

## 시스템 모델 다이어그램

> Mermaid Live Editor( https://mermaid.live )에 아래 코드를 붙여넣으면 다이어그램이 렌더링됩니다.

```mermaid
flowchart TB
    subgraph USER["사용자 (학생)"]
        MIC["마이크 (음성 입력)"]
        QST["질문 입력"]
    end

    subgraph FRONTEND["Frontend · Vue 3 + Vite + Tailwind CSS"]
        TRANS_VIEW["실시간 전사 화면\n(키워드 클릭 가능)"]
        CONCEPT_TAB["개념 설명 탭\n(전사 옆 즉시 표시)"]
        CHAT_VIEW["AI 채팅 패널\n(자유 질의응답)"]
        subgraph POST_TABS["수업 후처리 탭"]
            TAB_SUMMARY["요약"]
            TAB_QUIZ["퀴즈"]
            TAB_MINUTES["회의록"]
            TAB_KEYWORD["핵심 용어"]
        end
    end

    subgraph BACKEND["Backend · FastAPI + Python"]

        subgraph STT_PIPE["음성 처리 파이프라인"]
            WHISPER["Whisper\n(faster-whisper)\nSTT"]
            KOBART["KoBART\n(fine-tuned)\n한국어 교정"]
        end

        subgraph RAG_ENGINE["RAG 검색 엔진"]
            MULTI_Q["멀티쿼리 확장\n(규칙 기반 키워드 추출)"]
            HYBRID["하이브리드 검색"]
            VEC_SEARCH["벡터 검색\n(LlamaIndex + BGE-M3\n→ pgvector cosine)"]
            KW_SEARCH["키워드 검색\n(PostgreSQL ILIKE)"]
            CITE["출처 추출\n(mm:ss timestamp)"]
        end

        subgraph LLM_ENGINE["LLM 추론 · Qwen 3.5 (BnB 4bit)"]
            RAG_MODE["방식 A: RAG + LLM\n프롬프트에 텍스트 삽입"]
            MPRAG_MODE["방식 B: MergePRAG + LLM\nHyperNetwork → K,V 생성\nOrthogonal Merging\nCritical Layer Inject"]
        end

        subgraph POST_PIPE["후처리 파이프라인"]
            POST_GEN["수업 종료 시\n전사문 → LLM 태스크별 생성\n(요약 / 퀴즈 / 회의록 / 핵심용어)"]
        end

        EMBED["BGE-M3 임베딩\n(1024d 벡터 변환)"]
    end

    subgraph DB["PostgreSQL + pgvector"]
        TB_COURSE["courses\n(과목 정보)"]
        TB_SESSION["sessions\n(강의 회차)"]
        TB_TRANSCRIPT["transcripts\n(전사 텍스트)"]
        TB_CHUNK["chunks\n(검색용 청크\n+ embedding 1024d)"]
        TB_EXPLAIN["explanations\n(AI 답변 기록)"]
        TB_POST["summaries / quizzes\nminutes / keywords\n(후처리 결과물)"]
    end

    subgraph ENV["실행 환경"]
        WIN["Windows PC\nRTX 5070 Ti 16GB\nBackend + LLM + DB"]
        MAC["MacBook\nFrontend + 개발\nSSH / Tailscale 연결"]
    end

    %% 데이터 흐름
    MIC -- "음성 입력" --> TRANS_VIEW
    TRANS_VIEW -- "PCM 48kHz\nWebSocket" --> WHISPER
    WHISPER -- "raw 텍스트" --> TRANS_VIEW
    WHISPER --> KOBART
    KOBART -- "교정된 텍스트" --> TRANS_VIEW
    KOBART --> EMBED
    EMBED -- "전사문 + 벡터 저장" --> TB_TRANSCRIPT
    EMBED -- "청크 + 벡터 저장" --> TB_CHUNK
    TB_SESSION -.- TB_TRANSCRIPT
    TB_COURSE -.- TB_SESSION

    TRANS_VIEW -- "키워드 클릭" --> MULTI_Q
    CITE -- "근거 + 설명" --> CONCEPT_TAB

    QST -- "질문 텍스트" --> CHAT_VIEW
    CHAT_VIEW --> MULTI_Q
    MULTI_Q --> HYBRID
    HYBRID --> VEC_SEARCH
    HYBRID --> KW_SEARCH
    VEC_SEARCH -- "cosine 유사도" --> TB_CHUNK
    KW_SEARCH -- "ILIKE 매칭" --> TB_TRANSCRIPT
    VEC_SEARCH --> CITE
    KW_SEARCH --> CITE
    CITE -- "관련 문장 + 출처" --> RAG_MODE
    CITE -- "관련 문장 + 출처" --> MPRAG_MODE

    RAG_MODE -- "SSE 스트리밍\n답변 + 출처" --> CHAT_VIEW
    MPRAG_MODE -- "SSE 스트리밍\n답변 + 출처" --> CHAT_VIEW
    RAG_MODE --> TB_EXPLAIN
    MPRAG_MODE --> TB_EXPLAIN

    %% 후처리 파이프라인
    TB_CHUNK -- "수업 전체 전사문" --> POST_GEN
    POST_GEN --> TB_POST
    TB_POST --> TAB_SUMMARY
    TB_POST --> TAB_QUIZ
    TB_POST --> TAB_MINUTES
    TB_POST --> TAB_KEYWORD

    %% 스타일
    style USER fill:#E3F2FD,stroke:#1565C0,color:#000
    style FRONTEND fill:#E8F5E9,stroke:#2E7D32,color:#000
    style BACKEND fill:#FFF3E0,stroke:#E65100,color:#000
    style DB fill:#F3E5F5,stroke:#6A1B9A,color:#000
    style ENV fill:#ECEFF1,stroke:#546E7A,color:#000
    style STT_PIPE fill:#FFF8E1,stroke:#F9A825,color:#000
    style RAG_ENGINE fill:#E1F5FE,stroke:#0277BD,color:#000
    style LLM_ENGINE fill:#FCE4EC,stroke:#C62828,color:#000
    style POST_PIPE fill:#E0F2F1,stroke:#00695C,color:#000
    style POST_TABS fill:#E0F2F1,stroke:#00695C,color:#000
```

---

## 기능 단위 정리

| 기능 | 설명 | 기술 스택 | 데이터 흐름 |
|------|------|-----------|------------|
| **실시간 음성 전사** | 교수 음성을 텍스트로 변환 | Whisper (faster-whisper), WebSocket | 마이크 → Backend → `transcripts` 테이블 |
| **한국어 교정** | 전사 오류 자동 교정 | KoBART (fine-tuned) | `transcripts.original_text` → `transcripts.corrected_text` |
| **강의 내용 임베딩** | 전사문을 벡터로 변환·저장 | BGE-M3 (1024d), pgvector | `transcripts` → `chunks` 테이블 (embedding 컬럼) |
| **하이브리드 검색** | 질문에 관련된 강의 내용 검색 | pgvector (벡터), PostgreSQL (키워드) | 질문 → `chunks` 검색 → 관련 문장 + 출처 |
| **AI 질의응답** | 강의 맥락 기반 답변 생성 | Qwen 3.5 (BnB 4bit), SSE 스트리밍 | 검색 결과 + 질문 → LLM → 스트리밍 답변 |
| **MergePRAG 주입** | 강의 내용을 모델 내부에 직접 주입 | HyperNetwork, Cross-Attention, Orthogonal Merge | 검색 결과 → K,V 생성 → Critical Layer Inject |
| **과목별 메모리 관리** | 과목마다 독립된 지식 메모리 유지 | Orthogonal Merging (Gram-Schmidt) | `courses` 테이블 기준 메모리 분리 |
| **출처 표시** | 답변의 근거가 되는 강의 시점 표시 | Citation 추출, timestamp 매핑 | `chunks.start_time/end_time` → "mm:ss" 형식 |

---

## 기술 스택 요약

| 계층 | 기술 | 역할 |
|------|------|------|
| **Frontend** | Vue 3, Vite, Tailwind CSS | UI, 실시간 표시, 채팅 인터페이스 |
| **Backend** | FastAPI, Python, asyncio | API 서버, 음성 처리, RAG 검색 |
| **STT** | faster-whisper (large-v3-turbo) | 음성 → 텍스트 변환 |
| **교정** | KoBART (fine-tuned) | 한국어 전사 오류 교정 |
| **임베딩** | BAAI/bge-m3 (1024d) | 텍스트 → 벡터 변환 |
| **DB** | PostgreSQL + pgvector | 데이터 저장 + 벡터 검색 |
| **LLM** | Qwen 3.5 (BitsAndBytes 4bit) | 답변 생성 |
| **MergePRAG** | HyperNetwork + Cross-Attention | 지식 주입 (논문 기반) |
| **통신** | WebSocket, SSE | 실시간 양방향/단방향 스트리밍 |

---

## 실행 환경

| 장비 | 역할 | 사양 |
|------|------|------|
| **Windows PC** | Backend + LLM + DB | RTX 5070 Ti 16GB, CUDA |
| **MacBook** | Frontend + 개발 | SSH/Tailscale로 Windows 연결 |

모든 모델은 자체 서버에서 구동 (외부 API 의존 없음)
