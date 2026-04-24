# LLM 추론 파이프라인 상세 모델

> Mermaid Live Editor( https://mermaid.live )에서 렌더링

## 전체 추론 파이프라인 (3가지 방식 비교)

```mermaid
flowchart TB
    subgraph INPUT["입력"]
        USER_Q["사용자 질문\n'운영체제의 프로세스가 뭐야?'"]
        KEYWORD["키워드 클릭\n'프로세스'"]
    end

    subgraph RAG["RAG 검색 파이프라인"]
        direction TB
        KIWI["형태소 분석 (Kiwi)\n'운영체제', '프로세스'"]

        subgraph MULTI_Q["멀티쿼리 확장"]
            Q1["원본: '운영체제의 프로세스가 뭐야?'"]
            Q2["키워드: '운영체제 프로세스'"]
        end

        subgraph HYBRID["하이브리드 검색"]
            VEC["벡터 검색\nBGE-M3 → 1024d 벡터\n→ pgvector cosine 유사도"]
            KW["키워드 검색\nKiwi 형태소 → 명사 추출\n→ PostgreSQL ILIKE"]
        end

        MERGE["결과 병합 + 중복 제거\ntop_k=5"]
        CITE["출처 매핑\n(과목명, 회차, mm:ss)"]
    end

    subgraph DB["PostgreSQL + pgvector"]
        CHUNKS["chunks 테이블\nchunk_text + embedding(1024d)\n+ start_time, end_time"]
    end

    subgraph CONTEXT["검색 결과 (Context)"]
        CTX_TEXT["'프로세스는 실행 중인 프로그램이다.\n운영체제는 프로세스를 관리한다.\n...'"]
        CTX_CITE["출처: CS기초 3회차 12:30"]
    end

    %% =============================================
    %% 3가지 LLM 추론 방식
    %% =============================================

    subgraph LLM_MODES["LLM 추론 (Qwen 3.5-4B · BnB 4bit)"]
        direction TB

        subgraph MODE_A["방식 A: LLM Only"]
            A_IN["입력: 질문만\n'운영체제의 프로세스가 뭐야?'"]
            A_GEN["model.generate()\nmax_tokens=128"]
            A_OUT["출력: 일반 지식 기반 답변\n(강의 맥락 없음)"]
        end

        subgraph MODE_B["방식 B: RAG + LLM"]
            B_IN["입력: 프롬프트에 Context 삽입\n'참고자료:\\n{검색된 문장}\\n질문: ...'"]
            B_GEN["model.generate()\nmax_tokens=128"]
            B_OUT["출력: 강의 내용 기반 답변\n+ 출처 표시"]
        end

        subgraph MODE_C["방식 C: MergePRAG + LLM"]
            direction TB
            C_EMBED["검색된 문장 → embed_tokens()\n[1, T, 2560]"]
            C_HYPER["HyperNetwork (학습 완료)\nAttentivePooling → MLP → LinearProjection\n→ delta_K, delta_V [1, 16, 2560]"]
            C_ORTHO["Orthogonal Merging\n(Gram-Schmidt)\n여러 passage K,V → 간섭 없이 병합"]
            C_HOOK["Critical Layer 0에\nForward Hook 등록\nhidden += cross_attention(Q, K, V)"]
            C_GEN["model.generate()\nmax_tokens=128"]
            C_OUT["출력: 모델 내부에 지식 주입된 답변\n+ 출처 표시"]
        end
    end

    subgraph STREAMING["스트리밍 출력 (SSE)"]
        SSE_CITE["1. citations 이벤트\n{type:'citations', citations:[...]}"]
        SSE_TOKEN["2. 토큰 단위 스트리밍\n{type:'token', content:'프로'}"]
        SSE_DONE["3. 완료 이벤트\n{type:'done'}"]
    end

    subgraph OUTPUT["사용자 화면"]
        ANS["AI 답변 (실시간 표시)"]
        SRC["출처 링크 (과목·회차·시간)"]
    end

    %% =============================================
    %% 데이터 흐름 연결
    %% =============================================

    USER_Q --> KIWI
    KEYWORD --> KIWI
    KIWI --> MULTI_Q
    Q1 --> VEC
    Q2 --> VEC
    Q1 --> KW
    Q2 --> KW
    VEC --> CHUNKS
    KW --> CHUNKS
    CHUNKS --> VEC
    CHUNKS --> KW
    VEC --> MERGE
    KW --> MERGE
    MERGE --> CITE
    CITE --> CTX_TEXT
    CITE --> CTX_CITE

    %% 방식 A: 검색 안 함
    USER_Q -.->|"RAG 없이 직접"| A_IN
    A_IN --> A_GEN --> A_OUT

    %% 방식 B: Context를 텍스트로 삽입
    CTX_TEXT -->|"프롬프트에 텍스트 삽입"| B_IN
    B_IN --> B_GEN --> B_OUT

    %% 방식 C: Context를 K,V로 변환 후 주입
    CTX_TEXT -->|"passage 임베딩"| C_EMBED
    C_EMBED --> C_HYPER
    C_HYPER --> C_ORTHO
    C_ORTHO --> C_HOOK
    C_HOOK --> C_GEN --> C_OUT

    %% 출력
    A_OUT --> SSE_TOKEN
    B_OUT --> SSE_CITE
    B_OUT --> SSE_TOKEN
    C_OUT --> SSE_CITE
    C_OUT --> SSE_TOKEN
    SSE_CITE --> SRC
    SSE_TOKEN --> ANS
    SSE_DONE --> ANS

    %% 스타일
    style INPUT fill:#E3F2FD,stroke:#1565C0,color:#000
    style RAG fill:#E1F5FE,stroke:#0277BD,color:#000
    style DB fill:#F3E5F5,stroke:#6A1B9A,color:#000
    style CONTEXT fill:#E8F5E9,stroke:#2E7D32,color:#000
    style LLM_MODES fill:#FFF3E0,stroke:#E65100,color:#000
    style MODE_A fill:#FFECB3,stroke:#FF8F00,color:#000
    style MODE_B fill:#C8E6C9,stroke:#388E3C,color:#000
    style MODE_C fill:#FFCDD2,stroke:#C62828,color:#000
    style STREAMING fill:#F3E5F5,stroke:#7B1FA2,color:#000
    style OUTPUT fill:#E0F7FA,stroke:#00838F,color:#000
    style HYBRID fill:#FFF8E1,stroke:#F9A825,color:#000
    style MULTI_Q fill:#E8EAF6,stroke:#3949AB,color:#000
```

---

## 3가지 방식 비교 요약

| | 방식 A: LLM Only | 방식 B: RAG + LLM | 방식 C: MergePRAG + LLM |
|---|---|---|---|
| **입력** | 질문 텍스트만 | 질문 + 검색된 강의 내용 (텍스트) | 질문 + 검색된 강의 내용 (K,V 벡터) |
| **지식 주입 방식** | 없음 (모델 사전학습 지식만) | 프롬프트에 텍스트 삽입 | Critical Layer에 Cross-Attention Inject |
| **출처 표시** | 불가 | 가능 (mm:ss) | 가능 (mm:ss) |
| **컨텍스트 제한** | 없음 | 토큰 길이 제한에 포함 | K,V 16개로 압축 → 제한 없음 |
| **추론 엔진** | transformers | transformers (또는 vLLM) | transformers only (hook 필요) |
| **속도** | 가장 빠름 | 빠름 | 약간 느림 (HyperNetwork + hook) |

## MergePRAG 상세 흐름

```mermaid
flowchart LR
    subgraph PASSAGE["검색된 강의 문장"]
        P1["'프로세스는 실행 중인 프로그램'"]
        P2["'CPU 스케줄링으로 프로세스 관리'"]
    end

    subgraph EMBED_LAYER["Qwen embed_tokens()"]
        E1["passage 1 → [1, T₁, 2560]"]
        E2["passage 2 → [1, T₂, 2560]"]
    end

    subgraph HYPERNET["HyperNetwork (학습된 가중치)"]
        POOL["AttentivePooling\n[1,T,2560] → [1,2560]"]
        MLP["MLP (2-layer + LayerNorm)\n[1,2560] → [1,2560]"]
        LP["LinearProjection\n[1,2560] → K[1,16,2560], V[1,16,2560]"]
    end

    subgraph ORTHO["Orthogonal Merging"]
        GM["Gram-Schmidt 직교화\npassage 1의 K,V + passage 2의 K,V\n→ 간섭 없이 병합된 K_merged, V_merged"]
    end

    subgraph INJECT["Critical Layer 0 Injection"]
        HOOK["Forward Hook 등록"]
        CROSS["Cross-Attention (8-head)\nQ = hidden_states [1, seq, 2560]\nK = K_merged [1, 16, 2560]\nV = V_merged [1, 16, 2560]"]
        ADD["hidden += cross_attn(Q, K, V)\n(residual addition)"]
    end

    subgraph GENERATE["답변 생성"]
        GEN["model.generate()\nTextIteratorStreamer\n토큰 단위 스트리밍"]
    end

    P1 --> E1
    P2 --> E2
    E1 --> POOL
    E2 --> POOL
    POOL --> MLP --> LP
    LP -->|"K₁,V₁"| GM
    LP -->|"K₂,V₂"| GM
    GM -->|"K_merged, V_merged"| HOOK
    HOOK --> CROSS --> ADD
    ADD --> GEN

    style PASSAGE fill:#E8F5E9,stroke:#2E7D32,color:#000
    style EMBED_LAYER fill:#E3F2FD,stroke:#1565C0,color:#000
    style HYPERNET fill:#FFF3E0,stroke:#E65100,color:#000
    style ORTHO fill:#FCE4EC,stroke:#C62828,color:#000
    style INJECT fill:#F3E5F5,stroke:#6A1B9A,color:#000
    style GENERATE fill:#E0F7FA,stroke:#00838F,color:#000
```

## 입출력 데이터 형식

### API 요청 (Input)
```json
{
  "question": "운영체제의 프로세스가 뭐야?",
  "is_thinking": false
}
```

### API 응답 (Output · SSE 스트리밍)
```
data: {"type":"citations","citations":[{"source":"CS기초 3회차","time":"12:30"}]}

data: {"type":"token","content":"프로"}
data: {"type":"token","content":"세스는"}
data: {"type":"token","content":" 실행 중인"}
data: {"type":"token","content":" 프로그램입니다."}

data: {"type":"done"}
```
