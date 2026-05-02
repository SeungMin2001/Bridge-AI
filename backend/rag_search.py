"""
RAG 검색 모듈
- Hybrid Search (벡터 유사도 + 키워드 BM25)
- Query Rewriting + Multi-query
- Citation (출처 표시)
- 실시간 전사 임베딩 추가
"""
import logging
import os
import psycopg2
from kiwipiepy import Kiwi
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

# ── 형태소 분석기 (Kiwi) ──
_kiwi = Kiwi()

# 키워드 검색에 사용할 품사 태그 (명사, 동사 어간, 형용사 어간)
_KEYWORD_TAGS = {"NNG", "NNP", "VV", "VA"}  # 일반명사, 고유명사, 동사, 형용사


def extract_keywords(text: str) -> list[str]:
    """형태소 분석으로 명사/동사어간/형용사어간만 추출"""
    tokens = _kiwi.tokenize(text)
    keywords = []
    for token in tokens:
        if token.tag in _KEYWORD_TAGS and len(token.form) >= 2:
            keywords.append(token.form)
    return keywords

# ── 임베딩 모델 (서버 시작 시 1회 초기화) ──
_embed_model = None
_vector_store = None
_index = None
_initialized = False
_init_error = None

logger = logging.getLogger(__name__)


def _db_config() -> dict:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("DB_NAME", "rag"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD") or None,
    }


def init():
    """서버 시작 시 1회 호출. 임베딩 모델 + vector store 로드."""
    global _embed_model, _vector_store, _index, _initialized, _init_error
    if _initialized:
        return

    try:
        print("[RAG] 임베딩 모델 로딩 중...")
        _embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
        Settings.embed_model = _embed_model

        # 신창영이 임시 지움
        # _vector_store = PGVectorStore.from_params(
        #     database="shin",
        #     host="localhost",
        #     password="1234",
        #     port=5432,
        #     user="postgres",
        #     table_name="shin",
        #     embed_dim=1024,
        # )
        _vector_store = PGVectorStore.from_params(
            database=os.getenv("DB_NAME", "rag"),
            host=os.getenv("DB_HOST", "localhost"),
            password=os.getenv("DB_PASSWORD") or None,
            port=int(os.getenv("DB_PORT", 5432)),
            user=os.getenv("DB_USER", "postgres"),
            table_name=os.getenv("RAG_TABLE_NAME", "rag"),
            embed_dim=1024,
        )
        _index = VectorStoreIndex.from_vector_store(vector_store=_vector_store)
        _initialized = True
        print("[RAG] 초기화 완료")
    except Exception as exc:
        _init_error = exc
        _embed_model = None
        _vector_store = None
        _index = None
        _initialized = True
        logger.warning("[RAG] 초기화 실패; 벡터 검색 비활성화: %s", exc)


def add_document(text: str, metadata: dict):
    """실시간 전사 chunk를 임베딩하여 vector store에 추가"""
    init()
    if _index is None:
        logger.warning("[RAG] 벡터 스토어 미초기화로 문서 추가 스킵")
        return
    doc = Document(text=text, metadata=metadata)
    _index.insert(doc)
    print(f"[RAG] 문서 추가됨: {text[:30]}...")


# ── 키워드(BM25 대용) 검색: DB에서 직접 텍스트 매칭 ──
def _keyword_search(query: str, top_k: int = 5) -> list[dict]:
    """PostgreSQL ts_rank + LIKE 기반 키워드 검색"""
    # 신창영이 임시 지움
    # conn = psycopg2.connect(
    #     host="localhost", port=5432,
    #     database="shin", user="postgres", password="1234"
    # )
    conn = psycopg2.connect(**_db_config())
    cur = conn.cursor()

    # 형태소 분석으로 명사/동사/형용사 키워드 추출
    words = extract_keywords(query)
    if not words:
        cur.close()
        conn.close()
        return []

    # 신창영이 임시 지움
    # like_conditions = " OR ".join([f"c.chunk_text ILIKE %s" for _ in words])
    # match_score = " + ".join([f"CASE WHEN c.chunk_text ILIKE %s THEN 1 ELSE 0 END" for _ in words])
    # sql = f"""
    #     SELECT c.chunk_id, c.session_id, c.chunk_index, c.start_time, c.end_time, c.chunk_text,
    #            s.title as session_title, s.session_date,
    #            co.title as course_title,
    #            ({match_score}) as match_count
    #     FROM chunks c
    #     JOIN sessions s ON c.session_id = s.session_id
    #     JOIN courses co ON s.course_id = co.course_id
    #     WHERE {like_conditions}
    #     ORDER BY match_count DESC
    #     LIMIT %s
    # """

    # 각 단어에 대해 ILIKE OR 조건 + 매칭 키워드 수로 랭킹
    like_conditions = " OR ".join(["t.chunk_text ILIKE %s" for _ in words])
    like_values = [f"%{w}%" for w in words]

    # 키워드 매칭 개수를 점수로 계산하여 ORDER BY
    match_score = " + ".join(["CASE WHEN t.chunk_text ILIKE %s THEN 1 ELSE 0 END" for _ in words])
    score_values = [f"%{w}%" for w in words]

    sql = f"""
        SELECT t.transcript_id, t.session_id, t.chunk_index, t.start_time, t.end_time, t.chunk_text,
               s.title as session_title, s.session_date,
               co.title as course_title,
               ({match_score}) as match_count
        FROM transcripts t
        LEFT JOIN sessions s ON t.session_id = s.session_id
        LEFT JOIN courses co ON s.course_id = co.course_id
        WHERE {like_conditions}
        ORDER BY match_count DESC
        LIMIT %s
    """
    cur.execute(sql, score_values + like_values + [top_k])
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for row in rows:
        transcript_id, session_id, chunk_index, start_time, end_time, chunk_text, session_title, session_date, course_title, match_count = row
        results.append({
            "text": chunk_text,
            "course_title": course_title or "미분류",
            "session_title": session_title or "세션",
            "session_date": str(session_date) if session_date else "",
            "start_time": float(start_time or 0),
            "end_time": float(end_time or 0),
            "source": "keyword",
        })
    return results


# ── 벡터 검색 ──
def _vector_search(query: str, top_k: int = 5) -> list[dict]:
    init()
    if _index is None:
        if _init_error is not None:
            logger.warning("[RAG] 벡터 검색 비활성화: %s", _init_error)
        return []
    retriever = _index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    results = []
    for node in nodes:
        m = node.metadata
        results.append({
            "text": node.text,
            "course_title": m.get("course_title", ""),
            "session_title": m.get("session_title", ""),
            "session_date": m.get("session_date", ""),
            "start_time": m.get("start_time", 0),
            "end_time": m.get("end_time", 0),
            "score": node.score,
            "source": "vector",
        })
    return results


# ── Multi-query: 질문을 여러 검색 쿼리로 확장 ──
def _expand_queries(question: str) -> list[str]:
    """
    원본 질문 + 형태소 분석 기반 키워드 쿼리 생성.
    Kiwi 형태소 분석기로 명사/동사/형용사 어간만 추출.
    """
    queries = [question]

    # 형태소 분석으로 핵심 키워드 추출
    keywords = extract_keywords(question)

    if keywords:
        keyword_query = " ".join(keywords)
        if keyword_query != question:
            queries.append(keyword_query)

    return queries


# ── 시간 포맷팅 ──
def _format_time(seconds: float) -> str:
    """초를 mm:ss 형태로 변환"""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


# ── Reciprocal Rank Fusion (RRF) 병합 ──
def _merge_results(vector_results: list, keyword_results: list, top_k: int = 5, k: int = 60) -> list[dict]:
    """
    RRF로 벡터 + 키워드 결과를 통합 랭킹.
    score = Σ 1/(k + rank)  (k=60이 표준값)
    벡터 3위 + 키워드 1위인 문장이 벡터 1위만인 문장보다 높을 수 있음.
    """
    scores = {}  # text → {"score": float, "data": dict}

    # 벡터 결과: 이미 cosine 유사도 순으로 정렬되어 있음
    for rank, r in enumerate(vector_results):
        text = r["text"]
        if text not in scores:
            scores[text] = {"score": 0, "data": r}
        scores[text]["score"] += 1.0 / (k + rank + 1)

    # 키워드 결과: match_count DESC로 정렬되어 있음
    for rank, r in enumerate(keyword_results):
        text = r["text"]
        if text not in scores:
            scores[text] = {"score": 0, "data": r}
        scores[text]["score"] += 1.0 / (k + rank + 1)

    # RRF 점수 기준 정렬
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["data"] for item in ranked[:top_k]]


# ── Citation 포맷 ──
def _format_citation(result: dict) -> str:
    """출처 문자열 생성"""
    time_range = f"{_format_time(result['start_time'])}~{_format_time(result['end_time'])}"
    return f"{result['course_title']} > {result['session_title']} > {time_range}"


# ══════════════════════════════════════
#  메인 검색 함수 (main.py에서 호출)
# ══════════════════════════════════════
def search(question: str, top_k: int = 5) -> dict:
    """
    Hybrid Search + Multi-query + Citation

    Returns:
        {
            "context": "검색된 내용을 정리한 문자열 (LLM 프롬프트용)",
            "citations": [
                {"text": "chunk 내용", "citation": "컴퓨터공학개론 > 2주차 > 3:00~6:00"},
                ...
            ]
        }
    """
    # 1. Multi-query 확장
    queries = _expand_queries(question)

    # 2. 각 쿼리로 Hybrid Search
    all_vector = []
    all_keyword = []
    for q in queries:
        all_vector.extend(_vector_search(q, top_k=3))
        all_keyword.extend(_keyword_search(q, top_k=3))

    # 3. 병합 & 중복 제거
    results = _merge_results(all_vector, all_keyword, top_k=top_k)

    if not results:
        return {"context": "", "citations": []}

    # 4. Context 문자열 구성 (LLM에 전달할 참고자료)
    context_parts = []
    citations = []
    for i, r in enumerate(results, 1):
        citation = _format_citation(r)
        context_parts.append(f"[{i}] {r['text']} (출처: {citation})")
        citations.append({
            "text": r["text"],
            "citation": citation,
            "course_title": r["course_title"],
            "session_title": r["session_title"],
            "session_date": r["session_date"],
            "start_time": r["start_time"],
            "end_time": r["end_time"],
        })

    context = "\n".join(context_parts)

    return {"context": context, "citations": citations}


# 테스트용
if __name__ == "__main__":
    result = search("생필품이 뭐야?")
    print("=== Context (LLM에 전달) ===")
    print(result["context"])
    print()
    print("=== Citations ===")
    for c in result["citations"]:
        print(f"  - {c['citation']}: {c['text']}")
