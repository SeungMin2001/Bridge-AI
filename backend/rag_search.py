"""
RAG 검색 모듈
- Hybrid Search (벡터 유사도 + 키워드 BM25)
- Query Rewriting + Multi-query
- Citation (출처 표시)
- 실시간 전사 임베딩 추가
"""
import re
import psycopg2
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

# ── 임베딩 모델 (서버 시작 시 1회 초기화) ──
_embed_model = None
_vector_store = None
_index = None
_initialized = False


def init():
    """서버 시작 시 1회 호출. 임베딩 모델 + vector store 로드."""
    global _embed_model, _vector_store, _index, _initialized
    if _initialized:
        return

    print("[RAG] 임베딩 모델 로딩 중...")
    _embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    Settings.embed_model = _embed_model

    _vector_store = PGVectorStore.from_params(
        database="rag",
        host="localhost",
        password="1234",
        port=5432,
        user="postgres",
        table_name="rag",
        embed_dim=1024,
    )
    _index = VectorStoreIndex.from_vector_store(vector_store=_vector_store)
    _initialized = True
    print("[RAG] 초기화 완료")


def add_document(text: str, metadata: dict):
    """실시간 전사 chunk를 임베딩하여 vector store에 추가"""
    init()
    doc = Document(text=text, metadata=metadata)
    _index.insert(doc)
    print(f"[RAG] 문서 추가됨: {text[:30]}...")


# ── 키워드(BM25 대용) 검색: DB에서 직접 텍스트 매칭 ──
def _keyword_search(query: str, top_k: int = 5) -> list[dict]:
    """PostgreSQL ts_rank + LIKE 기반 키워드 검색"""
    conn = psycopg2.connect(
        host="localhost", port=5432,
        database="rag", user="postgres", password="1234"
    )
    cur = conn.cursor()

    # 쿼리를 단어로 분리해서 LIKE 검색
    words = [w.strip() for w in query.split() if len(w.strip()) >= 2]
    if not words:
        cur.close()
        conn.close()
        return []

    # 각 단어에 대해 ILIKE OR 조건 (교정문 우선)
    like_conditions = " OR ".join(["COALESCE(t.corrected_text, t.chunk_text) ILIKE %s" for _ in words])
    like_values = [f"%{w}%" for w in words]

    sql = f"""
        SELECT t.transcript_id, t.session_id, t.chunk_index, t.start_time, t.end_time,
               COALESCE(t.corrected_text, t.chunk_text) AS chunk_text,
               s.title as session_title, s.session_date,
               co.title as course_title
        FROM transcripts t
        JOIN sessions s ON t.session_id = s.session_id
        JOIN courses co ON s.course_id = co.course_id
        WHERE {like_conditions}
        LIMIT %s
    """
    cur.execute(sql, like_values + [top_k])
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for row in rows:
        _, session_id, chunk_index, start_time, end_time, chunk_text, session_title, session_date, course_title = row
        results.append({
            "text": chunk_text,
            "course_title": course_title,
            "session_title": session_title,
            "session_date": str(session_date),
            "start_time": float(start_time),
            "end_time": float(end_time),
            "source": "keyword",
        })
    return results


# ── 벡터 검색 ──
def _vector_search(query: str, top_k: int = 5) -> list[dict]:
    init()
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
    원본 질문 + 핵심 키워드 추출 쿼리를 생성.
    LLM 없이 규칙 기반으로 확장 (속도 우선).
    """
    queries = [question]

    # 불용어 제거 후 핵심어 조합
    stopwords = {"이", "가", "은", "는", "을", "를", "의", "에", "에서", "로", "으로",
                 "와", "과", "도", "만", "뭐", "뭘", "뭔", "어떻게", "무엇", "무슨",
                 "알려줘", "설명해줘", "말해줘", "뭐야", "이야", "인가", "인지", "대해"}
    words = [w for w in re.split(r'\s+', question) if w not in stopwords and len(w) >= 2]

    if words:
        # 핵심 키워드만으로 된 쿼리
        keyword_query = " ".join(words)
        if keyword_query != question:
            queries.append(keyword_query)

    return queries


# ── 시간 포맷팅 ──
def _format_time(seconds: float) -> str:
    """초를 mm:ss 형태로 변환"""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


# ── 중복 제거 및 병합 ──
def _merge_results(vector_results: list, keyword_results: list, top_k: int = 5) -> list[dict]:
    """벡터 + 키워드 결과를 병합하고 중복 제거"""
    seen_texts = set()
    merged = []

    # 벡터 결과 우선
    for r in vector_results:
        if r["text"] not in seen_texts:
            seen_texts.add(r["text"])
            merged.append(r)

    # 키워드 결과 추가
    for r in keyword_results:
        if r["text"] not in seen_texts:
            seen_texts.add(r["text"])
            merged.append(r)

    return merged[:top_k]


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
