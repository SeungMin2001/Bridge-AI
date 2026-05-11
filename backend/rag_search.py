"""
RAG 검색 모듈
- Hybrid Search (벡터 유사도 + 키워드 BM25)
- Query Rewriting + Multi-query
- Citation (출처 표시)
- 실시간 전사 임베딩 추가
"""
import logging
import os
import json
from datetime import datetime, timedelta
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
    # 신창영 : 키워드 검색과 벡터 검색이 서로 다른 DB를 보지 않도록 공통 DB 설정을 사용
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("DB_NAME", "shin"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "1234"),
    }


def _json_value(value, fallback):
    if value in (None, ""):
        return fallback
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return fallback


def _iter_resource_items(value, resource_key: str):
    items = _json_value(value, [])
    if not isinstance(items, list):
        return

    for entry in items:
        if not isinstance(entry, dict):
            continue

        nested_items = entry.get(resource_key)
        if isinstance(nested_items, list):
            for item in nested_items:
                if isinstance(item, dict):
                    yield item
        else:
            yield entry


def _default_recording_title(file_title: str) -> str:
    if not file_title:
        return "녹음본"
    if "전사" in file_title:
        return file_title.replace("전사", "녹음")
    return f"{file_title} 녹음"


def _parse_datetime(value):
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def _recording_title(recording: dict) -> str:
    return str(recording.get("title") or recording.get("name") or "").strip()


def _recording_id(recording: dict) -> str:
    return str(recording.get("id") or recording.get("recordingId") or recording.get("recording_id") or "").strip()


def _extract_recording_title(session_voicefile, file_title: str, transcript_created_at=None, recording_id: str = "") -> str:
    # 신창영 : 같은 파일 안의 여러 녹음본 중 전사 저장 시각과 가장 가까운 녹음본 제목을 citation에 사용
    recordings = list(_iter_resource_items(session_voicefile, "recordings"))
    if recording_id:
        for recording in recordings:
            if _recording_id(recording) == str(recording_id):
                title = _recording_title(recording)
                if title:
                    return title

    transcript_time = _parse_datetime(transcript_created_at)
    if transcript_time:
        candidates = []
        for recording in recordings:
            title = _recording_title(recording)
            if not title:
                continue
            started_at = _parse_datetime(recording.get("startedAt") or recording.get("started_at"))
            ended_at = _parse_datetime(recording.get("endedAt") or recording.get("ended_at"))
            if started_at and ended_at and started_at - timedelta(seconds=30) <= transcript_time <= ended_at + timedelta(seconds=30):
                return title
            if ended_at and transcript_time <= ended_at + timedelta(seconds=30):
                candidates.append((ended_at, title))
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]

    for recording in recordings:
        title = _recording_title(recording)
        if title:
            return title
    return _default_recording_title(file_title)


def _transcription_item_text(item: dict) -> str:
    if not isinstance(item, dict):
        return ""
    if item.get("text"):
        return str(item["text"]).strip()
    segments = item.get("segments")
    if isinstance(segments, list):
        return " ".join(
            str(segment.get("text") or "").strip()
            for segment in segments
            if isinstance(segment, dict) and str(segment.get("text") or "").strip()
        ).strip()
    return ""


def _recording_transcript_text(recording: dict) -> str:
    transcriptions = recording.get("transcriptions")
    if not isinstance(transcriptions, list):
        return ""
    return "\n\n".join(
        text for text in (_transcription_item_text(item) for item in transcriptions) if text
    )


def _find_recording(session_voicefile, file_title: str, recording_title: str = "", transcript_created_at=None, recording_id: str = "") -> dict | None:
    recordings = list(_iter_resource_items(session_voicefile, "recordings"))
    if not recordings:
        return None

    if recording_id:
        for recording in recordings:
            if _recording_id(recording) == str(recording_id):
                return recording

    normalized_title = str(recording_title or "").strip()
    if normalized_title:
        for recording in recordings:
            if _recording_title(recording) == normalized_title:
                return recording

    inferred_title = _extract_recording_title(session_voicefile, file_title, transcript_created_at)
    for recording in recordings:
        if _recording_title(recording) == inferred_title:
            return recording
    return None


def _get_session_label(session_id: str | None, transcript_created_at=None, recording_id: str = "") -> dict | None:
    """세션 id로 DB에 저장된 실제 파일명/녹음본명을 조회합니다."""
    if not session_id:
        return None

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**_db_config())
        cur = conn.cursor()
        cur.execute(
            """
            SELECT s.title as session_title,
                   s.session_date,
                   s.session_voicefile,
                   s.audio_path,
                   co.title as course_title
            FROM sessions s
            LEFT JOIN courses co ON s.course_id = co.course_id
            WHERE s.session_id = %s
            """,
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return None

        session_title, session_date, session_voicefile, audio_path, course_title = row
        file_title = session_title or "전사 파일"
        recording_title = _extract_recording_title(session_voicefile, file_title, transcript_created_at, recording_id)
        if recording_title == "녹음본" and audio_path:
            recording_title = os.path.basename(str(audio_path))

        return {
            "file_title": file_title,
            "recording_title": recording_title,
            "recording_id": recording_id,
            "course_title": course_title or "미분류",
            "session_title": file_title,
            "session_date": str(session_date) if session_date else "",
        }
    except Exception as exc:
        logger.warning("[RAG] 세션 제목 조회 실패: %s", exc)
        return None
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def _get_recording_transcript(session_id: str | None, recording_title: str = "", transcript_created_at=None, recording_id: str = "") -> str:
    """신창영 : 출처 패널에는 DB 전체 전사보다 저장된 녹음본 전사문을 우선 노출"""
    if not session_id:
        return ""

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**_db_config())
        cur = conn.cursor()
        cur.execute(
            """
            SELECT title, session_voicefile
            FROM sessions
            WHERE session_id = %s
            """,
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return ""

        file_title, session_voicefile = row
        recording = _find_recording(session_voicefile, file_title or "전사 파일", recording_title, transcript_created_at, recording_id)
        return _recording_transcript_text(recording) if recording else ""
    except Exception as exc:
        logger.warning("[RAG] 녹음본 전사문 조회 실패: %s", exc)
        return ""
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def _get_full_transcript(session_id: str | None) -> str:
    """출처 팝오버에서 보여줄 세션 전체 전사문을 조회합니다."""
    if not session_id:
        return ""

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**_db_config())
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COALESCE(corrected_text, chunk_text, '')
            FROM transcripts
            WHERE session_id = %s
            ORDER BY chunk_index ASC NULLS LAST, start_time ASC NULLS LAST, created_at ASC
            """,
            (session_id,),
        )
        return "\n\n".join(row[0] for row in cur.fetchall() if row[0])
    except Exception as exc:
        logger.warning("[RAG] 전체 전사문 조회 실패: %s", exc)
        return ""
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def init():
    """서버 시작 시 1회 호출. 임베딩 모델 + vector store 로드."""
    global _embed_model, _vector_store, _index, _initialized, _init_error
    if _initialized:
        return

    try:
        print("[RAG] 임베딩 모델 로딩 중...")
        _embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
        Settings.embed_model = _embed_model

        # 신창영 : 기존에는 shin DB와 postgres 계정을 코드에 고정, 현재는 실행 환경변수 DB 설정을 사용
        # _vector_store = PGVectorStore.from_params(
        #     database="shin",
        #     host="localhost",
        #     password="1234",
        #     port=5432,
        #     user="postgres",
        #     table_name="shin",
        #     embed_dim=1024,
        # )
        db_config = _db_config()
        _vector_store = PGVectorStore.from_params(
            database=db_config["database"],
            host=db_config["host"],
            password=db_config["password"],
            port=db_config["port"],
            user=db_config["user"],
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
        # logger.warning("[RAG] 초기화 실패; 벡터 검색 비활성화: %s", exc)


def add_document(text: str, metadata: dict):
    """실시간 전사 chunk를 임베딩하여 vector store에 추가"""
    init()
    if _index is None:
        # logger.warning("[RAG] 벡터 스토어 미초기화로 문서 추가 스킵")
        return
    doc = Document(text=text, metadata=metadata)
    _index.insert(doc)
    # print(f"[RAG] 문서 추가됨: {text[:30]}...")


# ── 키워드(BM25 대용) 검색: DB에서 직접 텍스트 매칭 ──
# 신창영 : 기존에는 session_id 필터 없이 전체 transcripts를 대상으로 키워드 검색
# def _keyword_search(query: str, top_k: int = 5) -> list[dict]:
def _keyword_search(query: str, top_k: int = 5, session_id: str | None = None) -> list[dict]:
    """PostgreSQL ts_rank + LIKE 기반 키워드 검색"""
    # 신창영 : DB 접속 정보는 _db_config()에서 환경변수 기반으로 통합
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

    # 신창영 : 기존 chunks 테이블 검색 코드, 현재는 실제 저장 테이블인 transcripts를 검색
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
    # 신창영 : 기존 검색 대상은 t.chunk_text만이었고, 현재는 corrected_text까지 검색
    # like_conditions = " OR ".join(["t.chunk_text ILIKE %s" for _ in words])
    search_text_expr = "COALESCE(t.corrected_text, t.chunk_text, '')"
    like_conditions = " OR ".join([f"{search_text_expr} ILIKE %s" for _ in words])
    like_values = [f"%{w}%" for w in words]

    # 키워드 매칭 개수를 점수로 계산하여 ORDER BY
    # 신창영 : 기존 점수 계산도 t.chunk_text만 기준이라 corrected_text 기준으로 보정
    # match_score = " + ".join(["CASE WHEN t.chunk_text ILIKE %s THEN 1 ELSE 0 END" for _ in words])
    match_score = " + ".join([f"CASE WHEN {search_text_expr} ILIKE %s THEN 1 ELSE 0 END" for _ in words])
    score_values = [f"%{w}%" for w in words]

    where_clauses = [f"({like_conditions})"]
    values = score_values + like_values
    if session_id:
        where_clauses.append("t.session_id = %s")
        values.append(session_id)

    # 신창영 : 기존 SQL은 WHERE {like_conditions}만 사용하여 다른 녹음의 전사문이 섞일 수 있었음
    # WHERE {like_conditions}
    sql = f"""
        SELECT t.transcript_id, t.session_id, t.recording_id, t.chunk_index, t.start_time, t.end_time, t.created_at,
               COALESCE(t.corrected_text, t.chunk_text, '') AS chunk_text,
               s.title as session_title, s.session_date, s.session_voicefile,
               co.title as course_title,
               ({match_score}) as match_count
        FROM transcripts t
        JOIN sessions s ON t.session_id = s.session_id
        LEFT JOIN courses co ON s.course_id = co.course_id
        WHERE {" AND ".join(where_clauses)}
        ORDER BY match_count DESC
        LIMIT %s
    """
    cur.execute(sql, values + [top_k])
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for row in rows:
        transcript_id, row_session_id, recording_id, chunk_index, start_time, end_time, created_at, chunk_text, session_title, session_date, session_voicefile, course_title, match_count = row
        file_title = session_title or "전사 파일"
        recording_title = _extract_recording_title(session_voicefile, file_title, created_at, recording_id or "")
        results.append({
            "text": chunk_text,
            "file_title": file_title,
            "recording_title": recording_title,
            "course_title": course_title or "미분류",
            "session_title": file_title,
            "session_date": str(session_date) if session_date else "",
            "start_time": float(start_time or 0),
            "end_time": float(end_time or 0),
            "transcript_id": str(transcript_id),
            "recording_id": recording_id or "",
            "chunk_index": chunk_index,
            "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or ""),
            "session_id": str(row_session_id) if row_session_id else "",
            "source": "keyword",
        })
    return results


# ── 벡터 검색 ──
# 신창영 : 기존에는 session_id 필터 없이 벡터 검색 top_k만 조회
# def _vector_search(query: str, top_k: int = 5) -> list[dict]:
def _vector_search(query: str, top_k: int = 5, session_id: str | None = None) -> list[dict]:
    init()
    if _index is None:
        if _init_error is not None:
            logger.warning("[RAG] 벡터 검색 비활성화: %s", _init_error)
        return []
    # 신창영 : session_id 필터 후 결과 부족을 줄이기 위해 후보를 더 넓게 조회
    # retriever = _index.as_retriever(similarity_top_k=top_k)
    retriever = _index.as_retriever(similarity_top_k=top_k * 4 if session_id else top_k)
    nodes = retriever.retrieve(query)
    session_label_cache = {}

    results = []
    for node in nodes:
        m = node.metadata
        row_session_id = str(m.get("session_id", ""))
        # 신창영 : 기존에는 이 필터가 없어 다른 세션의 벡터 결과가 섞일 수 있었음
        if session_id and row_session_id != str(session_id):
            continue

        # 신창영 : 파일 삭제 후 남은 오래된 벡터 노드는 세션 DB에 존재할 때만 노출
        # 신창영 : citation 제목은 stale metadata가 아니라 최신 sessions/session_voicefile 값으로 보정
        if not row_session_id:
            continue
        created_at = m.get("created_at", "")
        recording_id = str(m.get("recording_id", ""))
        label_cache_key = (row_session_id, str(created_at or ""), recording_id)
        if label_cache_key not in session_label_cache:
            session_label_cache[label_cache_key] = _get_session_label(row_session_id, created_at, recording_id)
        session_label = session_label_cache[label_cache_key]
        if session_label is None:
            continue

        results.append({
            "text": node.text,
            "file_title": session_label.get("file_title") or m.get("session_title", ""),
            "recording_title": session_label.get("recording_title") or m.get("session_title", ""),
            "course_title": session_label.get("course_title") or m.get("course_title", ""),
            "session_title": session_label.get("session_title") or m.get("session_title", ""),
            "session_date": session_label.get("session_date") or m.get("session_date", ""),
            "start_time": m.get("start_time", 0),
            "end_time": m.get("end_time", 0),
            "session_id": row_session_id,
            "recording_id": recording_id,
            "transcript_id": str(m.get("transcript_id", "")),
            "chunk_index": m.get("chunk_index", None),
            "created_at": created_at,
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
        elif scores[text]["data"].get("source") == "vector":
            scores[text]["data"] = {**scores[text]["data"], **r}
        scores[text]["score"] += 1.0 / (k + rank + 1)

    # RRF 점수 기준 정렬
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["data"] for item in ranked[:top_k]]


def _run_hybrid_search(queries: list[str], top_k: int = 5, session_id: str | None = None) -> list[dict]:
    """여러 검색어에 대해 벡터 검색과 키워드 검색을 실행한 뒤 RRF로 병합."""
    all_vector = []
    all_keyword = []

    for query in queries:
        all_vector.extend(_vector_search(query, top_k=top_k, session_id=session_id))
        all_keyword.extend(_keyword_search(query, top_k=top_k, session_id=session_id))

    return _merge_results(all_vector, all_keyword, top_k=top_k)


# ── Citation 포맷 ──
def _format_citation(result: dict) -> str:
    """출처 문자열 생성"""
    time_range = f"{_format_time(result['start_time'])}~{_format_time(result['end_time'])}"
    return f"{result['session_title']} > {time_range}"


# ══════════════════════════════════════
#  메인 검색 함수 (main.py에서 호출)
# ══════════════════════════════════════
# 신창영 : 기존에는 session_id를 받지 않고 전체 자료에서 검색
# def search(question: str, top_k: int = 5) -> dict:
def search(question: str, top_k: int = 5, session_id: str | None = None) -> dict:
    """
    Hybrid Search + Multi-query + Citation

    Returns:
        {
            "context": "검색된 내용을 정리한 문자열 (LLM 프롬프트용)",
            "citations": [
                {"text": "chunk 내용", "citation": "2주차 > 3:00~6:00"},
                ...
            ]
        }
    """
    # 1. Multi-query 확장
    queries = _expand_queries(question)

    # 2. 현재 파일에서 먼저 검색하고, 없으면 전체 파일에서 다시 검색
    # 신창영 : 선택 세션 기준 검색 후 결과가 없을 때만 전체 검색으로 fallback
    # all_vector.extend(_vector_search(q, top_k=3))
    # all_keyword.extend(_keyword_search(q, top_k=3))
    results = _run_hybrid_search(queries, top_k=top_k, session_id=session_id)
    search_scope = "current_file" if session_id else "all_files"
    if session_id and not results:
        results = _run_hybrid_search(queries, top_k=top_k, session_id=None)
        search_scope = "all_files_fallback"

    if not results:
        return {"context": "", "citations": []}

    # 4. Context 문자열 구성 (LLM에 전달할 참고자료)
    context_parts = []
    citations = []
    full_transcript_cache = {}
    recording_transcript_cache = {}
    for i, r in enumerate(results, 1):
        citation = _format_citation(r)
        result_session_id = r.get("session_id") or session_id
        # 신창영 : 참조 패널에는 가능하면 파일 전체 전사보다 해당 녹음본 전사문을 우선 제공
        recording_cache_key = (
            result_session_id,
            r.get("recording_id") or "",
            r.get("recording_title") or "",
            str(r.get("created_at") or ""),
        )
        if recording_cache_key not in recording_transcript_cache:
            recording_transcript_cache[recording_cache_key] = _get_recording_transcript(
                result_session_id,
                r.get("recording_title") or "",
                r.get("created_at"),
                r.get("recording_id") or "",
            )
        if result_session_id not in full_transcript_cache:
            full_transcript_cache[result_session_id] = _get_full_transcript(result_session_id)
        full_transcript = recording_transcript_cache.get(recording_cache_key) or full_transcript_cache.get(result_session_id) or r["text"]
        context_parts.append(f"[{i}] {r['text']} (출처: {citation})")
        citations.append({
            "text": r["text"],
            "citation": citation,
            "course_title": r["course_title"],
            "session_title": r["session_title"],
            "session_date": r["session_date"],
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "transcript_id": r.get("transcript_id", ""),
            "recording_id": r.get("recording_id", ""),
            "chunk_index": r.get("chunk_index", None),
            "created_at": r.get("created_at", ""),
            "session_id": r.get("session_id", ""),
            "search_scope": search_scope,
            "full_transcript": full_transcript,
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
