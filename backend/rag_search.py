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
import re
from datetime import datetime, timedelta
import psycopg2
from kiwipiepy import Kiwi
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from db_config import psycopg2_config
from materials.material_citation_service import build_material_citation, format_material_citation

# ── 형태소 분석기 (Kiwi) ──
_kiwi = Kiwi()

# 키워드 검색에 사용할 품사 태그 (명사, 동사 어간, 형용사 어간)
_KEYWORD_TAGS = {"NNG", "NNP", "VV", "VA", "SL"}  # 일반명사, 고유명사, 동사, 형용사, 외국어
_ALNUM_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]*")
_LOCATOR_QUERY_STOPWORDS = {
    "혹시",
    "무엇",
    "뭐",
    "어디",
    "어느",
    "위치",
    "찾",
    "찾아",
    "검색",
    "언급",
    "부분",
    "구간",
    "파일",
    "녹음",
    "녹음본",
    "전사",
    "자료",
    "내용",
    "말",
    "나오",
    "포함",
    "포함되",
    "있는",
    "있어",
    "있나요",
    "관련",
    "해당",
    "대한",
    "대해",
    "대해서",
    "특정",
    "단어",
    "표현",
    "키워드",
    "근거",
    "링크",
    "보여",
    "보여줘",
}
_GROUNDED_LOOKUP_TERMS = (
    "누구",
    "언제",
    "언제까지",
    "몇 시",
    "몇시",
    "마감",
    "마감일",
    "기한",
    "제출",
    "제출일",
    "제출해야",
    "까지",
    "무엇",
    "뭐야",
    "뭐여",
    "뭐냐",
    "뭐임",
    "뭐였",
    "뭐였지",
    "뭐라고",
    "뭐라",
    "뭐에요",
    "뭐예요",
    "뭔가",
    "뭔데",
    "무슨",
    "의미",
    "정의",
    "설명",
    "알려",
    "개념",
    "뜻",
    "이란",
    "란",
    "요약",
    "정리",
)
_GROUNDED_LOOKUP_STOPWORDS = _LOCATOR_QUERY_STOPWORDS | {
    "강의",
    "교수",
    "교수님",
    "녹음",
    "녹음본",
    "내용",
    "데이터",
    "비정형",
    "비정형데이터",
    "세션",
    "자료",
    "파일",
    "페이지",
    "pdf",
    "뭐라고",
    "뭐라",
    "뭐였",
    "뭐였지",
    "했어",
    "했었어",
    "했었지",
    "어떻게",
    "왜",
    "설명",
    "설명했어",
    "설명해",
    "알려",
}
_WEAK_RELEVANCE_TERMS = _GROUNDED_LOOKUP_STOPWORDS | {
    "함수",
    "정의",
    "의미",
    "개념",
    "설명",
    "정리",
    "요약",
    "역할",
    "특징",
    "차이",
    "관련",
    "답변",
    "질문",
}
_PROGRAMMING_QUERY_HINTS = (
    "코드",
    "구현",
    "컴파일",
    "에러",
    "오류",
    "프로그래밍",
    "c언어",
    "c 언어",
    "파이썬",
    "python",
    "java",
    "javascript",
    "함수 구현",
    "소스",
    "알고리즘",
)
_ASSIGNMENT_QUERY_HINTS = ("과제", "제출", "마감", "기한", "언제까지", "실습", "숙제", "보고서")
_ASSIGNMENT_CONTEXT_HINTS = (
    "과제",
    "제출",
    "마감",
    "실습",
    "다음 시간까지",
    "풀어오",
    "보고서",
)
_CODE_HEAVY_RE = re.compile(
    r"(#include|#define|printf\s*\(|scanf\s*\(|\bvoid\s+\w+\s*\(|"
    r"\bint\s+\w+\s*\(|\breturn\s+0\s*;|```|</?\w+>|[{};]{4,})",
    re.IGNORECASE,
)


def extract_keywords(text: str) -> list[str]:
    """형태소 분석으로 명사/동사어간/형용사어간만 추출"""
    tokens = _kiwi.tokenize(text)
    keywords = []
    for token in tokens:
        if token.tag in _KEYWORD_TAGS and len(token.form) >= 2:
            keywords.append(token.form)

    # 신창영: 수정 이유 - Kiwi 품사 필터만 쓰면 MMC 같은 영문 약어가 누락될 수 있어 원문에서 별도로 보존합니다.
    keywords.extend(_ALNUM_TERM_RE.findall(str(text or "")))

    deduped = []
    seen = set()
    for keyword in keywords:
        clean_keyword = str(keyword or "").strip()
        if not clean_keyword:
            continue
        key = clean_keyword.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clean_keyword)
    return deduped


def _extract_search_terms(
    query: str,
    *,
    locator_query: bool = False,
    grounded_lookup_query: bool = False,
) -> list[str]:
    """질문 유형에 맞게 DB 키워드 검색어를 정리합니다."""
    keywords = extract_keywords(query)
    if not locator_query and not grounded_lookup_query:
        return keywords

    if grounded_lookup_query:
        grounded_terms = _extract_grounded_lookup_terms(query)
        if grounded_terms:
            return grounded_terms

    lookup_terms = _extract_lookup_terms(query)
    if lookup_terms:
        return lookup_terms

    filtered = [
        keyword
        for keyword in keywords
        if keyword.lower() not in _LOCATOR_QUERY_STOPWORDS
        and keyword not in _LOCATOR_QUERY_STOPWORDS
    ]
    return filtered or keywords


def _clean_lookup_term(value: str) -> str:
    """질문에서 추출한 검색 대상 표현의 조사/불필요한 말을 정리합니다."""
    term = re.sub(r"^(혹시|그럼|그러면|저기|혹은|그|이|저)\s*", "", str(value or "")).strip()
    term = re.sub(r"[\s,.:;!?？]+$", "", term).strip()
    term = re.sub(r"(은|는|이|가|을|를|에|에서|으로|로|도|만)$", "", term).strip()
    return term


def _append_lookup_term(terms: list[str], value: str):
    term = _clean_lookup_term(value)
    if len(term) < 2:
        return
    if term.lower() in _LOCATOR_QUERY_STOPWORDS or term in _LOCATOR_QUERY_STOPWORDS:
        return
    if term not in terms:
        terms.append(term)


def _extract_lookup_terms(query: str) -> list[str]:
    """언급 위치/근거 찾기 질문에서 사용자가 찾는 실제 표현을 우선 추출합니다."""
    text = " ".join(str(query or "").split())
    terms: list[str] = []

    for match in re.finditer(r"['\"“”‘’](.+?)['\"“”‘’]", text):
        _append_lookup_term(terms, match.group(1))

    marker_patterns = (
        r"([A-Za-z][A-Za-z0-9_+#.-]*|[가-힣A-Za-z0-9_+#.-]{2,30})\s*(?:에\s*대한|에대한|에\s*대해|에대해|에\s*대해서|에대해서)",
        r"([A-Za-z][A-Za-z0-9_+#.-]*|[가-힣A-Za-z0-9_+#.-]{2,30})\s*(?:라고|이라는|라는)",
        r"(?:단어|표현|키워드)\s*[\"'“”‘’]?\s*([A-Za-z][A-Za-z0-9_+#.-]*|[가-힣A-Za-z0-9_+#.-]{2,30})",
    )
    for pattern in marker_patterns:
        for match in re.finditer(pattern, text):
            _append_lookup_term(terms, match.group(1))

    if terms:
        return terms

    for keyword in extract_keywords(text):
        if keyword.lower() in _LOCATOR_QUERY_STOPWORDS or keyword in _LOCATOR_QUERY_STOPWORDS:
            continue
        _append_lookup_term(terms, keyword)

    return terms


def _is_current_scope_query(question: str) -> bool:
    """사용자가 명시적으로 현재/선택 파일만 묻는지 판별합니다."""
    text = str(question or "")
    current_scope_terms = (
        "현재 파일",
        "이 파일",
        "여기 파일",
        "현재 여기에",
        "여기에 저장",
        "선택된",
        "열려 있는",
        "열려있는",
        "지금 파일",
        "이 강의",
        "이 자료",
        "이 내용",
        "여기 내용",
        "현재 내용",
    )
    return any(term in text for term in current_scope_terms)


def _is_grounded_lookup_query(question: str) -> bool:
    """특정 인물/용어의 정체나 의미를 저장 자료 근거로 묻는 질문인지 판별합니다."""
    text = str(question or "").strip()
    if not text:
        return False
    if _is_elliptic_grounded_lookup_query(text):
        return True
    if not any(term in text for term in _GROUNDED_LOOKUP_TERMS):
        return False
    return bool(_extract_grounded_lookup_terms(text) or _ALNUM_TERM_RE.search(text))


def _is_elliptic_grounded_lookup_query(question: str) -> bool:
    """'신승민은?'처럼 질문 술어가 생략된 짧은 조회형 질문을 키워드 우선 검색으로 보냅니다."""
    text = " ".join(str(question or "").split())
    if not text.endswith(("?", "？")):
        return False
    if any(term in text for term in ("어디", "어느", "몇", "위치", "파일", "자료", "녹음", "페이지", "구간")):
        return False
    body = text[:-1].strip()
    if len(body) < 2 or len(body) > 40:
        return False
    term = _clean_lookup_term(body)
    if len(term) < 2:
        return False
    return term.lower() not in _GROUNDED_LOOKUP_STOPWORDS and term not in _GROUNDED_LOOKUP_STOPWORDS


def _extract_grounded_lookup_terms(query: str) -> list[str]:
    """'오태진 교수 누구야' 같은 질문에서 실제 조회할 핵심 표현을 추출합니다."""
    text = " ".join(str(query or "").split())
    focus = text
    terms: list[str] = []

    # "교수님이 전위를 뭐라고 설명했지?"처럼 주어/발화자가 앞에 붙은 질문에서는
    # 발화자가 아니라 설명 대상 용어를 검색 핵심어로 잡아야 한다.
    object_patterns = (
        r"(?:교수님|교수|선생님|강의|수업).{0,20}?([가-힣A-Za-z0-9_+#.-]{2,30})\s*(?:은|는|이|가|을|를)?\s*(?:뭐라고|뭐라|어떻게)?\s*(?:설명|말|정의|언급)",
        r"([가-힣A-Za-z0-9_+#.-]{2,30})\s*(?:은|는|이|가|을|를)?\s*(?:뭐라고|뭐라|어떻게)?\s*(?:설명|정의|말|언급)",
    )
    for pattern in object_patterns:
        match = re.search(pattern, text)
        if match:
            _append_lookup_term(terms, match.group(1))
            if terms:
                return terms

    question_match = re.search(
        r"(.+?)(?:누구|무엇|뭐야|뭐여|뭐냐|뭐임|뭐였|뭐였지|뭐라고|뭐라|뭐에요|뭐예요|뭔가|뭔데|무슨|의미|정의|설명|알려|개념|뜻|이란|란|요약|정리)",
        text,
    )
    if question_match:
        focus = question_match.group(1)

    concept_match = re.search(r"(.+?)(?:의\s*)?(?:개념|정의|의미|뜻|이란|란|요약|정리)", focus)
    if concept_match:
        focus = concept_match.group(1)
    else:
        source_parts = re.split(r"\s*(?:의|에서|에는|에)\s+", focus)
        source_parts = [part.strip() for part in source_parts if part.strip()]
        if len(source_parts) > 1:
            focus = source_parts[-1]

    focus = re.sub(r"(은|는|이|가|을|를|의|에|에서|으로|로|도|만)\s*$", "", focus).strip()

    alnum_terms = _ALNUM_TERM_RE.findall(focus)
    if alnum_terms:
        return alnum_terms

    _append_lookup_term(terms, focus)
    if terms:
        return terms

    for match in re.finditer(r"[가-힣A-Za-z0-9_+#.-]{2,30}", focus):
        term = _clean_lookup_term(match.group(0))
        if len(term) < 2:
            continue
        if term.lower() in _GROUNDED_LOOKUP_STOPWORDS or term in _GROUNDED_LOOKUP_STOPWORDS:
            continue
        if term not in terms:
            terms.append(term)
    if terms:
        return terms

    for keyword in extract_keywords(focus):
        term = _clean_lookup_term(keyword)
        if len(term) < 2:
            continue
        if term.lower() in _GROUNDED_LOOKUP_STOPWORDS or term in _GROUNDED_LOOKUP_STOPWORDS:
            continue
        if term not in terms:
            terms.append(term)
    return terms

# ── 임베딩 모델 (서버 시작 시 1회 초기화) ──
_embed_model = None
_vector_store = None
_index = None
_initialized = False
_init_error = None

logger = logging.getLogger(__name__)
SELECTED_TRANSCRIPT_CONTEXT_MAX_CHARS = int(os.getenv("CHAT_SELECTED_TRANSCRIPT_CONTEXT_MAX_CHARS", "16000"))
RAG_DEBUG = os.getenv("CHAT_RAG_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
RAG_FAST_KEYWORD_FIRST = os.getenv("CHAT_RAG_FAST_KEYWORD_FIRST", "1").strip().lower() in {"1", "true", "yes", "on"}
RAG_FAST_KEYWORD_MIN_RESULTS = max(1, int(os.getenv("CHAT_RAG_FAST_KEYWORD_MIN_RESULTS", "2")))
RAG_FAST_KEYWORD_MIN_HITS = max(1, int(os.getenv("CHAT_RAG_FAST_KEYWORD_MIN_HITS", "1")))
RAG_USE_VECTOR_SEARCH = os.getenv("CHAT_RAG_USE_VECTOR_SEARCH", "1").strip().lower() in {"1", "true", "yes", "on"}
RAG_VECTOR_CANDIDATE_MULTIPLIER = max(1, int(os.getenv("CHAT_RAG_VECTOR_CANDIDATE_MULTIPLIER", "2")))
RAG_PREFETCH_FULL_TRANSCRIPT = os.getenv("CHAT_RAG_PREFETCH_FULL_TRANSCRIPT", "0").strip().lower() in {"1", "true", "yes", "on"}
RAG_CONTEXT_NEIGHBOR_CHUNKS = max(0, int(os.getenv("CHAT_RAG_CONTEXT_NEIGHBOR_CHUNKS", "1")))
RAG_CONTEXT_NEIGHBOR_MAX_CHARS = max(500, int(os.getenv("CHAT_RAG_CONTEXT_NEIGHBOR_MAX_CHARS", "1800")))
RAG_RELEVANT_SENTENCE_COUNT = max(1, int(os.getenv("CHAT_RAG_RELEVANT_SENTENCE_COUNT", "3")))
DEMO_PIPELINE_LOG = os.getenv("RAG_DEMO_PIPELINE_LOG", "0").strip().lower() in {"1", "true", "yes", "on"}


def _demo_log(message: str) -> None:
    if DEMO_PIPELINE_LOG:
        print(f"[DEMO:RAG] {message}", flush=True)


def _preview(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    return compact if len(compact) <= limit else f"{compact[:limit - 3]}..."


def _db_config() -> dict:
    # 신창영 : 키워드 검색과 벡터 검색이 서로 다른 DB를 보지 않도록 공통 DB 설정을 사용
    return psycopg2_config()


def _rag_table_name() -> str:
    """llama_index PGVectorStore가 실제로 쓰는 data_<table_name> 테이블명을 반환합니다."""
    raw_name = os.getenv("RAG_TABLE_NAME", "rag")
    safe_name = re.sub(r"[^A-Za-z0-9_]", "", raw_name) or "rag"
    return f"data_{safe_name}"


def _quote_ident(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _vector_metadata_column(cur, table_name: str) -> str | None:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
          AND column_name IN ('metadata_', 'metadata')
        ORDER BY CASE WHEN column_name = 'metadata_' THEN 0 ELSE 1 END
        LIMIT 1
        """,
        (table_name,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _vector_text_column(cur, table_name: str) -> str | None:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
          AND column_name IN ('text', 'text_', 'content')
        ORDER BY CASE
            WHEN column_name = 'text' THEN 0
            WHEN column_name = 'text_' THEN 1
            ELSE 2
        END
        LIMIT 1
        """,
        (table_name,),
    )
    row = cur.fetchone()
    return row[0] if row else None


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
    """출처 패널에 보여줄 녹음본 전사문을 DB 기준으로 조회합니다."""
    if not session_id:
        return ""

    db_transcript = _get_db_recording_transcript(session_id, recording_id)
    if db_transcript:
        return db_transcript

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


def _get_db_recording_transcript(session_id: str | None, recording_id: str = "") -> str:
    """검색/출처가 같은 DB row를 보도록 recording_id 기준 전사문을 시간순으로 조회합니다."""
    if not session_id:
        return ""

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**_db_config())
        cur = conn.cursor()
        values = [session_id]
        where = ["session_id = %s"]
        if recording_id:
            where.append("recording_id = %s")
            values.append(recording_id)
        cur.execute(
            f"""
            SELECT COALESCE(corrected_text, chunk_text, '')
            FROM transcripts
            WHERE {" AND ".join(where)}
            ORDER BY chunk_index ASC NULLS LAST, start_time ASC NULLS LAST, created_at ASC
            """,
            values,
        )
        return "\n\n".join(row[0] for row in cur.fetchall() if row[0])
    except Exception as exc:
        logger.warning("[RAG] DB 녹음본 전사문 조회 실패: %s", exc)
        return ""
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def _get_full_transcript(session_id: str | None) -> str:
    """출처 팝오버에서 보여줄 세션 전체 전사문을 DB 기준으로 조회합니다."""
    return _get_db_recording_transcript(session_id)


def _expand_transcript_context_window(result: dict, *, radius: int = RAG_CONTEXT_NEIGHBOR_CHUNKS) -> dict:
    """검색된 전사 chunk의 앞뒤 문맥을 함께 가져와 답변 누락을 줄입니다.

    반환된 text/start/end는 citation 범위와 함께 확장되므로, 모델에 전달된
    근거와 프론트에서 열리는 출처 범위가 어긋나지 않습니다.
    """
    if radius <= 0:
        return result
    if result.get("source_type", "transcript") != "transcript":
        return result
    if result.get("source") == "selected_source_context":
        return result

    session_id = str(result.get("session_id") or "").strip()
    recording_id = str(result.get("recording_id") or "").strip()
    chunk_index = result.get("chunk_index")
    if not session_id or chunk_index is None:
        return result

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**_db_config())
        cur = conn.cursor()
        values = [session_id, int(chunk_index) - radius, int(chunk_index) + radius]
        where = [
            "t.session_id::text = %s",
            "t.chunk_index BETWEEN %s AND %s",
        ]
        if recording_id:
            where.append("t.recording_id = %s")
            values.append(recording_id)
        cur.execute(
            f"""
            SELECT t.transcript_id, t.recording_id, t.chunk_index,
                   t.start_time, t.end_time, t.created_at,
                   COALESCE(t.corrected_text, t.chunk_text, '') AS chunk_text
            FROM transcripts t
            WHERE {" AND ".join(where)}
            ORDER BY t.chunk_index ASC NULLS LAST,
                     t.start_time ASC NULLS LAST,
                     t.created_at ASC
            """,
            values,
        )
        rows = cur.fetchall()
    except Exception as exc:
        logger.warning("[RAG] 주변 전사 문맥 조회 실패: %s", exc)
        return result
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    parts = []
    start_times = []
    end_times = []
    transcript_ids = []
    for transcript_id, row_recording_id, row_chunk_index, start_time, end_time, _created_at, chunk_text in rows:
        text = str(chunk_text or "").strip()
        if not text:
            continue
        parts.append(f"[{_format_time(float(start_time or 0))}~{_format_time(float(end_time or 0))}] {text}")
        start_times.append(float(start_time or 0))
        end_times.append(float(end_time or 0))
        transcript_ids.append(str(transcript_id))

    if len(parts) <= 1:
        return result

    expanded_text = "\n".join(parts).strip()
    if len(expanded_text) > RAG_CONTEXT_NEIGHBOR_MAX_CHARS:
        expanded_text = f"{expanded_text[:RAG_CONTEXT_NEIGHBOR_MAX_CHARS].rstrip()}\n...[주변 전사 문맥이 길어 일부만 사용했습니다]"

    return {
        **result,
        "text": expanded_text,
        "start_time": min(start_times) if start_times else result.get("start_time", 0),
        "end_time": max(end_times) if end_times else result.get("end_time", 0),
        "transcript_id": result.get("transcript_id") or (transcript_ids[0] if transcript_ids else ""),
        "context_expanded": True,
    }


def _select_sentence_level_excerpt(text: str, question: str, *, fallback: str = "") -> str:
    """질문과 가장 직접적으로 맞는 문장만 citation 하이라이트 대상으로 고릅니다."""
    source = str(text or "").strip()
    if not source:
        return str(fallback or "").strip()

    keywords = _question_relevance_terms(question)
    strong_terms = _strong_question_terms(question)
    grounded_lookup_query = _is_grounded_lookup_query(question)
    if not keywords:
        return _first_reasonable_sentence(source) or source[:260]

    candidates = []
    for line in source.splitlines():
        clean_line = re.sub(r"^\s*\[\d+:\d{2}~\d+:\d{2}\]\s*", "", line).strip()
        candidates.extend(_split_sentences(clean_line))

    if not candidates:
        candidates = _split_sentences(source)
    if not candidates:
        return source[:260]

    exact_candidates = [
        sentence
        for sentence in candidates
        if _strong_subject_score(sentence, strong_terms) > 0
    ] if grounded_lookup_query and strong_terms else []
    ranked_pool = exact_candidates or candidates
    ranked = sorted(
        ranked_pool,
        key=lambda sentence: (
            _definition_sentence_score(sentence, strong_terms),
            _strong_subject_score(sentence, strong_terms),
            _strong_subject_hit_count(sentence, strong_terms),
            _keyword_hit_count(sentence, keywords),
            _relevance_score(sentence, keywords),
            -abs(len(sentence) - 140),
        ),
        reverse=True,
    )
    best = ranked[0].strip()
    if _keyword_hit_count(best, keywords) <= 0 and _strong_subject_score(best, strong_terms) <= 0:
        fallback_sentence = _first_reasonable_sentence(fallback) if fallback else ""
        return fallback_sentence or best[:260]
    return best[:320]


def _select_relevant_evidence_sentences(text: str, question: str, *, max_sentences: int | None = None) -> list[str]:
    """LLM이 빠뜨리면 안 되는 질문 관련 문장을 근거 안에서 선별합니다."""
    source = str(text or "").strip()
    if not source:
        return []

    terms = _question_relevance_terms(question)
    strong_terms = _strong_question_terms(question)
    grounded_lookup_query = _is_grounded_lookup_query(question)
    max_sentences = max_sentences or RAG_RELEVANT_SENTENCE_COUNT
    candidates: list[tuple[int, int, int, int, int, str]] = []
    all_sentences: list[tuple[int, int, str]] = []
    for order, line in enumerate(source.splitlines() or [source]):
        clean_line = re.sub(r"^\s*\[\d+:\d{2}~\d+:\d{2}\]\s*", "", line).strip()
        for sentence_order, sentence in enumerate(_split_sentences(clean_line)):
            compact = " ".join(sentence.split())
            if len(compact) < 8:
                continue
            all_sentences.append((order, sentence_order, compact))
            score = _relevance_score(compact, terms)
            strong_score = _strong_subject_score(compact, strong_terms)
            strong_hits = _strong_subject_hit_count(compact, strong_terms)
            if score > 0 or strong_score > 0:
                candidates.append((strong_score, strong_hits, score, order, sentence_order, compact))

    if not candidates:
        fallback = _first_reasonable_sentence(source)
        return [fallback] if fallback else []

    # 개념 조회형 질문은 질문 핵심어가 직접 들어간 문장을 citation/highlight의 최우선 근거로 사용합니다.
    exact_candidates = [item for item in candidates if item[0] > 0] if grounded_lookup_query and strong_terms else []
    ranked_pool = exact_candidates or candidates
    top = sorted(
        ranked_pool,
        key=lambda item: (_definition_sentence_score(item[5], strong_terms), item[0], item[1], item[2], -item[3], -item[4], len(item[5])),
        reverse=True,
    )[:max_sentences]

    if exact_candidates and len(top) < max_sentences:
        selected_positions = {(item[3], item[4]) for item in top}
        exact_positions = sorted((item[3], item[4]) for item in exact_candidates)
        for exact_order, exact_sentence_order in exact_positions:
            for neighbor_order, neighbor_sentence_order, neighbor_sentence in all_sentences:
                if neighbor_order != exact_order:
                    continue
                if neighbor_sentence_order <= exact_sentence_order:
                    continue
                if (neighbor_order, neighbor_sentence_order) in selected_positions:
                    continue
                # 정의 문장 직후의 보조 설명은 질문 핵심어가 반복되지 않아도 답변 품질에 필요합니다.
                top.append((0, 0, _relevance_score(neighbor_sentence, terms), neighbor_order, neighbor_sentence_order, neighbor_sentence))
                selected_positions.add((neighbor_order, neighbor_sentence_order))
                if len(top) >= max_sentences:
                    break
            if len(top) >= max_sentences:
                break

    top = sorted(top, key=lambda item: (item[3], item[4]))

    sentences: list[str] = []
    seen = set()
    for _, _, _, _, _, sentence in top:
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        sentences.append(sentence[:420])
    return sentences


def _format_evidence_context_block(index: int, context_text: str, citation: str, relevant_sentences: list[str]) -> str:
    """검색 문맥을 핵심 참고문장 중심으로 재구성해 LLM 입력에 넣습니다."""
    context_text = str(context_text or "").strip()
    if relevant_sentences:
        evidence_lines = "\n".join(f"- {sentence}" for sentence in relevant_sentences)
        return (
            f"[{index}] 핵심 참고문장 (반드시 답변에 반영):\n{evidence_lines}\n"
            f"보조 문맥:\n{context_text}\n"
            f"(출처: {citation})"
        )
    return f"[{index}] {context_text} (출처: {citation})"


def _split_sentences(text: str) -> list[str]:
    """한국어 강의 전사 문장을 citation용으로 너무 길지 않게 나눕니다."""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return []
    pattern = re.compile(r".+?(?:다\.|요\.|입니다\.|습니다\.|[.!?。？！])(?:\s+|$)")
    sentences = []
    last_end = 0
    for match in pattern.finditer(value):
        sentences.append(match.group(0).strip())
        last_end = match.end()
    if sentences:
        tail = value[last_end:].strip()
        if tail:
            sentences.append(tail)
        return sentences
    return [value]


def _first_reasonable_sentence(text: str) -> str:
    for sentence in _split_sentences(text):
        if len(sentence.strip()) >= 12:
            return sentence.strip()
    return str(text or "").strip()[:260]


def _prioritize_selected_evidence(results: list[dict], question: str) -> list[dict]:
    """최종 근거 번호가 질문과 가장 직접적인 문장부터 시작되도록 재정렬합니다."""
    keywords = _question_relevance_terms(question)
    strong_terms = _strong_question_terms(question)
    programming_question = _is_programming_question(question)
    if not keywords or len(results) <= 1:
        return results

    ranked = sorted(
        enumerate(results),
        key=lambda item: (
            _strong_subject_score(item[1].get("text", ""), strong_terms),
            _strong_subject_hit_count(item[1].get("text", ""), strong_terms),
            not (_looks_code_heavy(item[1].get("text", "")) and not programming_question),
            _relevance_score(item[1].get("text", ""), keywords),
            any(word in str(item[1].get("text", "")).casefold() for word in keywords),
            float(item[1].get("score") or item[1].get("match_count") or 0.0),
            -item[0],
        ),
        reverse=True,
    )
    return [item for _, item in ranked]


def _fetch_transcript_row_for_vector_metadata(metadata: dict) -> dict | None:
    """벡터 검색 결과의 metadata를 DB row로 재확인해 stale vector text/citation을 방지합니다."""
    session_id = str(metadata.get("session_id") or "").strip()
    transcript_id = str(metadata.get("transcript_id") or "").strip()
    recording_id = str(metadata.get("recording_id") or "").strip()
    chunk_index = metadata.get("chunk_index")
    if not session_id and not transcript_id:
        return None

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**_db_config())
        cur = conn.cursor()
        values = []
        where_parts = []
        if transcript_id:
            where_parts.append("t.transcript_id::text = %s")
            values.append(transcript_id)
        elif session_id and recording_id and chunk_index is not None:
            where_parts.extend(["t.session_id::text = %s", "t.recording_id = %s", "t.chunk_index = %s"])
            values.extend([session_id, recording_id, int(chunk_index)])
        elif session_id and chunk_index is not None:
            where_parts.extend(["t.session_id::text = %s", "t.chunk_index = %s"])
            values.extend([session_id, int(chunk_index)])
        else:
            return None

        cur.execute(
            f"""
            SELECT t.transcript_id, t.session_id, t.recording_id, t.chunk_index,
                   t.start_time, t.end_time, t.created_at,
                   COALESCE(t.corrected_text, t.chunk_text, '') AS chunk_text,
                   s.title as session_title, s.session_date, s.session_voicefile,
                   co.title as course_title
            FROM transcripts t
            JOIN sessions s ON t.session_id = s.session_id
            LEFT JOIN courses co ON s.course_id = co.course_id
            WHERE {" AND ".join(where_parts)}
            ORDER BY t.created_at DESC NULLS LAST
            LIMIT 1
            """,
            values,
        )
        row = cur.fetchone()
        if not row:
            return None

        (
            row_transcript_id,
            row_session_id,
            row_recording_id,
            row_chunk_index,
            start_time,
            end_time,
            created_at,
            chunk_text,
            session_title,
            session_date,
            session_voicefile,
            course_title,
        ) = row
        if not str(chunk_text or "").strip():
            return None
        file_title = session_title or "전사 파일"
        return {
            "text": chunk_text,
            "file_title": file_title,
            "recording_title": _extract_recording_title(session_voicefile, file_title, created_at, row_recording_id or ""),
            "course_title": course_title or "미분류",
            "session_title": file_title,
            "session_date": str(session_date) if session_date else "",
            "start_time": float(start_time or 0),
            "end_time": float(end_time or 0),
            "session_id": str(row_session_id),
            "recording_id": row_recording_id or "",
            "transcript_id": str(row_transcript_id),
            "chunk_index": row_chunk_index,
            "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or ""),
            "source_type": "transcript",
        }
    except Exception as exc:
        logger.warning("[RAG] 벡터 결과 DB 검증 실패: %s", exc)
        return None
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def _fetch_transcript_rows_by_ids(transcript_ids: list[str]) -> dict[str, dict]:
    """벡터 후보들의 transcript row를 한 번의 DB 조회로 가져옵니다."""
    clean_ids = []
    seen = set()
    for transcript_id in transcript_ids:
        value = str(transcript_id or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        clean_ids.append(value)
    if not clean_ids:
        return {}

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**_db_config())
        cur = conn.cursor()
        cur.execute(
            """
            SELECT t.transcript_id, t.session_id, t.recording_id, t.chunk_index,
                   t.start_time, t.end_time, t.created_at,
                   COALESCE(t.corrected_text, t.chunk_text, '') AS chunk_text,
                   s.title as session_title, s.session_date, s.session_voicefile,
                   co.title as course_title
            FROM transcripts t
            JOIN sessions s ON t.session_id = s.session_id
            LEFT JOIN courses co ON s.course_id = co.course_id
            WHERE t.transcript_id::text = ANY(%s)
            """,
            (clean_ids,),
        )
        rows = cur.fetchall()
    except Exception as exc:
        logger.warning("[RAG] 벡터 후보 batch DB 검증 실패: %s", exc)
        return {}
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    results = {}
    for row in rows:
        (
            row_transcript_id,
            row_session_id,
            row_recording_id,
            row_chunk_index,
            start_time,
            end_time,
            created_at,
            chunk_text,
            session_title,
            session_date,
            session_voicefile,
            course_title,
        ) = row
        if not str(chunk_text or "").strip():
            continue
        file_title = session_title or "전사 파일"
        results[str(row_transcript_id)] = {
            "text": chunk_text,
            "file_title": file_title,
            "recording_title": _extract_recording_title(session_voicefile, file_title, created_at, row_recording_id or ""),
            "course_title": course_title or "미분류",
            "session_title": file_title,
            "session_date": str(session_date) if session_date else "",
            "start_time": float(start_time or 0),
            "end_time": float(end_time or 0),
            "session_id": str(row_session_id),
            "recording_id": row_recording_id or "",
            "transcript_id": str(row_transcript_id),
            "chunk_index": row_chunk_index,
            "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or ""),
            "source_type": "transcript",
        }
    return results


def _truncate_selected_transcript(text: str, max_chars: int = SELECTED_TRANSCRIPT_CONTEXT_MAX_CHARS) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}\n\n...[선택한 녹음본 전사가 길어 일부만 사용했습니다]"


def _selected_transcript_context_search(
    session_id: str | None,
    source_filter: dict | None,
) -> list[dict]:
    """선택한 녹음본/전사에 대해 검색어 매칭이 없어도 원문을 LLM context로 제공합니다."""
    if not session_id:
        return []

    filters = _normalize_source_filter(source_filter)
    if not filters["recording_ids"] and not filters["transcript_ids"]:
        return []

    source_clauses = []
    values = [session_id]
    if filters["recording_ids"]:
        source_clauses.append("t.recording_id = ANY(%s)")
        values.append(list(filters["recording_ids"]))
    if filters["transcript_ids"]:
        source_clauses.append("t.transcript_id::text = ANY(%s)")
        values.append(list(filters["transcript_ids"]))

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**_db_config())
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT t.transcript_id, t.session_id, t.recording_id, t.chunk_index,
                   t.start_time, t.end_time, t.created_at,
                   COALESCE(t.corrected_text, t.chunk_text, '') AS chunk_text,
                   s.title as session_title, s.session_date, s.session_voicefile,
                   co.title as course_title
            FROM transcripts t
            JOIN sessions s ON t.session_id = s.session_id
            LEFT JOIN courses co ON s.course_id = co.course_id
            WHERE t.session_id = %s
              AND ({" OR ".join(source_clauses)})
            ORDER BY t.recording_id ASC NULLS LAST,
                     t.chunk_index ASC NULLS LAST,
                     t.start_time ASC NULLS LAST,
                     t.created_at ASC
            """,
            values,
        )
        rows = cur.fetchall()
    except Exception as exc:
        logger.warning("[RAG] 선택 녹음본 전사 fallback 조회 실패: %s", exc)
        return []
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    groups = {}
    for row in rows:
        (
            transcript_id,
            row_session_id,
            recording_id,
            chunk_index,
            start_time,
            end_time,
            created_at,
            chunk_text,
            session_title,
            session_date,
            session_voicefile,
            course_title,
        ) = row
        text = str(chunk_text or "").strip()
        if not text:
            continue

        key = recording_id or "selected-transcripts"
        if key not in groups:
            file_title = session_title or "전사 파일"
            groups[key] = {
                "chunks": [],
                "file_title": file_title,
                "recording_title": _extract_recording_title(session_voicefile, file_title, created_at, recording_id or ""),
                "course_title": course_title or "미분류",
                "session_title": file_title,
                "session_date": str(session_date) if session_date else "",
                "session_id": str(row_session_id) if row_session_id else str(session_id),
                "recording_id": recording_id or "",
                "first_transcript_id": str(transcript_id),
                "first_chunk_index": chunk_index,
                "first_created_at": created_at,
                "start_time": float(start_time or 0),
                "end_time": float(end_time or 0),
            }

        groups[key]["chunks"].append({
            "text": text,
            "start_time": float(start_time or 0),
            "end_time": float(end_time or 0),
        })
        groups[key]["end_time"] = float(end_time or groups[key]["end_time"] or 0)

    results = []
    for group in groups.values():
        full_transcript = "\n".join(
            f"[{_format_time(chunk['start_time'])}~{_format_time(chunk['end_time'])}] {chunk['text']}"
            for chunk in group["chunks"]
        ).strip()
        if not full_transcript:
            continue

        results.append({
            "text": _truncate_selected_transcript(full_transcript),
            "full_transcript": full_transcript,
            "file_title": group["file_title"],
            "recording_title": group["recording_title"],
            "course_title": group["course_title"],
            "session_title": group["session_title"],
            "session_date": group["session_date"],
            "start_time": group["start_time"],
            "end_time": group["end_time"],
            "transcript_id": group["first_transcript_id"],
            "recording_id": group["recording_id"],
            "chunk_index": group["first_chunk_index"],
            "created_at": group["first_created_at"].isoformat()
            if hasattr(group["first_created_at"], "isoformat")
            else str(group["first_created_at"] or ""),
            "session_id": group["session_id"],
            "source_type": "transcript",
            "source": "selected_source_context",
        })

    return results


def _is_locator_query(question: str) -> bool:
    """특정 단어/주제가 어디에 언급됐는지 찾는 질문인지 간단히 판별합니다."""
    text = str(question or "").strip()
    if not text:
        return False

    locator_terms = ("어디", "어느", "몇 분", "몇초", "몇 초", "위치", "찾", "검색", "근거", "링크", "보여")
    mention_terms = ("언급", "나오", "포함", "있어", "있는", "말했", "다룬", "등장")
    strong_mention_terms = ("언급", "나오", "포함", "말했", "다룬", "등장")
    target_terms = ("파일", "녹음", "전사", "자료", "내용", "부분", "구간", "문장", "대목")
    has_lookup_term = bool(_extract_lookup_terms(text) or _ALNUM_TERM_RE.search(text))
    has_locator = any(term in text for term in locator_terms)
    has_mention = any(term in text for term in mention_terms)
    has_strong_mention = any(term in text for term in strong_mention_terms)
    has_target = any(term in text for term in target_terms)

    return (
        has_lookup_term
        and has_mention
        and (has_locator or has_target or has_strong_mention or "에 대한" in text or "에대한" in text or "라고" in text)
    )


def _normalize_source_filter(source_filter: dict | None) -> dict:
    if not isinstance(source_filter, dict):
        return {
            "material_ids": set(),
            "stored_names": set(),
            "recording_ids": set(),
            "transcript_ids": set(),
        }

    def clean_set(key: str, *, basename: bool = False) -> set[str]:
        values = source_filter.get(key) or []
        if not isinstance(values, list):
            values = [values]
        cleaned = set()
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            cleaned.add(os.path.basename(text) if basename else text)
        return cleaned

    return {
        "material_ids": clean_set("material_ids"),
        "stored_names": clean_set("stored_names", basename=True),
        "recording_ids": clean_set("recording_ids"),
        "transcript_ids": clean_set("transcript_ids"),
    }


def _has_source_filter(filters: dict) -> bool:
    return any(filters.get(key) for key in ("material_ids", "stored_names", "recording_ids", "transcript_ids"))


def _has_material_source_filter(filters: dict) -> bool:
    return bool(filters.get("material_ids") or filters.get("stored_names"))


def _matches_source_filter(metadata: dict, filters: dict) -> bool:
    if not _has_source_filter(filters):
        return True

    source_type = str(metadata.get("source_type") or "transcript")
    if source_type == "material":
        if not filters["material_ids"] and not filters["stored_names"]:
            return False
        material_id = str(metadata.get("material_id") or "")
        stored_name = os.path.basename(str(metadata.get("stored_name") or ""))
        return (
            (material_id and material_id in filters["material_ids"])
            or (stored_name and stored_name in filters["stored_names"])
        )

    if not filters["recording_ids"] and not filters["transcript_ids"]:
        return False

    recording_id = str(metadata.get("recording_id") or "")
    transcript_id = str(metadata.get("transcript_id") or "")
    return (
        (recording_id and recording_id in filters["recording_ids"])
        or (transcript_id and transcript_id in filters["transcript_ids"])
    )


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
    metadata = {**(metadata or {})}
    metadata.setdefault("source_type", "transcript")
    _delete_existing_vector_document(metadata)
    doc = Document(text=text, metadata=metadata)
    _index.insert(doc)
    # print(f"[RAG] 문서 추가됨: {text[:30]}...")


def _delete_existing_vector_document(metadata: dict) -> None:
    """같은 transcript/material chunk가 재색인될 때 오래된 벡터 row를 먼저 제거합니다."""
    transcript_id = str(metadata.get("transcript_id") or "").strip()
    material_id = str(metadata.get("material_id") or "").strip()
    chunk_index = metadata.get("chunk_index")
    if not transcript_id and not material_id:
        return

    table_name = _rag_table_name()
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**_db_config())
        cur = conn.cursor()
        metadata_column = _vector_metadata_column(cur, table_name)
        if not metadata_column:
            return

        quoted_table = _quote_ident(table_name)
        quoted_metadata = _quote_ident(metadata_column)
        if transcript_id:
            cur.execute(
                f"DELETE FROM {quoted_table} WHERE {quoted_metadata}->>'transcript_id' = %s",
                (transcript_id,),
            )
        elif material_id and chunk_index is not None:
            cur.execute(
                f"""
                DELETE FROM {quoted_table}
                WHERE {quoted_metadata}->>'material_id' = %s
                  AND {quoted_metadata}->>'chunk_index' = %s
                """,
                (material_id, str(chunk_index)),
            )
        conn.commit()
    except Exception as exc:
        logger.warning("[RAG] 기존 벡터 row 정리 실패: %s", exc)
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# ── 키워드(BM25 대용) 검색: DB에서 직접 텍스트 매칭 ──
# 신창영 : 기존에는 session_id 필터 없이 전체 transcripts를 대상으로 키워드 검색
# def _keyword_search(query: str, top_k: int = 5) -> list[dict]:
def _keyword_search(
    query: str,
    top_k: int = 5,
    session_id: str | None = None,
    source_filter: dict | None = None,
    locator_query: bool | None = None,
    grounded_lookup_query: bool | None = None,
) -> list[dict]:
    """PostgreSQL ts_rank + LIKE 기반 키워드 검색"""
    filters = _normalize_source_filter(source_filter)
    if _has_source_filter(filters) and not filters["recording_ids"] and not filters["transcript_ids"]:
        return []

    # 신창영 : DB 접속 정보는 _db_config()에서 환경변수 기반으로 통합
    # conn = psycopg2.connect(
    #     host="localhost", port=5432,
    #     database="shin", user="postgres", password="1234"
    # )
    conn = psycopg2.connect(**_db_config())
    cur = conn.cursor()

    # 형태소 분석으로 명사/동사/형용사/영문 약어 키워드 추출
    words = _extract_search_terms(
        query,
        locator_query=_is_locator_query(query) if locator_query is None else locator_query,
        grounded_lookup_query=_is_grounded_lookup_query(query)
        if grounded_lookup_query is None
        else grounded_lookup_query,
    )
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
    if filters["recording_ids"]:
        where_clauses.append("t.recording_id = ANY(%s)")
        values.append(list(filters["recording_ids"]))
    if filters["transcript_ids"]:
        where_clauses.append("t.transcript_id::text = ANY(%s)")
        values.append(list(filters["transcript_ids"]))

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
            "source_type": "transcript",
            "source": "keyword",
            "match_count": int(match_count or 0),
        })
    return results


def _material_keyword_search(
    query: str,
    top_k: int = 5,
    session_id: str | None = None,
    source_filter: dict | None = None,
    locator_query: bool | None = None,
    grounded_lookup_query: bool | None = None,
) -> list[dict]:
    """PDF RAG chunk도 전사문과 동일하게 키워드 후보로 검색합니다."""
    filters = _normalize_source_filter(source_filter)
    if _has_source_filter(filters) and not filters["material_ids"] and not filters["stored_names"]:
        return []

    words = _extract_search_terms(
        query,
        locator_query=_is_locator_query(query) if locator_query is None else locator_query,
        grounded_lookup_query=_is_grounded_lookup_query(query)
        if grounded_lookup_query is None
        else grounded_lookup_query,
    )
    if not words:
        return []

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**_db_config())
        cur = conn.cursor()
        table_name = _rag_table_name()
        metadata_column = _vector_metadata_column(cur, table_name)
        text_column = _vector_text_column(cur, table_name)
        if not metadata_column or not text_column:
            return []

        quoted_table = _quote_ident(table_name)
        quoted_metadata = _quote_ident(metadata_column)
        quoted_text = _quote_ident(text_column)
        text_expr = f"COALESCE({quoted_text}, '')"
        like_conditions = " OR ".join([f"{text_expr} ILIKE %s" for _ in words])
        match_score = " + ".join([f"CASE WHEN {text_expr} ILIKE %s THEN 1 ELSE 0 END" for _ in words])
        score_values = [f"%{word}%" for word in words]
        like_values = [f"%{word}%" for word in words]
        where_clauses = [
            f"{quoted_metadata}->>'source_type' = 'material'",
            f"({like_conditions})",
        ]
        values = score_values + like_values

        if session_id:
            where_clauses.append(f"{quoted_metadata}->>'session_id' = %s")
            values.append(str(session_id))
        if filters["material_ids"]:
            where_clauses.append(f"{quoted_metadata}->>'material_id' = ANY(%s)")
            values.append(list(filters["material_ids"]))
        if filters["stored_names"]:
            where_clauses.append(
                f"regexp_replace(COALESCE({quoted_metadata}->>'stored_name', ''), '^.*[\\\\/]', '') = ANY(%s)"
            )
            values.append(list(filters["stored_names"]))

        cur.execute(
            f"""
            SELECT {text_expr} AS chunk_text,
                   {quoted_metadata} AS metadata,
                   ({match_score}) AS match_count
            FROM {quoted_table}
            WHERE {" AND ".join(where_clauses)}
            ORDER BY match_count DESC
            LIMIT %s
            """,
            values + [top_k],
        )
        rows = cur.fetchall()
    except Exception as exc:
        logger.warning("[RAG] PDF 키워드 검색 실패: %s", exc)
        return []
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    label_cache = {}
    results = []
    for chunk_text, metadata_value, match_count in rows:
        metadata = _json_value(metadata_value, {})
        if not isinstance(metadata, dict):
            metadata = {}
        row_session_id = str(metadata.get("session_id") or "")
        if not row_session_id:
            continue
        if not _matches_source_filter(metadata, filters):
            continue

        if row_session_id not in label_cache:
            label_cache[row_session_id] = _get_session_label(row_session_id)
        session_label = label_cache.get(row_session_id) or {}
        results.append({
            "text": chunk_text,
            "file_title": session_label.get("file_title") or metadata.get("session_title", ""),
            "recording_title": "",
            "course_title": session_label.get("course_title") or metadata.get("course_title", ""),
            "session_title": session_label.get("session_title") or metadata.get("session_title", ""),
            "session_date": session_label.get("session_date") or metadata.get("session_date", ""),
            "start_time": 0,
            "end_time": 0,
            "session_id": row_session_id,
            "recording_id": "",
            "transcript_id": "",
            "material_id": str(metadata.get("material_id", "")),
            "material_name": str(metadata.get("material_name", "")),
            "stored_name": str(metadata.get("stored_name", "")),
            "page": metadata.get("page", 0),
            "chunk_index": metadata.get("chunk_index", None),
            "created_at": metadata.get("created_at", ""),
            "source_type": "material",
            "source": "keyword",
            "match_count": int(match_count or 0),
        })
    return results


def _keyword_search_all_sources(
    query: str,
    top_k: int = 5,
    session_id: str | None = None,
    source_filter: dict | None = None,
    locator_query: bool | None = None,
    grounded_lookup_query: bool | None = None,
) -> list[dict]:
    results = []
    results.extend(_keyword_search(
        query,
        top_k=top_k,
        session_id=session_id,
        source_filter=source_filter,
        locator_query=locator_query,
        grounded_lookup_query=grounded_lookup_query,
    ))
    results.extend(_material_keyword_search(
        query,
        top_k=top_k,
        session_id=session_id,
        source_filter=source_filter,
        locator_query=locator_query,
        grounded_lookup_query=grounded_lookup_query,
    ))
    return results


# ── 벡터 검색 ──
# 신창영 : 기존에는 session_id 필터 없이 벡터 검색 top_k만 조회
# def _vector_search(query: str, top_k: int = 5) -> list[dict]:
def _vector_search(
    query: str,
    top_k: int = 5,
    session_id: str | None = None,
    source_filter: dict | None = None,
) -> list[dict]:
    init()
    if _index is None:
        if _init_error is not None:
            logger.warning("[RAG] 벡터 검색 비활성화: %s", _init_error)
        return []
    # 신창영 : session_id 필터 후 결과 부족을 줄이기 위해 후보를 더 넓게 조회
    # retriever = _index.as_retriever(similarity_top_k=top_k)
    filters = _normalize_source_filter(source_filter)
    needs_wide_candidates = bool(session_id) or _has_source_filter(filters)
    retriever = _index.as_retriever(
        similarity_top_k=top_k * RAG_VECTOR_CANDIDATE_MULTIPLIER if needs_wide_candidates else top_k
    )
    nodes = retriever.retrieve(query)
    session_label_cache = {}

    results = []
    transcript_candidates = []
    for node in nodes:
        m = node.metadata
        row_session_id = str(m.get("session_id", ""))
        # 신창영 : 기존에는 이 필터가 없어 다른 세션의 벡터 결과가 섞일 수 있었음
        if session_id and row_session_id != str(session_id):
            continue
        if not _matches_source_filter(m, filters):
            continue

        # 신창영 : 파일 삭제 후 남은 오래된 벡터 노드는 세션 DB에 존재할 때만 노출
        # 신창영 : citation 제목은 stale metadata가 아니라 최신 sessions/session_voicefile 값으로 보정
        if not row_session_id:
            continue
        source_type = str(m.get("source_type") or "transcript")
        if source_type == "material":
            label_cache_key = ("material", row_session_id)
            if label_cache_key not in session_label_cache:
                session_label_cache[label_cache_key] = _get_session_label(row_session_id)
            session_label = session_label_cache[label_cache_key]
            if session_label is None:
                continue

            results.append({
                "text": node.text,
                "file_title": session_label.get("file_title") or m.get("session_title", ""),
                "recording_title": "",
                "course_title": session_label.get("course_title") or m.get("course_title", ""),
                "session_title": session_label.get("session_title") or m.get("session_title", ""),
                "session_date": session_label.get("session_date") or m.get("session_date", ""),
                "start_time": 0,
                "end_time": 0,
                "session_id": row_session_id,
                "recording_id": "",
                "transcript_id": "",
                "material_id": str(m.get("material_id", "")),
                "material_name": str(m.get("material_name", "")),
                "stored_name": str(m.get("stored_name", "")),
                "page": m.get("page", 0),
                "chunk_index": m.get("chunk_index", None),
                "created_at": m.get("created_at", ""),
                "score": node.score,
                "source_type": "material",
                "source": "vector",
            })
            continue

        transcript_candidates.append((node, m))

    transcript_ids = [
        str(metadata.get("transcript_id") or "").strip()
        for _, metadata in transcript_candidates
        if str(metadata.get("transcript_id") or "").strip()
    ]
    canonical_by_id = _fetch_transcript_rows_by_ids(transcript_ids)

    for node, m in transcript_candidates:
        transcript_id = str(m.get("transcript_id") or "").strip()
        canonical = canonical_by_id.get(transcript_id) if transcript_id else None
        if canonical is None and not transcript_id:
            canonical = _fetch_transcript_row_for_vector_metadata(m)
        if canonical is None:
            continue

        results.append({
            "text": canonical["text"],
            "file_title": canonical.get("file_title") or canonical.get("session_title", ""),
            "recording_title": canonical.get("recording_title") or canonical.get("session_title", ""),
            "course_title": canonical.get("course_title") or "",
            "session_title": canonical.get("session_title", ""),
            "session_date": canonical.get("session_date", ""),
            "start_time": canonical.get("start_time", 0),
            "end_time": canonical.get("end_time", 0),
            "session_id": canonical["session_id"],
            "recording_id": str(canonical.get("recording_id", "")),
            "transcript_id": canonical.get("transcript_id", ""),
            "chunk_index": canonical.get("chunk_index", None),
            "created_at": canonical.get("created_at", ""),
            "score": node.score,
            "source_type": "transcript",
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
    normalized_question = str(question or "").casefold()

    # 짧은 약어 질문은 원 질문만으로는 관련 passage 일부만 잡힐 수 있어
    # 강의 샘플에서 함께 설명되는 주변 개념까지 검색 쿼리로 확장합니다.
    if "prag" in normalized_question:
        queries.extend([
            "PRAG Parametric RAG 외부 지식 메모리 K/V 어텐션",
            "BridgePRAG 질문 passage pair K/V 메모리 질문 의도",
            "passage 질문 문맥 pair 메모리 생성",
        ])
    if "rag" in normalized_question:
        queries.extend([
            "RAG 검색 증강 생성 관련 문서 검색 문맥 답변",
            "passage top-k 노이즈 문맥 길이 재랭킹",
        ])

    # 형태소 분석으로 핵심 키워드 추출
    keywords = extract_keywords(question)

    if keywords:
        keyword_query = " ".join(keywords)
        if keyword_query != question:
            queries.append(keyword_query)

    deduped = []
    seen = set()
    for query in queries:
        compact = " ".join(str(query or "").split())
        key = compact.casefold()
        if not compact or key in seen:
            continue
        seen.add(key)
        deduped.append(compact)
    return deduped


# ── 시간 포맷팅 ──
def _format_time(seconds: float) -> str:
    """초를 mm:ss 형태로 변환"""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


# ── Reciprocal Rank Fusion (RRF) 병합 ──
def _keyword_hit_count(text: str, keywords: list[str]) -> int:
    haystack = str(text or "").casefold()
    return sum(1 for word in keywords if str(word or "").casefold() in haystack)


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").casefold())


def _is_programming_question(question: str) -> bool:
    text = str(question or "").casefold()
    return any(hint in text for hint in _PROGRAMMING_QUERY_HINTS)


def _is_assignment_or_schedule_question(question: str) -> bool:
    text = str(question or "")
    return any(hint in text for hint in _ASSIGNMENT_QUERY_HINTS)


def _looks_assignment_context(text: str) -> bool:
    value = str(text or "")
    return any(hint in value for hint in _ASSIGNMENT_CONTEXT_HINTS)


def _looks_code_heavy(text: str) -> bool:
    value = str(text or "")
    if not value:
        return False
    if _CODE_HEAVY_RE.search(value):
        return True
    code_marks = sum(value.count(mark) for mark in ("{", "}", ";", "()", "->", "==", "!="))
    return code_marks >= 5


def _strong_question_terms(question: str) -> list[str]:
    """'손실함수', '전위'처럼 근거에 반드시 있어야 하는 조회 대상어를 추출합니다."""
    terms: list[str] = []

    if _is_grounded_lookup_query(question):
        terms.extend(_extract_grounded_lookup_terms(question))

    for keyword in extract_keywords(question):
        value = _clean_lookup_term(keyword)
        if len(value) < 2:
            continue
        if value.casefold() in _WEAK_RELEVANCE_TERMS or value in _WEAK_RELEVANCE_TERMS:
            continue
        if value.startswith(("설명", "정리", "요약", "알려")) or value.endswith(("했어", "했었어", "했었지", "해줘")):
            continue
        terms.append(value)

    deduped: list[str] = []
    seen = set()
    for term in terms:
        normalized = str(term or "").strip().casefold()
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _strong_subject_score(text: str, strong_terms: list[str]) -> int:
    """질문 대상어가 근거에 직접 들어있는지 점수화합니다."""
    if not strong_terms:
        return 0
    haystack = str(text or "").casefold()
    compact_haystack = _compact_text(text)
    score = 0
    for term in strong_terms:
        token = str(term or "").strip().casefold()
        if not token:
            continue
        compact_token = _compact_text(token)
        if token in haystack:
            score += 3 if len(compact_token) >= 4 else 2
        elif compact_token and compact_token in compact_haystack:
            score += 3 if len(compact_token) >= 4 else 2
    return score


def _strong_subject_hit_count(text: str, strong_terms: list[str]) -> int:
    """질문 핵심어 중 몇 개가 근거에 직접 등장하는지 계산합니다."""
    if not strong_terms:
        return 0
    haystack = str(text or "").casefold()
    compact_haystack = _compact_text(text)
    hits = 0
    for term in strong_terms:
        token = str(term or "").strip().casefold()
        compact_token = _compact_text(token)
        if token and token in haystack:
            hits += 1
        elif compact_token and compact_token in compact_haystack:
            hits += 1
    return hits


def _definition_sentence_score(text: str, strong_terms: list[str]) -> int:
    """'전위는 ...입니다'처럼 질문 대상어를 직접 정의하는 문장을 우선합니다."""
    if not strong_terms:
        return 0
    value = re.sub(r"\s+", " ", str(text or "").strip().casefold())
    compact_value = _compact_text(value)
    if not value:
        return 0

    score = 0
    for term in strong_terms:
        token = str(term or "").strip().casefold()
        compact_token = _compact_text(token)
        if len(compact_token) < 2:
            continue
        definition_markers = (
            f"{compact_token}는",
            f"{compact_token}은",
            f"{compact_token}이란",
            f"{compact_token}란",
        )
        if any(marker in compact_value for marker in definition_markers):
            score += 8
        if compact_value.startswith(compact_token):
            score += 3
        if any(marker in value for marker in ("의미", "정의", "나타내", "말합니다", "입니다")):
            score += 2
    return score


def _has_long_subject_match(text: str, strong_terms: list[str]) -> bool:
    """수요곡선/가격탄력성처럼 긴 복합어가 직접 일치하면 단일 히트도 강한 근거로 봅니다."""
    haystack = str(text or "").casefold()
    compact_haystack = _compact_text(text)
    for term in strong_terms:
        compact_token = _compact_text(term)
        token = str(term or "").strip().casefold()
        if len(compact_token) < 4:
            continue
        if token in haystack or compact_token in compact_haystack:
            return True
    return False


def _relevance_score(text: str, terms: list[str]) -> int:
    """질문 관련어가 문장/근거에 얼마나 촘촘히 들어있는지 점수화합니다."""
    haystack = str(text or "").casefold()
    compact_haystack = _compact_text(text)
    if not haystack or not terms:
        return 0
    score = 0
    for term in terms:
        token = str(term or "").strip().casefold()
        if not token:
            continue
        count = haystack.count(token)
        compact_token = _compact_text(token)
        if count <= 0 and compact_token:
            count = compact_haystack.count(compact_token)
        if count <= 0:
            continue
        score += 2 if len(token) >= 4 else 1
        score += min(count - 1, 2)
    return score


def _question_relevance_terms(question: str) -> list[str]:
    """검색/근거 필터링에 사용할 질문 핵심어를 확장합니다."""
    terms = _strong_question_terms(question)
    for keyword in extract_keywords(question):
        value = _clean_lookup_term(keyword)
        if len(value) < 2:
            continue
        if value.casefold() in _WEAK_RELEVANCE_TERMS or value in _WEAK_RELEVANCE_TERMS:
            continue
        if value.startswith(("설명", "정리", "요약", "알려")) or value.endswith(("했어", "했었어", "했었지", "해줘")):
            continue
        terms.append(value)

        # 긴 한국어 복합어는 일부만 표현된 관련 문장도 놓치지 않도록 부분 단서를 추가합니다.
        if len(value) >= 4 and value.casefold() not in _WEAK_RELEVANCE_TERMS and value not in _WEAK_RELEVANCE_TERMS:
            for size in (2, 3):
                for index in range(0, len(value) - size + 1):
                    piece = value[index:index + size]
                    if len(piece) >= 2:
                        terms.append(piece)

    question_text = str(question or "").casefold()
    if "rag" in question_text:
        terms.extend(["rag", "검색", "문맥", "passage", "메모리", "k/v"])
    if "prag" in question_text:
        terms.extend(["prag", "parametric", "메모리", "k/v", "어텐션", "passage"])
    if "탄력" in question_text:
        terms.extend(["탄력", "비탄력", "가격", "수요", "수요량", "총수입", "민감"])

    deduped = []
    seen = set()
    for term in terms:
        normalized = str(term or "").strip().casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _filter_relevant_results(results: list[dict], question: str, *, min_keep: int = 1) -> list[dict]:
    """질문과 직접 관련 없는 검색 결과를 LLM 입력 전에 줄입니다."""
    if not results:
        return []
    terms = _question_relevance_terms(question)
    if not terms:
        return results

    grounded_lookup_query = _is_grounded_lookup_query(question)
    strong_terms = _strong_question_terms(question)
    programming_question = _is_programming_question(question)
    assignment_question = _is_assignment_or_schedule_question(question)
    scored = []
    kept = []
    for index, item in enumerate(results):
        text = item.get("text", "")
        score = _relevance_score(text, terms)
        strong_score = _strong_subject_score(text, strong_terms)
        strong_hits = _strong_subject_hit_count(text, strong_terms)
        has_long_subject = _has_long_subject_match(text, strong_terms)
        code_noise = _looks_code_heavy(text) and not programming_question
        scored.append((score, strong_score, code_noise, index, item))

        if grounded_lookup_query and strong_terms:
            if code_noise:
                continue
            if strong_score <= 0 and score < 12:
                continue
            if strong_score > 0 and len(strong_terms) >= 2 and strong_hits < 2 and not has_long_subject:
                continue
        elif score <= 0:
            continue
        elif strong_terms and len(strong_terms) >= 2 and strong_hits < 2 and not has_long_subject:
            continue
        elif code_noise and score < 3:
            continue
        kept.append(item)

    if kept and not assignment_question:
        non_assignment_kept = [item for item in kept if not _looks_assignment_context(item.get("text", ""))]
        if len(non_assignment_kept) >= min_keep:
            kept = non_assignment_kept

    if len(kept) >= min_keep:
        return kept

    # 개념/정의 조회형 질문에서 핵심어가 직접 들어간 근거를 못 찾은 경우,
    # 코드 조각이나 벡터-only 잡음을 억지로 citation으로 쓰면 답변 품질이 무너진다.
    # 이때는 빈 결과를 반환해 "근거 없음" 경로로 보내는 편이 더 안전하다.
    if grounded_lookup_query and strong_terms:
        return []

    fallback = [
        item
        for score, strong_score, code_noise, _, item in sorted(
            scored,
            key=lambda row: (row[2], -row[1], -row[0], row[3]),
        )
        if not code_noise
    ]
    if fallback:
        return fallback[:min_keep]
    return [item for _, _, _, _, item in sorted(scored, key=lambda row: row[3])[:min_keep]]


def _merge_results(vector_results: list, keyword_results: list, top_k: int = 5, k: int = 60, query_keywords: list[str] | None = None) -> list[dict]:
    """
    RRF로 벡터 + 키워드 결과를 통합 랭킹.
    score = Σ 1/(k + rank)  (k=60이 표준값)
    벡터 3위 + 키워드 1위인 문장이 벡터 1위만인 문장보다 높을 수 있음.
    """
    query_keywords = query_keywords or []
    scores = {}  # text -> score/data/lexical metadata

    # 벡터 결과: 이미 cosine 유사도 순으로 정렬되어 있음
    for rank, r in enumerate(vector_results):
        text = r["text"]
        if text not in scores:
            scores[text] = {"score": 0, "data": r, "keyword_hits": 0, "lexical_hits": 0}
        scores[text]["score"] += 1.0 / (k + rank + 1)

    # 키워드 결과: match_count DESC로 정렬되어 있음
    for rank, r in enumerate(keyword_results):
        text = r["text"]
        if text not in scores:
            scores[text] = {"score": 0, "data": r, "keyword_hits": 0, "lexical_hits": 0}
        elif scores[text]["data"].get("source") == "vector":
            scores[text]["data"] = {**scores[text]["data"], **r}
        scores[text]["keyword_hits"] = max(scores[text]["keyword_hits"], int(r.get("match_count") or 0))
        scores[text]["score"] += 1.0 / (k + rank + 1)

    for item in scores.values():
        item["lexical_hits"] = _keyword_hit_count(item["data"].get("text", ""), query_keywords)

    # 전체 워크스페이스 검색에서는 벡터-only 잡음보다 질문 키워드가 실제로 포함된 전사문을 우선한다.
    ranked = sorted(
        scores.values(),
        key=lambda x: (
            x["keyword_hits"] > 0 or x["lexical_hits"] > 0,
            x["keyword_hits"],
            x["lexical_hits"],
            x["score"],
        ),
        reverse=True,
    )
    return _diversify_ranked_results([item["data"] for item in ranked], top_k=top_k)


def _diversify_ranked_results(results: list[dict], *, top_k: int) -> list[dict]:
    """같은 근거가 반복되지 않도록 하되, 부족하면 원래 랭킹으로 채웁니다."""
    if top_k <= 0:
        return []

    selected: list[dict] = []
    seen_identity: set[tuple] = set()
    seen_text: set[str] = set()

    for item in results:
        identity = _result_identity(item)
        text_key = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()[:180]
        if identity in seen_identity or text_key in seen_text:
            continue
        selected.append(item)
        seen_identity.add(identity)
        seen_text.add(text_key)
        if len(selected) >= top_k:
            return selected

    for item in results:
        identity = _result_identity(item)
        if any(_result_identity(selected_item) == identity for selected_item in selected):
            continue
        selected.append(item)
        if len(selected) >= top_k:
            break
    return selected


def _keyword_results_are_confident(keyword_results: list[dict], query_keywords: list[str] | None = None) -> bool:
    """키워드 검색만으로도 질문과 직접 맞는 전사 근거를 찾은 경우를 판별합니다."""
    if not keyword_results:
        return False
    query_keywords = query_keywords or []
    top_result = keyword_results[0]
    if int(top_result.get("match_count") or 0) >= RAG_FAST_KEYWORD_MIN_HITS:
        return True
    return _keyword_hit_count(top_result.get("text", ""), query_keywords) >= RAG_FAST_KEYWORD_MIN_HITS


def _result_identity(result: dict) -> tuple:
    source_type = result.get("source_type") or "transcript"
    if source_type == "material":
        return (
            "material",
            result.get("session_id", ""),
            result.get("material_id", ""),
            result.get("stored_name", ""),
            result.get("page", ""),
            result.get("chunk_index", ""),
        )

    return (
        "transcript",
        result.get("session_id", ""),
        result.get("recording_id", ""),
        result.get("transcript_id", ""),
        result.get("chunk_index", ""),
        result.get("start_time", ""),
        result.get("end_time", ""),
    )


def _dedupe_vector_results(vector_results: list[dict], top_k: int = 5) -> list[dict]:
    best_by_key = {}
    for result in vector_results:
        key = _result_identity(result)
        score = float(result.get("score") or 0.0)
        current = best_by_key.get(key)
        if current is None or score > float(current.get("score") or 0.0):
            best_by_key[key] = result

    ranked = sorted(
        best_by_key.values(),
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )
    return ranked[:top_k]


def _prioritize_grounded_lookup_results(
    results: list[dict],
    *,
    grounded_lookup_query: bool,
    locator_query: bool,
    top_k: int,
    question: str = "",
) -> list[dict]:
    """개념/정의 질문에서는 자료 종류보다 질문 핵심어가 직접 일치하는 근거를 우선 배치합니다."""
    if not results:
        return []
    if not grounded_lookup_query or locator_query:
        return results[:top_k]

    strong_terms = _strong_question_terms(question)
    programming_question = _is_programming_question(question)
    ranked = sorted(
        enumerate(results),
        key=lambda item: (
            _strong_subject_score(item[1].get("text", ""), strong_terms),
            _strong_subject_hit_count(item[1].get("text", ""), strong_terms),
            not (_looks_code_heavy(item[1].get("text", "")) and not programming_question),
            (item[1].get("source_type") == "material"),
            float(item[1].get("score") or item[1].get("match_count") or 0),
            -item[0],
        ),
        reverse=True,
    )
    return _ensure_material_and_transcript_mix([item for _, item in ranked], top_k=top_k)


def _ensure_material_and_transcript_mix(results: list[dict], *, top_k: int) -> list[dict]:
    """PDF와 전사 후보가 모두 있으면 최종 근거에 최소 1개씩 포함합니다."""
    selected = list(results[:top_k])
    if top_k < 2 or len(selected) < top_k:
        return selected

    has_material_candidate = any(item.get("source_type") == "material" for item in results)
    has_transcript_candidate = any(item.get("source_type", "transcript") == "transcript" for item in results)
    if not has_material_candidate or not has_transcript_candidate:
        return selected

    has_material_selected = any(item.get("source_type") == "material" for item in selected)
    has_transcript_selected = any(item.get("source_type", "transcript") == "transcript" for item in selected)
    if has_material_selected and has_transcript_selected:
        return selected

    replacement_type = "transcript" if not has_transcript_selected else "material"
    replacement = next(
        (item for item in results[top_k:] if item.get("source_type", "transcript") == replacement_type),
        None,
    )
    if replacement is None:
        return selected

    selected[-1] = replacement
    return selected


def _run_vector_similarity_search(
    queries: list[str],
    top_k: int = 5,
    session_id: str | None = None,
    source_filter: dict | None = None,
) -> list[dict]:
    """PDF material chunk와 transcript chunk를 같은 벡터 후보군에서 유사도 순으로 랭킹."""
    vector_results = []
    for query in queries:
        vector_results.extend(_vector_search(query, top_k=top_k, session_id=session_id, source_filter=source_filter))
    return _dedupe_vector_results(vector_results, top_k=top_k)


def _run_hybrid_search(
    queries: list[str],
    top_k: int = 5,
    session_id: str | None = None,
    source_filter: dict | None = None,
) -> list[dict]:
    """여러 검색어에 대해 벡터 검색과 키워드 검색을 실행한 뒤 RRF로 병합."""
    all_keyword = []
    query_keywords = extract_keywords(queries[0]) if queries else []
    filters = _normalize_source_filter(source_filter)
    material_source_requested = _has_material_source_filter(filters)

    for query in queries:
        all_keyword.extend(_keyword_search_all_sources(
            query,
            top_k=top_k,
            session_id=session_id,
            source_filter=source_filter,
        ))

    keyword_results = _merge_results([], all_keyword, top_k=top_k, query_keywords=query_keywords)
    if (
        RAG_FAST_KEYWORD_FIRST
        and not material_source_requested
        and len(keyword_results) >= min(top_k, RAG_FAST_KEYWORD_MIN_RESULTS)
        and _keyword_results_are_confident(keyword_results, query_keywords)
    ):
        return keyword_results
    if not RAG_USE_VECTOR_SEARCH:
        return keyword_results

    all_vector = []
    # 벡터 검색은 원 질문 1회만 사용하고, 키워드 검색은 확장 쿼리를 모두 사용한다.
    # 이렇게 해도 hybrid 구조는 유지하면서 BGE embedding/retrieval 호출 수를 줄일 수 있다.
    for query in queries[:1]:
        all_vector.extend(_vector_search(query, top_k=top_k, session_id=session_id, source_filter=source_filter))

    return _merge_results(all_vector, all_keyword, top_k=top_k, query_keywords=query_keywords)


def _run_keyword_only_search(
    queries: list[str],
    top_k: int = 5,
    session_id: str | None = None,
    source_filter: dict | None = None,
    locator_query: bool = False,
    grounded_lookup_query: bool = False,
) -> list[dict]:
    """정확한 언급 위치를 찾는 질문에서는 키워드 매칭 결과를 우선 사용합니다."""
    all_keyword = []
    for query in queries:
        all_keyword.extend(_keyword_search_all_sources(
            query,
            top_k=top_k,
            session_id=session_id,
            source_filter=source_filter,
            locator_query=locator_query,
            grounded_lookup_query=grounded_lookup_query,
        ))
    return _merge_results([], all_keyword, top_k=top_k)


def _run_keyword_only_search(
    queries: list[str],
    top_k: int = 5,
    session_id: str | None = None,
    source_filter: dict | None = None,
    locator_query: bool = False,
    grounded_lookup_query: bool = False,
) -> list[dict]:
    """정확한 언급 위치를 찾는 질문에서는 키워드 매칭 결과를 우선 사용합니다."""
    all_keyword = []
    for query in queries:
        all_keyword.extend(_keyword_search_all_sources(
            query,
            top_k=top_k,
            session_id=session_id,
            source_filter=source_filter,
            locator_query=locator_query,
            grounded_lookup_query=grounded_lookup_query,
        ))
    return _merge_results([], all_keyword, top_k=top_k)


# ── Citation 포맷 ──
def _format_citation(result: dict) -> str:
    """출처 문자열 생성"""
    if result.get("source_type") == "material":
        return format_material_citation(result)

    time_range = f"{_format_time(result['start_time'])}~{_format_time(result['end_time'])}"
    source_title = result.get("recording_title") or result.get("session_title") or result.get("file_title") or "녹음본"
    return f"{source_title} > {time_range}"


# ══════════════════════════════════════
#  메인 검색 함수 (main.py에서 호출)
# ══════════════════════════════════════
# 신창영 : 기존에는 session_id를 받지 않고 전체 자료에서 검색
# def search(question: str, top_k: int = 5) -> dict:
def search(
    question: str,
    top_k: int = 5,
    session_id: str | None = None,
    source_filter: dict | None = None,
) -> dict:
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
    _demo_log(f"1) 질문 수신: '{_preview(question, 100)}'")
    _demo_log(f"2) 형태소/키워드 분석: keywords={extract_keywords(question)[:12]}")
    _demo_log(f"3) Multi-query 구성: queries={queries}")

    # 2. 현재 파일에서 먼저 검색하고, 없으면 전체 파일에서 다시 검색
    # 신창영 : 선택 세션 기준 검색 후 결과가 없을 때만 전체 검색으로 fallback
    # all_vector.extend(_vector_search(q, top_k=3))
    # all_keyword.extend(_keyword_search(q, top_k=3))
    filters = _normalize_source_filter(source_filter)
    has_filter = _has_source_filter(filters)
    locator_query = _is_locator_query(question)
    grounded_lookup_query = _is_grounded_lookup_query(question)
    strict_keyword_query = locator_query
    current_scope_query = _is_current_scope_query(question)
    candidate_top_k = top_k * 3 if grounded_lookup_query and not locator_query else top_k
    _demo_log(
        "4) 검색 전략 결정: "
        f"vector={'on' if RAG_USE_VECTOR_SEARCH and not strict_keyword_query else 'off'}, "
        f"keyword=on, top_k={top_k}, scope={'selected' if session_id else 'all'}"
    )

    if strict_keyword_query:
        results = _run_keyword_only_search(
            queries,
            top_k=top_k,
            session_id=session_id,
            source_filter=source_filter,
            locator_query=locator_query,
            grounded_lookup_query=grounded_lookup_query,
        )
        search_scope = "current_file_keyword" if session_id else "all_files_keyword"
    else:
        # 개념/정의 질문은 PDF 벡터 결과만으로는 전사 스크립트 후보가 밀릴 수 있어
        # 키워드 후보까지 함께 섞은 뒤 PDF/전사 근거를 균형 있게 고른다.
        if grounded_lookup_query:
            results = _run_hybrid_search(
                queries,
                top_k=candidate_top_k,
                session_id=session_id,
                source_filter=source_filter,
            )
            search_scope = "current_file_hybrid" if session_id else "all_files_hybrid"
        else:
            # 서비스 채팅에서는 질문의 핵심 단어가 전사에 직접 등장하는 경우가 많다.
            # 벡터 검색만 먼저 쓰면 "수학 과제"처럼 명확한 단서가 있어도 의미상 가까운
            # 무관 문장이 선택될 수 있으므로 기본 검색도 키워드 후보를 섞은 hybrid로 수행한다.
            results = _run_hybrid_search(
                queries,
                top_k=candidate_top_k,
                session_id=session_id,
                source_filter=source_filter,
            )
            search_scope = "current_file_hybrid" if session_id else "all_files_hybrid"

    if not results and not strict_keyword_query:
        results = _run_hybrid_search(queries, top_k=candidate_top_k, session_id=session_id, source_filter=source_filter)
        if results:
            search_scope = f"{search_scope}_keyword_fallback"

    if strict_keyword_query and session_id and not results and not current_scope_query:
        results = _run_keyword_only_search(
            queries,
            top_k=top_k,
            session_id=None,
            source_filter=None,
            locator_query=locator_query,
            grounded_lookup_query=grounded_lookup_query,
        )
        if results:
            search_scope = "all_files_keyword_fallback"

    if session_id and not results and not current_scope_query and not strict_keyword_query:
        # 프론트는 현재 파일의 PDF/녹음본을 source_filter로 자동 선택한다.
        # 다만 사용자가 "이 파일에서"라고 한정하지 않은 일반 개념 질문은
        # 선택 파일에서 실패했을 때 전체 저장 자료까지 확장해야 샘플 DB 근거를 놓치지 않는다.
        results = _run_hybrid_search(
            queries,
            top_k=candidate_top_k,
            session_id=None,
            source_filter=None,
        )
        if results:
            search_scope = "all_files_fallback"

    if (
        not results
        and session_id
        and (filters["recording_ids"] or filters["transcript_ids"])
        and not strict_keyword_query
    ):
        results = _selected_transcript_context_search(session_id, source_filter)
        if results:
            search_scope = "selected_source_context"

    if session_id and not results and not has_filter and not strict_keyword_query:
        results = _run_vector_similarity_search(queries, top_k=candidate_top_k, session_id=None, source_filter=None)
        if not results:
            results = _run_hybrid_search(queries, top_k=candidate_top_k, session_id=None, source_filter=None)
        search_scope = "all_files_fallback"

    if not results:
        _demo_log("5) 검색 결과 없음: context 비움")
        return {"context": "", "citations": []}

    results = _prioritize_grounded_lookup_results(
        results,
        grounded_lookup_query=grounded_lookup_query,
        locator_query=locator_query,
        top_k=candidate_top_k,
        question=question,
    )
    _demo_log(f"5) Hybrid 검색 완료: scope={search_scope}, similar_sentences={len(results)}")
    for rank, item in enumerate(results, 1):
        score_parts = []
        for score_key in ("score", "similarity", "distance", "lexical_score", "rank_score"):
            value = item.get(score_key)
            if isinstance(value, (int, float)):
                score_parts.append(f"{score_key}={value:.4f}")
        _demo_log(
            f"   유사문장#{rank}: source={item.get('source')}, "
            f"type={item.get('source_type', 'transcript')}, "
            f"citation='{_format_citation(item)}', "
            f"{' '.join(score_parts) if score_parts else 'score=n/a'}, "
            f"text='{_preview(item.get('text'), 160)}'"
        )

    if RAG_DEBUG:
        print(
            "[RAG:search]",
            f"scope={search_scope}",
            f"top_k={top_k}",
            f"session={session_id or '-'}",
            f"question={question}",
        )
        for rank, item in enumerate(results, 1):
            print(
                "[RAG:search]",
                f"#{rank}",
                f"source={item.get('source')}",
                f"type={item.get('source_type', 'transcript')}",
                f"tid={item.get('transcript_id', '')}",
                f"rid={item.get('recording_id', '')}",
                f"chunk={item.get('chunk_index', '')}",
                f"text={str(item.get('text', ''))[:120]}",
            )

    # 4. Context 문자열 구성 (LLM에 전달할 참고자료)
    context_parts = []
    citations = []
    full_transcript_cache = {}
    recording_transcript_cache = {}
    selected_results = _prioritize_selected_evidence(results, question)
    selected_results = _filter_relevant_results(selected_results, question, min_keep=1)
    selected_results = selected_results[:top_k]
    _demo_log(f"6) 최종 근거 선택: selected={len(selected_results)}, max_evidence={top_k}")
    for i, r in enumerate(selected_results, 1):
        original_text = r.get("text", "")
        if not locator_query:
            r = _expand_transcript_context_window(r)
        context_text = r.get("text", "")
        relevant_sentences = _select_relevant_evidence_sentences(context_text, question)
        citation_excerpt = " ".join(relevant_sentences).strip() or _select_sentence_level_excerpt(
            context_text,
            question,
            fallback=original_text,
        )
        citation = _format_citation(r)
        _demo_log(f"   근거#{i}: {citation}")
        result_session_id = r.get("session_id") or session_id
        if r.get("source_type") == "material":
            material_relevant_sentences = relevant_sentences or _select_relevant_evidence_sentences(r.get("text", ""), question)
            context_parts.append(_format_evidence_context_block(i, r.get("text", ""), citation, material_relevant_sentences))
            citations.append(build_material_citation(
                r,
                citation=citation,
                session_id=result_session_id,
                search_scope=search_scope,
            ))
            continue

        full_transcript = r.get("full_transcript") or context_text
        if RAG_PREFETCH_FULL_TRANSCRIPT and not r.get("full_transcript"):
            # 데모 응답 속도를 위해 기본값은 chunk만 내려보내고, 필요할 때만 전체 전사를 미리 붙입니다.
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
            full_transcript = (
                recording_transcript_cache.get(recording_cache_key)
                or full_transcript_cache.get(result_session_id)
                or r["text"]
            )
        if r.get("source") == "selected_source_context":
            context_parts.append(
                _format_evidence_context_block(
                    i,
                    f"선택된 녹음본 전체 전사\n{context_text}",
                    citation,
                    relevant_sentences,
                )
            )
        else:
            context_parts.append(_format_evidence_context_block(i, context_text, citation, relevant_sentences))
        citations.append({
            "text": citation_excerpt or context_text,
            "citation": citation,
            "file_title": r.get("file_title", ""),
            "recording_title": r.get("recording_title", ""),
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
            "source_type": "transcript",
            "full_transcript": full_transcript,
        })

    context = "\n".join(context_parts)
    _demo_log(f"7) LLM 전달용 RAG context 생성: context_chars={len(context)}, citations={len(citations)}")

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
