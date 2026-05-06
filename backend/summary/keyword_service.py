"""
키워드 추출 서비스 (Keyword Service)

역할:
  1. kiwipiepy 형태소 분석으로 토큰화
  2. textrankr 기반 키워드 추출
  3. textrankr 미설치 시 TextRank 알고리즘으로 키워드 추출

"""
import logging
import os
import re

import networkx as nx
from kiwipiepy import Kiwi

logger = logging.getLogger(__name__)

try:
    import textrankr as _textrankr
except ImportError:
    _textrankr = None

_kiwi = Kiwi()

KEYWORD_TAGS = {"NNG", "NNP", "VV", "VA"}
MIN_TOKEN_LEN = int(os.getenv("KEYWORD_MIN_TOKEN_LEN", 2))
DEFAULT_TOP_K = int(os.getenv("KEYWORD_TOP_K", 20))
DEFAULT_WINDOW = int(os.getenv("KEYWORD_WINDOW", 4))

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!。？！])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    """텍스트를 문장 단위로 분리합니다."""
    text = re.sub(r"\r\n?", "\n", text)
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part and part.strip()]


def _tokenize_sentence(sentence: str) -> list[str]:
    """형태소 분석으로 키워드 후보 토큰만 추출합니다."""
    tokens = _kiwi.tokenize(sentence)
    return [t.form for t in tokens if t.tag in KEYWORD_TAGS and len(t.form) >= MIN_TOKEN_LEN]


def _tokenize_sentences(text: str) -> list[list[str]]:
    """문장별로 토큰 리스트를 구성합니다."""
    sentences = _split_sentences(text)
    return [tokens for sentence in sentences if (tokens := _tokenize_sentence(sentence))]


def _normalize_keywords(raw) -> list[tuple[str, float]]:
    """textrankr 반환 타입을 (keyword, score) 형태로 정규화합니다."""
    if not raw:
        return []
    if isinstance(raw, dict):
        return [(k, float(v)) for k, v in raw.items()]
    if isinstance(raw, list):
        if raw and isinstance(raw[0], (tuple, list)) and len(raw[0]) >= 2:
            return [(str(item[0]), float(item[1])) for item in raw]
        if raw and isinstance(raw[0], str):
            return [(item, 1.0) for item in raw]
    return []


def _extract_keywords_textrankr(text: str, tokenized_sentences: list[list[str]], top_k: int) -> list[tuple[str, float]] | None:
    """textrankr가 설치된 경우 우선 사용합니다."""
    if _textrankr is None:
        return None

    try:
        for func_name in ("keywords", "extract_keywords", "summarize_keywords"):
            func = getattr(_textrankr, func_name, None)
            if callable(func):
                raw = func(tokenized_sentences, top_k=top_k)
                normalized = _normalize_keywords(raw)
                if normalized:
                    return normalized

        for cls_name in ("TextRank", "KeywordSummarizer", "KeywordExtractor"):
            cls = getattr(_textrankr, cls_name, None)
            if cls is None:
                continue
            instance = cls() if callable(cls) else None
            if instance is None:
                continue
            for method_name in ("keywords", "get_keywords", "extract_keywords", "summarize"):
                method = getattr(instance, method_name, None)
                if callable(method):
                    raw = method(tokenized_sentences, top_k=top_k)
                    normalized = _normalize_keywords(raw)
                    if normalized:
                        return normalized

        for func_name in ("keywords", "extract_keywords", "summarize_keywords"):
            func = getattr(_textrankr, func_name, None)
            if callable(func):
                raw = func(text, top_k=top_k)
                normalized = _normalize_keywords(raw)
                if normalized:
                    return normalized
    except Exception as exc:
        logger.warning("[KEYWORD] textrankr 처리 실패: %s", exc)

    return None


def _textrank_keywords(tokenized_sentences: list[list[str]], top_k: int, window_size: int) -> list[tuple[str, float]]:
    """textrankr가 없을 때 자체 TextRank로 키워드를 계산합니다."""
    graph = nx.Graph()
    for tokens in tokenized_sentences:
        for i, token in enumerate(tokens):
            if token not in graph:
                graph.add_node(token)
            for j in range(i + 1, min(i + window_size, len(tokens))):
                neighbor = tokens[j]
                if token == neighbor:
                    continue
                if graph.has_edge(token, neighbor):
                    graph[token][neighbor]["weight"] += 1.0
                else:
                    graph.add_edge(token, neighbor, weight=1.0)

    if len(graph) == 0:
        return []

    scores = nx.pagerank(graph, weight="weight")
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def extract_keywords_from_transcripts(
    transcripts: list[dict],
    top_k: int | None = None,
    window_size: int | None = None,
) -> list[dict]:
    """전사문 리스트에서 키워드를 추출해 DB 저장용 형태로 반환합니다."""
    top_k = top_k or DEFAULT_TOP_K
    window_size = window_size or DEFAULT_WINDOW

    keywords: list[dict] = []
    for transcript in transcripts:
        text = transcript.get("text") or ""
        if not text.strip():
            continue

        tokenized_sentences = _tokenize_sentences(text)
        if not tokenized_sentences:
            continue

        ranked = _extract_keywords_textrankr(text, tokenized_sentences, top_k)
        if ranked is None:
            ranked = _textrank_keywords(tokenized_sentences, top_k, window_size)

        for rank_order, (keyword, score) in enumerate(ranked, start=1):
            keywords.append({
                "transcript_id": transcript.get("transcript_id"),
                "keyword_text": keyword,
                "score": float(score),
                "rank_order": rank_order,
            })

    return keywords
