"""
PDF 자료 요약 전처리용 TextRank 유틸.

이 파일의 역할은 "요약문 작성"이 아니라,
PDF에서 뽑힌 긴 텍스트 중 LLM에 먼저 보여줄 핵심 원문 문장 후보를 고르는 것이다.
"""
import math
import re

import networkx as nx
from kiwipiepy import Kiwi


_kiwi = Kiwi()
_SENTENCE_END_RE = re.compile(r"(?<=[\.\?\!。？！])\s+|\n+")
_PAGE_TAG_RE = re.compile(r"\[PDF page (\d+)\]")
_PDF_METADATA_RE = re.compile(r"\[PDF [^\]]+\]\s*")
_KEYWORD_TAGS = {"NNG", "NNP", "VV", "VA"}
DEMO_PIPELINE_LOG = True


def _demo_log(message: str) -> None:
    if DEMO_PIPELINE_LOG:
        print(f"[DEMO:SUMMARY:TextRank] {message}", flush=True)


def _preview(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    return compact if len(compact) <= limit else f"{compact[:limit - 3]}..."


def clean_material_sentence(text: str) -> str:
    """PDF 메타데이터, bullet 문자, 과도한 공백을 제거한다."""
    text = _PDF_METADATA_RE.sub("", text or "")
    text = re.sub(r"^[\-\*\u2022\u25aa\u25cf\u25e6\u2756\s]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_too_noisy(text: str) -> bool:
    """수식/표/깨진 행렬처럼 요약 재료로 쓰기 어려운 문장을 걸러낸다."""
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True

    symbols = sum(1 for char in compact if char in "[]{}=+*/<>|_\\")
    if symbols >= max(5, len(compact) * 0.18):
        return True

    digits = sum(1 for char in compact if char.isdigit())
    if digits >= max(8, len(compact) * 0.35):
        return True

    letters = re.sub(r"[^가-힣A-Za-z]", "", compact)
    return len(letters) < 4


def _split_long_text(text: str, max_chars: int = 260) -> list[str]:
    """PDF 추출 결과가 한 줄로 길게 붙는 경우를 대비해 적당히 자른다."""
    if len(text) <= max_chars:
        return [text]

    parts = []
    words = text.split()
    current = []
    current_len = 0
    for word in words:
        next_len = current_len + len(word) + (1 if current else 0)
        if current and next_len > max_chars:
            parts.append(" ".join(current))
            current = [word]
            current_len = len(word)
            continue
        current.append(word)
        current_len = next_len

    if current:
        parts.append(" ".join(current))
    return parts


def extract_sentence_candidates(material_text: str, min_chars: int = 12) -> list[dict]:
    """
    PDF 전체 텍스트를 TextRank 계산 대상 문장 후보로 바꾼다.

    반환 item 예:
        {"page": 3, "order": 12, "text": "배열은 같은 자료형의 데이터를 저장한다."}
    """
    candidates = []
    current_page = None
    seen = set()
    order = 0

    for raw_line in material_text.splitlines():
        page_match = _PAGE_TAG_RE.search(raw_line)
        if page_match:
            current_page = int(page_match.group(1))
            raw_line = _PAGE_TAG_RE.sub("", raw_line)

        line = clean_material_sentence(raw_line)
        if not line:
            continue
        if line.startswith("자료명:") or line.startswith("전체 페이지:"):
            continue

        for part in _SENTENCE_END_RE.split(line):
            cleaned = clean_material_sentence(part)
            if len(cleaned) < min_chars:
                continue
            if _looks_too_noisy(cleaned):
                continue

            for sentence in _split_long_text(cleaned):
                normalized = re.sub(r"\W+", "", sentence.lower())
                if normalized in seen:
                    continue
                seen.add(normalized)
                order += 1
                candidates.append(
                    {
                        "page": current_page,
                        "order": order,
                        "text": sentence,
                    }
                )

    return candidates


def _tokenize_for_rank(text: str) -> set[str]:
    """한 문장을 TextRank 비교용 단어 묶음으로 바꾼다."""
    return {
        token.form
        for token in _kiwi.tokenize(text)
        if token.tag in _KEYWORD_TAGS and len(token.form) >= 2
    }


def _sentence_similarity(left: set[str], right: set[str]) -> float:
    """두 문장이 공유하는 단어가 많을수록 높은 유사도를 준다."""
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    return overlap / math.sqrt(len(left) * len(right))


def rank_sentences(candidates: list[dict], top_k: int = 16) -> list[dict]:
    """문장 후보에 TextRank 점수를 매긴 뒤 상위 top_k개를 반환한다."""
    if not candidates:
        return []

    token_sets = [_tokenize_for_rank(item["text"]) for item in candidates]
    graph = nx.Graph()
    for index, _item in enumerate(candidates):
        graph.add_node(index)

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            weight = _sentence_similarity(token_sets[left], token_sets[right])
            if weight > 0:
                graph.add_edge(left, right, weight=weight)

    if graph.number_of_edges() == 0:
        scores = {index: 1.0 for index in range(len(candidates))}
    else:
        scores = nx.pagerank(graph, weight="weight")

    ranked = [
        {
            **item,
            "score": float(scores.get(index, 0.0)),
        }
        for index, item in enumerate(candidates)
    ]
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def extract_ranked_sentences(
    material_text: str,
    top_k: int = 16,
    min_chars: int = 12,
) -> tuple[list[dict], list[dict]]:
    """PDF 텍스트에서 문장 후보 전체와 TextRank 상위 문장을 함께 반환한다."""
    _demo_log(f"1) 문장 후보 추출 시작: input_chars={len(material_text or '')}, top_k={top_k}")
    candidates = extract_sentence_candidates(material_text, min_chars=min_chars)
    _demo_log(f"2) 문장 후보 추출 완료: candidates={len(candidates)}")
    for index, item in enumerate(candidates[:5], start=1):
        tokens = sorted(_tokenize_for_rank(item["text"]))[:10]
        _demo_log(f"   형태소 후보#{index}: sentence='{_preview(item['text'], 90)}' tokens={tokens}")
    _demo_log("3) TextRank PageRank 계산 시작")
    ranked = rank_sentences(candidates, top_k=top_k)
    _demo_log(f"4) TextRank 핵심문장 선택 완료: selected={len(ranked)}")
    for index, item in enumerate(ranked[:5], start=1):
        _demo_log(
            f"   핵심문장#{index}: score={item.get('score', 0):.4f}, "
            f"page={item.get('page')}, text='{_preview(item.get('text'), 110)}'"
        )
    return candidates, ranked


def format_ranked_sentences_for_prompt(ranked_sentences: list[dict]) -> str:
    """LLM 프롬프트에 넣기 좋은 형태로 TextRank 문장을 정리한다."""
    if not ranked_sentences:
        return "(TextRank로 선택된 문장이 없습니다.)"

    lines = []
    for index, item in enumerate(sorted(ranked_sentences, key=lambda row: row["order"]), start=1):
        page = f"p.{item['page']}" if item.get("page") else "page unknown"
        lines.append(f"{index}. [{page}] {item['text']}")
    return "\n".join(lines)
