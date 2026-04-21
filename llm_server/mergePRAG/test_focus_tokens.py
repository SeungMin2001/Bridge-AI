import re
from collections import Counter

import torch

from .config import (
    ENABLE_FOCUS_WEIGHT,
    FOCUS_WEIGHT_COLOR,
    FOCUS_WEIGHT_DATE,
    FOCUS_WEIGHT_DAY,
    FOCUS_WEIGHT_MAX,
    FOCUS_WEIGHT_NUMBER,
    FOCUS_WEIGHT_QUESTION_OVERLAP,
    FOCUS_WEIGHT_RARE,
)

_WORD_RE = re.compile(r"[A-Za-z0-9가-힣]+")

# 영어/한국어 fact 토큰(색/요일/날짜) 중심 사전
_COLOR_TERMS = {
    "red", "blue", "green", "yellow", "black", "white", "purple", "orange", "pink", "brown", "gray", "grey",
    "빨강", "빨간", "빨간색", "파랑", "파란", "파란색", "초록", "초록색", "녹색", "노랑", "노란", "노란색",
    "검정", "검은", "검은색", "하양", "하얀", "하얀색", "보라", "보라색", "주황", "주황색",
}
_DAY_TERMS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일",
}
_DATE_TERMS = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "년", "월", "일",
}
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "for", "and", "or", "with", "by", "at",
    "what", "which", "who", "when", "where", "why", "how", "did", "does", "do", "it", "this", "that",
    "은", "는", "이", "가", "을", "를", "에", "의", "도", "로", "으로", "와", "과", "그리고", "또는",
}


def _normalize_token_piece(piece: str) -> str:
    if not piece:
        return ""
    text = piece.replace("Ġ", "").replace("▁", "").strip().lower()
    m = _WORD_RE.search(text)
    return m.group(0) if m else ""


def _extract_terms(text: str) -> set[str]:
    if not text:
        return set()
    terms = {t.lower() for t in _WORD_RE.findall(text)}
    return {t for t in terms if len(t) >= 2 and t not in _STOPWORDS}


def build_focus_weight(tokenizer, input_ids: torch.Tensor, passage_mask: torch.Tensor, question_text: str = "") -> torch.Tensor:
    """규칙 기반 핵심 토큰 가중치.

    - 숫자/날짜/요일/색상/질문 중복 토큰 가중치를 높여 pooling 단계에서 사실 토큰 영향력을 강화한다.
    - 출력 shape: [B, T], 값 범위: [1, FOCUS_WEIGHT_MAX]
    """
    base = torch.ones_like(input_ids, dtype=torch.float32)
    if not ENABLE_FOCUS_WEIGHT:
        return base

    question_terms = _extract_terms(question_text)

    for b in range(input_ids.size(0)):
        ids = input_ids[b].tolist()
        mask = passage_mask[b].tolist()
        pieces = tokenizer.convert_ids_to_tokens(ids)
        normalized = [_normalize_token_piece(p) for p in pieces]

        passage_terms = [normalized[i] for i, m in enumerate(mask) if m == 1 and normalized[i]]
        freq = Counter(passage_terms)

        for i, is_passage in enumerate(mask):
            if is_passage != 1:
                continue

            term = normalized[i]
            if not term:
                continue

            w = 1.0
            if any(ch.isdigit() for ch in term):
                w *= FOCUS_WEIGHT_NUMBER
            if term in _COLOR_TERMS:
                w *= FOCUS_WEIGHT_COLOR
            if term in _DAY_TERMS:
                w *= FOCUS_WEIGHT_DAY
            if term in _DATE_TERMS:
                w *= FOCUS_WEIGHT_DATE
            if question_terms and term in question_terms:
                w *= FOCUS_WEIGHT_QUESTION_OVERLAP
            if len(term) >= 3 and term not in _STOPWORDS and freq.get(term, 0) == 1:
                w *= FOCUS_WEIGHT_RARE

            base[b, i] = min(float(base[b, i].item()) * w, FOCUS_WEIGHT_MAX)

    return base

