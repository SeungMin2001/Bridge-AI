import os
import re
import logging

from materials.material_citation_service import build_material_citation, format_material_citation
from materials.material_ocr_service import ocr_pdf_page
from materials.material_text_service import (
    MaterialTextError,
    extract_pdf_text,
    get_session_pdf_materials,
    stored_name_from_material,
)


PAGE_SEARCH_TOP_K = int(os.getenv("CHAT_MATERIAL_PAGE_SEARCH_TOP_K", "3"))
PAGE_CONTEXT_MAX_CHARS = int(os.getenv("CHAT_MATERIAL_PAGE_CONTEXT_CHARS", "2200"))
MIN_PAGE_SCORE = float(os.getenv("CHAT_MATERIAL_PAGE_MIN_SCORE", "2.0"))
OCR_MIN_TEXT_CHARS = int(os.getenv("CHAT_MATERIAL_OCR_MIN_TEXT_CHARS", "120"))

logger = logging.getLogger(__name__)

_PAGE_BLOCK_RE = re.compile(r"\[PDF page (\d+)\]\s*(.*?)(?=\n\[PDF page \d+\]|\Z)", re.DOTALL)
_WORD_RE = re.compile(r"[가-힣A-Za-z0-9_]+")
_CODE_RE = re.compile(r"[A-Za-z_]\w*\([^)]*\)|[A-Za-z_]\w*")
_DIRECT_PAGE_RE = re.compile(r"(?:p\.?\s*|page\s*)?(\d+)\s*(?:페이지|쪽)|(?:p\.?|page)\s*(\d+)", re.IGNORECASE)

_STOPWORDS = {
    "이",
    "그",
    "저",
    "것",
    "거",
    "여기",
    "에서",
    "어디",
    "부분",
    "내용",
    "자료",
    "파일",
    "pdf",
    "대해",
    "대한",
    "대해서",
    "설명",
    "설명해줘",
    "알려줘",
    "알려줄래",
    "알고",
    "싶어",
    "무슨",
    "어떤",
    "해주세요",
    "해줘",
    "나요",
    "인가",
    "있는",
    "있어",
    "하고",
    "그리고",
    "뭐야",
    "뭔가",
    "어디야",
    "어디에",
    "페이지",
    "쪽",
    "언급",
    "언급한",
    "나온",
    "나와",
    "보여줘",
}

_KOREAN_PARTICLE_SUFFIXES = (
    "으로부터",
    "에서부터",
    "로부터",
    "에게서",
    "한테서",
    "에서는",
    "으로는",
    "에게",
    "한테",
    "에서",
    "까지",
    "부터",
    "처럼",
    "보다",
    "으로",
    "라고",
    "이라",
    "라는",
    "이란",
    "이나",
    "이며",
    "하고",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "도",
    "만",
    "로",
    "와",
    "과",
)


async def build_material_page_context(
    question: str,
    session_id: str | None,
    source_filter: dict | None,
) -> tuple[str, list[dict]]:
    if not session_id:
        return "", []

    material_ids = source_filter.get("material_ids") if isinstance(source_filter, dict) else None
    stored_names = source_filter.get("stored_names") if isinstance(source_filter, dict) else None

    try:
        materials = await get_session_pdf_materials(
            session_id,
            material_ids=material_ids,
            stored_names=stored_names,
        )
    except MaterialTextError:
        return "", []

    profile = _build_query_profile(question)
    requested_pages = _extract_requested_pages(question)
    candidates = []

    for material in materials:
        stored_name = stored_name_from_material(material) or ""
        material_name = material.get("name") or material.get("title") or stored_name or "강의자료"

        try:
            material_text = extract_pdf_text(material)
        except MaterialTextError:
            continue

        pages = _split_pdf_pages(material_text)
        pages_by_number = {page["page"]: page for page in pages}

        for page_number in requested_pages:
            page = pages_by_number.get(page_number)
            if not page:
                continue
            page_text = _page_text_with_optional_ocr(material, page["page"], page["text"])
            candidates.append({
                "score": 1000.0,
                "page": page["page"],
                "text": page_text,
                "material": material,
                "stored_name": stored_name,
                "material_name": material_name,
            })

        for page in pages:
            score = _score_page(page["text"], profile)
            if score <= 0:
                continue
            page_text = _page_text_with_optional_ocr(material, page["page"], page["text"])
            candidates.append({
                "score": score,
                "page": page["page"],
                "text": page_text,
                "material": material,
                "stored_name": stored_name,
                "material_name": material_name,
            })

        if not profile["keywords"] and not profile["phrases"] and not profile["code_terms"]:
            fallback_page = _first_text_page(pages)
            if fallback_page:
                page_text = _page_text_with_optional_ocr(material, fallback_page["page"], fallback_page["text"])
                candidates.append({
                    "score": 1,
                    "page": fallback_page["page"],
                    "text": page_text,
                    "material": material,
                    "stored_name": stored_name,
                    "material_name": material_name,
                })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    ranked_candidates = _dedupe_page_candidates(candidates)
    selected = [item for item in ranked_candidates if item["score"] >= MIN_PAGE_SCORE][:PAGE_SEARCH_TOP_K]
    if not selected and ranked_candidates:
        selected = ranked_candidates[:1]

    if not selected:
        return "", []

    logger.info(
        "[MATERIAL:PAGE_SEARCH] question=%r selected=%s",
        question[:80],
        [
            {
                "material": item["material_name"],
                "page": item["page"],
                "score": round(item["score"], 2),
            }
            for item in selected
        ],
    )

    context_parts = []
    citations = []
    for index, item in enumerate(selected, 1):
        excerpt = _best_excerpt(item["text"], profile, PAGE_CONTEXT_MAX_CHARS)
        citation_result = {
            "text": excerpt,
            "full_transcript": item["text"],
            "material_id": str(item["material"].get("id") or ""),
            "material_name": item["material_name"],
            "stored_name": item["stored_name"],
            "page": item["page"],
            "chunk_index": None,
        }
        citation = format_material_citation(citation_result)
        context_parts.append(
            f"[PDF {index}] 자료명: {item['material_name']}\n"
            f"페이지: {item['page']}\n"
            f"{excerpt}\n"
            f"(출처: {citation})"
        )
        citations.append(build_material_citation(
            citation_result,
            citation=citation,
            session_id=session_id,
            search_scope="selected_pdf_page_search",
        ))

    return "\n\n".join(context_parts), citations


def _dedupe_page_candidates(candidates: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for item in candidates:
        material = item.get("material") or {}
        key = (
            str(material.get("id") or ""),
            item.get("stored_name") or "",
            item.get("page"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _extract_requested_pages(question: str) -> list[int]:
    pages = []
    seen = set()
    for match in _DIRECT_PAGE_RE.finditer(question or ""):
        page = _safe_int(match.group(1) or match.group(2))
        if page <= 0 or page in seen:
            continue
        seen.add(page)
        pages.append(page)
    return pages


def _page_text_with_optional_ocr(material: dict, page_number: int, page_text: str) -> str:
    text = _clean_text(page_text)
    if not _needs_ocr(text):
        return text

    ocr_text = ocr_pdf_page(material, page_number)
    if not ocr_text:
        return text
    if "텍스트 추출 없음" in text:
        return f"[OCR text p={page_number}]\n{ocr_text}"
    return f"{text}\n[OCR text p={page_number}]\n{ocr_text}"


def _needs_ocr(text: str) -> bool:
    cleaned = _compact(text)
    return "텍스트추출없음" in cleaned or len(cleaned) < OCR_MIN_TEXT_CHARS


def _split_pdf_pages(material_text: str) -> list[dict]:
    pages = []
    for match in _PAGE_BLOCK_RE.finditer(material_text or ""):
        page_number = _safe_int(match.group(1))
        text = _clean_text(match.group(2))
        if page_number > 0 and text:
            pages.append({"page": page_number, "text": text})
    return pages


def _build_query_profile(question: str) -> dict:
    normalized = _normalize(question)
    keywords = []
    seen = set()
    for token in _WORD_RE.findall(normalized):
        for variant in _keyword_variants(token):
            if variant not in seen:
                seen.add(variant)
                keywords.append(variant)

    code_terms = []
    seen_code = set()
    for term in _CODE_RE.findall(question or ""):
        normalized_term = _normalize(term)
        if len(normalized_term) < 2 or normalized_term in _STOPWORDS:
            continue
        if re.search(r"[A-Za-z_]", normalized_term) and normalized_term not in seen_code:
            seen_code.add(normalized_term)
            code_terms.append(normalized_term)

    phrases = []
    seen_phrase = set()
    for raw_part in re.split(r"[\n\r?!.。,，;；:：]+", question or ""):
        phrase = _normalize(raw_part)
        phrase = _strip_common_tail(phrase)
        compact = _compact(phrase)
        if len(compact) < 4 or compact in seen_phrase:
            continue
        seen_phrase.add(compact)
        phrases.append(phrase)

    return {
        "keywords": keywords,
        "code_terms": code_terms,
        "phrases": phrases,
    }


def _keyword_variants(token: str) -> list[str]:
    token = _normalize(token)
    compact_token = _compact(token)
    if len(compact_token) < 2:
        return []

    variants = [token]
    stemmed_to_stopword = False
    for suffix in _KOREAN_PARTICLE_SUFFIXES:
        if not token.endswith(suffix):
            continue
        stem = token[:-len(suffix)].strip()
        if len(_compact(stem)) >= 2:
            if stem in _STOPWORDS or _compact(stem) in _STOPWORDS:
                stemmed_to_stopword = True
                continue
            variants.append(stem)

    cleaned = []
    seen = set()
    for variant in variants:
        variant = _normalize(variant)
        compact_variant = _compact(variant)
        if len(compact_variant) < 2:
            continue
        if variant == token and stemmed_to_stopword:
            continue
        if variant in _STOPWORDS or compact_variant in _STOPWORDS:
            continue
        if compact_variant in seen:
            continue
        seen.add(compact_variant)
        cleaned.append(variant)
    return cleaned


def _score_page(text: str, profile: dict) -> float:
    normalized_text = _normalize(text)
    compact_text = _compact(text)
    heading_text = _page_heading_text(text)
    heading_compact = _compact(heading_text)

    score = 0.0
    for phrase in profile["phrases"]:
        compact_phrase = _compact(phrase)
        if not compact_phrase:
            continue
        if compact_phrase in compact_text:
            score += min(24.0, 6.0 + len(compact_phrase) * 0.35)
        if heading_compact and compact_phrase in heading_compact:
            score += 10.0

    for token in profile["keywords"]:
        compact_token = _compact(token)
        if not compact_token:
            continue
        count = compact_text.count(compact_token)
        if count:
            score += min(10.0, count * 2.2)
        if heading_compact and compact_token in heading_compact:
            score += 4.0

    for term in profile["code_terms"]:
        compact_term = _compact(term)
        bare_term = _normalize(term.replace("()", ""))
        if compact_term and compact_term in compact_text:
            score += 8.0
        elif bare_term and bare_term in normalized_text:
            score += 4.0

    return score


def _best_excerpt(text: str, profile: dict, max_chars: int) -> str:
    text = _clean_text(text)
    if len(text) <= max_chars:
        return text

    lower_text = text.lower()
    search_terms = []
    search_terms.extend(profile["phrases"])
    search_terms.extend(profile["code_terms"])
    search_terms.extend(profile["keywords"])

    positions = []
    for term in search_terms:
        term = _normalize(term)
        if not term:
            continue
        pos = lower_text.find(term)
        if pos >= 0:
            positions.append(pos)

    if positions:
        center = min(positions)
        start = max(0, center - max_chars // 4)
    else:
        start = 0

    end = min(len(text), start + max_chars)
    excerpt = text[start:end].strip()
    if start > 0:
        excerpt = f"... {excerpt}"
    if end < len(text):
        excerpt = f"{excerpt} ..."
    return excerpt


def _first_text_page(pages: list[dict]) -> dict | None:
    for page in pages:
        text = page.get("text") or ""
        if "텍스트 추출 없음" not in text and len(text) > 30:
            return page
    return pages[0] if pages else None


def _page_heading_text(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    heading_lines = []
    for line in lines[:8]:
        if line.startswith("[PDF heading hints"):
            heading_lines.append(line)
            continue
        if len(line) <= 90:
            heading_lines.append(line)
    return " ".join(heading_lines[:4])


def _strip_common_tail(text: str) -> str:
    text = re.sub(r"\b(알려줘|알려줄래|설명해줘|해줘|해주세요)\b", "", text)
    text = re.sub(r"(에\s*)?(대해서|대한)\s*", " ", text)
    return _normalize(text)


def _clean_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in (text or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣_]+", "", _normalize(text))


def _safe_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
