import math
import os
import re
import sys

import networkx as nx
from fastapi import APIRouter, HTTPException
from kiwipiepy import Kiwi
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from materials.material_text_service import MaterialTextError, build_material_text


router = APIRouter(prefix="/summary/test", tags=["summary-test"])

_kiwi = Kiwi()
_SENTENCE_END_RE = re.compile(r"(?<=[\.\?\!。？！])\s+|\n+")
_PAGE_TAG_RE = re.compile(r"\[PDF page (\d+)\]")
_PDF_METADATA_RE = re.compile(r"\[PDF [^\]]+\]\s*")
_KEYWORD_TAGS = {"NNG", "NNP", "VV", "VA"}


class MaterialTextRankRequest(BaseModel):
    session_id: str
    material_ids: list[str] = Field(default_factory=list)
    stored_names: list[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=30)
    min_sentence_chars: int = Field(default=12, ge=4, le=80)


def _clean_candidate(text: str) -> str:
    text = _PDF_METADATA_RE.sub("", text or "")
    text = re.sub(r"^[\-\*\u2022\u25aa\u25cf\u25e6\s]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_long_text(text: str, max_chars: int = 240) -> list[str]:
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


def _extract_sentence_candidates(material_text: str, min_chars: int) -> list[dict]:
    candidates = []
    current_page = None
    order = 0

    for raw_line in material_text.splitlines():
        page_match = _PAGE_TAG_RE.search(raw_line)
        if page_match:
            current_page = int(page_match.group(1))
            raw_line = _PAGE_TAG_RE.sub("", raw_line)

        line = _clean_candidate(raw_line)
        if not line:
            continue
        if line.startswith("자료명:") or line.startswith("전체 페이지:"):
            continue

        for part in _SENTENCE_END_RE.split(line):
            cleaned = _clean_candidate(part)
            if len(cleaned) < min_chars:
                continue
            for sentence in _split_long_text(cleaned):
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
    return {
        token.form
        for token in _kiwi.tokenize(text)
        if token.tag in _KEYWORD_TAGS and len(token.form) >= 2
    }


def _sentence_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    return overlap / math.sqrt(len(left) * len(right))


def _rank_sentences(candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []

    token_sets = [_tokenize_for_rank(item["text"]) for item in candidates]
    graph = nx.Graph()
    for index, item in enumerate(candidates):
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

    ranked = []
    for index, item in enumerate(candidates):
        ranked.append(
            {
                **item,
                "score": float(scores.get(index, 0.0)),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def _format_markdown(ranked_sentences: list[dict], candidate_count: int) -> str:
    if not ranked_sentences:
        return "## TextRank 문장 추출 결과\n\nPDF에서 표시할 문장 후보를 찾지 못했습니다."

    lines = [
        "## TextRank 문장 추출 결과",
        "",
        f"- 문장 후보: {candidate_count}개",
        f"- 표시 문장: {len(ranked_sentences)}개",
        "",
    ]

    for rank, item in enumerate(ranked_sentences, start=1):
        page = f"p.{item['page']}" if item.get("page") else "page unknown"
        score = f"{item['score']:.4f}"
        lines.append(f"{rank}. **{page}** · score {score}")
        lines.append(f"   - {item['text']}")

    return "\n".join(lines)


@router.post("/material/textrank")
async def material_textrank_preview(req: MaterialTextRankRequest):
    try:
        material_text, materials = await build_material_text(
            session_id=req.session_id,
            material_ids=req.material_ids,
            stored_names=req.stored_names,
        )
    except MaterialTextError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    candidates = _extract_sentence_candidates(material_text, req.min_sentence_chars)
    ranked_sentences = _rank_sentences(candidates, req.top_k)
    source_materials = [
        {
            "id": material.get("id"),
            "name": material.get("name"),
            "storedName": material.get("storedName"),
        }
        for material in materials
    ]
    first_material = source_materials[0] if source_materials else {}
    material_key = (
        first_material.get("id")
        or first_material.get("storedName")
        or first_material.get("name")
        or "unknown"
    )

    markdown = _format_markdown(ranked_sentences, len(candidates))

    return {
        "summary_id": None,
        "session_id": req.session_id,
        "recording_id": f"textrank:{material_key}",
        "speaker_id": "MATERIAL_TEXTRANK_TEST",
        "session_summary": markdown,
        "material_summary": markdown,
        "summary_level": "textrank",
        "source_materials": source_materials,
        "candidate_count": len(candidates),
        "ranked_sentences": ranked_sentences,
    }
