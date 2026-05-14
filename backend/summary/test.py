"""
PDF TextRank 실험용 API.

목표:
    1. 프론트에서 선택한 PDF 강의자료 텍스트를 읽는다.
    2. PDF 텍스트를 "문장 후보"로 잘게 나눈다.
    3. 문장끼리 비슷한 정도를 계산해서 그래프를 만든다.
    4. PageRank(TextRank 방식) 점수가 높은 문장만 화면에 보여준다.

주의:
    - 이 파일은 완성된 PDF 요약 기능이 아니라 화면 확인용 preview API다.
    - DB에 요약 결과를 저장하지 않는다.
"""
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

# Kiwi는 한국어 문장에서 명사/동사/형용사 같은 의미 있는 단어를 뽑기 위해 사용한다.
_kiwi = Kiwi()

# PDF에서 뽑힌 긴 텍스트를 문장 단위로 자르기 위한 정규식.
_SENTENCE_END_RE = re.compile(r"(?<=[\.\?\!。？！])\s+|\n+")

# materials.material_text_service가 페이지마다 붙여주는 "[PDF page N]" 태그를 찾는다.
_PAGE_TAG_RE = re.compile(r"\[PDF page (\d+)\]")
_PDF_METADATA_RE = re.compile(r"\[PDF [^\]]+\]\s*")

# TextRank 점수 계산에 사용할 품사. 조사/어미는 버리고 의미 있는 단어 위주로 비교한다.
_KEYWORD_TAGS = {"NNG", "NNP", "VV", "VA"}


class MaterialTextRankRequest(BaseModel):
    """프론트에서 PDF TextRank preview를 요청할 때 받는 값."""

    session_id: str
    material_ids: list[str] = Field(default_factory=list)
    stored_names: list[str] = Field(default_factory=list)

    # 최종 화면에 보여줄 상위 문장 개수.
    top_k: int = Field(default=10, ge=1, le=30)

    # 너무 짧은 제목/조각은 후보에서 제외하기 위한 최소 글자 수.
    min_sentence_chars: int = Field(default=12, ge=4, le=80)


def _clean_candidate(text: str) -> str:
    """PDF 메타데이터 태그, bullet 문자, 불필요한 공백을 제거한다."""
    text = _PDF_METADATA_RE.sub("", text or "")
    text = re.sub(r"^[\-\*\u2022\u25aa\u25cf\u25e6\s]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_long_text(text: str, max_chars: int = 240) -> list[str]:
    """PDF 추출 결과가 너무 길게 붙은 경우 화면에 보기 좋은 길이로 나눈다."""
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
    """
    PDF 전체 텍스트에서 TextRank에 넣을 "문장 후보"를 만든다.

    문장 후보란:
        아직 중요하다고 판단된 문장이 아니라,
        중요도 계산 대상이 될 수 있는 모든 문장 조각이다.
    """
    candidates = []
    current_page = None
    order = 0

    for raw_line in material_text.splitlines():
        # 현재 줄에 페이지 태그가 있으면 이후 후보 문장에 페이지 번호를 붙인다.
        page_match = _PAGE_TAG_RE.search(raw_line)
        if page_match:
            current_page = int(page_match.group(1))
            raw_line = _PAGE_TAG_RE.sub("", raw_line)

        line = _clean_candidate(raw_line)
        if not line:
            continue

        # 자료명/전체 페이지 안내 문구는 실제 강의 내용이 아니므로 제외한다.
        if line.startswith("자료명:") or line.startswith("전체 페이지:"):
            continue

        for part in _SENTENCE_END_RE.split(line):
            cleaned = _clean_candidate(part)

            # 너무 짧은 조각은 제목/페이지 번호/깨진 텍스트일 가능성이 높아서 제외한다.
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
    """한 문장을 TextRank 비교용 단어 묶음으로 바꾼다."""
    return {
        token.form
        for token in _kiwi.tokenize(text)
        if token.tag in _KEYWORD_TAGS and len(token.form) >= 2
    }


def _sentence_similarity(left: set[str], right: set[str]) -> float:
    """
    두 문장이 얼마나 비슷한지 계산한다.

    두 문장이 공유하는 단어가 많을수록 값이 커진다.
    값이 0이면 두 문장을 그래프에서 연결하지 않는다.
    """
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    return overlap / math.sqrt(len(left) * len(right))


def _rank_sentences(candidates: list[dict], top_k: int) -> list[dict]:
    """
    문장 후보 전체에 TextRank 점수를 매기고 상위 top_k개만 반환한다.

    TextRank의 핵심 아이디어:
        "다른 중요한 문장들과 많이 연결된 문장일수록 중요하다."
    """
    if not candidates:
        return []

    # 각 후보 문장을 단어 묶음으로 바꾼다. 이후 문장끼리 단어 겹침을 비교한다.
    token_sets = [_tokenize_for_rank(item["text"]) for item in candidates]

    # graph의 node 하나가 문장 후보 하나라고 보면 된다.
    graph = nx.Graph()
    for index, item in enumerate(candidates):
        graph.add_node(index)

    # 모든 문장 쌍을 비교해서, 비슷하면 두 문장 사이에 선(edge)을 만든다.
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            weight = _sentence_similarity(token_sets[left], token_sets[right])
            if weight > 0:
                graph.add_edge(left, right, weight=weight)

    if graph.number_of_edges() == 0:
        # 문장끼리 겹치는 단어가 전혀 없으면 모두 같은 점수로 둔다.
        scores = {index: 1.0 for index in range(len(candidates))}
    else:
        # 여기서 PageRank를 돌리는 부분이 TextRank의 핵심이다.
        scores = nx.pagerank(graph, weight="weight")

    ranked = []
    for index, item in enumerate(candidates):
        ranked.append(
            {
                **item,
                "score": float(scores.get(index, 0.0)),
            }
        )

    # 점수가 높은 문장부터 정렬한 뒤, 화면에 보여줄 개수만 자른다.
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def _format_markdown(ranked_sentences: list[dict], candidate_count: int) -> str:
    """프론트가 바로 표시할 수 있도록 TextRank 결과를 Markdown 문자열로 만든다."""
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

        # score는 TextRank 중요도 점수다. 점수가 높을수록 더 중요하다고 본다.
        lines.append(f"{rank}. **{page}** · score {score}")
        lines.append(f"   - {item['text']}")

    return "\n".join(lines)


@router.post("/material/textrank")
async def material_textrank_preview(req: MaterialTextRankRequest):
    """
    프론트의 자료요약 버튼이 호출하는 실험용 API.

    반환값은 기존 자료요약 화면 구조에 맞추기 위해
    material_summary/session_summary 형태로 맞춰서 내려준다.
    """
    try:
        # 업로드된 PDF 파일을 찾아 텍스트로 추출한다.
        material_text, materials = await build_material_text(
            session_id=req.session_id,
            material_ids=req.material_ids,
            stored_names=req.stored_names,
        )
    except MaterialTextError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    # 1단계: PDF 텍스트에서 TextRank 계산 대상이 될 문장 후보를 만든다.
    candidates = _extract_sentence_candidates(material_text, req.min_sentence_chars)

    # 2단계: 후보 문장에 점수를 매긴 뒤 상위 top_k개만 고른다.
    ranked_sentences = _rank_sentences(candidates, req.top_k)

    # 프론트 카드 하단에 "어떤 PDF에서 뽑았는지" 보여주기 위한 메타데이터.
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

    # 3단계: 화면에 표시할 Markdown 본문을 만든다.
    markdown = _format_markdown(ranked_sentences, len(candidates))

    # DB 저장은 하지 않고, 프론트가 요약 카드로 바로 보여줄 수 있는 형태만 반환한다.
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
