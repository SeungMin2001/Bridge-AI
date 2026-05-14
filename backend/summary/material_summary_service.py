"""
PDF 강의자료 요약 서비스.

흐름:
    PDF 텍스트
    -> TextRank로 중요한 문장 후보 추림
    -> LLM으로 사람이 읽기 좋은 Markdown 요약 생성
"""
import logging
import os

from summary.material_textrank import (
    extract_ranked_sentences,
    format_ranked_sentences_for_prompt,
)
from summary.summary_service import (
    MOCK_MODE,
    _build_messages,
    _call_llm,
)


logger = logging.getLogger(__name__)

MATERIAL_TEXTRANK_TOP_K = int(os.getenv("SUMMARY_MATERIAL_TEXTRANK_TOP_K", "24"))
MATERIAL_MIN_SENTENCE_CHARS = int(os.getenv("SUMMARY_MATERIAL_MIN_SENTENCE_CHARS", "12"))
MATERIAL_SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MATERIAL_MAX_TOKENS", "2400"))


MATERIAL_SUMMARY_LEVEL_ALIASES = {
    "short": "brief",
    "simple": "brief",
    "basic": "standard",
    "normal": "standard",
    "detail": "detailed",
    "full": "detailed",
    "page_by_page": "page",
    "pages": "page",
}

MATERIAL_SUMMARY_LEVELS = {
    "brief": {
        "label": "간단 요약",
        "topic_count": 4,
        "format": """1. 큰 주제 제목
   - **핵심 개념**: 설명
   - **중요 내용**: 설명
   - **관련 페이지**: p.1-3""",
        "guidance": "- 큰 주제는 3~5개로 압축\n- 빠른 복습용으로 핵심만 남김",
    },
    "standard": {
        "label": "표준 요약",
        "topic_count": 8,
        "format": """1. 큰 주제 제목
   - **핵심 개념**: 설명
   - **중요 내용**: 설명
   - **예시/비교**: 설명
   - **관련 페이지**: p.1-3""",
        "guidance": "- 큰 주제는 6~10개 안팎으로 정리\n- 시험 대비에 필요한 개념, 예시, 비교를 균형 있게 포함",
    },
    "detailed": {
        "label": "상세 요약",
        "topic_count": 14,
        "format": """1. 큰 주제 제목
   - **핵심 개념**: 설명
   - **세부 내용**: 설명
   - **예시/비교**: 설명
   - **주의점/시험 포인트**: 설명
   - **관련 페이지**: p.1-3""",
        "guidance": "- 큰 주제는 10~15개 안팎으로 자세히 정리\n- 구현 방식, 예시, 비교, 시험 포인트를 구체적으로 작성",
    },
    "page": {
        "label": "페이지별 요약",
        "topic_count": 20,
        "format": """1. p.1 제목 또는 핵심 주제
   - **핵심 내용**: 설명
   - **중요 용어**: 설명

2. p.2 제목 또는 핵심 주제
   - **핵심 내용**: 설명
   - **중요 용어**: 설명""",
        "guidance": "- 페이지 번호가 있는 문장은 페이지 순서를 최대한 유지\n- 페이지별 핵심 내용을 짧게 정리",
    },
}

MATERIAL_SUMMARY_PROMPT_TEMPLATE = """아래는 PDF 강의자료에서 TextRank로 먼저 고른 핵심 원문 문장입니다.
이 문장들을 근거로 {level_label} 수준의 한국어 Markdown 요약을 작성하세요.

요약 형식:
{format_instructions}

조건:
- 반드시 한국어로 작성
- 핵심 용어는 **굵게** 표시
- 원문에 없는 사실은 추가하지 말 것
- 깨진 수식/표/코드 조각은 중요한 내용이 아니면 무시
- 관련 페이지는 "p.10"처럼 표시
{level_guidance}

[TextRank 핵심 문장]
{ranked_sentences}

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "summary_text": "Markdown 요약"
}}
"""


def normalize_material_summary_level(summary_level: str | None) -> str:
    """프론트에서 들어온 요약 단계 값을 내부 표준값으로 맞춘다."""
    level = (summary_level or "standard").strip().lower()
    level = MATERIAL_SUMMARY_LEVEL_ALIASES.get(level, level)
    return level if level in MATERIAL_SUMMARY_LEVELS else "standard"


def _mock_material_summary(ranked_sentences: list[dict], level: str) -> str:
    """LLM 없이 동작 확인할 때 쓰는 간단한 요약 흉내."""
    if not ranked_sentences:
        return "PDF에서 요약할 핵심 문장을 찾지 못했습니다."

    lines = []
    for index, item in enumerate(ranked_sentences[: MATERIAL_SUMMARY_LEVELS[level]["topic_count"]], start=1):
        page = f"p.{item['page']}" if item.get("page") else "page unknown"
        lines.append(f"{index}. TextRank 핵심 문장")
        lines.append(f"   - **핵심 내용**: {item['text']}")
        lines.append(f"   - **관련 페이지**: {page}")
    return "\n".join(lines)


async def generate_material_summary_with_textrank(
    material_text: str,
    summary_level: str | None = None,
    top_k: int | None = None,
) -> tuple[str, dict]:
    """
    PDF 텍스트를 TextRank로 압축한 뒤 LLM 요약을 생성한다.

    반환:
        summary_text: 프론트에 보여줄 Markdown 요약
        metadata: source_text에 저장할 후보/선택 문장 정보
    """
    if not material_text.strip():
        raise ValueError("요약할 강의자료 텍스트가 없습니다")

    level = normalize_material_summary_level(summary_level)
    config = MATERIAL_SUMMARY_LEVELS[level]
    target_top_k = top_k or max(MATERIAL_TEXTRANK_TOP_K, config["topic_count"] * 3)

    candidates, ranked_sentences = extract_ranked_sentences(
        material_text,
        top_k=target_top_k,
        min_chars=MATERIAL_MIN_SENTENCE_CHARS,
    )
    if not ranked_sentences:
        raise ValueError("TextRank로 요약에 사용할 문장을 찾지 못했습니다")

    metadata = {
        "summaryLevel": level,
        "candidateCount": len(candidates),
        "rankedSentenceCount": len(ranked_sentences),
        "rankedSentences": ranked_sentences,
    }

    if MOCK_MODE:
        logger.info("[SUMMARY] MOCK_MODE: TextRank 기반 자료 요약 반환")
        return _mock_material_summary(ranked_sentences, level), metadata

    prompt = MATERIAL_SUMMARY_PROMPT_TEMPLATE.format(
        level_label=config["label"],
        format_instructions=config["format"],
        level_guidance=config["guidance"],
        ranked_sentences=format_ranked_sentences_for_prompt(ranked_sentences),
    )
    summary_text = await _call_llm(
        _build_messages(prompt),
        max_tokens=MATERIAL_SUMMARY_MAX_TOKENS,
    )
    return summary_text, metadata
