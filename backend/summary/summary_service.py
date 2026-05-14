"""
요약 서비스 (Summary Service)

역할:
    1. 화자 텍스트 기반 요약 생성
    2. 키워드/화자 요약을 참고한 세션 요약 생성
    3. 키워드/세션/화자 요약을 참고한 코스 요약 생성
"""
import json
import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

# ── 설정 ──
MOCK_MODE = os.getenv("SUMMARY_MOCK_MODE", "false").lower() == "true"
LLM_URL = os.getenv("SUMMARY_LLM_URL", os.getenv("LLM_URL", "http://localhost:8001"))
LLM_MODEL = os.getenv("SUMMARY_LLM_MODEL", os.getenv("LLM_MODEL", "QuantTrio/Qwen3.5-4B-AWQ"))
LLM_API_KEY = os.getenv("SUMMARY_LLM_API_KEY", os.getenv("LLM_API_KEY", "test-key"))

DEFAULT_SUMMARY_SENTENCES = int(os.getenv("SUMMARY_SENTENCES", 3))
MAX_SPEAKER_CHARS = int(os.getenv("SUMMARY_SPEAKER_MAX_CHARS", 6000))
MAX_SESSION_CHARS = int(os.getenv("SUMMARY_SESSION_MAX_CHARS", 4000))
MAX_COURSE_CHARS = int(os.getenv("SUMMARY_COURSE_MAX_CHARS", 4000))
MAX_KEYWORDS = int(os.getenv("SUMMARY_MAX_KEYWORDS", 30))
MAX_SPEAKER_SUMMARIES = int(os.getenv("SUMMARY_MAX_SPEAKER_SUMMARIES", 10))
MAX_SESSION_SUMMARIES = int(os.getenv("SUMMARY_MAX_SESSION_SUMMARIES", 10))

SUMMARY_SYSTEM_PROMPT = (
    "당신은 강의 내용을 간결하게 요약하는 AI입니다. "
    "반드시 아래 JSON 형식으로만 응답하세요. JSON 외 텍스트는 포함하지 마세요."
)

SUMMARY_SPEAKER_PROMPT_TEMPLATE = """아래는 화자 {speaker_id}의 전사문입니다:

{transcript_text}

위 내용을 바탕으로 {summary_sentences}문장 이내의 한국어 요약을 작성하세요.
- 중복을 제거하고 핵심만 요약
- 새로운 사실을 추가하지 말 것

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "summary_text": "요약 텍스트"
}}
"""

SUMMARY_SESSION_PROMPT_TEMPLATE = """아래는 세션 요약을 위한 정보입니다:

[핵심 키워드]
{keywords}

[화자별 요약]
{speaker_summaries}

위 내용을 바탕으로 {summary_sentences}문장 이내의 한국어 요약을 작성하세요.
- 중복을 제거하고 핵심만 요약
- 새로운 사실을 추가하지 말 것

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "summary_text": "요약 텍스트"
}}
"""

SUMMARY_SESSION_TEXT_PROMPT_TEMPLATE = """아래는 하나의 녹음 세션 전체 전사문입니다:

[핵심 키워드]
{keywords}

[전체 전사문]
{transcript_text}

위 내용을 바탕으로 {summary_sentences}문장 이내의 한국어 요약을 작성하세요.
- 중복을 제거하고 핵심만 요약
- 새로운 사실을 추가하지 말 것

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "summary_text": "요약 텍스트"
}}
"""

SUMMARY_COURSE_PROMPT_TEMPLATE = """아래는 과목 요약을 위한 정보입니다:

[핵심 키워드]
{keywords}

[세션 요약]
{session_summaries}

[화자별 요약]
{speaker_summaries}

위 내용을 바탕으로 {summary_sentences}문장 이내의 한국어 요약을 작성하세요.
- 중복을 제거하고 핵심만 요약
- 새로운 사실을 추가하지 말 것

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "summary_text": "요약 텍스트"
}}
"""

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!。？！])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    """문장 단위 분리를 통해 간단한 요약/미리보기 용도를 지원합니다."""
    text = re.sub(r"\r\n?", "\n", text)
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part and part.strip()]


def _truncate_text(text: str, max_chars: int) -> str:
    """LLM 입력 길이를 제한합니다."""
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _format_keywords(keywords: list[dict], limit: int) -> str:
    """키워드 리스트를 프롬프트용 문자열로 변환합니다."""
    if not keywords:
        return "(없음)"
    items = keywords[:limit]
    return "\n".join(f"- {kw['keyword_text']}" for kw in items)


def _format_speaker_label(speaker_id: str | None) -> str | None:
    """내부 speaker_id를 사용자에게 보일 화자명으로 변환합니다."""
    normalized = str(speaker_id or "").strip()
    if not normalized or normalized == "UNKNOWN":
        return None

    match = re.match(r"^SPEAKER_(\d+)$", normalized, re.IGNORECASE)
    if match:
        return f"화자 {int(match.group(1)) + 1}"

    return normalized


def _speaker_summary_sort_key(item: tuple[int, dict]) -> tuple[int, float | int]:
    index, summary = item
    start_time = summary.get("source_start_time")
    if isinstance(start_time, (int, float)):
        return (0, float(start_time))
    return (1, index)


def _format_speaker_summaries(summaries: list[dict], limit: int) -> str:
    """화자 요약 목록을 프롬프트용 문자열로 변환합니다."""
    if not summaries:
        return "(없음)"

    lines = []
    speaker_label_map = {}
    ordered_summaries = [
        summary for _, summary in sorted(enumerate(summaries), key=_speaker_summary_sort_key)
    ]

    for summary in ordered_summaries:
        speaker_id = str(summary.get("speaker_id") or "").strip()
        if not speaker_id or speaker_id == "UNKNOWN":
            continue

        if speaker_id not in speaker_label_map:
            fallback_label = _format_speaker_label(speaker_id)
            speaker_label_map[speaker_id] = (
                f"화자 {len(speaker_label_map) + 1}"
                if re.match(r"^SPEAKER_\d+$", speaker_id, re.IGNORECASE)
                else fallback_label
            )

        speaker_label = speaker_label_map[speaker_id]
        speaker_summary = str(summary.get("speaker_summary") or "").strip()
        if not speaker_label or not speaker_summary:
            continue

        lines.append(f"- {speaker_label}: {speaker_summary}")
        if len(lines) >= limit:
            break

    return "\n".join(lines) if lines else "(없음)"


def _format_session_summaries(summaries: list[dict], limit: int) -> str:
    """세션 요약 목록을 프롬프트용 문자열로 변환합니다."""
    if not summaries:
        return "(없음)"
    items = summaries[:limit]
    return "\n".join(
        f"- {s.get('session_id')}: {s.get('session_summary', '')}" for s in items
    )


def _parse_summary_json(raw_text: str) -> str:
    """LLM 응답에서 summary_text만 안전하게 파싱합니다."""
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("요약 JSON 파싱 실패: %s", exc)
        raise ValueError(f"LLM 응답을 JSON으로 파싱할 수 없습니다: {exc}")

    summary_text = payload.get("summary_text")
    if not summary_text or not isinstance(summary_text, str):
        raise ValueError("LLM 응답에 summary_text가 없습니다")

    return summary_text.strip()


def _mock_summary(text: str, summary_sentences: int) -> str:
    """Mock 모드에서 문장 앞부분만 사용해 요약을 흉내냅니다."""
    sentences = _split_sentences(text)
    if not sentences:
        return text.strip()
    return " ".join(sentences[:summary_sentences]).strip()


def _build_messages(user_prompt: str) -> list[dict]:
    """OpenAI 호환 LLM messages 형식을 구성합니다."""
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


async def _call_llm(messages: list[dict], max_tokens: int = 256) -> str:
    """LLM 호출 후 요약 문자열을 반환합니다."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=120.0)) as client:
        res = await client.post(
            f"{LLM_URL}/v1/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        )
        res.raise_for_status()

    data = res.json()
    raw_answer = data["choices"][0]["message"]["content"]
    logger.info("[SUMMARY] LLM 응답 수신: %d chars", len(raw_answer))
    return _parse_summary_json(raw_answer)


async def generate_speaker_summary(
    speaker_text: str,
    speaker_id: str,
    summary_sentences: int | None = None,
) -> str:
    """화자 텍스트를 기반으로 요약을 생성합니다."""
    if not speaker_text.strip():
        raise ValueError("요약할 화자 텍스트가 없습니다")

    summary_sentences = summary_sentences or DEFAULT_SUMMARY_SENTENCES
    transcript_text = _truncate_text(speaker_text, MAX_SPEAKER_CHARS)

    if MOCK_MODE:
        logger.info("[SUMMARY] MOCK_MODE: 화자 요약 반환")
        return _mock_summary(transcript_text, summary_sentences)

    user_prompt = SUMMARY_SPEAKER_PROMPT_TEMPLATE.format(
        speaker_id=speaker_id,
        transcript_text=transcript_text,
        summary_sentences=summary_sentences,
    )

    return await _call_llm(_build_messages(user_prompt))


async def generate_session_summary(
    keywords: list[dict],
    speaker_summaries: list[dict],
    summary_sentences: int | None = None,
) -> str:
    """키워드와 화자 요약을 참고해 세션 요약을 생성합니다."""
    if not keywords and not speaker_summaries:
        raise ValueError("세션 요약에 사용할 데이터가 없습니다")

    summary_sentences = summary_sentences or DEFAULT_SUMMARY_SENTENCES

    keywords_text = _format_keywords(keywords, MAX_KEYWORDS)
    speaker_text = _format_speaker_summaries(speaker_summaries, MAX_SPEAKER_SUMMARIES)

    payload_text = f"{keywords_text}\n\n{speaker_text}"
    payload_text = _truncate_text(payload_text, MAX_SESSION_CHARS)

    if MOCK_MODE:
        logger.info("[SUMMARY] MOCK_MODE: 세션 요약 반환")
        return _mock_summary(payload_text, summary_sentences)

    user_prompt = SUMMARY_SESSION_PROMPT_TEMPLATE.format(
        keywords=keywords_text,
        speaker_summaries=speaker_text,
        summary_sentences=summary_sentences,
    )

    return await _call_llm(_build_messages(user_prompt))


async def generate_session_summary_from_text(
    session_text: str,
    keywords: list[dict] | None = None,
    summary_sentences: int | None = None,
) -> str:
    """화자 구분 없이 전체 전사문을 직접 요약합니다."""
    if not session_text.strip():
        raise ValueError("요약할 세션 전사문이 없습니다")

    summary_sentences = summary_sentences or DEFAULT_SUMMARY_SENTENCES
    transcript_text = _truncate_text(session_text, MAX_SESSION_CHARS)
    keywords_text = _format_keywords(keywords or [], MAX_KEYWORDS)

    if MOCK_MODE:
        logger.info("[SUMMARY] MOCK_MODE: 전체 전사 기반 세션 요약 반환")
        return _mock_summary(transcript_text, summary_sentences)

    user_prompt = SUMMARY_SESSION_TEXT_PROMPT_TEMPLATE.format(
        keywords=keywords_text,
        transcript_text=transcript_text,
        summary_sentences=summary_sentences,
    )

    return await _call_llm(_build_messages(user_prompt))


async def generate_course_summary(
    keywords: list[dict],
    session_summaries: list[dict],
    speaker_summaries: list[dict],
    summary_sentences: int | None = None,
) -> str:
    """키워드/세션 요약/화자 요약을 참고해 코스 요약을 생성합니다."""
    if not keywords and not session_summaries and not speaker_summaries:
        raise ValueError("코스 요약에 사용할 데이터가 없습니다")

    summary_sentences = summary_sentences or DEFAULT_SUMMARY_SENTENCES

    keywords_text = _format_keywords(keywords, MAX_KEYWORDS)
    session_text = _format_session_summaries(session_summaries, MAX_SESSION_SUMMARIES)
    speaker_text = _format_speaker_summaries(speaker_summaries, MAX_SPEAKER_SUMMARIES)

    payload_text = f"{keywords_text}\n\n{session_text}\n\n{speaker_text}"
    payload_text = _truncate_text(payload_text, MAX_COURSE_CHARS)

    if MOCK_MODE:
        logger.info("[SUMMARY] MOCK_MODE: 코스 요약 반환")
        return _mock_summary(payload_text, summary_sentences)

    user_prompt = SUMMARY_COURSE_PROMPT_TEMPLATE.format(
        keywords=keywords_text,
        session_summaries=session_text,
        speaker_summaries=speaker_text,
        summary_sentences=summary_sentences,
    )

    return await _call_llm(_build_messages(user_prompt))
