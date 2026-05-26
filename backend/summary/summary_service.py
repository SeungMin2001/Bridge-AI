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
DEFAULT_LLM_URL = "http://localhost:8001"
DEFAULT_LLM_MODEL = "bridgeprag-qwen25-3b-kv64"

LLM_URL = os.getenv("SUMMARY_LLM_URL", os.getenv("LLM_URL", DEFAULT_LLM_URL))
LLM_MODEL = os.getenv("SUMMARY_LLM_MODEL", os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL))
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

위 내용을 바탕으로 핵심을 요약하여 가독성 좋은 한국어로 {summary_sentences}문장 내외로 작성하세요.

[요약 가이드라인]
1. 분량의 유연성 (중요):
   - 입력된 전사문의 정보량이 풍부하고 길다면, 절대 단순 1줄로 끝내지 마세요. 핵심 주제와 주요 세부 논점을 골고루 포함하여 최소 3줄(문장) 이상의 충분하고 풍성한 내용으로 작성해야 합니다.
   - 반대로 전사문이 한두 마디 정도로 아주 짧거나 중요 정보가 없다면, 불필요하게 말을 지어내어 늘리지 말고 핵심만 1줄 이내로 매우 간결하게 요약하세요.
2. 서식 및 가독성 극대화:
   - 주요 포인트가 여럿이거나 의제가 나뉘는 경우, 글머리 기호('-') 등을 사용하여 일목요연하고 깔끔한 구조로 가독성 좋게 정리하세요.
3. 객관성 유지:
   - 중복되거나 무의미한 표현은 배제하고, 원문에 없는 허구의 사실이나 임의의 추측을 덧붙이지 마세요.

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

위 내용을 종합하여 전체 강의/세션의 흐름과 핵심 결론을 가독성 높은 한국어로 {summary_sentences}문장 내외로 작성하세요.

[요약 가이드라인]
1. 세션의 핵심 및 맥락 구조화 (최소 3줄 이상):
   - 이 세션은 전사된 대화 전체를 대표하는 요약이므로, 정보가 길고 의제가 다양하다면 강의/세션의 핵심 주제, 논의의 흐름, 그리고 최종 결론이나 행동 지침을 모두 담아 최소 2~3줄(문장) 이상의 완성도 높은 분량으로 도출하세요.
   - 단, 전체 정보가 극도로 짧고 논의 내용이 없다면 억지로 부풀리지 않고 핵심 결론 1줄로 정제하여 작성합니다.
2. 가독성을 높이는 서식 구성:
   - 내용이 논점별로 나뉠 경우, 글머리 기호('-')와 중요 단어 강조 등을 통해 한눈에 보기 편한 직관적인 서식으로 가독성 좋게 표현하세요.
3. 중복되는 정보는 합쳐서 간결화하고, 원문 사실만을 객관적으로 정리하세요.

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

위 내용을 바탕으로 전체 맥락과 핵심을 정밀하게 분석하여 가독성 좋은 한국어로 {summary_sentences}문장 내외의 요약을 작성하세요.

[요약 가이드라인]
1. 내용의 상세성과 깊이 확보 (최소 3줄 이상):
   - 긴 전사문 전체를 반영하는 요약인 만큼, 전사문 정보량이 많다면 1줄 수준의 지나치게 짧은 요약을 피하고 주요 의제와 결과를 포함하여 최소 2~3줄(문장) 이상으로 깊이 있게 상세히 작성하세요.
   - 전사문 자체가 한두 마디 수준으로 매우 짧을 경우에는 억지로 내용을 늘릴 필요 없이 핵심 결론만 1줄로 간결하게 끝내세요.
2. 서식 및 구조화:
   - 주요 쟁점이나 구조가 뚜렷한 경우, 글머리 기호('-')와 깔끔한 서식을 사용해 가독성을 크게 개선하세요.
3. 중복이나 겉도는 표현은 지양하고 철저히 본문의 사실관계에만 기반하여 요약하세요.

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

위 내용을 바탕으로 전체 과목 강의들의 대맥을 짚어 가독성 높은 한국어로 {summary_sentences}문장 내외의 종합 요약을 작성하세요.

[요약 가이드라인]
1. 종합적 깊이 및 탄력적 요약 (최소 2~3줄 이상):
   - 코스 전체를 아우르는 요약이므로, 여러 세션의 정보를 종합하여 핵심 교육 목표, 주요 학습 내용, 최종적 흐름을 녹여내 최소 2~3줄(문장) 이상으로 탄탄하게 도출하세요.
   - 다만 입력 자료가 매우 부실하거나 분량이 극도로 짧은 경우에는 억지로 늘릴 필요 없이 간결한 1줄로 핵심을 정리합니다.
2. 시각적 가독성 제공:
   - 명확한 개조식 서식('-')이나 구조화된 단락을 사용하여 사용자가 강의의 전반적 흐름을 한눈에 쉽게 훑어볼 수 있도록 가독성 좋은 포맷을 사용하세요.
3. 원문 기반의 엄격한 사실 준수와 중복 표현 정리를 수행하세요.

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


async def _call_llm(messages: list[dict], max_tokens: int = 512) -> str:
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
