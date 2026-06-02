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
from collections import Counter

import httpx
from kiwipiepy import Kiwi

logger = logging.getLogger(__name__)
DEMO_PIPELINE_LOG = True
_kiwi = Kiwi()

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
    "당신은 강의 내용을 보고서형 학습 자료로 정리하는 AI입니다. "
    "원문에 없는 사실은 추가하지 말고, 중요한 개념과 흐름을 구조화해 충분히 설명하세요. "
    "가능하면 아래 JSON 형식으로만 응답하세요."
)

SUMMARY_SPEAKER_PROMPT_TEMPLATE = """아래는 화자 {speaker_id}의 전사문입니다:

{transcript_text}

위 내용을 바탕으로 한국어 Markdown 보고서형 요약을 작성하세요.
- 최소 {summary_sentences}개 이상의 핵심 항목을 포함
- 전체 흐름, 주요 개념, 세부 설명, 학습 포인트를 나누어 정리
- 짧은 메모가 아니라 발표/복습에 바로 사용할 수 있는 보고서처럼 작성
- 중복 표현은 줄이되 중요한 내용은 충분히 설명
- 새로운 사실을 추가하지 말 것

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "summary_text": "## 핵심 요약\\n...\\n## 주요 내용\\n...\\n## 학습 포인트\\n..."
}}
"""

SUMMARY_SESSION_PROMPT_TEMPLATE = """아래는 세션 요약을 위한 정보입니다:

[핵심 키워드]
{keywords}

[화자별 요약]
{speaker_summaries}

위 내용을 바탕으로 한국어 Markdown 보고서형 요약을 작성하세요.
- 최소 {summary_sentences}개 이상의 핵심 항목을 포함
- 전체 흐름, 주요 개념, 세부 설명, 학습 포인트를 나누어 정리
- 짧은 메모가 아니라 발표/복습에 바로 사용할 수 있는 보고서처럼 작성
- 중복 표현은 줄이되 중요한 내용은 충분히 설명
- 새로운 사실을 추가하지 말 것

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "summary_text": "## 핵심 요약\\n...\\n## 주요 내용\\n...\\n## 학습 포인트\\n..."
}}
"""

SUMMARY_SESSION_TEXT_PROMPT_TEMPLATE = """아래는 하나의 녹음 세션 전체 전사문입니다:

[핵심 키워드]
{keywords}

[전체 전사문]
{transcript_text}

위 내용을 바탕으로 한국어 Markdown 보고서형 요약을 작성하세요.
- 최소 {summary_sentences}개 이상의 핵심 항목을 포함
- 전체 흐름, 주요 개념, 세부 설명, 학습 포인트를 나누어 정리
- 짧은 메모가 아니라 발표/복습에 바로 사용할 수 있는 보고서처럼 작성
- 중복 표현은 줄이되 중요한 내용은 충분히 설명
- 새로운 사실을 추가하지 말 것

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "summary_text": "## 핵심 요약\\n...\\n## 주요 내용\\n...\\n## 학습 포인트\\n..."
}}
"""

SUMMARY_COURSE_PROMPT_TEMPLATE = """아래는 과목 요약을 위한 정보입니다:

[핵심 키워드]
{keywords}

[세션 요약]
{session_summaries}

[화자별 요약]
{speaker_summaries}

위 내용을 바탕으로 한국어 Markdown 보고서형 요약을 작성하세요.
- 최소 {summary_sentences}개 이상의 핵심 항목을 포함
- 전체 흐름, 주요 개념, 세부 설명, 학습 포인트를 나누어 정리
- 여러 세션을 연결해 과목 단위의 학습 보고서처럼 작성
- 중복 표현은 줄이되 중요한 내용은 충분히 설명
- 새로운 사실을 추가하지 말 것

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "summary_text": "## 핵심 요약\\n...\\n## 주요 내용\\n...\\n## 학습 포인트\\n..."
}}
"""

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!。？！])\s+|\n+")
_MORPHEME_LOG_TAGS = {"NNG", "NNP", "VV", "VA", "SL"}
SUMMARY_REQUIRED_SECTIONS = ("## 핵심 요약", "## 주요 내용", "## 학습 포인트")


def _demo_log(message: str) -> None:
    if DEMO_PIPELINE_LOG:
        print(f"[DEMO:SUMMARY] {message}", flush=True)


def _preview(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    return compact if len(compact) <= limit else f"{compact[:limit - 3]}..."


def _split_sentences(text: str) -> list[str]:
    """문장 단위 분리를 통해 간단한 요약/미리보기 용도를 지원합니다."""
    text = re.sub(r"\r\n?", "\n", text)
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part and part.strip()]


def _tokenize_for_demo(sentence: str) -> list[str]:
    """시연 로그용 형태소 토큰을 추출합니다."""
    return [
        token.form
        for token in _kiwi.tokenize(sentence)
        if token.tag in _MORPHEME_LOG_TAGS and len(token.form) >= 2
    ][:12]


def _log_sentence_morphemes(sentences: list[str], *, label: str) -> None:
    """문장 분리 후 형태소 분석 산출물을 로그로 남깁니다."""
    if not sentences:
        _demo_log(f"{label} 형태소 분석 생략: 문장 없음")
        return
    _demo_log(f"{label} 형태소 분석 시작: sentences={len(sentences)}, sample_sentences={min(len(sentences), 5)}")
    for index, sentence in enumerate(sentences[:5], start=1):
        preprocessed = re.sub(r"\s+", " ", sentence).strip()
        tokens = _tokenize_for_demo(preprocessed)
        _demo_log(f"   전처리문장#{index}: '{_preview(preprocessed, 100)}'")
        _demo_log(
            f"   형태소분리#{index}: tokens={tokens}"
        )
    _demo_log(f"{label} 형태소 분석 종료")


def _rank_key_sentences_for_demo(sentences: list[str], *, limit: int = 5) -> list[dict]:
    """시연 로그용 핵심 문장 후보를 산출합니다. 실제 요약 입력/출력에는 사용하지 않습니다."""
    candidates = []
    token_counts: Counter[str] = Counter()
    for order, sentence in enumerate(sentences, start=1):
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if len(cleaned) < 8:
            continue
        tokens = list(dict.fromkeys(_tokenize_for_demo(cleaned)))
        if not tokens:
            continue
        token_counts.update(tokens)
        candidates.append({"order": order, "text": cleaned, "tokens": tokens})

    if not candidates:
        return []

    ranked = []
    for item in candidates:
        tokens = item["tokens"]
        score = sum(token_counts[token] for token in tokens) / max(len(tokens), 1)
        ranked.append({**item, "score": float(score)})

    ranked.sort(key=lambda item: (-item["score"], item["order"]))
    return ranked[:limit]


def _log_key_sentences(sentences: list[str], *, label: str) -> None:
    """문장 분리 결과에서 핵심 문장 후보를 로그로 남깁니다."""
    ranked = _rank_key_sentences_for_demo(sentences)
    if not ranked:
        _demo_log(f"{label} 핵심문장 선택 생략: 후보 없음")
        return
    _demo_log(f"{label} 핵심문장 선택 완료: selected={len(ranked)}")
    for index, item in enumerate(ranked, start=1):
        _demo_log(
            f"   핵심문장#{index}: score={item.get('score', 0):.4f}, "
            f"text='{_preview(item.get('text'), 110)}'"
        )


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
        logger.warning("[SUMMARY] JSON 파싱 실패, 원문 요약 텍스트로 복구: %s", exc)
        if text:
            match = re.search(r'"summary_text"\s*:\s*"(.*)"\s*\}?\s*$', text, re.DOTALL)
            recovered = match.group(1) if match else text
            recovered = recovered.replace("\\n", "\n").replace('\\"', '"').strip()
            recovered = re.sub(r'"\s*\}\s*$', "", recovered).strip()
            return recovered
        raise ValueError(f"LLM 응답을 JSON으로 파싱할 수 없습니다: {exc}")

    summary_text = payload.get("summary_text")
    if not summary_text or not isinstance(summary_text, str):
        raise ValueError("LLM 응답에 summary_text가 없습니다")

    return summary_text.strip()


def _ensure_structured_summary(summary_text: str, source_text: str = "") -> str:
    """LLM/복구/mock 결과가 한 문단으로만 나오지 않도록 보고서형 Markdown으로 보정합니다."""
    text = str(summary_text or "").strip()
    if text and sum(section in text for section in SUMMARY_REQUIRED_SECTIONS) >= 2:
        return text

    sentences = _split_sentences(text) or _split_sentences(source_text)
    if not sentences:
        return text

    overview = sentences[:2]
    details = sentences[2:8] or sentences[: min(len(sentences), 5)]
    points = sentences[8:12] or details[:3]

    lines = ["## 핵심 요약"]
    for sentence in overview:
        lines.append(f"- {sentence}")

    lines.extend(["", "## 주요 내용"])
    for sentence in details:
        lines.append(f"- {sentence}")

    lines.extend(["", "## 학습 포인트"])
    for sentence in points:
        lines.append(f"- {sentence}")

    lines.extend(["", "## 복습 체크"])
    lines.append("- 위 핵심 개념과 세부 내용을 연결해 설명할 수 있는지 확인하세요.")
    return "\n".join(lines).strip()


def _mock_summary(text: str, summary_sentences: int) -> str:
    """Mock 모드에서 문장 앞부분만 사용해 요약을 흉내냅니다."""
    sentences = _split_sentences(text)
    if not sentences:
        return text.strip()
    return _ensure_structured_summary(" ".join(sentences[:summary_sentences]).strip(), text)


def _build_messages(user_prompt: str) -> list[dict]:
    """OpenAI 호환 LLM messages 형식을 구성합니다."""
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


async def _call_llm(messages: list[dict], max_tokens: int = 1200, source_text: str = "") -> str:
    """LLM 호출 후 요약 문자열을 반환합니다."""
    prompt_chars = sum(len(str(item.get("content") or "")) for item in messages)
    _demo_log(f"모델 전달: messages={len(messages)}, prompt_chars={prompt_chars}, max_tokens={max_tokens}")
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=120.0)) as client:
        res = await client.post(
            f"{LLM_URL}/v1/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "bridgeprag_alpha": 0.0,
                "bridgeprag_disable_memory": True,
                "demo_feature": "summary",
                "chat_template_kwargs": {"enable_thinking": False},
            },
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        )
        res.raise_for_status()

    data = res.json()
    raw_answer = data["choices"][0]["message"]["content"]
    logger.info("[SUMMARY] LLM 응답 수신: %d chars", len(raw_answer))
    summary_text = _parse_summary_json(raw_answer)
    summary_text = _ensure_structured_summary(summary_text, source_text)
    _demo_log(f"요약 생성 완료: output_chars={len(summary_text)}")
    return summary_text


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
    sentences = _split_sentences(transcript_text)
    _demo_log(f"화자 요약 시작: speaker={speaker_id}, input_chars={len(speaker_text)}, sentences={len(sentences)}")
    for index, sentence in enumerate(sentences[:5], start=1):
        _demo_log(f"   문장분리#{index}: {_preview(sentence, 100)}")
    _log_sentence_morphemes(sentences, label="화자 요약")
    _log_key_sentences(sentences, label="화자 요약")

    if MOCK_MODE:
        logger.info("[SUMMARY] MOCK_MODE: 화자 요약 반환")
        return _mock_summary(transcript_text, summary_sentences)

    user_prompt = SUMMARY_SPEAKER_PROMPT_TEMPLATE.format(
        speaker_id=speaker_id,
        transcript_text=transcript_text,
        summary_sentences=summary_sentences,
    )

    return await _call_llm(_build_messages(user_prompt), source_text=transcript_text)


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
    sentences = _split_sentences(payload_text)
    _demo_log(
        f"세션 요약 입력 분석: input_chars={len(payload_text)}, "
        f"sentences={len(sentences)}, keywords={len(keywords)}"
    )
    _log_sentence_morphemes(sentences, label="세션 요약")
    _log_key_sentences(sentences, label="세션 요약")

    if MOCK_MODE:
        logger.info("[SUMMARY] MOCK_MODE: 세션 요약 반환")
        return _mock_summary(payload_text, summary_sentences)

    user_prompt = SUMMARY_SESSION_PROMPT_TEMPLATE.format(
        keywords=keywords_text,
        speaker_summaries=speaker_text,
        summary_sentences=summary_sentences,
    )

    return await _call_llm(_build_messages(user_prompt), source_text=payload_text)


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
    sentences = _split_sentences(transcript_text)
    _demo_log(
        f"전사 요약 시작: input_chars={len(session_text)}, "
        f"used_chars={len(transcript_text)}, sentences={len(sentences)}, keywords={len(keywords or [])}"
    )
    for index, sentence in enumerate(sentences[:5], start=1):
        _demo_log(f"   문장분리#{index}: {_preview(sentence, 100)}")
    _log_sentence_morphemes(sentences, label="전사 요약")
    _log_key_sentences(sentences, label="전사 요약")
    if keywords:
        _demo_log(f"   핵심 키워드: {[kw.get('keyword_text') for kw in keywords[:10]]}")

    if MOCK_MODE:
        logger.info("[SUMMARY] MOCK_MODE: 전체 전사 기반 세션 요약 반환")
        return _mock_summary(transcript_text, summary_sentences)

    user_prompt = SUMMARY_SESSION_TEXT_PROMPT_TEMPLATE.format(
        keywords=keywords_text,
        transcript_text=transcript_text,
        summary_sentences=summary_sentences,
    )

    return await _call_llm(_build_messages(user_prompt), source_text=transcript_text)


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
    sentences = _split_sentences(payload_text)
    _demo_log(
        f"코스 요약 입력 분석: input_chars={len(payload_text)}, "
        f"sentences={len(sentences)}, keywords={len(keywords)}"
    )
    _log_sentence_morphemes(sentences, label="코스 요약")
    _log_key_sentences(sentences, label="코스 요약")

    if MOCK_MODE:
        logger.info("[SUMMARY] MOCK_MODE: 코스 요약 반환")
        return _mock_summary(payload_text, summary_sentences)

    user_prompt = SUMMARY_COURSE_PROMPT_TEMPLATE.format(
        keywords=keywords_text,
        session_summaries=session_text,
        speaker_summaries=speaker_text,
        summary_sentences=summary_sentences,
    )

    return await _call_llm(_build_messages(user_prompt), source_text=payload_text)
