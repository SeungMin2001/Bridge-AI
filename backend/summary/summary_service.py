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
DEMO_PIPELINE_LOG = os.getenv("SUMMARY_DEMO_PIPELINE_LOG", "0").strip().lower() in {"1", "true", "yes", "on"}
_kiwi = Kiwi()

# ── 설정 ──
MOCK_MODE = os.getenv("SUMMARY_MOCK_MODE", "false").lower() == "true"
DEFAULT_LLM_URL = "http://localhost:8001"
DEFAULT_LLM_MODEL = "bridgeprag-qwen25-3b-kv64"

LLM_URL = os.getenv("SUMMARY_LLM_URL", os.getenv("LLM_URL", DEFAULT_LLM_URL))
LLM_MODEL = os.getenv("SUMMARY_LLM_MODEL", os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL))
LLM_API_KEY = os.getenv("SUMMARY_LLM_API_KEY", os.getenv("LLM_API_KEY", "test-key"))

DEFAULT_SUMMARY_SENTENCES = int(os.getenv("SUMMARY_SENTENCES", 6))
MAX_SPEAKER_CHARS = int(os.getenv("SUMMARY_SPEAKER_MAX_CHARS", 8000))
MAX_SESSION_CHARS = int(os.getenv("SUMMARY_SESSION_MAX_CHARS", 9000))
MAX_COURSE_CHARS = int(os.getenv("SUMMARY_COURSE_MAX_CHARS", 7000))
MAX_KEYWORDS = int(os.getenv("SUMMARY_MAX_KEYWORDS", 30))
MAX_SPEAKER_SUMMARIES = int(os.getenv("SUMMARY_MAX_SPEAKER_SUMMARIES", 10))
MAX_SESSION_SUMMARIES = int(os.getenv("SUMMARY_MAX_SESSION_SUMMARIES", 10))

SUMMARY_SYSTEM_PROMPT = (
    "당신은 강의 내용을 보고서형 학습 자료로 정리하는 AI입니다. "
    "반드시 원문에 나온 구체적 개념, 정의, 예시, 비교, 절차를 직접 서술하세요. "
    "예: '뉴런은 입력 신호에 가중치를 곱한 뒤 활성화 함수를 통해 출력을 생성한다.' "
    "원문에 없는 사실은 절대 만들지 마세요. "
    "메타 설명(예: '정리했습니다', '반영했습니다', '작성했습니다')은 출력하지 마세요. "
    "반드시 아래 JSON 형식으로만 응답하세요."
)

SUMMARY_SPEAKER_PROMPT_TEMPLATE = """아래는 화자 {speaker_id}의 전사문입니다:

{transcript_text}

[반드시 반영할 핵심 원문 문장]
{evidence_sentences}

[핵심 개념어]
{concept_keywords}

위 내용을 바탕으로 한국어 Markdown 보고서형 요약을 작성하세요.
- 최소 {summary_sentences}개 이상의 핵심 항목을 포함
- 전체 흐름, 주요 개념, 세부 설명, 학습 포인트를 나누어 정리
- [반드시 반영할 핵심 원문 문장]의 구체 내용(개념 이름, 정의, 관계, 절차 등)을 직접 서술할 것
- 각 항목은 원문의 핵심 내용을 요약하여 서술형으로 작성 (예: "역전파는 손실 함수의 기울기를 출력층에서 입력층으로 전파하여 가중치를 갱신하는 알고리즘이다.")
- 「정리했습니다」「반영했습니다」「작성했습니다」같은 메타 설명은 출력하지 말 것

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

[반드시 반영할 핵심 원문 문장]
{evidence_sentences}

[핵심 개념어]
{concept_keywords}

위 내용을 바탕으로 한국어 Markdown 보고서형 요약을 작성하세요.
- 최소 {summary_sentences}개 이상의 핵심 항목을 포함
- 전체 흐름, 주요 개념, 세부 설명, 학습 포인트를 나누어 정리
- [반드시 반영할 핵심 원문 문장]의 구체 내용(개념 이름, 정의, 관계, 절차 등)을 직접 서술할 것
- 각 항목은 원문의 핵심 내용을 요약하여 서술형으로 작성
- 「정리했습니다」「반영했습니다」「작성했습니다」같은 메타 설명은 출력하지 말 것

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

[반드시 반영할 핵심 원문 문장]
{evidence_sentences}

[핵심 개념어]
{concept_keywords}

위 내용을 바탕으로 한국어 Markdown 보고서형 요약을 작성하세요.
- 최소 {summary_sentences}개 이상의 핵심 항목을 포함
- 전체 흐름, 주요 개념, 세부 설명, 학습 포인트를 나누어 정리
- [반드시 반영할 핵심 원문 문장]의 구체 내용(개념 이름, 정의, 관계, 절차 등)을 직접 서술할 것
- 각 항목은 원문의 핵심 내용을 요약하여 서술형으로 작성 (예: "역전파는 손실 함수의 기울기를 출력층에서 입력층으로 전파하여 가중치를 갱신하는 알고리즘이다.")
- 「정리했습니다」「반영했습니다」「작성했습니다」같은 메타 설명은 출력하지 말 것

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

[반드시 반영할 핵심 원문 문장]
{evidence_sentences}

[핵심 개념어]
{concept_keywords}

위 내용을 바탕으로 한국어 Markdown 보고서형 요약을 작성하세요.
- 최소 {summary_sentences}개 이상의 핵심 항목을 포함
- 전체 흐름, 주요 개념, 세부 설명, 학습 포인트를 나누어 정리
- 여러 세션을 연결해 과목 단위의 학습 보고서처럼 작성
- [반드시 반영할 핵심 원문 문장]의 구체 내용(개념 이름, 정의, 관계, 절차 등)을 직접 서술할 것
- 「정리했습니다」「반영했습니다」「작성했습니다」같은 메타 설명은 출력하지 말 것

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


def _extract_concept_keywords(text: str, *, limit: int = 18) -> list[str]:
    """원문에서 요약에 반드시 남겨야 할 개념어를 빈도 기반으로 추출합니다."""
    counter: Counter[str] = Counter()
    for sentence in _split_sentences(text):
        for token in _tokenize_for_demo(sentence):
            if len(token) < 2:
                continue
            counter[token] += 1
    return [word for word, _count in counter.most_common(limit)]


def _format_concept_keywords(text: str, *, limit: int = 18) -> str:
    """핵심 개념어를 프롬프트용 문자열로 변환합니다."""
    keywords = _extract_concept_keywords(text, limit=limit)
    return ", ".join(keywords) if keywords else "(없음)"


def _build_evidence_sentences(text: str, *, limit: int = 12) -> list[str]:
    """TextRank 유사 점수로 원문 핵심 문장을 고른 뒤 원래 흐름 순서로 정렬합니다."""
    sentences = _split_sentences(text)
    ranked = _rank_key_sentences_for_demo(sentences, limit=limit)
    if not ranked:
        return sentences[:limit]
    ordered = sorted(ranked, key=lambda item: item.get("order", 0))
    return [item["text"] for item in ordered if item.get("text")]


def _format_evidence_sentences(text: str, *, limit: int = 12) -> str:
    """핵심 원문 문장을 프롬프트용 번호 목록으로 변환합니다."""
    evidence = _build_evidence_sentences(text, limit=limit)
    if not evidence:
        return "(없음)"
    return "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(evidence, start=1))


def _summary_lacks_source_detail(summary_text: str, source_text: str) -> bool:
    """요약이 형식 설명에 머무르거나 원문 핵심어를 충분히 반영하지 못했는지 판별합니다."""
    text = str(summary_text or "").strip()
    if len(text) < 220:
        return True

    generic_markers = (
        "보고서는 전체 흐름",
        "학습 포인트를 명확하게 정리",
        "새로운 사실은 추가하지 않았",
        "중요한 개념과 흐름을 구조화",
        "위 핵심 개념과 세부 내용",
        "전사문/PDF에 실제로 나온",
        "형식만 설명하지 말고",
        "학습자가 바로 복습할 수 있는",
        "구체적인 내용을 작성",
        "원문에 없는 사실은",
        "전체 흐름과 주요 개념, 세부 설명, 학습 포인트를 나누어 정리",
        "반드시 반영할 핵심 원문 문장",
        "연결해 설명할 수 있는지 확인",
    )
    generic_hits = sum(1 for marker in generic_markers if marker in text)
    if generic_hits >= 2:
        return True

    concepts = _extract_concept_keywords(source_text, limit=10)
    if not concepts:
        return False
    covered = sum(1 for concept in concepts if concept in text)
    return covered < max(3, min(5, len(concepts) // 2))


def _build_extractive_report_summary(source_text: str, *, max_items: int = 12) -> str:
    """LLM 출력이 빈약할 때 원문 핵심 문장 기반으로 풍부한 보고서형 요약을 구성합니다."""
    evidence = _build_evidence_sentences(source_text, limit=max_items)
    if not evidence:
        return str(source_text or "").strip()

    overview = evidence[:3]
    details = evidence[3:10] or evidence[: min(len(evidence), 7)]
    points = evidence[10:12] or evidence[:3]
    concepts = _extract_concept_keywords(source_text, limit=8)

    lines = ["## 핵심 요약"]
    for sentence in overview:
        lines.append(f"- {sentence}")

    lines.extend(["", "## 주요 내용"])
    for index, sentence in enumerate(details, start=1):
        lines.append(f"{index}. {sentence}")

    if concepts:
        lines.extend(["", "## 핵심 개념"])
        lines.append("- " + ", ".join(concepts))

    lines.extend(["", "## 학습 포인트"])
    for sentence in points:
        lines.append(f"- {sentence}")

    lines.extend(["", "## 복습 체크"])
    lines.append("- 위 개념의 정의, 역할, 서로 연결되는 흐름을 원문 문장에 근거해 다시 설명할 수 있는지 확인하세요.")
    return "\n".join(lines).strip()


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
        if source_text and _summary_lacks_source_detail(text, source_text):
            _demo_log("요약 품질 보정: LLM 출력의 원문 반영 부족으로 핵심문장 기반 보고서 재구성")
            return _build_extractive_report_summary(source_text)
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


async def _call_llm(messages: list[dict], max_tokens: int = 1800, source_text: str = "") -> str:
    """요약용 LLM 호출을 수행합니다. JSON 응답을 명시적으로 요청해 채팅용 문장/글자 제한을 우회합니다."""
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
                "response_format": {"type": "json_object"},
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

    summary_sentences = max(summary_sentences or DEFAULT_SUMMARY_SENTENCES, DEFAULT_SUMMARY_SENTENCES)
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
        evidence_sentences=_format_evidence_sentences(transcript_text),
        concept_keywords=_format_concept_keywords(transcript_text),
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

    summary_sentences = max(summary_sentences or DEFAULT_SUMMARY_SENTENCES, DEFAULT_SUMMARY_SENTENCES)

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
        evidence_sentences=_format_evidence_sentences(payload_text),
        concept_keywords=_format_concept_keywords(payload_text),
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

    summary_sentences = max(summary_sentences or DEFAULT_SUMMARY_SENTENCES, DEFAULT_SUMMARY_SENTENCES)
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
        evidence_sentences=_format_evidence_sentences(transcript_text),
        concept_keywords=_format_concept_keywords(transcript_text),
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

    summary_sentences = max(summary_sentences or DEFAULT_SUMMARY_SENTENCES, DEFAULT_SUMMARY_SENTENCES)

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
        evidence_sentences=_format_evidence_sentences(payload_text),
        concept_keywords=_format_concept_keywords(payload_text),
        summary_sentences=summary_sentences,
    )

    return await _call_llm(_build_messages(user_prompt), source_text=payload_text)
