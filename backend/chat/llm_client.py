"""AI 채팅 LLM 호출 클라이언트.

OpenAI 호환 Chat Completions API와 Ollama native chat API 호출을 감싸고,
비스트리밍/스트리밍 응답에서 사용자에게 보여줄 답변 텍스트만 정리합니다.
"""

import os
import re
import json
import time

import httpx

from chat.context_service import source_filter_has_any_source


DEFAULT_LLM_URL = "http://localhost:8001"
DEFAULT_LLM_MODEL = "bridgeprag-qwen25-3b-kv64"

llm_server_url = os.getenv("LLM_URL", DEFAULT_LLM_URL)
llm_model_name = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
llm_api_key = os.getenv("LLM_API_KEY", "test-key")
# Demo-safe defaults are fixed in code so the presentation path does not depend on shell env vars.
CHAT_MAX_TOKENS = 240
CHAT_SOURCE_MAX_TOKENS = 300
CHAT_ANSWER_MAX_CHARS = 800
CHAT_ANSWER_MAX_SENTENCES = 3
CHAT_STREAM_HOLD_CHARS = 1
CHAT_STREAM_MODE = "fast"
CHAT_GROUNDED_FAST_PATH = True
CHAT_OLLAMA_NATIVE = os.getenv("CHAT_OLLAMA_NATIVE", "auto").strip().lower()
# AI 채팅은 RAG 근거가 있을 때 낮은 강도의 BridgePRAG K/V 주입을 기본 사용합니다.
CHAT_DISABLE_BRIDGEPRAG = False
CHAT_BRIDGEPRAG_REFERENCE_ALPHA = 0.0
CHAT_BRIDGEPRAG_TOPIC_ALPHA = 0.05
CHAT_BRIDGEPRAG_SUMMARY_ALPHA = 0.0
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.1"))
CHAT_LLM_READ_TIMEOUT = float(os.getenv("CHAT_LLM_READ_TIMEOUT", "90.0"))
CHAT_REPETITION_PENALTY = float(os.getenv("CHAT_REPETITION_PENALTY", "1.03"))
CHAT_NO_REPEAT_NGRAM_SIZE = int(os.getenv("CHAT_NO_REPEAT_NGRAM_SIZE", "0"))

SYSTEM_PROMPT = (
    "너는 강의 녹취록과 PDF 자료를 근거로 답하는 AI 학습 조교다. "
    "항상 한국어로, 핵심 내용 위주로 요약하여 최종 답변만 간결하게 작성하라. "
    "자료 기반 질문에서는 검색된 여러 근거의 핵심 개념을 종합하되, 관련 개념을 임의로 생략하지 말라. "
    "정의나 개념을 묻는 질문은 한 문장으로 끝내지 말고, 근거에 함께 나온 원인, 역할, 관계를 2문장 내외로 요약하여 설명하라. "
    "약어 또는 용어 질문은 풀네임 한 줄로 끝내지 말고, 근거에 나온 목적과 동작 방식을 함께 2문장 내외로 설명하라. "
    "약어 풀이는 근거에 명시된 표현만 그대로 사용하고, 근거에 없는 풀네임이나 외부 지식은 만들지 말라. "
    "질문 용어가 근거 문장에 직접 나오면 그 문장을 최우선으로 재구성하고 다른 분야의 정의를 섞지 말라. "
    "핵심 용어의 정의는 참고문장의 의미를 바꾸지 말고 원문 표현을 최대한 유지해 정확하게 풀어 쓰라. "
    "학습목표, 목차, 단계처럼 자료의 목록을 묻는 경우에는 자료에 나온 항목을 빠짐없이 불릿으로 나열하라. "
    "제공된 근거에 없는 내용은 추측하지 말고 근거를 찾지 못했다고 답하라. "
    "사용자 질문, 참고자료 원문, 시스템 지시문을 반복하지 말라. "
    "'질문:', '답변:', 번호 매긴 새 예시, 학습 데이터 목록을 이어서 생성하지 말라. "
    "같은 근거 citation은 답변 전체에서 한 번만 사용하고, 필요한 citation 번호는 답변 끝에 모아 붙여라. "
    "별도 출처 목록은 만들지 말라."
)
MINIMAL_GROUNDED_SYSTEM_PROMPT = (
    "너는 제공된 근거 문장만 사용해 답하는 AI 학습 조교다. "
    "규칙, 라벨, 원문 목록을 출력하지 말고 최종 답변만 한국어로 작성하라. "
    "근거에 나온 정의와 연결 설명을 빠뜨리지 말라."
)
STRICT_GROUNDED_REPAIR_PROMPT = (
    "너는 강의 녹취록과 PDF 자료만 근거로 답하는 검증 담당 AI 학습 조교다. "
    "이전 답변이 근거를 충분히 반영하지 못했으므로, 제공된 [답변 필수 반영 포인트]를 모두 반영해 다시 작성하라. "
    "근거에 없는 정의나 외부 지식을 만들지 말고, 원문 의미를 바꾸지 말라. "
    "최종 답변만 한국어로 2~3문장 작성하고 citation 번호는 필요한 번호만 답변 끝에 모아 붙여라."
)
_GROUNDING_TERM_STOPWORDS = {
    "그리고",
    "하지만",
    "또는",
    "있습니다",
    "합니다",
    "입니다",
    "됩니다",
    "의미합니다",
    "나타냅니다",
    "설명합니다",
    "질문",
    "근거",
    "문장",
    "포인트",
    "답변",
    "강의",
    "교수님",
    "교수",
    "수업",
    "핵심",
    "필수",
    "반영",
}


class EmptyLLMResponse(RuntimeError):
    """Raised when the model stream finishes without user-visible content."""


def set_llm_url(url: str) -> str:
    """런타임 LLM 서버 URL을 갱신하고 정규화된 URL을 반환합니다."""
    global llm_server_url
    llm_server_url = url.rstrip("/")
    print(f"[LLM] URL updated: {llm_server_url}")
    return llm_server_url


def remove_thinking(text: str, prompt: str = "") -> str:
    """모델 응답에 섞인 <think> 블록이나 chat template 잔여 문자열을 제거합니다."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text[:text.index("<think>")]
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1]
    return _clean_visible_answer(text, prompt)


def _clean_visible_answer(text: str, prompt: str = "") -> str:
    """서비스 화면에 보여줄 최종 답변만 남기고 과생성된 예시/질문 반복을 잘라냅니다."""
    text = text.replace("\r\n", "\n").strip()
    text, _ = _truncate_at_stop_pattern(text)
    text = re.sub(r"^\s*(assistant|답변|Answer)\s*[:：]?\s*", "", text, flags=re.IGNORECASE)

    stop_patterns = (
        r"\n\s*(질문|Question|사용자|User)\s*[:：]",
        r"\n\s*\[검색된 참고자료\]",
        r"\n\s*선택된\s+녹음본\s+전체\s+전사",
        r"\n\s*시스템\s*[:：]",
        r"\n\s*System\s*[:：]",
    )
    for pattern in stop_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            text = text[:match.start()].strip()

    is_keyword = "라는 단어의 뜻을" in prompt or "단어의 뜻을 한국어로" in prompt
    max_sentences = 1 if is_keyword else CHAT_ANSWER_MAX_SENTENCES
    max_chars = 200 if is_keyword else CHAT_ANSWER_MAX_CHARS

    text = _strip_boilerplate(text)
    text = _truncate_before_repeated_sentence(text)
    text = _trim_sentences(text, max_sentences)
    if len(text) > max_chars:
        text = _trim_to_char_budget(text, max_chars)
    text = _deduplicate_trailing_citations(text)
    return text.strip()


def _strip_boilerplate(text: str, *, strip_edges: bool = True) -> str:
    """인사/도움말성 상투 문구가 답변 본문을 밀어내지 않도록 제거합니다."""
    boilerplate_patterns = (
        r"\s*질문에\s+대해\s+답변해\s+드리겠습니다\.?",
        r"\s*질문이\s+있으시면\s+언제든지\s+말씀해\s+주세요\.?",
        r"\s*도움이\s+필요하시면\s+언제든지\s+말씀해\s+주세요\.?",
    )
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() if strip_edges else text


def _trim_sentences(text: str, max_sentences: int) -> str:
    """한국어/영어 문장 경계를 기준으로 답변을 너무 길지 않게 제한합니다."""
    if max_sentences <= 0:
        return text.strip()
    matches = list(re.finditer(r"[.!?。？！](?:\s+|$)|다\.(?:\s+|$)|요\.(?:\s+|$)", text))
    if len(matches) <= max_sentences:
        return text.strip()
    end = _extend_end_over_citation(text, matches[max_sentences - 1].end())
    return text[:end].strip()


def _trim_to_char_budget(text: str, max_chars: int) -> str:
    """문자 수 예산 안에서 가능한 마지막 문장 경계까지 자릅니다."""
    if len(text) <= max_chars:
        return text.strip()
    clipped = text[:max_chars].rstrip()
    sentence_ends = [m.end() for m in re.finditer(r"[.!?。？！]|다\.|요\.", clipped)]
    if sentence_ends:
        clipped = clipped[: sentence_ends[-1]]
    return clipped.rstrip(" ,;:：-")


def _stream_chars(text: str):
    """SSE 체감 속도를 높이기 위해 정리된 조각을 글자 단위로 내보냅니다."""
    for char in str(text or ""):
        yield char


def build_chat_messages(prompt: str, system_prompt: str | None = None) -> list[dict]:
    """시스템 프롬프트와 사용자 prompt를 LLM chat messages 형식으로 묶습니다."""
    return [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


async def complete_answer(
    prompt: str,
    source_filter: dict | None,
    *,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
) -> str:
    """비스트리밍 LLM 호출을 수행하고 최종 답변 문자열만 반환합니다."""
    if CHAT_GROUNDED_FAST_PATH and _needs_grounded_answer_validation(prompt):
        fast_answer = await _complete_minimal_grounded_answer(
            prompt,
            source_filter,
            max_tokens=max_tokens,
        )
        if fast_answer:
            return fast_answer

    messages = build_chat_messages(prompt, system_prompt)
    answer = await _complete_messages(messages, source_filter, max_tokens=max_tokens)
    if _needs_grounded_answer_validation(prompt):
        answer = await _repair_grounded_answer_if_needed(
            prompt,
            answer,
            source_filter,
            max_tokens=max_tokens,
        )
    return answer


async def _complete_minimal_grounded_answer(
    prompt: str,
    source_filter: dict | None,
    *,
    max_tokens: int | None = None,
) -> str:
    """긴 RAG prompt 대신 핵심 근거만 먼저 사용해 grounded 답변을 빠르게 생성합니다."""
    minimal_prompt = _build_minimal_grounded_prompt(prompt)
    if not minimal_prompt:
        return ""

    started_at = time.perf_counter()
    answer = await _complete_messages(
        build_chat_messages(minimal_prompt, MINIMAL_GROUNDED_SYSTEM_PROMPT),
        source_filter,
        max_tokens=max_tokens or CHAT_SOURCE_MAX_TOKENS,
    )
    elapsed = time.perf_counter() - started_at
    print(
        f"[LLM:grounded-fast] answer_chars={len(answer or '')} "
        f"valid={_answer_covers_required_points(answer, prompt)} elapsed={elapsed:.3f}s"
    )
    if answer and _answer_covers_required_points(answer, prompt):
        return answer
    return ""


async def _complete_messages(
    messages: list[dict],
    source_filter: dict | None,
    *,
    max_tokens: int | None = None,
) -> str:
    """LLM chat messages를 한 번 호출하고 화면용 답변으로 정리합니다."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=CHAT_LLM_READ_TIMEOUT)) as client:
        if _use_ollama_native_chat():
            res = await client.post(
                _llm_url("/api/chat"),
                json=_ollama_chat_payload(
                    messages,
                    source_filter,
                    stream=False,
                    thinking=False,
                    max_tokens=max_tokens,
                ),
            )
        else:
            res = await client.post(
                _llm_url("/v1/chat/completions"),
                json=_openai_chat_payload(
                    messages,
                    source_filter,
                    stream=False,
                    thinking=False,
                    max_tokens=max_tokens,
                ),
                headers={"Authorization": f"Bearer {llm_api_key}"},
            )
    res.raise_for_status()
    data = res.json()
    if _use_ollama_native_chat():
        raw_answer = (data.get("message") or {}).get("content", "")
    else:
        raw_answer = data["choices"][0]["message"]["content"]
    user_prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "")
            break
    return remove_thinking(raw_answer, user_prompt)


def _needs_grounded_answer_validation(prompt: str) -> bool:
    """RAG 필수 포인트가 있는 자료 기반 답변은 생성 후 근거 반영 여부를 확인합니다."""
    return "[답변 필수 반영 포인트]" in str(prompt or "")


async def _repair_grounded_answer_if_needed(
    prompt: str,
    answer: str,
    source_filter: dict | None,
    *,
    max_tokens: int | None = None,
) -> str:
    """답변이 필수 근거를 반영하지 못하면 LLM에게 같은 근거로 한 번 더 재작성시킵니다."""
    if _answer_covers_required_points(answer, prompt):
        return answer

    repair_prompt = (
        "이전 답변은 검색 근거의 핵심 내용을 충분히 반영하지 못했습니다. "
        "아래 원래 근거와 질문만 사용해서 다시 답하세요. "
        "[답변 필수 반영 포인트]의 문장을 의미를 바꾸지 않고 모두 반영하세요.\n\n"
        f"{prompt}\n\n"
        f"이전 답변(무시하고 재작성):\n{answer}\n\n"
        "수정 답변:"
    )
    repair_messages = build_chat_messages(repair_prompt, STRICT_GROUNDED_REPAIR_PROMPT)
    repaired = await _complete_messages(
        repair_messages,
        source_filter,
        max_tokens=max_tokens or CHAT_SOURCE_MAX_TOKENS,
    )
    if repaired and _answer_covers_required_points(repaired, prompt):
        return repaired

    if repaired and not _answer_leaks_prompt_or_system(repaired):
        return repaired
    return answer


def _answer_covers_required_points(answer: str, prompt: str) -> bool:
    """생성 답변이 RAG 필수 포인트의 핵심 내용을 실제로 반영했는지 가볍게 검증합니다."""
    clean_answer = _compact_for_grounding(answer)
    if len(clean_answer) < 12:
        return False
    if _answer_leaks_prompt_or_system(answer):
        return False

    required_points = _extract_required_points(prompt)
    if not required_points:
        return True
    if len(required_points) >= 2 and _visible_sentence_count(answer) < 2:
        return False

    # 여러 citation이 있을 때는 상위 근거 중심으로 검증한다.
    # 모든 보조 근거를 강제하면 불필요한 재작성 호출이 늘어 시연 응답성이 떨어진다.
    checked_points = required_points[:3]
    covered = 0
    for point in checked_points:
        terms = _important_terms_from_point(point)
        if not terms:
            covered += 1
            continue
        hit_count = sum(1 for term in terms if _compact_for_grounding(term) in clean_answer)
        required_hits = 1 if len(terms) <= 2 else min(3, max(2, len(terms) // 3))
        if hit_count >= required_hits:
            covered += 1

    required_covered = len(checked_points) if len(checked_points) <= 2 else 2
    return covered >= required_covered


def _answer_leaks_prompt_or_system(answer: str) -> bool:
    """시스템/프롬프트 규칙이 답변 본문으로 새어 나온 경우를 탐지합니다."""
    value = str(answer or "")
    leak_markers = (
        "[답변 필수 반영 포인트]",
        "[검색된 참고자료]",
        "시스템 지시문",
        "참고자료 원문",
        "사용자 질문",
        "같은 근거 citation",
        "citation 번호",
        "별도 출처 목록",
        "학습목표, 목차, 단계",
        "자료에 나온 항목",
        "2~3문장",
        "2~4문장",
        "2~5문장",
        "원인, 역할, 관계",
        "근거에 함께 나온",
        "답변은 한 문장",
        "제공된 근거",
        "검색된 여러 근거",
        "핵심 참고문장",
        "보조 문맥",
        "규칙, 라벨",
        "최종 답변만",
        "자료의 목록",
        "목록을 묻는",
        "빠짐없이",
        "추측하지",
        "번호를 매긴 새 예시",
        "질문:",
        "답변:",
        "최종 답변:",
    )
    return any(marker in value for marker in leak_markers)


def _visible_sentence_count(text: str) -> int:
    """사용자에게 보이는 답변이 근거 여러 문장을 설명할 만큼 충분한지 확인합니다."""
    value = re.sub(r"\[[0-9,\s]+\]", "", str(text or "")).strip()
    if not value:
        return 0
    matches = re.findall(r"[.!?。？！]|다\.|요\.", value)
    if matches:
        return len(matches)
    return 1


def _extract_required_points(prompt: str) -> list[str]:
    """context_service가 prompt에 넣은 '- [1] 근거문장' 목록만 추출합니다."""
    value = str(prompt or "")
    marker = "[답변 필수 반영 포인트]"
    if marker not in value:
        return []
    section = value.split(marker, 1)[1]
    section = section.split("\n\n", 1)[0]
    points: list[str] = []
    for line in section.splitlines():
        match = re.match(r"\s*-\s*\[\d+\]\s*(.+)", line)
        if match:
            point = re.sub(r"\s+", " ", match.group(1)).strip()
            if point:
                points.append(point)
    return points


def _build_minimal_grounded_prompt(prompt: str) -> str:
    """LLM이 긴 규칙을 복사할 때 사용할 근거/질문만 남긴 재작성 prompt를 만듭니다."""
    points = _extract_required_points(prompt)
    if not points:
        return ""
    question = _extract_user_question(prompt)
    evidence_lines = "\n".join(f"- {point}" for point in points)
    return (
        "근거 문장:\n"
        f"{evidence_lines}\n\n"
        f"질문: {question}\n\n"
        "위 근거 문장의 핵심을 빠뜨리지 말고 2~3문장으로 설명하세요. 답변:"
    )


def _extract_user_question(prompt: str) -> str:
    """prompt 마지막 질문만 추출해 재작성 prompt에 사용합니다."""
    matches = list(re.finditer(r"(?:^|\n)\s*질문\s*[:：]\s*(.+)", str(prompt or "")))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def _important_terms_from_point(point: str) -> list[str]:
    """근거 문장에서 답변에 반드시 남아야 할 개념어를 추출합니다."""
    text = str(point or "")
    raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9_+#./-]*|[가-힣]{2,}", text)
    terms: list[str] = []
    seen = set()
    for raw in raw_terms:
        term = re.sub(r"(은|는|이|가|을|를|에|에서|으로|로|도|만|와|과|의|이다|입니다|합니다|됩니다)$", "", raw)
        term = term.strip()
        if len(term) < 2:
            continue
        if term.casefold() in _GROUNDING_TERM_STOPWORDS or term in _GROUNDING_TERM_STOPWORDS:
            continue
        compact = _compact_for_grounding(term)
        if len(compact) < 2 or compact in seen:
            continue
        seen.add(compact)
        terms.append(term)
    return terms[:12]


def _compact_for_grounding(value: str) -> str:
    """띄어쓰기/기호 차이를 무시하고 근거 핵심어 포함 여부를 비교합니다."""
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", str(value or "").casefold())


async def stream_answer(
    prompt: str,
    source_filter: dict | None,
    *,
    thinking: bool = False,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
):
    """스트리밍 LLM 호출을 수행하고 사용자에게 보여줄 토큰만 async iterator로 반환합니다."""
    if CHAT_STREAM_MODE in {"buffered", "complete", "nonstream", "non-stream"}:
        answer = await complete_answer(prompt, source_filter, max_tokens=max_tokens, system_prompt=system_prompt)
        if answer:
            yield answer
            return
        raise EmptyLLMResponse("모델이 표시 가능한 답변을 반환하지 않았습니다. 다시 질문해 주세요.")

    if _needs_grounded_answer_validation(prompt):
        answer = await complete_answer(prompt, source_filter, max_tokens=max_tokens, system_prompt=system_prompt)
        if answer:
            for char in _stream_chars(answer):
                yield char
            return
        raise EmptyLLMResponse("모델이 표시 가능한 답변을 반환하지 않았습니다. 다시 질문해 주세요.")

    messages = build_chat_messages(prompt, system_prompt)
    state = {"saw_thinking_only": False}
    emitted_content = False

    raw_chunks = _raw_stream_answer(
        messages,
        source_filter,
        thinking=thinking,
        state=state,
        max_tokens=max_tokens,
    )
    cleaner = _clean_token_stream_chunks if CHAT_STREAM_MODE in {"token", "raw", "fast"} else _clean_stream_chunks
    async for chunk in cleaner(raw_chunks, prompt=prompt):
        emitted_content = True
        yield chunk

    if not emitted_content:
        if state["saw_thinking_only"]:
            raise EmptyLLMResponse("모델이 thinking 출력만 반환하고 최종 답변을 반환하지 않았습니다. 다시 질문해 주세요.")
        raise EmptyLLMResponse("모델이 표시 가능한 답변을 반환하지 않았습니다. 다시 질문해 주세요.")


async def _raw_stream_answer(
    messages: list[dict],
    source_filter: dict | None,
    *,
    thinking: bool,
    state: dict,
    max_tokens: int | None = None,
):
    """LLM 서버에서 받은 raw 스트림 토큰을 그대로 생성합니다."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=CHAT_LLM_READ_TIMEOUT)) as client:
        if _use_ollama_native_chat():
            async with client.stream(
                "POST",
                _llm_url("/api/chat"),
                json=_ollama_chat_payload(
                    messages,
                    source_filter,
                    stream=True,
                    thinking=thinking,
                    max_tokens=max_tokens,
                ),
            ) as stream:
                await _raise_for_llm_stream_error(stream)
                async for line in stream.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise RuntimeError(str(chunk["error"]))
                    message = chunk.get("message") or {}
                    if message.get("thinking"):
                        state["saw_thinking_only"] = True
                    content = message.get("content") or ""
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
        else:
            async with client.stream(
                "POST",
                _llm_url("/v1/chat/completions"),
                json=_openai_chat_payload(
                    messages,
                    source_filter,
                    stream=True,
                    thinking=thinking,
                    max_tokens=max_tokens,
                ),
                headers={"Authorization": f"Bearer {llm_api_key}"},
            ) as stream:
                await _raise_for_llm_stream_error(stream)
                async for line in stream.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})
                    if delta.get("thinking") or delta.get("reasoning_content"):
                        state["saw_thinking_only"] = True
                    content = delta.get("content") or (choice.get("message") or {}).get("content") or ""
                    if content:
                        yield content


async def _clean_stream_chunks(raw_chunks, prompt: str = ""):
    """raw 토큰을 문장 단위로 정리해, 미완성 문장이 화면에 찍히지 않게 합니다."""
    buffer = ""
    emitted_sentences = 0
    emitted_any = False
    stopped = False

    is_keyword = "라는 단어의 뜻을" in prompt or "단어의 뜻을 한국어로" in prompt
    max_sentences = 1 if is_keyword else CHAT_ANSWER_MAX_SENTENCES

    async for raw in raw_chunks:
        buffer += raw
        buffer, stopped = _truncate_at_stop_pattern(buffer)
        if not emitted_any:
            buffer = _strip_leading_answer_noise(buffer)
        buffer = _strip_boilerplate(buffer, strip_edges=False)

        while emitted_sentences < max_sentences:
            end = _first_sentence_end(buffer)
            if end is None:
                break
            piece = buffer[:end].strip()
            buffer = buffer[end:].lstrip()
            if not piece:
                continue
            emitted_any = True
            emitted_sentences += 1
            yield piece + (" " if emitted_sentences < max_sentences else "")

        if stopped:
            fallback = _clean_visible_answer(buffer, prompt)
            if fallback and emitted_sentences < max_sentences and _ends_like_complete_sentence(fallback):
                yield fallback
            return

        if emitted_sentences >= max_sentences:
            return

    if emitted_any:
        fallback = _clean_visible_answer(buffer, prompt)
        if fallback and emitted_sentences < max_sentences and _ends_like_complete_sentence(fallback):
            yield fallback
        return

    fallback = _clean_visible_answer(buffer, prompt)
    if fallback and _ends_like_complete_sentence(fallback):
        yield fallback


async def _clean_token_stream_chunks(raw_chunks, prompt: str = ""):
    """raw 토큰을 빠르게 흘려보내되, 과생성 라벨이 화면에 찍히기 전 작은 버퍼로 잡아냅니다."""
    buffer = ""
    pending = ""
    visible_text = ""
    emitted_any = False

    is_keyword = "라는 단어의 뜻을" in prompt or "단어의 뜻을 한국어로" in prompt
    max_sentences = 1 if is_keyword else CHAT_ANSWER_MAX_SENTENCES
    max_chars = 200 if is_keyword else CHAT_ANSWER_MAX_CHARS

    async for raw in raw_chunks:
        buffer += raw
        buffer, stopped = _truncate_at_stop_pattern(buffer)
        if not emitted_any:
            buffer = _strip_leading_answer_noise(buffer)
            if _looks_like_partial_answer_label(buffer):
                continue

        if not buffer:
            if stopped:
                return
            continue

        piece = buffer
        buffer = ""
        if not emitted_any:
            piece = _strip_boilerplate(piece, strip_edges=False).lstrip()
        if not piece:
            if stopped:
                return
            continue

        pending += piece
        if not stopped and len(pending) <= CHAT_STREAM_HOLD_CHARS:
            continue

        if stopped:
            piece = pending
            pending = ""
        else:
            safe_length = max(0, len(pending) - CHAT_STREAM_HOLD_CHARS)
            piece = pending[:safe_length]
            pending = pending[safe_length:]

        candidate = visible_text + piece
        candidate = _truncate_before_repeated_sentence(candidate)
        candidate = _trim_sentences(candidate, max_sentences)
        if len(candidate) <= len(visible_text):
            return
        piece = candidate[len(visible_text):]

        remaining = max_chars - len(visible_text)
        if remaining <= 0:
            return
        if len(piece) > remaining:
            piece = _trim_to_char_budget(piece, remaining)
            stopped = True
        if not piece:
            return

        emitted_any = True
        visible_text += piece
        for char in _stream_chars(piece):
            yield char

        if stopped or _stream_sentence_count(visible_text) >= max_sentences:
            return

    if pending:
        pending, _ = _truncate_at_stop_pattern(pending)
        pending = _clean_visible_answer(pending, prompt)
        pending = _truncate_before_repeated_sentence(visible_text + pending)[len(visible_text):]
        remaining = max_chars - len(visible_text)
        if pending and remaining > 0:
            pending = _trim_to_char_budget(pending, remaining)
            for char in _stream_chars(pending):
                yield char
            emitted_any = True

    if not emitted_any:
        fallback = _clean_visible_answer(buffer, prompt)
        if fallback:
            for char in _stream_chars(fallback):
                yield char


def _looks_like_partial_answer_label(text: str) -> bool:
    stripped = str(text or "").strip().lower()
    if not stripped:
        return False
    labels = ("assistant", "answer", "답변")
    return any(label.startswith(stripped) and stripped != label for label in labels)


def _stream_sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?。？！](?:\s+|$)|다\.(?:\s+|$)|요\.(?:\s+|$)", text))


def _truncate_before_repeated_sentence(text: str) -> str:
    """동일한 완성 문장이 반복되기 시작하면 두 번째 반복 직전에서 자릅니다."""
    seen: set[str] = set()
    start = 0
    sentence_end_re = re.compile(r"(?:[.!?。？！]|다\.|요\.)(?:\s*(?:\[\d+\]|\d+))?(?:\s+|$)")
    for match in sentence_end_re.finditer(text):
        end = match.end()
        sentence_start = start
        sentence = text[start:end].strip()
        start = end
        if not sentence:
            continue
        signature = re.sub(r"(?:\[\d+\]|\b\d+\b)", "", sentence)
        signature = re.sub(r"[\s.!?。？！,，;:：]+", "", signature).lower()
        if len(signature) < 12:
            continue
        if signature in seen:
            return text[:sentence_start].rstrip()
        seen.add(signature)
    return text


def _deduplicate_trailing_citations(text: str) -> str:
    """같은 citation 번호가 여러 문장에 반복되면 마지막 한 번만 남깁니다."""
    original = str(text or "")
    if not original.strip():
        return original

    citation_re = re.compile(
        r"(?P<lead>[ \t]*)(?P<token>\[(?P<bracket>\d{1,2})\]|(?<![\w가-힣])(?P<bare>\d{1,2})(?![\w가-힣]))"
    )

    valid_matches = []
    for match in citation_re.finditer(original):
        citation_no = match.group("bracket") or match.group("bare")
        before = original[: match.start()].rstrip()
        after = original[match.end():]
        next_char = after[:1]
        if not before:
            continue

        # Bare numbers are citation-like only when they trail a completed Korean/English sentence.
        previous_looks_complete = (
            before.endswith((".", "!", "?", "。", "？", "！", "다", "요"))
            or before.endswith(("다.", "요."))
        )
        followed_by_boundary = not next_char or next_char.isspace()
        if not (previous_looks_complete and followed_by_boundary):
            continue
        valid_matches.append((match, citation_no))

    if not valid_matches:
        return original

    last_match_by_no = {citation_no: match for match, citation_no in valid_matches}
    pieces = []
    cursor = 0
    for match, citation_no in valid_matches:
        pieces.append(original[cursor: match.start()])
        if last_match_by_no[citation_no] is match:
            lead = match.group("lead") or " "
            pieces.append(f"{lead}[{citation_no}]")
        cursor = match.end()
    pieces.append(original[cursor:])

    cleaned = "".join(pieces)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned.strip()


def _truncate_at_stop_pattern(text: str) -> tuple[str, bool]:
    """질문/자료 라벨처럼 답변 이후에 이어지는 과생성 시작점을 찾습니다."""
    stop_patterns = (
        r"(?:^|\n|\r)\s*(질문|Question|사용자|User|학생|Human|Prompt)\s*[:：]",
        r"(?:^|\n|\r)\s*(출처|참고자료|Sources?|References?)\s*[:：]",
        r"\n\s*\[검색된 참고자료\]",
        r"\n\s*선택된\s+녹음본\s+전체\s+전사",
        r"\n\s*시스템\s*[:：]",
        r"\n\s*System\s*[:：]",
    )
    for pattern in stop_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return text[:match.start()].strip(), True
    return text, False


def _strip_leading_answer_noise(text: str) -> str:
    """스트림 시작부의 chat template 잔여 라벨을 제거합니다."""
    text = text.replace("\r\n", "\n").lstrip()
    return re.sub(r"^\s*(assistant|답변|Answer)\s*[:：]?\s*", "", text, flags=re.IGNORECASE)


def _first_sentence_end(text: str) -> int | None:
    """첫 번째 완성 문장의 끝 위치를 반환합니다."""
    match = re.search(r"[.!?。？！](?:\s+|$)|다\.(?:\s+|$)|요\.(?:\s+|$)", text)
    if not match:
        return None

    end = match.end()
    if end >= len(text):
        return None

    extended_end = _extend_end_over_citation(text, end)
    if extended_end != end:
        end = extended_end
        if end >= len(text):
            return None
    return end


def _extend_end_over_citation(text: str, end: int) -> int:
    """문장 끝 바로 뒤에 citation이 있으면 함께 포함합니다."""
    citation_match = re.match(r"\s*(?:\[\d+\]|\d+)(?:\s+|$)", text[end:])
    if not citation_match:
        return end
    return end + citation_match.end()


def _ends_like_complete_sentence(text: str) -> bool:
    """최종 flush 시 미완성 어구를 내보내지 않기 위한 간단한 완결성 검사입니다."""
    return bool(re.search(r"(?:[.!?。？！]|\[\d+\]|\d+)\s*$", text.strip()))


def _chat_max_tokens(source_filter: dict | None, max_tokens: int | None = None) -> int:
    """선택 파일 컨텍스트가 있을 때는 더 긴 답변 예산을 사용합니다."""
    if max_tokens is not None:
        return max(1, int(max_tokens))
    return CHAT_SOURCE_MAX_TOKENS if source_filter_has_any_source(source_filter) else CHAT_MAX_TOKENS


def _llm_url(path: str) -> str:
    """현재 LLM 서버 base URL과 API path를 안전하게 결합합니다."""
    return f"{llm_server_url.rstrip('/')}{path}"


def _use_ollama_native_chat() -> bool:
    """환경변수와 URL 힌트를 기준으로 Ollama native chat API 사용 여부를 결정합니다."""
    if CHAT_OLLAMA_NATIVE in {"1", "true", "yes", "on"}:
        return True
    if CHAT_OLLAMA_NATIVE in {"0", "false", "no", "off"}:
        return False
    return ":11434" in llm_server_url or "ollama" in llm_server_url.lower()


def _ollama_chat_payload(
    messages: list[dict],
    source_filter: dict | None,
    *,
    stream: bool,
    thinking: bool = False,
    max_tokens: int | None = None,
) -> dict:
    """Ollama /api/chat 요청 payload를 생성합니다."""
    return {
        "model": llm_model_name,
        "messages": messages,
        "stream": stream,
        "think": bool(thinking),
        "options": {
            "num_predict": _chat_max_tokens(source_filter, max_tokens),
            "temperature": CHAT_TEMPERATURE,
        },
    }


def _openai_chat_payload(
    messages: list[dict],
    source_filter: dict | None,
    *,
    stream: bool,
    thinking: bool = False,
    max_tokens: int | None = None,
) -> dict:
    """OpenAI 호환 /v1/chat/completions 요청 payload를 생성합니다."""
    payload = {
        "model": llm_model_name,
        "messages": messages,
        "max_tokens": _chat_max_tokens(source_filter, max_tokens),
        "temperature": CHAT_TEMPERATURE,
        "stream": stream,
        "stop": [
            "\n질문:",
            "\nQuestion:",
            "\n사용자:",
            "\nUser:",
            "\n학생:",
            "\n답변:",
            "\n[검색된 참고자료]",
        ],
        "repetition_penalty": CHAT_REPETITION_PENALTY,
        "no_repeat_ngram_size": CHAT_NO_REPEAT_NGRAM_SIZE,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    bridgeprag_alpha = _bridgeprag_alpha_for_prompt(messages)
    payload["bridgeprag_alpha"] = bridgeprag_alpha
    return payload


def _bridgeprag_alpha_for_prompt(messages: list[dict]) -> float:
    """RAG 근거가 있는 AI 채팅에서 K/V 주입 강도를 안정적으로 선택합니다.

    BridgePRAG 메모리 주입은 연구 주제 설명에는 유용하지만, 물리/경제처럼
    일반 강의 정의를 묻는 데서는 frozen LLM의 문맥 추종을 흔들 수 있습니다.
    그래서 일반 자료 질문은 RAG prompt를 우선하고, PRAG/BridgePRAG 자체를
    묻는 경우에만 낮은 alpha로 주입합니다.
    """
    if CHAT_DISABLE_BRIDGEPRAG:
        return 0.0

    prompt = "\n".join(str(item.get("content") or "") for item in messages)
    if "검색된 참고자료 전체를 종합" in prompt:
        return CHAT_BRIDGEPRAG_SUMMARY_ALPHA
    if "[검색된 참고자료]" in prompt:
        prompt_lc = prompt.casefold()
        if any(term in prompt_lc for term in ("bridgeprag", "prag", "parametric retrieval", "k/v 메모리")):
            return CHAT_BRIDGEPRAG_TOPIC_ALPHA
        return CHAT_BRIDGEPRAG_REFERENCE_ALPHA
    return 0.0


async def _raise_for_llm_stream_error(stream):
    """스트리밍 응답이 HTTP 오류이면 본문 일부를 포함해 예외로 변환합니다."""
    if stream.status_code < 400:
        return
    raw = await stream.aread()
    detail = raw.decode("utf-8", errors="replace").strip()
    if len(detail) > 500:
        detail = f"{detail[:500]}..."
    raise RuntimeError(f"LLM 서버 오류 {stream.status_code}: {detail or '응답 본문 없음'}")
