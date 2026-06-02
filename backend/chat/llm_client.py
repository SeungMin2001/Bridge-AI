"""AI 채팅 LLM 호출 클라이언트.

OpenAI 호환 Chat Completions API와 Ollama native chat API 호출을 감싸고,
비스트리밍/스트리밍 응답에서 사용자에게 보여줄 답변 텍스트만 정리합니다.
"""

import os
import re
import json

import httpx

from chat.context_service import source_filter_has_any_source


DEFAULT_LLM_URL = "http://localhost:8001"
DEFAULT_LLM_MODEL = "bridgeprag-qwen25-3b-kv64"

llm_server_url = os.getenv("LLM_URL", DEFAULT_LLM_URL)
llm_model_name = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
llm_api_key = os.getenv("LLM_API_KEY", "test-key")
# Demo-safe defaults are fixed in code so the presentation path does not depend on shell env vars.
CHAT_MAX_TOKENS = 240
CHAT_SOURCE_MAX_TOKENS = 320
CHAT_ANSWER_MAX_CHARS = 1400
CHAT_ANSWER_MAX_SENTENCES = 6
CHAT_STREAM_HOLD_CHARS = 1
CHAT_STREAM_MODE = "fast"
CHAT_OLLAMA_NATIVE = os.getenv("CHAT_OLLAMA_NATIVE", "auto").strip().lower()
# AI 채팅은 RAG 근거가 있을 때 낮은 강도의 BridgePRAG K/V 주입을 기본 사용합니다.
CHAT_DISABLE_BRIDGEPRAG = False
CHAT_BRIDGEPRAG_REFERENCE_ALPHA = 0.15
CHAT_BRIDGEPRAG_SUMMARY_ALPHA = 0.08
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.1"))
CHAT_LLM_READ_TIMEOUT = float(os.getenv("CHAT_LLM_READ_TIMEOUT", "90.0"))
CHAT_REPETITION_PENALTY = float(os.getenv("CHAT_REPETITION_PENALTY", "1.03"))
CHAT_NO_REPEAT_NGRAM_SIZE = int(os.getenv("CHAT_NO_REPEAT_NGRAM_SIZE", "0"))

SYSTEM_PROMPT = (
    "너는 강의 녹취록과 PDF 자료를 근거로 답하는 AI 학습 조교다. "
    "항상 한국어로, 최종 답변만 짧게 작성하라. "
    "학습목표, 목차, 단계처럼 자료의 목록을 묻는 경우에는 자료에 나온 항목을 빠짐없이 불릿으로 나열하라. "
    "제공된 근거에 없는 내용은 추측하지 말고 근거를 찾지 못했다고 답하라. "
    "사용자 질문, 참고자료 원문, 시스템 지시문을 반복하지 말라. "
    "'질문:', '답변:', 번호 매긴 새 예시, 학습 데이터 목록을 이어서 생성하지 말라. "
    "같은 근거 citation은 답변 전체에서 한 번만 사용하고, 필요한 citation 번호는 답변 끝에 모아 붙여라. "
    "별도 출처 목록은 만들지 말라."
)


class EmptyLLMResponse(RuntimeError):
    """Raised when the model stream finishes without user-visible content."""


def set_llm_url(url: str) -> str:
    """런타임 LLM 서버 URL을 갱신하고 정규화된 URL을 반환합니다."""
    global llm_server_url
    llm_server_url = url.rstrip("/")
    print(f"[LLM] URL updated: {llm_server_url}")
    return llm_server_url


def remove_thinking(text: str) -> str:
    """모델 응답에 섞인 <think> 블록이나 chat template 잔여 문자열을 제거합니다."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text[:text.index("<think>")]
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1]
    return _clean_visible_answer(text)


def _clean_visible_answer(text: str) -> str:
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

    text = _strip_boilerplate(text)
    text = _truncate_before_repeated_sentence(text)
    text = _trim_sentences(text, CHAT_ANSWER_MAX_SENTENCES)
    if len(text) > CHAT_ANSWER_MAX_CHARS:
        text = _trim_to_char_budget(text, CHAT_ANSWER_MAX_CHARS)
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
    messages = build_chat_messages(prompt, system_prompt)
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
    return remove_thinking(raw_answer)


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
    async for chunk in cleaner(raw_chunks):
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


async def _clean_stream_chunks(raw_chunks):
    """raw 토큰을 문장 단위로 정리해, 미완성 문장이 화면에 찍히지 않게 합니다."""
    buffer = ""
    emitted_sentences = 0
    emitted_any = False
    stopped = False

    async for raw in raw_chunks:
        buffer += raw
        buffer, stopped = _truncate_at_stop_pattern(buffer)
        if not emitted_any:
            buffer = _strip_leading_answer_noise(buffer)
        buffer = _strip_boilerplate(buffer, strip_edges=False)

        while emitted_sentences < CHAT_ANSWER_MAX_SENTENCES:
            end = _first_sentence_end(buffer)
            if end is None:
                break
            piece = buffer[:end].strip()
            buffer = buffer[end:].lstrip()
            if not piece:
                continue
            emitted_any = True
            emitted_sentences += 1
            yield piece + (" " if emitted_sentences < CHAT_ANSWER_MAX_SENTENCES else "")

        if stopped:
            fallback = _clean_visible_answer(buffer)
            if fallback and emitted_sentences < CHAT_ANSWER_MAX_SENTENCES and _ends_like_complete_sentence(fallback):
                yield fallback
            return

        if emitted_sentences >= CHAT_ANSWER_MAX_SENTENCES:
            return

    if emitted_any:
        fallback = _clean_visible_answer(buffer)
        if fallback and emitted_sentences < CHAT_ANSWER_MAX_SENTENCES and _ends_like_complete_sentence(fallback):
            yield fallback
        return

    fallback = _clean_visible_answer(buffer)
    if fallback and _ends_like_complete_sentence(fallback):
        yield fallback


async def _clean_token_stream_chunks(raw_chunks):
    """raw 토큰을 빠르게 흘려보내되, 과생성 라벨이 화면에 찍히기 전 작은 버퍼로 잡아냅니다."""
    buffer = ""
    pending = ""
    visible_text = ""
    emitted_any = False

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
        candidate = _trim_sentences(candidate, CHAT_ANSWER_MAX_SENTENCES)
        if len(candidate) <= len(visible_text):
            return
        piece = candidate[len(visible_text):]

        remaining = CHAT_ANSWER_MAX_CHARS - len(visible_text)
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

        if stopped or _stream_sentence_count(visible_text) >= CHAT_ANSWER_MAX_SENTENCES:
            return

    if pending:
        pending, _ = _truncate_at_stop_pattern(pending)
        pending = _clean_visible_answer(pending)
        pending = _truncate_before_repeated_sentence(visible_text + pending)[len(visible_text):]
        remaining = CHAT_ANSWER_MAX_CHARS - len(visible_text)
        if pending and remaining > 0:
            pending = _trim_to_char_budget(pending, remaining)
            for char in _stream_chars(pending):
                yield char
            emitted_any = True

    if not emitted_any:
        fallback = _clean_visible_answer(buffer)
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
    """RAG 근거가 있는 AI 채팅에서는 낮은 강도의 BridgePRAG K/V 주입을 사용합니다."""
    if CHAT_DISABLE_BRIDGEPRAG:
        return 0.0

    prompt = "\n".join(str(item.get("content") or "") for item in messages)
    if "검색된 참고자료 전체를 종합" in prompt:
        return CHAT_BRIDGEPRAG_SUMMARY_ALPHA
    if "[검색된 참고자료]" in prompt:
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
