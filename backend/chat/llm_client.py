"""AI 채팅 LLM 호출 클라이언트.

OpenAI 호환 Chat Completions API와 Ollama native chat API 호출을 감싸고,
비스트리밍/스트리밍 응답에서 사용자에게 보여줄 답변 텍스트만 정리합니다.
"""

import os
import re
import json

import httpx

from chat.context_service import source_filter_has_any_source


llm_server_url = os.getenv("LLM_URL", "http://localhost:8001")
llm_model_name = os.getenv("LLM_MODEL", "QuantTrio/Qwen3.5-4B-AWQ")
llm_api_key = os.getenv("LLM_API_KEY", "test-key")
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "512"))
CHAT_SOURCE_MAX_TOKENS = int(os.getenv("CHAT_SOURCE_MAX_TOKENS", "768"))
CHAT_OLLAMA_NATIVE = os.getenv("CHAT_OLLAMA_NATIVE", "auto").strip().lower()

SYSTEM_PROMPT = (
    "You are a helpful lecture assistant. Answer in Korean. "
    "반드시 3문장 이내로 핵심만 답변해. 불필요한 부연설명 하지 마."
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
    return text.strip()


def build_chat_messages(prompt: str) -> list[dict]:
    """시스템 프롬프트와 사용자 prompt를 LLM chat messages 형식으로 묶습니다."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


async def complete_answer(prompt: str, source_filter: dict | None) -> str:
    """비스트리밍 LLM 호출을 수행하고 최종 답변 문자열만 반환합니다."""
    messages = build_chat_messages(prompt)
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=300.0)) as client:
        if _use_ollama_native_chat():
            res = await client.post(
                _llm_url("/api/chat"),
                json=_ollama_chat_payload(messages, source_filter, stream=False, thinking=False),
            )
        else:
            res = await client.post(
                _llm_url("/v1/chat/completions"),
                json=_openai_chat_payload(messages, source_filter, stream=False, thinking=False),
                headers={"Authorization": f"Bearer {llm_api_key}"},
            )
    res.raise_for_status()
    data = res.json()
    if _use_ollama_native_chat():
        raw_answer = (data.get("message") or {}).get("content", "")
    else:
        raw_answer = data["choices"][0]["message"]["content"]
    return remove_thinking(raw_answer)


async def stream_answer(prompt: str, source_filter: dict | None, *, thinking: bool = False):
    """스트리밍 LLM 호출을 수행하고 사용자에게 보여줄 토큰만 async iterator로 반환합니다."""
    messages = build_chat_messages(prompt)
    emitted_content = False
    saw_thinking_only = False

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=300.0)) as client:
        if _use_ollama_native_chat():
            async with client.stream(
                "POST",
                _llm_url("/api/chat"),
                json=_ollama_chat_payload(
                    messages,
                    source_filter,
                    stream=True,
                    thinking=thinking,
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
                        saw_thinking_only = True
                    content = message.get("content") or ""
                    if content:
                        emitted_content = True
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
                        saw_thinking_only = True
                    content = delta.get("content") or (choice.get("message") or {}).get("content") or ""
                    if content:
                        emitted_content = True
                        yield content

    if not emitted_content:
        if saw_thinking_only:
            raise EmptyLLMResponse("모델이 thinking 출력만 반환하고 최종 답변을 반환하지 않았습니다. 다시 질문해 주세요.")
        raise EmptyLLMResponse("모델이 표시 가능한 답변을 반환하지 않았습니다. 다시 질문해 주세요.")


def _chat_max_tokens(source_filter: dict | None) -> int:
    """선택 파일 컨텍스트가 있을 때는 더 긴 답변 예산을 사용합니다."""
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


def _ollama_chat_payload(messages: list[dict], source_filter: dict | None, *, stream: bool, thinking: bool = False) -> dict:
    """Ollama /api/chat 요청 payload를 생성합니다."""
    return {
        "model": llm_model_name,
        "messages": messages,
        "stream": stream,
        "think": bool(thinking),
        "options": {
            "num_predict": _chat_max_tokens(source_filter),
            "temperature": 0.7,
        },
    }


def _openai_chat_payload(messages: list[dict], source_filter: dict | None, *, stream: bool, thinking: bool = False) -> dict:
    """OpenAI 호환 /v1/chat/completions 요청 payload를 생성합니다."""
    return {
        "model": llm_model_name,
        "messages": messages,
        "max_tokens": _chat_max_tokens(source_filter),
        "temperature": 0.7,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }


async def _raise_for_llm_stream_error(stream):
    """스트리밍 응답이 HTTP 오류이면 본문 일부를 포함해 예외로 변환합니다."""
    if stream.status_code < 400:
        return
    raw = await stream.aread()
    detail = raw.decode("utf-8", errors="replace").strip()
    if len(detail) > 500:
        detail = f"{detail[:500]}..."
    raise RuntimeError(f"LLM 서버 오류 {stream.status_code}: {detail or '응답 본문 없음'}")
