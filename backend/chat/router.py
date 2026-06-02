"""AI 채팅 HTTP API 라우터.

프론트의 채팅바 요청을 받아 컨텍스트 생성 서비스와 LLM 클라이언트를 연결합니다.
엔드포인트 경로는 기존 프론트 호환성을 위해 /chat, /chat/stream, /register-llm을 유지합니다.
"""

import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chat.context_service import (
    build_prompt_and_citations,
    ensure_material_rag_for_chat,
    is_smalltalk_question,
)
from chat.llm_client import EmptyLLMResponse, complete_answer, set_llm_url, stream_answer


router = APIRouter(tags=["chat"])

DEMO_PIPELINE_LOG = True


def _demo_log(message: str) -> None:
    if DEMO_PIPELINE_LOG:
        print(f"[DEMO:CHAT] {message}", flush=True)


def _preview(text: str, limit: int = 120) -> str:
    compact = " ".join(str(text or "").split())
    return compact if len(compact) <= limit else f"{compact[:limit - 3]}..."


class ChatRequest(BaseModel):
    """AI 채팅 요청 본문.

    session_id와 source_filter는 프론트 호환을 위해 받지만, AI 채팅 검색 범위는 항상 전체 워크스페이스입니다.
    """

    question: str
    is_thinking: bool = False
    # 선택 파일 기준 RAG 검색을 위해 session_id를 받는다.
    session_id: str | None = None
    # 왼쪽 사이드바에서 고른 PDF/녹음본만 AI 채팅 근거로 쓰기 위한 필터.
    source_filter: dict | None = None


class RegisterRequest(BaseModel):
    """런타임에서 사용할 LLM 서버 URL 등록 요청 본문."""

    url: str


def _global_chat_scope() -> tuple[None, None]:
    """AI 질문은 파일 선택과 무관하게 모든 전사문/PDF 자료를 검색합니다."""
    return None, None


@router.post("/register-llm")
async def register_llm(req: RegisterRequest):
    """LLM 서버 URL을 변경해 로컬/원격 모델 서버를 전환합니다."""
    return {"status": "ok", "url": set_llm_url(req.url)}


@router.post("/chat")
async def chat(req: ChatRequest):
    """기존 비스트리밍 채팅 엔드포인트입니다.

    PDF/RAG 인덱싱을 확인하고, 현재 워크스페이스 컨텍스트를 포함한 prompt를 만들어 한 번에 답변을 반환합니다.
    """
    print(f"[CHAT] 요청 수신: {req.question}")
    _demo_log(f"1) 질문 수신: '{_preview(req.question, 100)}'")
    try:
        if is_smalltalk_question(req.question):
            prompt, citations = req.question, []
            _demo_log("2) 일반 대화로 분류: RAG 검색 생략, LLM에 직접 전달")
        else:
            search_session_id, search_source_filter = _global_chat_scope()
            # 기존 프론트가 session_id를 보내도 검색 범위는 전체로 풀고,
            # 해당 파일 PDF는 누락된 인덱스가 있으면 증분 보강만 수행합니다.
            _demo_log("2) 자료 기반 질문으로 분류: PDF/전사 RAG 인덱스 확인")
            await ensure_material_rag_for_chat(req.session_id, req.source_filter)
            prompt, citations = await build_prompt_and_citations(req.question, search_session_id, search_source_filter)
            _demo_log(f"3) RAG prompt 구성 완료: prompt_chars={len(prompt)}, citations={len(citations)}")
        _demo_log("4) LLM 서버에 답변 생성 요청")
        answer = await complete_answer(prompt, None)
        _demo_log(f"5) LLM 답변 수신: answer_chars={len(answer or '')}")
        if not answer:
            answer = "모델이 표시 가능한 답변을 반환하지 않았습니다. 다시 질문해 주세요."
        return {"thinking": "", "answer": answer, "citations": citations}
    except Exception as e:
        print(f"[CHAT] 에러: {e}")
        return {"thinking": "", "answer": f"오류: {e}", "citations": []}


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 방식으로 citations와 LLM 토큰을 순차 전송하는 채팅 엔드포인트입니다."""
    request_started_at = time.perf_counter()
    print(f"[CHAT STREAM] 요청 수신: {req.question}")
    _demo_log(f"1) 스트리밍 질문 수신: '{_preview(req.question, 100)}'")

    if is_smalltalk_question(req.question):
        prompt, citations = req.question, []
        _demo_log("2) 일반 대화로 분류: RAG 검색 생략, LLM 스트리밍으로 직접 전달")
    else:
        search_session_id, search_source_filter = _global_chat_scope()
        # 요약/퀴즈는 선택 파일 기준을 유지하지만, AI 채팅은 항상 전체 자료 검색으로 고정합니다.
        _demo_log("2) 자료 기반 질문으로 분류: 전체 PDF/전사 자료에서 RAG 검색 준비")
        await ensure_material_rag_for_chat(req.session_id, req.source_filter)
        prompt, citations = await build_prompt_and_citations(req.question, search_session_id, search_source_filter)
        _demo_log(f"3) RAG prompt 구성 완료: prompt_chars={len(prompt)}, citations={len(citations)}")
    t_rag = time.perf_counter()
    print(f"⏱️ [RAG 검색] {(t_rag - request_started_at)*1000:.0f}ms")
    _demo_log(f"4) RAG 단계 완료: elapsed_ms={(t_rag - request_started_at)*1000:.0f}")

    async def generate():
        """SSE 이벤트 형식으로 citations, token, error, DONE 메시지를 생성합니다."""
        first_token_logged = False
        first_token_elapsed = None
        emitted_content = False
        emitted_error = False

        yield f"data: {json.dumps({'type': 'citations', 'citations': citations}, ensure_ascii=False)}\n\n"

        try:
            _demo_log("5) LLM 서버에 스트리밍 답변 생성 요청")
            async for token in stream_answer(
                prompt,
                None,
                thinking=req.is_thinking,
            ):
                emitted_content = True
                if not first_token_logged:
                    first_token_logged = True
                    first_token_elapsed = time.perf_counter() - request_started_at
                    print(f"[CHAT STREAM] 첫 토큰 도착: {first_token_elapsed:.3f}s")
                    _demo_log(f"6) 첫 글자 수신: elapsed_s={first_token_elapsed:.3f}")
                yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"
        except EmptyLLMResponse as e:
            emitted_error = True
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"[CHAT STREAM] 에러: {e}")
            emitted_error = True
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            total_elapsed = time.perf_counter() - request_started_at
            first_token_text = f"{first_token_elapsed:.3f}s" if first_token_elapsed is not None else "N/A"
            print(
                f"[CHAT STREAM] 응답 종료: first_token={first_token_text}, "
                f"content={emitted_content}, error={emitted_error}, total={total_elapsed:.3f}s"
            )
            _demo_log(f"7) 스트리밍 종료: total_s={total_elapsed:.3f}, content={emitted_content}, error={emitted_error}")

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
