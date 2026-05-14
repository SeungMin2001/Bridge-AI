"""
LLM Server API (MergePRAG 통합)

엔드포인트:
  POST /generate          - LLM Only (기존)
  POST /generate/rag      - RAG context를 프롬프트에 삽입
  POST /generate/mergeprag - MergePRAG: 과목 메모리에서 K,V inject
  POST /generate/stream   - 스트리밍 버전 (방식 선택 가능)
  POST /memory/add        - 과목 메모리에 passage 추가
  POST /memory/add-batch  - 여러 passage 한번에 추가
  GET  /memory/list       - 과목별 메모리 상태 조회
  POST /memory/clear      - 과목 메모리 초기화

사용법: cd llm_server && uvicorn api:app --host 0.0.0.0 --port 8001
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import re
import json
import torch
from threading import Thread
from transformers import TextIteratorStreamer

from run_model import run_model
from mergePRAG.config import build_chat_text
from mergePRAG.main import CourseMemoryManager, make_hook, CRITICAL_LAYER

# ── 글로벌 ──
model = None
tokenizer = None
memory_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, memory_manager
    model, tokenizer = run_model()
    device = next(model.parameters()).device
    memory_manager = CourseMemoryManager(model, tokenizer, device)
    print(f"[API] 서버 준비 완료 (device={device})")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청 스키마 ──
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    enable_thinking: bool = False


class RAGRequest(BaseModel):
    prompt: str
    context: str = ""
    citations: list = []
    max_new_tokens: int = 256
    enable_thinking: bool = False


class MergePRAGRequest(BaseModel):
    prompt: str
    course_id: str
    passages: list[str] = []  # 추가로 inject할 passage (optional)
    max_new_tokens: int = 256
    enable_thinking: bool = False


class StreamRequest(BaseModel):
    prompt: str
    mode: str = "rag"  # "llm_only", "rag", "mergeprag"
    course_id: str = ""
    context: str = ""
    citations: list = []
    passages: list[str] = []
    max_new_tokens: int = 256
    enable_thinking: bool = False


class MemoryAddRequest(BaseModel):
    course_id: str
    passage: str


class MemoryBatchRequest(BaseModel):
    course_id: str
    passages: list[str]


class MemoryClearRequest(BaseModel):
    course_id: str


# ── 유틸 ──
def remove_thinking(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    if '<think>' in text:
        text = text[:text.index('<think>')]
    if 'assistant\n' in text:
        text = text.split('assistant\n')[-1]
    return text.strip()


def clean_special_tokens(text: str) -> str:
    return re.sub(r'<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>', '', text).strip()


def build_messages(prompt: str, context: str = "", enable_thinking: bool = False):
    if context:
        user_content = (
            f"다음은 강의 내용에서 검색된 참고자료입니다:\n\n{context}\n\n"
            f"위 참고자료를 바탕으로 답변하고, 답변 마지막에 참고한 출처를 '[출처]' 형식으로 표시해주세요.\n\n"
            f"질문: {prompt}"
        )
    else:
        user_content = prompt

    return build_chat_text(
        tokenizer,
        question=user_content,
        enable_thinking=enable_thinking,
    )


def generate_sync(input_text: str, max_new_tokens: int = 256):
    """동기 생성 (non-streaming)"""
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


def generate_with_hook(input_text: str, K, V, max_new_tokens: int = 256):
    """MergePRAG hook inject 후 생성"""
    target_layer = model.model.layers[CRITICAL_LAYER]
    hook = target_layer.register_forward_hook(make_hook(K, V))
    try:
        result = generate_sync(input_text, max_new_tokens)
    finally:
        hook.remove()
    return result


def parse_output(raw_output: str):
    think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    answer = remove_thinking(raw_output)
    answer = clean_special_tokens(answer)
    return thinking, answer


# ══════════════════════════════════════
#  엔드포인트: LLM Only
# ══════════════════════════════════════
@app.post("/generate")
async def generate(req: GenerateRequest):
    input_text = build_messages(req.prompt, enable_thinking=req.enable_thinking)
    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, generate_sync, input_text, req.max_new_tokens)
    thinking, answer = parse_output(raw)
    return {"thinking": thinking, "answer": answer}


# ══════════════════════════════════════
#  엔드포인트: RAG + LLM
# ══════════════════════════════════════
@app.post("/generate/rag")
async def generate_rag(req: RAGRequest):
    input_text = build_messages(req.prompt, context=req.context, enable_thinking=req.enable_thinking)
    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, generate_sync, input_text, req.max_new_tokens)
    thinking, answer = parse_output(raw)
    return {"thinking": thinking, "answer": answer, "citations": req.citations}


# ══════════════════════════════════════
#  엔드포인트: MergePRAG + LLM
# ══════════════════════════════════════
@app.post("/generate/mergeprag")
async def generate_mergeprag(req: MergePRAGRequest):
    # 추가 passage가 있으면 메모리에 병합
    if req.passages:
        memory_manager.add_passages(req.course_id, req.passages)

    # 과목 메모리에서 K, V 조회
    K, V = memory_manager.get_memory(req.course_id, question=req.prompt)
    if K is None:
        return {"thinking": "", "answer": "해당 과목의 메모리가 없습니다. 먼저 passage를 추가해주세요.", "citations": []}

    input_text = build_messages(req.prompt, enable_thinking=req.enable_thinking)
    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, generate_with_hook, input_text, K, V, req.max_new_tokens)
    thinking, answer = parse_output(raw)
    return {"thinking": thinking, "answer": answer}


# ══════════════════════════════════════
#  엔드포인트: 스트리밍 (SSE, 방식 선택)
# ══════════════════════════════════════
@app.post("/generate/stream")
async def generate_stream(req: StreamRequest):
    # 프롬프트 구성
    if req.mode == "rag":
        input_text = build_messages(req.prompt, context=req.context, enable_thinking=req.enable_thinking)
    else:
        input_text = build_messages(req.prompt, enable_thinking=req.enable_thinking)

    # MergePRAG: 추가 passage 병합 + hook 준비
    hook_handle = None
    if req.mode == "mergeprag" and req.course_id:
        if req.passages:
            memory_manager.add_passages(req.course_id, req.passages)
        K, V = memory_manager.get_memory(req.course_id, question=req.prompt)
        if K is not None:
            target_layer = model.model.layers[CRITICAL_LAYER]
            hook_handle = target_layer.register_forward_hook(make_hook(K, V))

    async def event_generator():
        try:
            # citations 먼저 전송
            if req.citations:
                yield f"data: {json.dumps({'type': 'citations', 'citations': req.citations}, ensure_ascii=False)}\n\n"

            # TextIteratorStreamer로 토큰 단위 스트리밍
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            gen_kwargs = {
                **{k: v for k, v in inputs.items()},
                "max_new_tokens": req.max_new_tokens,
                "do_sample": False,
                "streamer": streamer,
            }

            thread = Thread(target=lambda: model.generate(**gen_kwargs))
            thread.start()

            in_thinking = False
            for token_text in streamer:
                # <think> 태그 필터링
                if '<think>' in token_text:
                    in_thinking = True
                    continue
                if '</think>' in token_text:
                    in_thinking = False
                    continue
                if in_thinking:
                    continue

                # special token 제거
                cleaned = clean_special_tokens(token_text)
                if cleaned:
                    yield f"data: {json.dumps({'type': 'token', 'content': cleaned}, ensure_ascii=False)}\n\n"

            thread.join()
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        finally:
            # hook 해제
            if hook_handle is not None:
                hook_handle.remove()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ══════════════════════════════════════
#  과목 메모리 관리 API
# ══════════════════════════════════════
@app.post("/memory/add")
async def memory_add(req: MemoryAddRequest):
    count = memory_manager.add_passage(req.course_id, req.passage)
    return {"course_id": req.course_id, "passage_count": count}


@app.post("/memory/add-batch")
async def memory_add_batch(req: MemoryBatchRequest):
    count = memory_manager.add_passages(req.course_id, req.passages)
    return {"course_id": req.course_id, "passage_count": count}


@app.get("/memory/list")
async def memory_list():
    return {"memories": memory_manager.list_courses()}


@app.post("/memory/clear")
async def memory_clear(req: MemoryClearRequest):
    memory_manager.clear_memory(req.course_id)
    return {"course_id": req.course_id, "cleared": True}


#  과목 메모리 통계 API (검증용)
@app.get("/memory/stats")
async def memory_stats():
    """전체 과목 메모리 상태를 상세 반환한다."""
    courses = memory_manager.list_courses()
    detailed = []
    for course in courses:
        cid = course["course_id"]
        mem = memory_manager.memories.get(cid)
        info = {
            "course_id": cid,
            "passage_count": course["passage_count"],
        }
        if mem:
            info["k_shape"] = list(mem["K"].shape)
            info["v_shape"] = list(mem["V"].shape)
            info["k_norm"] = round(float(mem["K"].norm()), 4)
            info["v_norm"] = round(float(mem["V"].norm()), 4)
            passages = mem.get("passages", [])
            info["passage_lengths"] = [len(p) for p in passages]
            info["total_chars"] = sum(len(p) for p in passages)
        detailed.append(info)
    return {
        "total_courses": len(courses),
        "critical_layer": CRITICAL_LAYER,
        "courses": detailed,
    }


@app.get("/memory/stats/{course_id}")
async def memory_stats_course(course_id: str):
    """특정 과목의 메모리 상태를 상세 반환한다."""
    mem = memory_manager.memories.get(course_id)
    if mem is None:
        return {"course_id": course_id, "error": "no memory found"}
    passages = mem.get("passages", [])
    return {
        "course_id": course_id,
        "passage_count": mem["count"],
        "k_shape": list(mem["K"].shape),
        "v_shape": list(mem["V"].shape),
        "k_norm": round(float(mem["K"].norm()), 4),
        "v_norm": round(float(mem["V"].norm()), 4),
        "total_chars": sum(len(p) for p in passages),
        "passages": [
            {"index": i, "length": len(p), "preview": p[:80]}
            for i, p in enumerate(passages)
        ],
    }


# ── 헬스체크 ──
@app.get("/health")
async def health():
    return {"status": "ok", "model": "Qwen/Qwen2.5-3B", "mergeprag": True}
