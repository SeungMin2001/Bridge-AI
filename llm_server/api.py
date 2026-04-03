from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from threading import Thread
import asyncio, json, traceback
from transformers import TextIteratorStreamer
from run_model import run_model

model = None
tokenizer = None

_DONE = object()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    model, tokenizer = run_model()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512


@app.post("/generate")
async def generate(req: GenerateRequest):
    messages = [
        {"role": "system", "content": "You are a helpful lecture assistant. Answer in Korean."},
        {"role": "user", "content": req.prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    q = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def run_generate():
        try:
            print("[LLM] 생성 시작...")
            model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                streamer=streamer,
            )
            print("[LLM] 생성 완료")
        except Exception as e:
            print(f"[LLM] 생성 에러: {e}")
            traceback.print_exc()
            # 에러 발생 시에도 streamer에 종료 신호를 보내야 feed_from_streamer가 끝남
            streamer.end()

    def feed_from_streamer():
        try:
            for token_text in streamer:
                loop.call_soon_threadsafe(q.put_nowait, token_text)
        except Exception as e:
            print(f"[LLM] streamer 에러: {e}")
            loop.call_soon_threadsafe(q.put_nowait, Exception(str(e)))
        finally:
            loop.call_soon_threadsafe(q.put_nowait, _DONE)

    Thread(target=run_generate, daemon=True).start()
    Thread(target=feed_from_streamer, daemon=True).start()

    async def event_stream():
        phase = "thinking"

        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=120)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'error', 'token': '응답 시간 초과'}, ensure_ascii=False)}\n\n"
                break

            if item is _DONE:
                break

            if isinstance(item, Exception):
                yield f"data: {json.dumps({'type': 'error', 'token': str(item)}, ensure_ascii=False)}\n\n"
                break

            token_text = item

            if "<think>" in token_text:
                token_text = token_text.replace("<think>", "")
                phase = "thinking"

            if "</think>" in token_text:
                token_text = token_text.replace("</think>", "")
                if token_text.strip():
                    yield f"data: {json.dumps({'type': 'thinking', 'token': token_text}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'thinking_done'}, ensure_ascii=False)}\n\n"
                phase = "answer"
                continue

            if not token_text:
                continue

            if token_text.strip() in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
                continue

            yield f"data: {json.dumps({'type': phase, 'token': token_text}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
