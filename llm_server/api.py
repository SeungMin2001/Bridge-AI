from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from threading import Thread
import asyncio, json
from transformers import TextIteratorStreamer
from run_model import run_model

model = None
tokenizer = None

_DONE = object()  # 센티널: 생성 완료 신호


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

    gen_kwargs = {
        **inputs,
        "max_new_tokens": req.max_new_tokens,
        "do_sample": False,
        "streamer": streamer,
    }

    q = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def run_and_feed():
        """생성 스레드: streamer에서 토큰을 읽어 asyncio.Queue에 넣는다."""
        try:
            model.generate(**gen_kwargs)
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, e)
        # streamer 소진 후 토큰들을 큐에 전달
        for token_text in streamer:
            loop.call_soon_threadsafe(q.put_nowait, token_text)
        loop.call_soon_threadsafe(q.put_nowait, _DONE)

    # 수정: generate가 끝나야 streamer가 끝나므로, 별도 스레드에서 읽기
    def feed_from_streamer():
        """streamer에서 blocking으로 읽어 큐에 넣는다."""
        for token_text in streamer:
            loop.call_soon_threadsafe(q.put_nowait, token_text)
        loop.call_soon_threadsafe(q.put_nowait, _DONE)

    def run_generate():
        model.generate(**gen_kwargs)

    Thread(target=run_generate, daemon=True).start()
    Thread(target=feed_from_streamer, daemon=True).start()

    async def event_stream():
        phase = "thinking"

        while True:
            item = await q.get()

            if item is _DONE:
                break

            if isinstance(item, Exception):
                yield f"data: {json.dumps({'type': 'error', 'token': str(item)}, ensure_ascii=False)}\n\n"
                break

            token_text = item

            # <think> 태그 시작
            if "<think>" in token_text:
                token_text = token_text.replace("<think>", "")
                phase = "thinking"

            # </think> 태그 끝
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
