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

    thread = Thread(target=lambda: model.generate(**gen_kwargs))
    thread.start()

    async def event_stream():
        phase = "thinking"
        loop = asyncio.get_event_loop()

        while True:
            # blocking iterator를 스레드에서 실행하여 이벤트 루프 차단 방지
            try:
                token_text = await loop.run_in_executor(None, next, streamer)
            except StopIteration:
                break

            # <think> 태그 시작
            if "<think>" in token_text:
                token_text = token_text.replace("<think>", "")
                phase = "thinking"

            # </think> 태그 끝 → answer 단계로 전환
            if "</think>" in token_text:
                token_text = token_text.replace("</think>", "")
                if token_text.strip():
                    yield f"data: {json.dumps({'type': 'thinking', 'token': token_text}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'thinking_done'}, ensure_ascii=False)}\n\n"
                phase = "answer"
                continue

            if not token_text:
                continue

            # special token 필터링
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
