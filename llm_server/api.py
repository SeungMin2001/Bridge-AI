from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from threading import Thread
import asyncio, json, torch, queue
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
        phase = "thinking"  # thinking → answer
        inside_think = False

        for token_text in streamer:
            # <think> 태그 시작
            if "<think>" in token_text:
                inside_think = True
                token_text = token_text.replace("<think>", "")
                phase = "thinking"

            # </think> 태그 끝 → answer 단계로 전환
            if "</think>" in token_text:
                inside_think = False
                token_text = token_text.replace("</think>", "")
                # thinking 남은 부분 전송
                if token_text.strip():
                    yield f"data: {json.dumps({'type': 'thinking', 'token': token_text}, ensure_ascii=False)}\n\n"
                # phase 전환 신호
                yield f"data: {json.dumps({'type': 'thinking_done'}, ensure_ascii=False)}\n\n"
                phase = "answer"
                continue

            # 빈 토큰 스킵
            if not token_text:
                continue

            # special token 필터링
            if token_text.strip() in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
                continue

            yield f"data: {json.dumps({'type': phase, 'token': token_text}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
