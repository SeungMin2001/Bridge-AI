from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from threading import Thread
import torch
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
    max_new_tokens: int = 256


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
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def run_generation():
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                streamer=streamer,
            )

    thread = Thread(target=run_generation)
    thread.start()

    def token_stream():
        for token in streamer:
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")
