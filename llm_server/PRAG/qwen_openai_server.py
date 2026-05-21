"""Minimal OpenAI-compatible server for local Qwen augmentation.

This is intended for data augmentation when Docker/vLLM is unavailable.
It exposes the two endpoints used by PRAG augmentation scripts:

  GET  /health
  GET  /v1/models
  POST /v1/completions
  POST /v1/chat/completions

Example:
  python -m llm_server.PRAG.qwen_openai_server --model Qwen/Qwen3.5-4B --port 8001
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = 0.35
    top_p: float = 0.9
    stop: str | list[str] | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = 0.35
    top_p: float = 0.9
    stop: str | list[str] | None = None


def make_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="Local Qwen OpenAI-compatible Server")
    executor = ThreadPoolExecutor(max_workers=1)

    state: dict[str, Any] = {
        "model_id": args.model,
        "device": "cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
        "model": None,
        "tokenizer": None,
    }

    @app.on_event("startup")
    def load_model() -> None:
        dtype = torch.float32 if state["device"] == "cpu" else torch.float16
        print(f"[qwen-openai-server] loading model={args.model} device={state['device']} dtype={dtype}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="auto" if state["device"] == "cuda" else None,
            trust_remote_code=True,
        )
        if state["device"] == "cpu":
            model = model.to("cpu")
        model.eval()
        state["tokenizer"] = tokenizer
        state["model"] = model
        print("[qwen-openai-server] ready")

    def apply_stop(text: str, stop: str | list[str] | None) -> str:
        if not stop:
            return text
        stops = [stop] if isinstance(stop, str) else stop
        cut = len(text)
        for token in stops:
            if not token:
                continue
            idx = text.find(token)
            if idx >= 0:
                cut = min(cut, idx)
        return text[:cut]

    def generate_text(prompt: str, req: CompletionRequest | ChatCompletionRequest) -> str:
        tokenizer = state["tokenizer"]
        model = state["model"]
        if tokenizer is None or model is None:
            raise RuntimeError("Model is not loaded yet")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        do_sample = req.temperature is not None and req.temperature > 0
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=do_sample,
                temperature=max(float(req.temperature), 1e-5) if do_sample else None,
                top_p=float(req.top_p),
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return apply_stop(text.strip(), req.stop).strip()

    async def run_generation(prompt: str, req: CompletionRequest | ChatCompletionRequest) -> str:
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, generate_text, prompt, req)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok" if state["model"] is not None else "loading",
            "model_id": state["model_id"],
            "device": state["device"],
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": state["model_id"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local-transformers",
                    "root": state["model_id"],
                    "parent": None,
                }
            ],
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> dict[str, Any]:
        text = await run_generation(req.prompt, req)
        return {
            "id": f"cmpl-local-{int(time.time() * 1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model or state["model_id"],
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        tokenizer = state["tokenizer"]
        if tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded yet")
        messages = [m.model_dump() for m in req.messages]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        text = await run_generation(prompt, req)
        return {
            "id": f"chatcmpl-local-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or state["model_id"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local Qwen OpenAI-compatible server.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(make_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
