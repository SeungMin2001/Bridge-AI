"""OpenAI-compatible BridgePRAG inference server.

This server replaces the vLLM process for the service path. It exposes the
minimal Chat Completions endpoints that the existing backend already calls,
while internally running a local Transformers Qwen base model plus a trained
BridgePRAG HyperKV checkpoint.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from threading import Lock, Thread
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .memory import (
    HyperKVGenerator,
    build_chat_prompt,
    deterministic_generation_config,
    encode_merged_memory,
    make_memory_hook,
    model_num_heads,
)


DEFAULT_CHECKPOINT = (
    "llm_server/PRAG/"
    "prag_orthomerge_entity_simple4000_epoch_bestslot_fixedlayer_kv64_ep10_qp_kvadapt_memory_checkpoint.pt"
)

MODEL_ID = os.getenv("BRIDGEPRAG_MODEL_ID", "bridgeprag-qwen25-3b-kv64")
BASE_MODEL = os.getenv("BRIDGEPRAG_MODEL", "Qwen/Qwen2.5-3B")
CHECKPOINT_PATH = os.getenv("BRIDGEPRAG_CHECKPOINT", DEFAULT_CHECKPOINT)
MAX_PASSAGES = int(os.getenv("BRIDGEPRAG_MAX_PASSAGES", "4"))
MAX_INPUT_TOKENS = int(os.getenv("BRIDGEPRAG_MAX_INPUT_TOKENS", "2048"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("BRIDGEPRAG_MAX_NEW_TOKENS", "512"))
GENERATION_PROMPT_MODE = os.getenv("BRIDGEPRAG_GENERATION_PROMPT", "full").strip().lower()
DTYPE = os.getenv("BRIDGEPRAG_DTYPE", "float16").strip().lower()
STRICT_MODEL_ID = os.getenv("BRIDGEPRAG_STRICT_MODEL_ID", "0").strip().lower() in {"1", "true", "yes", "on"}
LOG_REQUESTS = os.getenv("BRIDGEPRAG_LOG_REQUESTS", "1").strip().lower() in {"1", "true", "yes", "on"}


model = None
tokenizer = None
hypernet = None
device = None
target_layer = None
runtime_config: dict[str, Any] = {}
generation_lock = Lock()


def _torch_dtype():
    if DTYPE in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if DTYPE in {"fp32", "float32"}:
        return torch.float32
    return torch.float16


def _load_base_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": _torch_dtype(),
    }
    try:
        mdl = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except TypeError:
        dtype = kwargs.pop("torch_dtype")
        mdl = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(target_device)
    mdl.eval()
    for param in mdl.parameters():
        param.requires_grad = False
    return mdl, tok, target_device


def _load_bridgeprag_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BridgePRAG checkpoint not found: {path}. "
            "Set BRIDGEPRAG_CHECKPOINT to the trained *_memory_checkpoint.pt path."
        )
    return torch.load(path, map_location="cpu")


def _make_hypernet_from_checkpoint(ckpt: dict[str, Any]):
    config = ckpt.get("config") or {}
    d_model = int(getattr(model.config, "hidden_size"))
    feature_dim = int(config.get("feature_dim") or d_model)
    num_kv = int(config.get("num_kv") or 64)
    hidden_dim = int(config.get("hidden_dim") or 1024)
    question_fusion = str(config.get("question_fusion") or "kv_adapter")

    net = HyperKVGenerator(
        d_model=d_model,
        num_kv=num_kv,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        question_fusion=question_fusion,
        legacy=False,
    ).float()
    state = ckpt.get("hypernet", ckpt)
    net.load_state_dict(state, strict=True)
    net.to(device)
    net.eval()
    return net, config


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, hypernet, device, target_layer, runtime_config

    print(f"[BridgePRAG] loading base model={BASE_MODEL}")
    model, tokenizer, device = _load_base_model(BASE_MODEL)
    print(f"[BridgePRAG] loading checkpoint={CHECKPOINT_PATH}")
    ckpt = _load_bridgeprag_checkpoint(CHECKPOINT_PATH)
    hypernet, config = _make_hypernet_from_checkpoint(ckpt)

    layer_idx = int(os.getenv("BRIDGEPRAG_CRITICAL_LAYER", config.get("critical_layer", 23)))
    target_layer = model.model.layers[layer_idx]
    runtime_config = {
        "model_id": MODEL_ID,
        "base_model": BASE_MODEL,
        "checkpoint": CHECKPOINT_PATH,
        "critical_layer": layer_idx,
        "num_kv": int(config.get("num_kv", getattr(hypernet, "num_kv", 0))),
        "question_conditioned_memory": bool(config.get("question_conditioned_memory", True)),
        "question_fusion": str(config.get("question_fusion", getattr(hypernet, "question_fusion", "none"))),
        "injection_mode": str(config.get("injection_mode", "attention")),
        "generation_prompt_mode": GENERATION_PROMPT_MODE,
        "alpha": float(os.getenv("BRIDGEPRAG_ALPHA", config.get("alpha", 1.0))),
        "device": str(device),
    }
    print(f"[BridgePRAG] ready {runtime_config}")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", **runtime_config}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "bridgeprag",
                "root": BASE_MODEL,
                "parent": None,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(payload: dict[str, Any]):
    _validate_model(payload.get("model"))
    messages = payload.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    max_tokens = int(payload.get("max_tokens") or payload.get("max_new_tokens") or DEFAULT_MAX_NEW_TOKENS)
    stream = bool(payload.get("stream", False))
    request = _build_request(messages, max_tokens=max_tokens)

    if stream:
        return StreamingResponse(_stream_openai_chunks(request), media_type="text/event-stream")

    text = await asyncio.to_thread(_generate_text, request)
    return {
        "id": f"chatcmpl-bridgeprag-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "bridgeprag": _request_trace(request),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def _validate_model(requested_model: str | None) -> None:
    if not STRICT_MODEL_ID:
        return
    allowed = {MODEL_ID, BASE_MODEL}
    if requested_model not in allowed:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"The model {requested_model} does not exist.",
                    "type": "NotFoundError",
                    "param": "model",
                    "code": 404,
                }
            },
        )


def _build_request(messages: list[dict[str, Any]], *, max_tokens: int) -> dict[str, Any]:
    system_texts = [str(item.get("content") or "") for item in messages if item.get("role") == "system"]
    user_texts = [str(item.get("content") or "") for item in messages if item.get("role") == "user"]
    user_prompt = user_texts[-1] if user_texts else str(messages[-1].get("content") or "")
    question = _extract_question(user_prompt)
    passages = _extract_passages(user_prompt)[:MAX_PASSAGES]
    generation_text = _build_generation_text(system_texts, user_prompt, question, has_memory=bool(passages))
    return {
        "question": question,
        "passages": passages,
        "generation_text": generation_text,
        "max_tokens": max_tokens,
        "reference_prompt": "[검색된 참고자료]" in user_prompt,
    }


def _extract_question(text: str) -> str:
    matches = list(re.finditer(r"(?:^|\n)질문:\s*(.+)", text, flags=re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    matches = list(re.finditer(r"(?:^|\n)Question:\s*(.+)", text, flags=re.DOTALL | re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip()
    return text.strip()


def _extract_passages(text: str) -> list[str]:
    section = text
    marker = "[검색된 참고자료]"
    if marker in text:
        section = text.split(marker, 1)[1]
    stop_match = re.search(r"\n\n(?:사용자가|PDF|페이지|답변|질문:)", section)
    if stop_match:
        section = section[: stop_match.start()]

    passages: list[str] = []
    pattern = re.compile(r"(?ms)^\[(\d+)\]\s*(.*?)(?=^\[\d+\]\s|\Z)")
    for match in pattern.finditer(section):
        passage = match.group(2).strip()
        passage = re.sub(r"\s*\(출처:\s*.*?\)\s*$", "", passage, flags=re.DOTALL).strip()
        passage = re.sub(r"^선택된\s+녹음본\s+전체\s+전사\s*\(출처:\s*.*?\)\s*", "", passage).strip()
        if passage and passage not in passages:
            passages.append(passage)

    if not passages:
        for line in section.splitlines():
            line = line.strip()
            match = re.match(r"^\[\d+\]\s*(.+?)(?:\s*\(출처:\s*.*?\))?$", line)
            if match:
                passage = match.group(1).strip()
                if passage and passage not in passages:
                    passages.append(passage)
    return passages


def _build_generation_text(system_texts: list[str], user_prompt: str, question: str, *, has_memory: bool) -> str:
    if GENERATION_PROMPT_MODE == "full" or not has_memory:
        content = "\n".join(part for part in [*system_texts, user_prompt] if part).strip()
    else:
        content = question.strip()
    return build_chat_prompt(tokenizer, content)


def _encode_memory_for_request(request: dict[str, Any]):
    passages = request["passages"]
    if not passages:
        return None
    return encode_merged_memory(
        model,
        tokenizer,
        hypernet,
        passages,
        device,
        question=request["question"],
        question_conditioned=bool(runtime_config.get("question_conditioned_memory", True)),
    )


def _generation_kwargs(request: dict[str, Any], streamer: TextIteratorStreamer | None = None) -> dict[str, Any]:
    inputs = tokenizer(
        request["generation_text"],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(device)
    kwargs = {
        **inputs,
        "generation_config": deterministic_generation_config(tokenizer, request["max_tokens"]),
    }
    if streamer is not None:
        kwargs["streamer"] = streamer
    return kwargs


def _generate_text(request: dict[str, Any]) -> str:
    with generation_lock, torch.no_grad():
        memory = _encode_memory_for_request(request)
        _log_request_trace(request, memory)
        hook = _register_memory_hook(memory)
        kwargs = _generation_kwargs(request)
        try:
            generated = model.generate(**kwargs)
        finally:
            if hook is not None:
                hook.remove()
        prompt_len = kwargs["input_ids"].shape[1]
        text = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
        return _clean_output(text)


def _stream_openai_chunks(request: dict[str, Any]):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    done_payload = "data: [DONE]\n\n"

    with generation_lock, torch.no_grad():
        memory = _encode_memory_for_request(request)
        _log_request_trace(request, memory)
        hook = _register_memory_hook(memory)
        def _worker():
            with torch.no_grad():
                model.generate(**_generation_kwargs(request, streamer=streamer))

        thread = Thread(target=_worker)
        thread.start()
        try:
            for token in streamer:
                cleaned = _clean_stream_token(token)
                if not cleaned:
                    continue
                chunk = {
                    "id": f"chatcmpl-bridgeprag-{int(time.time() * 1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {"content": cleaned}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            thread.join()
        finally:
            if hook is not None:
                hook.remove()
        yield done_payload


def _register_memory_hook(memory: dict[str, Any] | None):
    if not memory:
        return None
    return target_layer.register_forward_hook(
        make_memory_hook(
            memory["K"],
            memory["V"],
            model_num_heads(model),
            alpha=float(runtime_config.get("alpha", 1.0)),
            injection_mode=str(runtime_config.get("injection_mode", "attention")),
        )
    )


def _request_trace(request: dict[str, Any], memory: dict[str, Any] | None = None) -> dict[str, Any]:
    passages = request.get("passages") or []
    return {
        "memory_active": bool(passages),
        "passage_count": len(passages),
        "reference_prompt": bool(request.get("reference_prompt")),
        "merged_count": int((memory or {}).get("merged_count") or len(passages) or 0),
        "merge_mode": (memory or {}).get("merge_mode", "orthogonal" if len(passages) > 1 else "single"),
        "critical_layer": runtime_config.get("critical_layer"),
        "num_kv": runtime_config.get("num_kv"),
        "question_fusion": runtime_config.get("question_fusion"),
        "injection_mode": runtime_config.get("injection_mode"),
        "generation_prompt_mode": runtime_config.get("generation_prompt_mode"),
    }


def _log_request_trace(request: dict[str, Any], memory: dict[str, Any] | None) -> None:
    if not LOG_REQUESTS:
        return
    trace = _request_trace(request, memory)
    question = re.sub(r"\s+", " ", str(request.get("question") or "")).strip()
    if len(question) > 80:
        question = f"{question[:77]}..."
    print(
        "[BridgePRAG:request] "
        f"memory_active={trace['memory_active']} "
        f"passages={trace['passage_count']} "
        f"merged={trace['merged_count']} "
        f"layer={trace['critical_layer']} "
        f"num_kv={trace['num_kv']} "
        f"fusion={trace['question_fusion']} "
        f"injection={trace['injection_mode']} "
        f"prompt={trace['generation_prompt_mode']} "
        f"question={question!r}"
    )


def _clean_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL)
    text = re.sub(r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>", "", text)
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1]
    return text.strip()


def _clean_stream_token(text: str) -> str:
    return re.sub(r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>", "", str(text or ""))
