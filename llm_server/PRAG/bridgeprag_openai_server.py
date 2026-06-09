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
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

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
SERVICE_DEFAULT_ALPHA = 0.0
SERVICE_REPETITION_PENALTY = float(os.getenv("BRIDGEPRAG_REPETITION_PENALTY", "1.03"))
SERVICE_NO_REPEAT_NGRAM_SIZE = int(os.getenv("BRIDGEPRAG_NO_REPEAT_NGRAM_SIZE", "0"))
SERVICE_MAX_ANSWER_CHARS = 700
SERVICE_MAX_ANSWER_SENTENCES = 3
DTYPE = os.getenv("BRIDGEPRAG_DTYPE", "float16").strip().lower()
STRICT_MODEL_ID = os.getenv("BRIDGEPRAG_STRICT_MODEL_ID", "0").strip().lower() in {"1", "true", "yes", "on"}
LOG_REQUESTS = os.getenv("BRIDGEPRAG_LOG_REQUESTS", "1").strip().lower() in {"1", "true", "yes", "on"}
DEMO_PIPELINE_LOG = True


model = None
tokenizer = None
hypernet = None
device = None
target_layer = None
runtime_config: dict[str, Any] = {}
generation_lock = Lock()


def _demo_log(stage: str, message: str) -> None:
    if DEMO_PIPELINE_LOG:
        print(f"[DEMO:{stage}] {message}", flush=True)


def _preview(text: Any, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    return compact if len(compact) <= limit else f"{compact[:limit - 3]}..."


def _tensor_shape(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "unknown"
    return "x".join(str(part) for part in tuple(shape))


def _tensor_preview(value: Any, limit: int = 6) -> str:
    try:
        tensor = value.detach().float().flatten()[:limit].cpu().tolist()
    except Exception:
        return "[]"
    return "[" + ", ".join(f"{item:.4f}" for item in tensor) + "]"


class ServiceStopCriteria(StoppingCriteria):
    """서비스 답변이 반복/라벨/길이 제한에 도달하면 실제 generation을 중단합니다."""

    def __init__(self, tokenizer, prompt_len: int, stop_sequences: list[str]):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.stop_sequences = stop_sequences

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0, self.prompt_len:]
        if generated_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        _visible_text, stopped = _clean_generated_prefix(text, self.stop_sequences)
        return bool(stopped)


class JsonCompletionStopCriteria(StoppingCriteria):
    """Top-level JSON array/object가 완성되면 generation을 즉시 중단합니다."""

    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0, self.prompt_len:]
        if generated_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return _has_complete_top_level_json(text)


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


def _runtime_alpha(config: dict[str, Any]) -> float:
    """Use PRAG as a service-side helper rather than the dominant signal.

    The checkpoint was trained/evaluated with alpha=1.0, but the product path
    also passes retrieved text through the prompt. A lower default keeps RAG
    text as the main grounding source while still allowing K/V memory to nudge
    generation. BRIDGEPRAG_ALPHA can still override this for ablations.
    """
    override = os.getenv("BRIDGEPRAG_ALPHA")
    if override not in (None, ""):
        return float(override)
    return SERVICE_DEFAULT_ALPHA


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
        "alpha": _runtime_alpha(config),
        "checkpoint_alpha": float(config.get("alpha", 1.0)),
        "device": str(device),
    }
    print(f"[BridgePRAG] ready {_public_runtime_config()}")
    yield


app = FastAPI(lifespan=lifespan)


def _public_runtime_config() -> dict[str, Any]:
    return {
        key: value
        for key, value in runtime_config.items()
        if key not in {"alpha", "checkpoint_alpha"}
    }


@app.get("/health")
async def health():
    return {"status": "ok", **_public_runtime_config()}


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
    request = _build_request(messages, max_tokens=max_tokens, payload=payload)
    request["alpha"] = _payload_alpha(payload)

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


def _payload_alpha(payload: dict[str, Any]) -> float | None:
    value = payload.get("bridgeprag_alpha")
    if value is None:
        value = payload.get("alpha")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="bridgeprag_alpha must be a float")


def _build_request(messages: list[dict[str, Any]], *, max_tokens: int, payload: dict[str, Any]) -> dict[str, Any]:
    system_texts = [str(item.get("content") or "") for item in messages if item.get("role") == "system"]
    user_texts = [str(item.get("content") or "") for item in messages if item.get("role") == "user"]
    user_prompt = user_texts[-1] if user_texts else str(messages[-1].get("content") or "")
    question = _extract_question(user_prompt)
    passages = _extract_passages(user_prompt)[:MAX_PASSAGES]

    disable_bridgeprag_memory = bool(payload.get("bridgeprag_disable_memory"))
    alpha = _payload_alpha(payload)
    if alpha is None:
        alpha = float(runtime_config.get("alpha", 1.0))
    memory_active = bool(passages) and not disable_bridgeprag_memory and alpha > 0.0

    generation_text = _build_generation_text(
        system_texts,
        user_prompt,
        question,
        has_memory=bool(passages),
        memory_active=memory_active,
        messages=messages,
    )
    return {
        "question": question,
        "passages": passages,
        "generation_text": generation_text,
        "max_tokens": max_tokens,
        "disable_bridgeprag_memory": disable_bridgeprag_memory,
        "demo_feature": str(payload.get("demo_feature") or payload.get("feature") or "chat"),
        "stop": _payload_stop_sequences(payload),
        "repetition_penalty": _payload_float(payload, "repetition_penalty", SERVICE_REPETITION_PENALTY),
        "no_repeat_ngram_size": _payload_int(payload, "no_repeat_ngram_size", SERVICE_NO_REPEAT_NGRAM_SIZE),
        "json_mode": _payload_wants_json(payload, messages),
        "reference_prompt": "[검색된 참고자료]" in user_prompt,
    }


def _payload_wants_json(payload: dict[str, Any], messages: list[dict[str, Any]]) -> bool:
    """퀴즈/요약 저장용 JSON 생성은 채팅용 길이 제한 후처리에서 제외합니다."""
    response_format = payload.get("response_format")
    if isinstance(response_format, dict) and "json" in str(response_format.get("type", "")).lower():
        return True
    # summary/quiz feature는 항상 JSON 응답을 기대합니다.
    feature = str(payload.get("demo_feature") or payload.get("feature") or "").lower()
    if feature in {"summary", "quiz"}:
        return True
    joined = "\n".join(str(item.get("content") or "") for item in messages)
    if "json" in joined.lower() and ("JSON" in joined or "json" in joined.lower()):
        return True
    # 프롬프트에 JSON 형식 지시가 포함된 경우도 감지합니다.
    if '"summary_text"' in joined or "summary_text" in joined:
        return True
    return False


def _payload_stop_sequences(payload: dict[str, Any]) -> list[str]:
    """OpenAI 호환 stop 값을 서비스 기본 stop 문자열과 합칩니다."""
    defaults = [
        "\n질문:",
        "\nQuestion:",
        "\n사용자:",
        "\nUser:",
        "\n학생:",
        "\n답변:",
        "\n[검색된 참고자료]",
    ]
    value = payload.get("stop")
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = [str(item) for item in value if item]
    else:
        candidates = []

    merged: list[str] = []
    for item in [*defaults, *candidates]:
        if item and item not in merged:
            merged.append(item)
    return merged


def _payload_float(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"{key} must be a float")


def _payload_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"{key} must be an integer")


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
        passage = _normalize_reference_passage_for_memory(passage)
        if passage and passage not in passages:
            passages.append(passage)

    if not passages:
        for line in section.splitlines():
            line = line.strip()
            match = re.match(r"^\[\d+\]\s*(.+?)(?:\s*\(출처:\s*.*?\))?$", line)
            if match:
                passage = match.group(1).strip()
                passage = _normalize_reference_passage_for_memory(passage)
                if passage and passage not in passages:
                    passages.append(passage)
    return passages


def _normalize_reference_passage_for_memory(passage: str) -> str:
    """LLM 프롬프트용 라벨을 제거하고 K/V 메모리에는 핵심 근거 문장만 넣습니다."""
    value = str(passage or "").strip()
    if not value:
        return ""

    core_marker = "핵심 참고문장"
    support_marker = "보조 문맥:"
    if core_marker in value:
        core_part = value.split(core_marker, 1)[1]
        if ":" in core_part[:40]:
            core_part = core_part.split(":", 1)[1]
        if support_marker in core_part:
            core_part = core_part.split(support_marker, 1)[0]
        core_lines = []
        for line in core_part.splitlines():
            clean = re.sub(r"^\s*[-•]\s*", "", line).strip()
            if clean:
                core_lines.append(clean)
        if core_lines:
            return " ".join(core_lines)

    if support_marker in value:
        value = value.split(support_marker, 1)[1].strip()
    value = re.sub(r"\s*\(출처:\s*.*?\)\s*$", "", value, flags=re.DOTALL).strip()
    value = re.sub(r"^\s*핵심\s+참고문장.*?:\s*", "", value, flags=re.DOTALL).strip()
    return value


def _build_generation_text(
    system_texts: list[str],
    user_prompt: str,
    question: str,
    *,
    has_memory: bool,
    memory_active: bool,
    messages: list[dict[str, Any]],
) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            pass

    content = "\n".join(part for part in [*system_texts, user_prompt] if part).strip()
    # PRAG 메모리가 활성화되어도 검색 근거와 답변 규칙은 텍스트 프롬프트에 남겨 둡니다.
    # K/V 메모리는 보조 신호이고, 최종 답변의 사실 근거는 RAG 원문이 직접 통제해야 합니다.
    return build_chat_prompt(tokenizer, content)


def _encode_memory_for_request(request: dict[str, Any]):
    if request.get("disable_bridgeprag_memory"):
        _demo_log(
            "LLM",
            f"일반 생성 경로 사용: feature={request.get('demo_feature', 'unknown')}, PRAG 주입 경로 미사용",
        )
        return None
    if not _request_uses_memory(request):
        _demo_log(
            "BRIDGEPRAG",
            f"K/V 메모리 생성 생략: passages={len(request.get('passages') or [])}",
        )
        return None
    passages = request["passages"]
    if not passages:
        return None
    _demo_log(
        "BRIDGEPRAG",
        f"1) Question-Passage pair 구성: question='{_preview(request.get('question'), 80)}', passages={len(passages)}",
    )
    for index, passage in enumerate(passages[:MAX_PASSAGES], start=1):
        _demo_log("BRIDGEPRAG", f"   pair#{index}: {_preview(passage, 130)}")
    _demo_log(
        "BRIDGEPRAG",
        f"2) HyperNetwork 인코딩 시작: fusion={runtime_config.get('question_fusion')}, num_kv={runtime_config.get('num_kv')}",
    )
    _demo_log("BRIDGEPRAG", "   Question+Context 임베딩 및 Attention Pooling 수행")
    _demo_log("BRIDGEPRAG", "   MLP 변환 시작: Linear -> GELU -> LayerNorm -> Linear -> GELU")
    memory = encode_merged_memory(
        model,
        tokenizer,
        hypernet,
        passages,
        device,
        question=request["question"],
        question_conditioned=bool(runtime_config.get("question_conditioned_memory", True)),
    )
    _demo_log(
        "BRIDGEPRAG",
        "3) K/V 메모리 생성 완료: "
        f"K_shape={_tensor_shape(memory.get('K'))}, V_shape={_tensor_shape(memory.get('V'))}, "
        f"merge={memory.get('merge_mode', 'orthogonal')}, merged={memory.get('merged_count', len(passages))}",
    )
    _demo_log("BRIDGEPRAG", f"   K preview={_tensor_preview(memory.get('K'))}")
    _demo_log("BRIDGEPRAG", f"   V preview={_tensor_preview(memory.get('V'))}")
    _demo_log(
        "BRIDGEPRAG",
        f"   Orthogonal Merge 완료: merge={memory.get('merge_mode', 'orthogonal')}, merged={memory.get('merged_count', len(passages))}",
    )
    return memory


def _generation_kwargs(request: dict[str, Any], streamer: TextIteratorStreamer | None = None, cancelled: list[bool] | None = None) -> dict[str, Any]:
    inputs = tokenizer(
        request["generation_text"],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(device)
    generation_config = deterministic_generation_config(tokenizer, request["max_tokens"])
    generation_config.repetition_penalty = float(request.get("repetition_penalty") or SERVICE_REPETITION_PENALTY)
    generation_config.no_repeat_ngram_size = int(request.get("no_repeat_ngram_size") or SERVICE_NO_REPEAT_NGRAM_SIZE)
    prompt_len = int(inputs["input_ids"].shape[1])

    stopping_criteria = []
    if request.get("json_mode"):
        stopping_criteria.append(JsonCompletionStopCriteria(tokenizer, prompt_len))
    else:
        stopping_criteria.append(ServiceStopCriteria(tokenizer, prompt_len, request.get("stop") or []))

    if cancelled is not None:
        class CancelStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return bool(cancelled[0])
        stopping_criteria.append(CancelStoppingCriteria())

    kwargs = {
        **inputs,
        "generation_config": generation_config,
        "stopping_criteria": StoppingCriteriaList(stopping_criteria),
    }
    if streamer is not None:
        kwargs["streamer"] = streamer
    return kwargs


def _generate_text(request: dict[str, Any]) -> str:
    with generation_lock, torch.no_grad():
        memory = _encode_memory_for_request(request)
        _log_request_trace(request, memory)
        hook = None if request.get("disable_bridgeprag_memory") else _register_memory_hook(memory, request.get("alpha"))
        kwargs = _generation_kwargs(request)
        _demo_log("BRIDGEPRAG", f"5) 답변 생성 시작: max_new_tokens={request.get('max_tokens')}")
        try:
            generated = model.generate(**kwargs)
        finally:
            if hook is not None:
                hook.remove()
        prompt_len = kwargs["input_ids"].shape[1]
        text = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
        cleaned = _clean_output(text, request.get("stop") or [], json_mode=bool(request.get("json_mode")))
        _demo_log("BRIDGEPRAG", f"6) 답변 생성 완료: chars={len(cleaned)}")
        return cleaned


def _stream_openai_chunks(request: dict[str, Any]):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    done_payload = "data: [DONE]\n\n"
    cancelled = [False]

    with generation_lock, torch.no_grad():
        memory = _encode_memory_for_request(request)
        _log_request_trace(request, memory)
        hook = None if request.get("disable_bridgeprag_memory") else _register_memory_hook(memory, request.get("alpha"))
        _demo_log("BRIDGEPRAG", f"5) 스트리밍 생성 시작: max_new_tokens={request.get('max_tokens')}")
        def _worker():
            with torch.no_grad():
                try:
                    model.generate(**_generation_kwargs(request, streamer=streamer, cancelled=cancelled))
                except Exception as e:
                    print(f"[BridgePRAG:worker] generate error: {e}", flush=True)

        thread = Thread(target=_worker)
        thread.start()
        try:
            generated_text = ""
            emitted_text = ""
            for token in streamer:
                cleaned = _clean_stream_token(token)
                if not cleaned:
                    continue
                if request.get("json_mode"):
                    chunk = {
                        "id": f"chatcmpl-bridgeprag-{int(time.time() * 1000)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {"content": cleaned}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    continue
                generated_text += cleaned
                visible_text, should_stop = _clean_generated_prefix(generated_text, request.get("stop") or [])
                if len(visible_text) <= len(emitted_text):
                    if should_stop:
                        break
                    continue
                delta = visible_text[len(emitted_text):]
                emitted_text = visible_text
                chunk = {
                    "id": f"chatcmpl-bridgeprag-{int(time.time() * 1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                if should_stop:
                    break
            thread.join()
        except GeneratorExit:
            cancelled[0] = True
            for _ in streamer:
                pass
            thread.join()
            raise
        except Exception:
            cancelled[0] = True
            for _ in streamer:
                pass
            thread.join()
            raise
        finally:
            if hook is not None:
                hook.remove()
        yield done_payload


def _register_memory_hook(memory: dict[str, Any] | None, request_alpha: float | None = None):
    if not memory:
        _demo_log("BRIDGEPRAG", "4) Attention 주입 생략: K/V 메모리 없음")
        return None
    alpha = float(runtime_config.get("alpha", 1.0) if request_alpha is None else request_alpha)
    if alpha <= 0.0:
        _demo_log("BRIDGEPRAG", "4) Attention 주입 생략: 주입 비활성 요청")
        return None
    _demo_log(
        "BRIDGEPRAG",
        f"4) Transformer layer {runtime_config.get('critical_layer')} Attention에 K/V 메모리 주입: mode={runtime_config.get('injection_mode')}",
    )
    return target_layer.register_forward_hook(
        make_memory_hook(
            memory["K"],
            memory["V"],
            model_num_heads(model),
            alpha=alpha,
            injection_mode=str(runtime_config.get("injection_mode", "attention")),
        )
    )


def _request_uses_memory(request: dict[str, Any]) -> bool:
    if request.get("disable_bridgeprag_memory"):
        return False
    if not request.get("passages"):
        return False
    alpha = float(runtime_config.get("alpha", 1.0) if request.get("alpha") is None else request.get("alpha"))
    return alpha > 0.0


def _request_trace(request: dict[str, Any], memory: dict[str, Any] | None = None) -> dict[str, Any]:
    passages = request.get("passages") or []
    return {
        "memory_active": bool(memory) and _request_uses_memory(request),
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
    if request.get("disable_bridgeprag_memory"):
        question = re.sub(r"\s+", " ", str(request.get("question") or "")).strip()
        if len(question) > 80:
            question = f"{question[:77]}..."
        print(
            "[LLM:request] "
            f"feature={request.get('demo_feature', 'unknown')} "
            f"memory_active=False "
            f"prompt={runtime_config.get('generation_prompt_mode')} "
            f"question={question!r}"
        )
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


def _clean_output(text: str, stop_sequences: list[str] | None = None, *, json_mode: bool = False) -> str:
    text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL)
    text = re.sub(r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>", "", text)
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1]
    if json_mode:
        return _clean_json_output(text)
    text, _ = _clean_generated_prefix(text, stop_sequences or [])
    return text.strip()


def _clean_stream_token(text: str) -> str:
    return re.sub(r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>", "", str(text or ""))


def _clean_json_output(text: str) -> str:
    """JSON 생성 요청은 내용 보존을 우선하고 chat template 잔여물만 제거합니다."""
    text = re.sub(r"^\s*(assistant|답변|Answer)\s*[:：]?\s*", "", str(text or ""), flags=re.IGNORECASE).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    first_array = text.find("[")
    first_object = text.find("{")
    starts = [idx for idx in (first_array, first_object) if idx >= 0]
    if starts:
        return text[min(starts):].strip()
    return text


def _has_complete_top_level_json(text: str) -> bool:
    """문자열 내부 괄호를 제외하고 최상위 JSON 닫힘 여부를 확인합니다."""
    text = str(text or "")
    start = None
    expected_closer = None
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"[": "]", "{": "}"}

    for index, char in enumerate(text):
        if start is None:
            if char in pairs:
                start = index
                expected_closer = pairs[char]
                stack.append(expected_closer)
            elif char.strip():
                # JSON 앞에 chat label 등이 붙는 경우는 조금 더 기다립니다.
                continue
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char in pairs:
            stack.append(pairs[char])
            continue
        if stack and char == stack[-1]:
            stack.pop()
            if not stack and char == expected_closer:
                trailing = text[index + 1:].strip()
                return not trailing

    return False


def _clean_generated_prefix(text: str, stop_sequences: list[str]) -> tuple[str, bool]:
    """생성 중 stop/반복/길이 조건을 적용한 화면용 prefix를 반환합니다."""
    text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL)
    text = re.sub(r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>", "", text)
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1]
    text = re.sub(r"^\s*(assistant|답변|Answer)\s*[:：]?\s*", "", text, flags=re.IGNORECASE)

    stopped = False
    for stop in stop_sequences or []:
        index = text.find(stop)
        if index >= 0:
            text = text[:index]
            stopped = True

    text, repeated = _truncate_before_repeated_sentence(text)
    stopped = stopped or repeated

    text, sentence_limited = _truncate_to_sentence_limit(text, SERVICE_MAX_ANSWER_SENTENCES)
    stopped = stopped or sentence_limited

    if len(text) > SERVICE_MAX_ANSWER_CHARS:
        text = _trim_to_char_budget(text, SERVICE_MAX_ANSWER_CHARS)
        stopped = True

    return text, stopped


_SENTENCE_END_RE = re.compile(r"(?:[.!?。？！]|다\.|요\.)(?:\s*(?:\[\d+\]|\d+))?(?:\s+|$)")


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            spans.append((start, end, sentence))
        start = end
    return spans


def _sentence_signature(sentence: str) -> str:
    signature = re.sub(r"(?:\[\d+\]|\b\d+\b)", "", sentence)
    signature = re.sub(r"[\s.!?。？！,，;:：]+", "", signature).lower()
    return signature


def _truncate_before_repeated_sentence(text: str) -> tuple[str, bool]:
    seen: set[str] = set()
    for start, _end, sentence in _sentence_spans(text):
        signature = _sentence_signature(sentence)
        if len(signature) < 12:
            continue
        if signature in seen:
            return text[:start].rstrip(), True
        seen.add(signature)
    return text, False


def _truncate_to_sentence_limit(text: str, max_sentences: int) -> tuple[str, bool]:
    if max_sentences <= 0:
        return text, False
    spans = _sentence_spans(text)
    if len(spans) <= max_sentences:
        return text, False
    return text[: spans[max_sentences - 1][1]].rstrip(), True


def _trim_to_char_budget(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text.strip()
    clipped = text[:max_chars].rstrip()
    sentence_ends = [match.end() for match in _SENTENCE_END_RE.finditer(clipped)]
    if sentence_ends:
        clipped = clipped[: sentence_ends[-1]].rstrip()
    return clipped.rstrip(" ,;:：-")
