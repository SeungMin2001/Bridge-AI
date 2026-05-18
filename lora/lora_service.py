"""
MergePRAG LoRA 변환 서비스 (포트 9001)

역할:
  기존 llm_server의 HyperNetwork(CourseMemoryManager)를 그대로 사용하여
  K,V를 추출한 뒤, LoRA 어댑터로 변환하고 vLLM에 핫로드한다.

  llm_server의 코드를 import 한다.

실행: cd lora && python lora_service.py
포트: 9001
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# llm_server 패키지를 import하기 위해 상위 경로 추가
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LLM_SERVER_DIR = os.path.join(_PROJECT_ROOT, "llm_server")
if _LLM_SERVER_DIR not in sys.path:
    sys.path.insert(0, _LLM_SERVER_DIR)

# llm_server의 기존 코드를 수정 없이 import
from run_model import run_model
from mergePRAG.config import ALPHA, NUM_KV, load_critical_layer
from mergePRAG.main import CourseMemoryManager

# lora 모듈 (같은 폴더)
from lora_converter import build_lora_adapter

# ── 환경변수 설정 ──
# vLLM Docker 서버 (LoRA 핫로드 대상)
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8001")
# 어댑터 저장 루트 (Docker 볼륨과 공유)
ADAPTER_ROOT = os.getenv("LORA_ADAPTER_ROOT", os.path.join(os.path.dirname(__file__), "adapters"))
# Docker 내부에서 보이는 어댑터 경로
ADAPTER_CONTAINER_ROOT = os.getenv("LORA_ADAPTER_CONTAINER_ROOT", "/lora/adapters")
# 서비스 포트
SERVICE_PORT = int(os.getenv("LORA_SERVICE_PORT", "9001"))

CRITICAL_LAYER = load_critical_layer()


@dataclass
class ConversionStats:
    """LoRA 변환/핫로드 통계 추적"""
    total_conversions: int = 0
    total_hotloads: int = 0
    failed_hotloads: int = 0
    last_conversion_time: float | None = None
    last_adapter_name: str | None = None
    errors: list[str] = field(default_factory=list)


stats = ConversionStats()

# 글로벌 (lifespan에서 초기화)
model = None
tokenizer = None
memory_manager: CourseMemoryManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서비스 시작 시 기존 llm_server의 모델 + HyperNetwork를 로드한다."""
    global model, tokenizer, memory_manager

    print("[LoRA Service] 모델 + HyperNetwork 로딩 중...")
    model, tokenizer = run_model()
    device = next(model.parameters()).device
    # llm_server의 CourseMemoryManager를 그대로 사용
    memory_manager = CourseMemoryManager(model, tokenizer, device)

    os.makedirs(ADAPTER_ROOT, exist_ok=True)
    print(
        f"[LoRA Service] 준비 완료\n"
        f"  포트: {SERVICE_PORT}\n"
        f"  vLLM: {VLLM_URL}\n"
        f"  어댑터 저장: {ADAPTER_ROOT}\n"
        f"  컨테이너 경로: {ADAPTER_CONTAINER_ROOT}\n"
        f"  critical_layer={CRITICAL_LAYER}, num_kv={NUM_KV}, alpha={ALPHA}"
    )
    yield


app = FastAPI(title="MergePRAG LoRA Converter", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청 스키마 ──
class InjectRequest(BaseModel):
    """텍스트 → K,V → LoRA → vLLM 핫로드"""
    course_id: str
    passage: str


class InjectBatchRequest(BaseModel):
    """여러 passage 일괄 주입"""
    course_id: str
    passages: list[str]


class ConvertAndLoadRequest(BaseModel):
    """기존 메모리를 LoRA로 변환+로드만"""
    course_id: str


class MemoryClearRequest(BaseModel):
    course_id: str


# ── vLLM 핫로드 ──
async def hotload_to_vllm(adapter_name: str, adapter_path: str) -> dict:
    """vLLM의 /v1/load_lora_adapter API로 어댑터를 핫로드한다."""
    relative = os.path.relpath(adapter_path, ADAPTER_ROOT)
    container_path = os.path.join(ADAPTER_CONTAINER_ROOT, relative).replace("\\", "/")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            resp = await client.post(
                f"{VLLM_URL}/v1/load_lora_adapter",
                json={"lora_name": adapter_name, "lora_path": container_path},
            )
            resp.raise_for_status()
            stats.total_hotloads += 1
            stats.last_adapter_name = adapter_name
            print(f"[LoRA Service] vLLM 핫로드 성공: {adapter_name}")
            return {"success": True, "adapter_name": adapter_name, "container_path": container_path}
    except Exception as e:
        stats.failed_hotloads += 1
        stats.errors.append(f"{time.strftime('%H:%M:%S')} hotload: {e}")
        print(f"[LoRA Service] vLLM 핫로드 실패: {e}")
        return {"success": False, "error": str(e)}


def _convert_and_save(course_id: str) -> dict | None:
    """현재 메모리의 K,V를 LoRA로 변환하고 저장한다."""
    K, V = memory_manager.get_memory(course_id)
    if K is None:
        return None
    return build_lora_adapter(
        K=K, V=V,
        alpha=ALPHA,
        target_layer=CRITICAL_LAYER,
        rank=NUM_KV,
        adapter_name=f"mergeprag-{course_id}",
        adapter_root=ADAPTER_ROOT,
    )


# ── 엔드포인트 ──
@app.post("/inject")
async def inject_passage(req: InjectRequest):
    """텍스트 → HyperNetwork K,V → LoRA 변환 → vLLM 핫로드."""
    count = memory_manager.add_passage(req.course_id, req.passage)

    result = _convert_and_save(req.course_id)
    if result is None:
        return {"error": "K,V extraction failed"}

    hotload_result = await hotload_to_vllm(
        f"mergeprag-{req.course_id}", result["adapter_path"],
    )

    stats.total_conversions += 1
    stats.last_conversion_time = time.time()

    return {
        "course_id": req.course_id,
        "passage_count": count,
        "lora": result,
        "hotload": hotload_result,
    }


@app.post("/inject-batch")
async def inject_batch(req: InjectBatchRequest):
    """여러 passage를 한번에 주입하고 최종 LoRA를 핫로드한다."""
    for p in req.passages:
        memory_manager.add_passage(req.course_id, p)

    result = _convert_and_save(req.course_id)
    if result is None:
        return {"error": "no memory after injection"}

    hotload_result = await hotload_to_vllm(
        f"mergeprag-{req.course_id}", result["adapter_path"],
    )

    stats.total_conversions += 1
    stats.last_conversion_time = time.time()

    return {
        "course_id": req.course_id,
        "passage_count": memory_manager.memories[req.course_id]["count"],
        "lora": result,
        "hotload": hotload_result,
    }


@app.post("/convert")
async def convert_only(req: ConvertAndLoadRequest):
    """이미 메모리에 있는 K,V를 LoRA로 변환+로드만 한다."""
    result = _convert_and_save(req.course_id)
    if result is None:
        return {"error": f"course '{req.course_id}' has no memory"}

    hotload_result = await hotload_to_vllm(
        f"mergeprag-{req.course_id}", result["adapter_path"],
    )
    return {"lora": result, "hotload": hotload_result}


@app.get("/memory/list")
async def memory_list():
    """과목별 메모리 현황."""
    return {"memories": memory_manager.list_courses()}


@app.post("/memory/clear")
async def memory_clear(req: MemoryClearRequest):
    """과목 메모리 초기화."""
    memory_manager.clear_memory(req.course_id)
    return {"course_id": req.course_id, "cleared": True}


@app.get("/stats")
async def get_stats():
    """변환/핫로드 통계."""
    return {
        "total_conversions": stats.total_conversions,
        "total_hotloads": stats.total_hotloads,
        "failed_hotloads": stats.failed_hotloads,
        "last_conversion_time": stats.last_conversion_time,
        "last_adapter_name": stats.last_adapter_name,
        "recent_errors": stats.errors[-5:],
        "critical_layer": CRITICAL_LAYER,
        "num_kv": NUM_KV,
        "alpha": ALPHA,
        "vllm_url": VLLM_URL,
    }


@app.get("/health")
async def health():
    """헬스체크."""
    return {
        "status": "ok",
        "service": "mergeprag-lora-converter",
        "port": SERVICE_PORT,
        "model_loaded": model is not None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
