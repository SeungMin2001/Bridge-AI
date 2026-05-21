"""
MergePRAG K,V → LoRA 어댑터 변환 모듈

llm_server의 HyperNetwork가 추출한 K,V 텐서를
vLLM 호환 LoRA 어댑터로 변환한다.

수학적 근사:
  원본: δ = softmax(H·K^T / √d_k)·V  (동적, 비선형)
  근사: ΔW = α · K^T · V              (정적, 선형)
  분해: ΔW ≈ LoRA_B · LoRA_A  (SVD top-r)

이 모듈은 llm_server의 코드를 수정하지 않고,
CourseMemoryManager에서 K,V를 받아서 변환만 한다.
"""

import json
import os
import shutil

import torch

# safetensors가 없으면 torch.save fallback
try:
    from safetensors.torch import save_file as safetensors_save
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# ── 기본 설정 ──
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B"
DEFAULT_ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "adapters")


def kv_to_delta_w(K: torch.Tensor, V: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """K,V 텐서를 정적 가중치 변화량 ΔW로 변환한다.

    Args:
        K: Key 텐서, shape [num_kv, d_model] 또는 [1, num_kv, d_model]
        V: Value 텐서, shape [num_kv, d_model] 또는 [1, num_kv, d_model]
        alpha: 스케일링 계수

    Returns:
        ΔW: shape [d_model, d_model]
    """
    if K.dim() == 3:
        K = K.squeeze(0)
    if V.dim() == 3:
        V = V.squeeze(0)
    # ΔW = α · K^T · V = [d_model, num_kv] @ [num_kv, d_model] = [d_model, d_model]
    return alpha * (K.T.float() @ V.float())


def delta_w_to_lora(
    delta_w: torch.Tensor,
    rank: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """ΔW를 SVD 분해하여 LoRA A, B 행렬을 추출한다.

    Args:
        delta_w: [d_model, d_model] 가중치 변화량
        rank: LoRA rank (SVD top-r)

    Returns:
        lora_A: [rank, d_model]
        lora_B: [d_model, rank]
        energy_ratio: 상위 r개 특이값의 에너지 비율 (근사 품질 지표)
    """
    U, S, Vh = torch.linalg.svd(delta_w.float(), full_matrices=False)
    effective_rank = min(rank, S.shape[0])
    sqrt_S = torch.sqrt(S[:effective_rank])

    # LoRA_B = U[:, :r] · √S[:r]  → [d_model, rank]
    lora_B = U[:, :effective_rank] * sqrt_S.unsqueeze(0)
    # LoRA_A = √S[:r] · V^T[:r, :] → [rank, d_model]
    lora_A = sqrt_S.unsqueeze(1) * Vh[:effective_rank, :]

    total_energy = (S ** 2).sum().item()
    kept_energy = (S[:effective_rank] ** 2).sum().item()
    energy_ratio = kept_energy / (total_energy + 1e-10)

    return lora_A, lora_B, energy_ratio


def save_lora_adapter(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    target_layer: int,
    output_dir: str,
    base_model: str = DEFAULT_BASE_MODEL,
    rank: int = 16,
    lora_alpha: int = 16,
    target_module: str = "o_proj",
) -> str:
    """LoRA A,B를 PEFT 호환 어댑터 디렉토리로 저장한다.

    vLLM의 /v1/load_lora_adapter가 이 디렉토리를 로드할 수 있다.
    """
    os.makedirs(output_dir, exist_ok=True)

    # adapter_config.json
    config = {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "r": rank,
        "revision": None,
        "target_modules": [target_module],
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 가중치 (PEFT naming)
    key_prefix = f"base_model.model.model.layers.{target_layer}.self_attn.{target_module}"
    state_dict = {
        f"{key_prefix}.lora_A.weight": lora_A.contiguous().to(torch.float16),
        f"{key_prefix}.lora_B.weight": lora_B.contiguous().to(torch.float16),
    }

    if HAS_SAFETENSORS:
        safetensors_save(state_dict, os.path.join(output_dir, "adapter_model.safetensors"))
    else:
        torch.save(state_dict, os.path.join(output_dir, "adapter_model.bin"))

    return output_dir


def build_lora_adapter(
    K: torch.Tensor,
    V: torch.Tensor,
    alpha: float = 0.1,
    target_layer: int = 19,
    rank: int = 16,
    lora_alpha: int = 16,
    max_delta_norm: float | None = None,
    adapter_name: str = "mergeprag",
    adapter_root: str = DEFAULT_ADAPTER_DIR,
    base_model: str = DEFAULT_BASE_MODEL,
    target_module: str = "o_proj",
) -> dict:
    """K,V → LoRA 어댑터 전체 파이프라인 원콜 함수.

    Returns:
        dict: adapter_path, delta_w_norm, energy_ratio, rank 등
    """
    delta_w = kv_to_delta_w(K, V, alpha=alpha)
    delta_w_norm_raw = float(delta_w.norm())
    delta_w_scale = 1.0
    if max_delta_norm is not None and max_delta_norm > 0 and delta_w_norm_raw > max_delta_norm:
        delta_w_scale = max_delta_norm / (delta_w_norm_raw + 1e-8)
        delta_w = delta_w * delta_w_scale
    lora_A, lora_B, energy_ratio = delta_w_to_lora(delta_w, rank=rank)

    output_dir = os.path.join(adapter_root, adapter_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    save_lora_adapter(
        lora_A=lora_A, lora_B=lora_B,
        target_layer=target_layer, output_dir=output_dir,
        base_model=base_model, rank=rank,
        lora_alpha=lora_alpha, target_module=target_module,
    )

    return {
        "adapter_path": os.path.abspath(output_dir),
        "delta_w_norm": round(float(delta_w.norm()), 4),
        "delta_w_norm_raw": round(delta_w_norm_raw, 4),
        "delta_w_scale": round(delta_w_scale, 6),
        "energy_ratio": round(energy_ratio, 4),
        "rank": rank,
        "lora_A_shape": list(lora_A.shape),
        "lora_B_shape": list(lora_B.shape),
    }
