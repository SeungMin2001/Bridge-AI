import json
import os


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


MODEL_NAME = os.getenv("MERGEPRAG_MODEL_NAME", "Qwen/Qwen3.5-4B")
# 논문 기본: num_kv=1. slot 수를 늘려도 되지만 논문 재현은 1부터.
NUM_KV = int(os.getenv("MERGEPRAG_NUM_KV", "1"))
# 논문: single_layer=9 (Llama-3.1). Qwen의 경우 find_critical_layers.py 결과 사용.
DEFAULT_CRITICAL_LAYER = int(os.getenv("MERGEPRAG_DEFAULT_LAYER", "9"))
# 현재 Qwen/lecture QA 조건에서는 alpha=1.0가 hidden을 과도하게 덮어쓰는 경우가 많아
# 보수적으로 낮춘다. 필요시 환경변수로 다시 올릴 수 있다.
ALPHA = float(os.getenv("MERGEPRAG_ALPHA", "0.1"))
MAX_SEQ_LEN = 512
# 논문은 token embedding only를 썼지만, 현재처럼 역할이 뒤바뀐 near-counterfactual passage에서는
# 순서/구문 정보를 잃기 쉬워 contextual hidden이 더 안정적이다.
USE_CONTEXTUAL_PASSAGE_ENCODER = _get_bool("MERGEPRAG_USE_CONTEXTUAL_ENCODER", True)
USE_QUESTION_CONDITIONED_MEMORY = _get_bool("MERGEPRAG_USE_QUESTION_CONDITIONED_MEMORY", False)
# pooled_only는 현재 실험에서 pooled projection collapse를 그대로 증폭했다.
# 기본은 MLP 경로를 우선 사용하고, 필요할 때만 hybrid/pooled_only를 실험한다.
KV_PATH_MODE = os.getenv("MERGEPRAG_KV_PATH_MODE", "mlp_only").strip().lower()
USE_POOLED_KV_SKIP = _get_bool("MERGEPRAG_USE_POOLED_KV_SKIP", True)
POOLED_KV_SKIP_SCALE = float(os.getenv("MERGEPRAG_POOLED_KV_SKIP_SCALE", "1.0"))
POOLED_K_SKIP_SCALE = float(os.getenv("MERGEPRAG_POOLED_K_SKIP_SCALE", str(POOLED_KV_SKIP_SCALE)))
POOLED_V_SKIP_SCALE = float(os.getenv("MERGEPRAG_POOLED_V_SKIP_SCALE", str(POOLED_KV_SKIP_SCALE)))
SYSTEM_PROMPT = (
    "You are a helpful lecture assistant. "
    "Answer in Korean. 반드시 3문장 이내로 핵심만 답변해. "
    "불필요한 부연설명 하지 마."
)

_BASE_DIR = os.path.dirname(__file__)
CRITICAL_LAYERS_PATH = os.path.join(_BASE_DIR, "critical_layers.json")
WEIGHTS_PATH = os.path.join(_BASE_DIR, "hypernet_weights.pt")
CHECKPOINT_PATH = os.path.join(_BASE_DIR, "hypernet_checkpoint.pt")
LOG_PATH = os.path.join(_BASE_DIR, "train_log.json")
CHART_PATH = os.path.join(_BASE_DIR, "train_loss_curve.png")
TRAIN_DATA_PATH = os.getenv(
    "MERGEPRAG_TRAIN_DATA_PATH",
    r"C:\Users\user\Documents\last_project\data\SQuAD_train_processed.jsonl",
)
VALID_DATA_PATH = os.getenv(
    "MERGEPRAG_VALID_DATA_PATH",
    r"C:\Users\user\Documents\last_project\data\SQuAD_valid_processed.jsonl",
)


def load_critical_layer() -> int:
    env_override = os.getenv("MERGEPRAG_CRITICAL_LAYER")
    if env_override is not None:
        return int(env_override)

    if not os.path.exists(CRITICAL_LAYERS_PATH):
        return DEFAULT_CRITICAL_LAYER

    try:
        with open(CRITICAL_LAYERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        layers = data.get("critical_layers") or []
        if layers:
            return int(layers[0])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass

    return DEFAULT_CRITICAL_LAYER


def build_chat_text(tokenizer, question: str, answer: str = "", enable_thinking: bool = False) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if answer:
        messages.append({"role": "assistant", "content": answer})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not bool(answer),
        enable_thinking=enable_thinking,
    )


def load_hypernet_state_dict(map_location=None):
    """기본은 최신 파일을 사용한다.

    MERGEPRAG_LOAD_SOURCE:
      - latest (default): 수정 시각이 더 최신인 weights/checkpoint 사용
      - weights: hypernet_weights.pt 우선
      - checkpoint: hypernet_checkpoint.pt 우선
    """
    load_source = os.getenv("MERGEPRAG_LOAD_SOURCE", "latest").strip().lower()

    weights_exists = os.path.exists(WEIGHTS_PATH)
    checkpoint_exists = os.path.exists(CHECKPOINT_PATH)

    def load_weights():
        state = torch_load(WEIGHTS_PATH, map_location=map_location)
        return state, {
            "source": WEIGHTS_PATH,
            "kind": "weights",
            "step": None,
        }

    def load_checkpoint():
        ckpt = torch_load(CHECKPOINT_PATH, map_location=map_location)
        if isinstance(ckpt, dict) and "hypernet" in ckpt:
            return ckpt["hypernet"], {
                "source": CHECKPOINT_PATH,
                "kind": "checkpoint",
                "step": ckpt.get("step"),
            }
        raise ValueError(f"Checkpoint format invalid: {CHECKPOINT_PATH}")

    if load_source == "weights" and weights_exists:
        return load_weights()
    if load_source == "checkpoint" and checkpoint_exists:
        return load_checkpoint()

    if load_source == "latest":
        candidates = []
        if weights_exists:
            candidates.append(("weights", os.path.getmtime(WEIGHTS_PATH)))
        if checkpoint_exists:
            candidates.append(("checkpoint", os.path.getmtime(CHECKPOINT_PATH)))
        if candidates:
            latest_kind = max(candidates, key=lambda x: x[1])[0]
            if latest_kind == "checkpoint":
                return load_checkpoint()
            return load_weights()

    if weights_exists:
        return load_weights()
    if checkpoint_exists:
        return load_checkpoint()

    raise FileNotFoundError(
        f"Neither hypernet weights nor checkpoint found: {WEIGHTS_PATH}, {CHECKPOINT_PATH}"
    )


def torch_load(path, map_location=None):
    import torch
    return torch.load(path, map_location=map_location)
