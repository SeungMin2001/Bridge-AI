import json
import os


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


MODEL_NAME = os.getenv("MERGEPRAG_MODEL_NAME", "Qwen/Qwen3.5-4B")
# num_kv=1은 K가 softmax 선택 역할을 못 해서 V collapse에 취약했다.
# 현재 기본은 V 분리와 slot 선택을 같이 보기 위해 4로 둔다.
NUM_KV = _get_int("MERGEPRAG_NUM_KV", 4)
# 논문: single_layer=9 (Llama-3.1). Qwen의 경우 find_critical_layers.py 결과 사용.
DEFAULT_CRITICAL_LAYER = _get_int("MERGEPRAG_DEFAULT_LAYER", 9)
# 현재 Qwen/lecture QA 조건에서는 alpha=1.0가 hidden을 과도하게 덮어쓰는 경우가 많아
# 보수적으로 낮춘다. 필요시 환경변수로 다시 올릴 수 있다.
ALPHA = _get_float("MERGEPRAG_ALPHA", 0.1)
MAX_SEQ_LEN = _get_int("MERGEPRAG_MAX_SEQ_LEN", 512)
# 논문은 token embedding only를 썼지만, 현재처럼 역할이 뒤바뀐 near-counterfactual passage에서는
# 순서/구문 정보를 잃기 쉬워 contextual hidden이 더 안정적이다.
USE_CONTEXTUAL_PASSAGE_ENCODER = _get_bool("MERGEPRAG_USE_CONTEXTUAL_ENCODER", True)
USE_QUESTION_CONDITIONED_MEMORY = _get_bool("MERGEPRAG_USE_QUESTION_CONDITIONED_MEMORY", True)
QUERY_POOL_SCALE = _get_float("MERGEPRAG_QUERY_POOL_SCALE", 4.0)
MEMORY_ENCODER_INSTRUCTION = os.getenv(
    "MERGEPRAG_MEMORY_ENCODER_INSTRUCTION",
    (
        "Memory task: encode the passage for answering the question. "
        "Preserve who did what to whom, comparison direction, numbers, dates, "
        "negation, and the exact entity that answers the question."
    ),
)
# K MLP가 near-counterfactual 차이를 다시 뭉개는 경우가 있어 K에도 pooled skip을 섞는다.
# V는 passage pooled 정보를 직접 싣도록 skip 경로를 기본으로 둔다.
KV_PATH_MODE = os.getenv("MERGEPRAG_KV_PATH_MODE", "k_hybrid_v_skip").strip().lower()
USE_POOLED_KV_SKIP = _get_bool("MERGEPRAG_USE_POOLED_KV_SKIP", True)
POOLED_KV_SKIP_SCALE = _get_float("MERGEPRAG_POOLED_KV_SKIP_SCALE", 1.0)
POOLED_K_SKIP_SCALE = _get_float("MERGEPRAG_POOLED_K_SKIP_SCALE", POOLED_KV_SKIP_SCALE)
POOLED_V_SKIP_SCALE = _get_float("MERGEPRAG_POOLED_V_SKIP_SCALE", 1.0)
USE_V_RMS_CLAMP = _get_bool("MERGEPRAG_USE_V_RMS_CLAMP", True)
V_RMS_CLAMP = _get_float("MERGEPRAG_V_RMS_CLAMP", 0.25)
USE_K_RMS_CLAMP = _get_bool("MERGEPRAG_USE_K_RMS_CLAMP", True)
K_RMS_CLAMP = _get_float("MERGEPRAG_K_RMS_CLAMP", 0.25)
# plain: 논문/기존 실험 형식 "Question: ...\nAnswer:"
# chat: 실제 서비스 API와 같은 chat template 형식. lecture-domain 재학습 때 권장.
TRAIN_PROMPT_FORMAT = os.getenv("MERGEPRAG_TRAIN_PROMPT_FORMAT", "chat").strip().lower()

# 학습 목적 함수 설정. SQuAD 같은 쉬운 global negative만으로는 passage flip을 못 배우므로
# hard negative/contrastive 데이터에서는 아래 loss가 실제 grounding 방향을 잡아준다.
NEGATIVE_MARGIN = _get_float("MERGEPRAG_NEGATIVE_MARGIN", 0.2)
NEGATIVE_LOSS_WEIGHT = _get_float("MERGEPRAG_NEGATIVE_LOSS_WEIGHT", 0.25)
REPULSION_LOSS_WEIGHT = _get_float("MERGEPRAG_REPULSION_LOSS_WEIGHT", 1.0)
HIDDEN_SIM_TARGET = _get_float("MERGEPRAG_HIDDEN_SIM_TARGET", 0.97)
K_SIM_TARGET = _get_float("MERGEPRAG_K_SIM_TARGET", 0.95)
V_SIM_TARGET = _get_float("MERGEPRAG_V_SIM_TARGET", 0.65)
V_REPULSION_MULTIPLIER = _get_float("MERGEPRAG_V_REPULSION_MULTIPLIER", 4.0)
QUESTION_REPULSION_LOSS_WEIGHT = _get_float("MERGEPRAG_QUESTION_REPULSION_LOSS_WEIGHT", 1.25)
QUESTION_NEGATIVE_LOSS_WEIGHT = _get_float("MERGEPRAG_QUESTION_NEGATIVE_LOSS_WEIGHT", 0.50)
SLOT_DIVERSITY_LOSS_WEIGHT = _get_float("MERGEPRAG_SLOT_DIVERSITY_LOSS_WEIGHT", 0.1)
SLOT_DIVERSITY_TARGET = _get_float("MERGEPRAG_SLOT_DIVERSITY_TARGET", 0.5)

# 설정이 바뀐 채 예전 checkpoint를 자동 재개하면 collapse 원인 분석이 꼬인다.
# 새 checkpoint에는 config snapshot을 저장하고, legacy checkpoint는 명시적으로 허용할 때만 재개한다.
ALLOW_LEGACY_CHECKPOINT_RESUME = _get_bool("MERGEPRAG_ALLOW_LEGACY_CHECKPOINT_RESUME", False)
ALLOW_CONFIG_MISMATCH_RESUME = _get_bool("MERGEPRAG_ALLOW_CONFIG_MISMATCH_RESUME", False)
SYSTEM_PROMPT = os.getenv(
    "MERGEPRAG_SYSTEM_PROMPT",
    (
        "You are a helpful lecture assistant. "
        "Answer in the same language as the user's question. "
        "Use only the provided lecture content as the grounding source, "
        "and keep the answer concise."
    ),
)

_BASE_DIR = os.path.dirname(__file__)
DEFAULT_DATA_DIR = os.getenv("MERGEPRAG_DATA_DIR", r"C:\Users\user\Documents\last_project\data")
CRITICAL_LAYERS_PATH = os.path.join(_BASE_DIR, "critical_layers.json")
WEIGHTS_PATH = os.path.join(_BASE_DIR, "hypernet_weights.pt")
CHECKPOINT_PATH = os.path.join(_BASE_DIR, "hypernet_checkpoint.pt")
LOG_PATH = os.path.join(_BASE_DIR, "train_log.json")
CHART_PATH = os.path.join(_BASE_DIR, "train_loss_curve.png")
TRAIN_DATA_PATH = os.getenv(
    "MERGEPRAG_TRAIN_DATA_PATH",
    os.path.join(DEFAULT_DATA_DIR, "ServiceHardPair_train.jsonl"),
)
VALID_DATA_PATH = os.getenv(
    "MERGEPRAG_VALID_DATA_PATH",
    os.path.join(DEFAULT_DATA_DIR, "ServiceHardPair_valid.jsonl"),
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
        if isinstance(state, dict) and "hypernet" in state:
            return state["hypernet"], {
                "source": WEIGHTS_PATH,
                "kind": "weights",
                "step": state.get("step"),
                "config": state.get("config"),
            }
        return state, {
            "source": WEIGHTS_PATH,
            "kind": "weights",
            "step": None,
            "config": None,
        }

    def load_checkpoint():
        ckpt = torch_load(CHECKPOINT_PATH, map_location=map_location)
        if isinstance(ckpt, dict) and "hypernet" in ckpt:
            return ckpt["hypernet"], {
                "source": CHECKPOINT_PATH,
                "kind": "checkpoint",
                "step": ckpt.get("step"),
                "config": ckpt.get("config"),
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
