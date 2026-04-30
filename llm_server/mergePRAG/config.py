import json
import os


MODEL_NAME = os.getenv("MERGEPRAG_MODEL_NAME", "Qwen/Qwen3.5-4B")
NUM_KV = int(os.getenv("MERGEPRAG_NUM_KV", "16"))
DEFAULT_CRITICAL_LAYER = int(os.getenv("MERGEPRAG_DEFAULT_LAYER", "7"))
ALPHA = float(os.getenv("MERGEPRAG_ALPHA", "1.0"))
MAX_SEQ_LEN = 512
ENABLE_FOCUS_WEIGHT = os.getenv("MERGEPRAG_ENABLE_FOCUS_WEIGHT", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
FOCUS_WEIGHT_COLOR = float(os.getenv("MERGEPRAG_FOCUS_WEIGHT_COLOR", "3.0"))
FOCUS_WEIGHT_DAY = float(os.getenv("MERGEPRAG_FOCUS_WEIGHT_DAY", "3.0"))
FOCUS_WEIGHT_NUMBER = float(os.getenv("MERGEPRAG_FOCUS_WEIGHT_NUMBER", "2.2"))
FOCUS_WEIGHT_DATE = float(os.getenv("MERGEPRAG_FOCUS_WEIGHT_DATE", "2.4"))
FOCUS_WEIGHT_QUESTION_OVERLAP = float(os.getenv("MERGEPRAG_FOCUS_WEIGHT_QUESTION_OVERLAP", "1.8"))
FOCUS_WEIGHT_RARE = float(os.getenv("MERGEPRAG_FOCUS_WEIGHT_RARE", "1.3"))
FOCUS_WEIGHT_MAX = float(os.getenv("MERGEPRAG_FOCUS_WEIGHT_MAX", "12.0"))
SYSTEM_PROMPT = (
    "You are a helpful lecture assistant. "
    "Answer in Korean. 반드시 3문장 이내로 핵심만 답변해. "
    "불필요한 부연설명 하지 마."
)

_BASE_DIR = os.path.dirname(__file__)


def _resolve_optional_path(path_value: str, base_dir: str) -> str:
    """환경변수 경로를 절대경로로 정규화한다.

    - 절대경로면 그대로 사용
    - 상대경로면 mergePRAG 디렉터리 기준으로 해석
    """
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


CRITICAL_LAYERS_PATH = os.path.join(_BASE_DIR, "critical_layers.json")
WEIGHTS_PATH = _resolve_optional_path(
    os.getenv("MERGEPRAG_WEIGHTS_PATH", "hypernet_weights.pt"),
    _BASE_DIR,
)
CHECKPOINT_PATH = _resolve_optional_path(
    os.getenv("MERGEPRAG_CHECKPOINT_PATH", "hypernet_checkpoint.pt"),
    _BASE_DIR,
)
LOG_PATH = os.path.join(_BASE_DIR, "train_log.json")
CHART_PATH = os.path.join(_BASE_DIR, "train_loss_curve.png")
TRAIN_DATA_PATH = os.getenv(
    "MERGEPRAG_TRAIN_DATA_PATH",
    r"C:\Users\user\Documents\last_project\data\HotPot_train_processed.jsonl",
)
VALID_DATA_PATH = os.getenv(
    "MERGEPRAG_VALID_DATA_PATH",
    r"C:\Users\user\Documents\last_project\data\HotPot_valid_processed.jsonl",
)


def load_critical_layer() -> int:
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
    """최종 weight 우선, 없으면 중간 checkpoint의 hypernet state_dict와 메타정보를 반환."""
    if os.path.exists(WEIGHTS_PATH):
        state = torch_load(WEIGHTS_PATH, map_location=map_location)
        return state, {
            "source": WEIGHTS_PATH,
            "kind": "weights",
            "step": None,
        }

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch_load(CHECKPOINT_PATH, map_location=map_location)
        if isinstance(ckpt, dict) and "hypernet" in ckpt:
            return ckpt["hypernet"], {
                "source": CHECKPOINT_PATH,
                "kind": "checkpoint",
                "step": ckpt.get("step"),
            }
        raise ValueError(f"Checkpoint format invalid: {CHECKPOINT_PATH}")

    raise FileNotFoundError(
        f"Neither hypernet weights nor checkpoint found: {WEIGHTS_PATH}, {CHECKPOINT_PATH}"
    )


def torch_load(path, map_location=None):
    import torch
    return torch.load(path, map_location=map_location)
