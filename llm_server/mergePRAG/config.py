import json
import os


MODEL_NAME = "Qwen/Qwen3.5-4B"
NUM_KV = 1
DEFAULT_CRITICAL_LAYER = 0
ALPHA = 0.01
MAX_SEQ_LEN = 512
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
