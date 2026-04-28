"""Configuration for the clean PRAG service-memory implementation."""

from __future__ import annotations

import json
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
DATA_DIR = Path(os.getenv("PRAG_DATA_DIR", str(PROJECT_ROOT / "data")))

MODEL_NAME = os.getenv("PRAG_MODEL_NAME", os.getenv("MERGEPRAG_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"))
AUGMENT_MODEL_NAME = os.getenv("PRAG_AUGMENT_MODEL_NAME", MODEL_NAME)

SOURCE_DATA_PATH = Path(os.getenv("PRAG_SOURCE_DATA_PATH", str(DATA_DIR / "ServiceHardPair_train.jsonl")))
RAW_PASSAGES_PATH = Path(os.getenv("PRAG_RAW_PASSAGES_PATH", str(DATA_DIR / "PRAG_raw_passages.jsonl")))
AUGMENTED_TRAIN_PATH = Path(os.getenv("PRAG_AUGMENTED_TRAIN_PATH", str(DATA_DIR / "PRAG_augmented_train.jsonl")))
AUGMENTED_VALID_PATH = Path(os.getenv("PRAG_AUGMENTED_VALID_PATH", str(DATA_DIR / "PRAG_augmented_valid.jsonl")))

CHECKPOINT_PATH = Path(os.getenv("PRAG_CHECKPOINT_PATH", str(BASE_DIR / "prag_memory_checkpoint.pt")))
WEIGHTS_PATH = Path(os.getenv("PRAG_WEIGHTS_PATH", str(BASE_DIR / "prag_memory_weights.pt")))
LOG_PATH = Path(os.getenv("PRAG_LOG_PATH", str(BASE_DIR / "prag_train_log.json")))
CRITICAL_LAYERS_PATH = Path(os.getenv("PRAG_CRITICAL_LAYERS_PATH", str(BASE_DIR / "critical_layers.json")))

DEFAULT_CRITICAL_LAYER = int(os.getenv("PRAG_DEFAULT_LAYER", "19"))
NUM_KV = int(os.getenv("PRAG_NUM_KV", "8"))
HIDDEN_DIM = int(os.getenv("PRAG_HIDDEN_DIM", "1024"))
ALPHA = float(os.getenv("PRAG_ALPHA", "0.3"))
MAX_MEMORY_TOKENS = int(os.getenv("PRAG_MAX_MEMORY_TOKENS", "256"))
MAX_SEQ_LEN = int(os.getenv("PRAG_MAX_SEQ_LEN", "512"))
RANK_MARGIN = float(os.getenv("PRAG_RANK_MARGIN", "0.5"))
RANK_WEIGHT = float(os.getenv("PRAG_RANK_WEIGHT", "1.0"))
LR = float(os.getenv("PRAG_LR", "8e-5"))
LR_MIN = float(os.getenv("PRAG_LR_MIN", "1e-6"))
EPOCHS = int(os.getenv("PRAG_EPOCHS", "1"))
LOG_EVERY = int(os.getenv("PRAG_LOG_EVERY", "25"))
SAVE_EVERY = int(os.getenv("PRAG_SAVE_EVERY", "250"))
EVAL_EVERY = int(os.getenv("PRAG_EVAL_EVERY", "250"))
EVAL_MAX_SAMPLES = int(os.getenv("PRAG_EVAL_MAX_SAMPLES", "120"))


def contains_hangul(text: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in str(text or ""))


def load_critical_layer() -> int:
    env_layer = os.getenv("PRAG_CRITICAL_LAYER")
    if env_layer is not None:
        return int(env_layer)
    if not CRITICAL_LAYERS_PATH.exists():
        return DEFAULT_CRITICAL_LAYER
    try:
        data = json.loads(CRITICAL_LAYERS_PATH.read_text(encoding="utf-8"))
        if data.get("model") and data["model"] != MODEL_NAME:
            return DEFAULT_CRITICAL_LAYER
        layers = data.get("critical_layers") or []
        return int(layers[0]) if layers else DEFAULT_CRITICAL_LAYER
    except Exception:
        return DEFAULT_CRITICAL_LAYER
