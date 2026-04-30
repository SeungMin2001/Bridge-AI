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

DIVERSE_SOURCE_PATH = Path(os.getenv("PRAG_DIVERSE_SOURCE_PATH", str(DATA_DIR / "PRAG_diverse_sources.jsonl")))
SOURCE_DATA_PATH = Path(os.getenv("PRAG_SOURCE_DATA_PATH", str(DIVERSE_SOURCE_PATH)))
RAW_PASSAGES_PATH = Path(os.getenv("PRAG_RAW_PASSAGES_PATH", str(DATA_DIR / "PRAG_raw_passages.jsonl")))
AUGMENTED_TRAIN_PATH = Path(os.getenv("PRAG_AUGMENTED_TRAIN_PATH", str(DATA_DIR / "PRAG_augmented_train.jsonl")))
AUGMENTED_VALID_PATH = Path(os.getenv("PRAG_AUGMENTED_VALID_PATH", str(DATA_DIR / "PRAG_augmented_valid.jsonl")))
MULTIFACT_SOURCE_PATH = Path(os.getenv("PRAG_MULTIFACT_SOURCE_PATH", str(DATA_DIR / "PRAG_multifact_sources.jsonl")))
MULTIFACT_AUGMENTED_TRAIN_PATH = Path(os.getenv("PRAG_MULTIFACT_AUGMENTED_TRAIN_PATH", str(DATA_DIR / "PRAG_multifact_augmented_train.jsonl")))
MULTIFACT_AUGMENTED_VALID_PATH = Path(os.getenv("PRAG_MULTIFACT_AUGMENTED_VALID_PATH", str(DATA_DIR / "PRAG_multifact_augmented_valid.jsonl")))
LECTURE_SCRIPT_DIR = Path(os.getenv("PRAG_LECTURE_SCRIPT_DIR", str(PROJECT_ROOT / "Group-Chat-agent" / "data")))
LECTURE_SOURCE_PATH = Path(os.getenv("PRAG_LECTURE_SOURCE_PATH", str(DATA_DIR / "PRAG_lecture_sources.jsonl")))
LECTURE_AUGMENTED_TRAIN_PATH = Path(os.getenv("PRAG_LECTURE_AUGMENTED_TRAIN_PATH", str(DATA_DIR / "PRAG_lecture_augmented_train.jsonl")))
LECTURE_AUGMENTED_VALID_PATH = Path(os.getenv("PRAG_LECTURE_AUGMENTED_VALID_PATH", str(DATA_DIR / "PRAG_lecture_augmented_valid.jsonl")))
AIHUB_LECTURE_DIR = Path(os.getenv("PRAG_AIHUB_LECTURE_DIR", str(DATA_DIR / "aihub_univ_lecture")))
AIHUB_LECTURE_TRAIN_SOURCE_PATH = Path(os.getenv("PRAG_AIHUB_LECTURE_TRAIN_SOURCE_PATH", str(DATA_DIR / "PRAG_aihub_lecture_sources_train.jsonl")))
AIHUB_LECTURE_VALID_SOURCE_PATH = Path(os.getenv("PRAG_AIHUB_LECTURE_VALID_SOURCE_PATH", str(DATA_DIR / "PRAG_aihub_lecture_sources_valid.jsonl")))
AIHUB_LECTURE_AUGMENTED_TRAIN_PATH = Path(os.getenv("PRAG_AIHUB_LECTURE_AUGMENTED_TRAIN_PATH", str(DATA_DIR / "PRAG_aihub_lecture_augmented_train.jsonl")))
AIHUB_LECTURE_AUGMENTED_VALID_PATH = Path(os.getenv("PRAG_AIHUB_LECTURE_AUGMENTED_VALID_PATH", str(DATA_DIR / "PRAG_aihub_lecture_augmented_valid.jsonl")))
KORQUAD_AUGMENTED_TRAIN_PATH = Path(os.getenv("PRAG_KORQUAD_AUGMENTED_TRAIN_PATH", str(DATA_DIR / "PRAG_korquad_augmented_train.jsonl")))
KORQUAD_AUGMENTED_VALID_PATH = Path(os.getenv("PRAG_KORQUAD_AUGMENTED_VALID_PATH", str(DATA_DIR / "PRAG_korquad_augmented_valid.jsonl")))

CHECKPOINT_PATH = Path(os.getenv("PRAG_CHECKPOINT_PATH", str(BASE_DIR / "prag_memory_checkpoint.pt")))
WEIGHTS_PATH = Path(os.getenv("PRAG_WEIGHTS_PATH", str(BASE_DIR / "prag_memory_weights.pt")))
LOG_PATH = Path(os.getenv("PRAG_LOG_PATH", str(BASE_DIR / "prag_train_log.json")))
MULTIFACT_CHECKPOINT_PATH = Path(os.getenv("PRAG_MULTIFACT_CHECKPOINT_PATH", str(BASE_DIR / "prag_multifact_memory_checkpoint.pt")))
MULTIFACT_WEIGHTS_PATH = Path(os.getenv("PRAG_MULTIFACT_WEIGHTS_PATH", str(BASE_DIR / "prag_multifact_memory_weights.pt")))
MULTIFACT_LOG_PATH = Path(os.getenv("PRAG_MULTIFACT_LOG_PATH", str(BASE_DIR / "prag_multifact_train_log.json")))
LECTURE_CHECKPOINT_PATH = Path(os.getenv("PRAG_LECTURE_CHECKPOINT_PATH", str(BASE_DIR / "prag_lecture_memory_checkpoint.pt")))
LECTURE_WEIGHTS_PATH = Path(os.getenv("PRAG_LECTURE_WEIGHTS_PATH", str(BASE_DIR / "prag_lecture_memory_weights.pt")))
LECTURE_LOG_PATH = Path(os.getenv("PRAG_LECTURE_LOG_PATH", str(BASE_DIR / "prag_lecture_train_log.json")))
AIHUB_LECTURE_CHECKPOINT_PATH = Path(os.getenv("PRAG_AIHUB_LECTURE_CHECKPOINT_PATH", str(BASE_DIR / "prag_aihub_lecture_memory_checkpoint.pt")))
AIHUB_LECTURE_WEIGHTS_PATH = Path(os.getenv("PRAG_AIHUB_LECTURE_WEIGHTS_PATH", str(BASE_DIR / "prag_aihub_lecture_memory_weights.pt")))
AIHUB_LECTURE_LOG_PATH = Path(os.getenv("PRAG_AIHUB_LECTURE_LOG_PATH", str(BASE_DIR / "prag_aihub_lecture_train_log.json")))
KORQUAD_CHECKPOINT_PATH = Path(os.getenv("PRAG_KORQUAD_CHECKPOINT_PATH", str(BASE_DIR / "prag_korquad_memory_checkpoint.pt")))
KORQUAD_WEIGHTS_PATH = Path(os.getenv("PRAG_KORQUAD_WEIGHTS_PATH", str(BASE_DIR / "prag_korquad_memory_weights.pt")))
KORQUAD_LOG_PATH = Path(os.getenv("PRAG_KORQUAD_LOG_PATH", str(BASE_DIR / "prag_korquad_train_log.json")))
CRITICAL_LAYERS_PATH = Path(os.getenv("PRAG_CRITICAL_LAYERS_PATH", str(BASE_DIR / "critical_layers.json")))

DEFAULT_CRITICAL_LAYER = int(os.getenv("PRAG_DEFAULT_LAYER", "19"))
NUM_KV = int(os.getenv("PRAG_NUM_KV", "16"))
HIDDEN_DIM = int(os.getenv("PRAG_HIDDEN_DIM", "1024"))
ALPHA = float(os.getenv("PRAG_ALPHA", "1.0"))
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
