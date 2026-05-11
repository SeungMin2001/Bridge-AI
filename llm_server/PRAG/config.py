"""Configuration for the clean PRAG service-memory implementation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
DATA_DIR = Path(os.getenv("PRAG_DATA_DIR", str(PROJECT_ROOT / "data")))

# Fix the PRAG training target to Qwen2.5-7B. Windows user-level env vars from
# older 3B runs can otherwise silently override the intended model in new shells.
MODEL_NAME = "Qwen/Qwen2.5-7B"
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
KORQUAD_SERVICE_AUGMENTED_TRAIN_PATH = Path(os.getenv(
    "PRAG_KORQUAD_SERVICE_AUGMENTED_TRAIN_PATH",
    str(DATA_DIR / "PRAG_korquad_service_augmented_train.jsonl"),
))
KORQUAD_SERVICE_AUGMENTED_VALID_PATH = Path(os.getenv(
    "PRAG_KORQUAD_SERVICE_AUGMENTED_VALID_PATH",
    str(DATA_DIR / "PRAG_korquad_service_augmented_valid.jsonl"),
))
EXTERNAL_QA_AUGMENTED_TRAIN_PATH = Path(os.getenv("PRAG_EXTERNAL_QA_AUGMENTED_TRAIN_PATH", str(DATA_DIR / "PRAG_external_qa_augmented_train.jsonl")))
EXTERNAL_QA_AUGMENTED_VALID_PATH = Path(os.getenv("PRAG_EXTERNAL_QA_AUGMENTED_VALID_PATH", str(DATA_DIR / "PRAG_external_qa_augmented_valid.jsonl")))
TRANSCRIPT_SOURCE_PATH = Path(os.getenv("PRAG_TRANSCRIPT_SOURCE_PATH", str(DATA_DIR / "PRAG_transcript_sources.jsonl")))
TRANSCRIPT_AUGMENTED_TRAIN_PATH = Path(os.getenv("PRAG_TRANSCRIPT_AUGMENTED_TRAIN_PATH", str(DATA_DIR / "PRAG_transcript_augmented_train.jsonl")))
TRANSCRIPT_AUGMENTED_VALID_PATH = Path(os.getenv("PRAG_TRANSCRIPT_AUGMENTED_VALID_PATH", str(DATA_DIR / "PRAG_transcript_augmented_valid.jsonl")))

CHECKPOINT_PATH = Path(os.getenv("PRAG_CHECKPOINT_PATH", str(BASE_DIR / "prag_memory_checkpoint.pt")))
WEIGHTS_PATH = Path(os.getenv("PRAG_WEIGHTS_PATH", str(BASE_DIR / "prag_memory_weights.pt")))
LOG_PATH = Path(os.getenv("PRAG_LOG_PATH", str(BASE_DIR / "prag_train_log.json")))
MULTIFACT_CHECKPOINT_PATH = Path(os.getenv(
    "PRAG_MULTIFACT_CHECKPOINT_PATH",
    str(BASE_DIR / "prag_multifact_qwen25_7b_qp_embed_phrase_clean_ko_checkpoint.pt"),
))
MULTIFACT_WEIGHTS_PATH = Path(os.getenv(
    "PRAG_MULTIFACT_WEIGHTS_PATH",
    str(BASE_DIR / "prag_multifact_qwen25_7b_qp_embed_phrase_clean_ko_weights.pt"),
))
MULTIFACT_LOG_PATH = Path(os.getenv(
    "PRAG_MULTIFACT_LOG_PATH",
    str(BASE_DIR / "prag_multifact_qwen25_7b_qp_embed_phrase_clean_ko_train_log.json"),
))
LECTURE_CHECKPOINT_PATH = Path(os.getenv("PRAG_LECTURE_CHECKPOINT_PATH", str(BASE_DIR / "prag_lecture_memory_checkpoint.pt")))
LECTURE_WEIGHTS_PATH = Path(os.getenv("PRAG_LECTURE_WEIGHTS_PATH", str(BASE_DIR / "prag_lecture_memory_weights.pt")))
LECTURE_LOG_PATH = Path(os.getenv("PRAG_LECTURE_LOG_PATH", str(BASE_DIR / "prag_lecture_train_log.json")))
AIHUB_LECTURE_CHECKPOINT_PATH = Path(os.getenv("PRAG_AIHUB_LECTURE_CHECKPOINT_PATH", str(BASE_DIR / "prag_aihub_lecture_memory_checkpoint.pt")))
AIHUB_LECTURE_WEIGHTS_PATH = Path(os.getenv("PRAG_AIHUB_LECTURE_WEIGHTS_PATH", str(BASE_DIR / "prag_aihub_lecture_memory_weights.pt")))
AIHUB_LECTURE_LOG_PATH = Path(os.getenv("PRAG_AIHUB_LECTURE_LOG_PATH", str(BASE_DIR / "prag_aihub_lecture_train_log.json")))
KORQUAD_CHECKPOINT_PATH = Path(os.getenv("PRAG_KORQUAD_CHECKPOINT_PATH", str(BASE_DIR / "prag_korquad_memory_checkpoint.pt")))
KORQUAD_WEIGHTS_PATH = Path(os.getenv("PRAG_KORQUAD_WEIGHTS_PATH", str(BASE_DIR / "prag_korquad_memory_weights.pt")))
KORQUAD_LOG_PATH = Path(os.getenv("PRAG_KORQUAD_LOG_PATH", str(BASE_DIR / "prag_korquad_train_log.json")))
KORQUAD_SERVICE_CHECKPOINT_PATH = Path(os.getenv(
    "PRAG_KORQUAD_SERVICE_CHECKPOINT_PATH",
    str(BASE_DIR / "prag_korquad_service_memory_checkpoint.pt"),
))
KORQUAD_SERVICE_WEIGHTS_PATH = Path(os.getenv(
    "PRAG_KORQUAD_SERVICE_WEIGHTS_PATH",
    str(BASE_DIR / "prag_korquad_service_memory_weights.pt"),
))
KORQUAD_SERVICE_LOG_PATH = Path(os.getenv(
    "PRAG_KORQUAD_SERVICE_LOG_PATH",
    str(BASE_DIR / "prag_korquad_service_train_log.json"),
))
MIXED_KOR_SERVICE_CHECKPOINT_PATH = Path(os.getenv(
    "PRAG_MIXED_KOR_SERVICE_CHECKPOINT_PATH",
    str(BASE_DIR / "prag_mixed_kor_service_memory_checkpoint.pt"),
))
MIXED_KOR_SERVICE_WEIGHTS_PATH = Path(os.getenv(
    "PRAG_MIXED_KOR_SERVICE_WEIGHTS_PATH",
    str(BASE_DIR / "prag_mixed_kor_service_memory_weights.pt"),
))
MIXED_KOR_SERVICE_LOG_PATH = Path(os.getenv(
    "PRAG_MIXED_KOR_SERVICE_LOG_PATH",
    str(BASE_DIR / "prag_mixed_kor_service_train_log.json"),
))
EXTERNAL_QA_CHECKPOINT_PATH = Path(os.getenv("PRAG_EXTERNAL_QA_CHECKPOINT_PATH", str(BASE_DIR / "prag_external_qa_memory_checkpoint.pt")))
EXTERNAL_QA_WEIGHTS_PATH = Path(os.getenv("PRAG_EXTERNAL_QA_WEIGHTS_PATH", str(BASE_DIR / "prag_external_qa_memory_weights.pt")))
EXTERNAL_QA_LOG_PATH = Path(os.getenv("PRAG_EXTERNAL_QA_LOG_PATH", str(BASE_DIR / "prag_external_qa_train_log.json")))
TRANSCRIPT_CHECKPOINT_PATH = Path(os.getenv("PRAG_TRANSCRIPT_CHECKPOINT_PATH", str(BASE_DIR / "prag_transcript_memory_checkpoint.pt")))
TRANSCRIPT_WEIGHTS_PATH = Path(os.getenv("PRAG_TRANSCRIPT_WEIGHTS_PATH", str(BASE_DIR / "prag_transcript_memory_weights.pt")))
TRANSCRIPT_LOG_PATH = Path(os.getenv("PRAG_TRANSCRIPT_LOG_PATH", str(BASE_DIR / "prag_transcript_train_log.json")))
CRITICAL_LAYERS_PATH = Path(os.getenv("PRAG_CRITICAL_LAYERS_PATH", str(BASE_DIR / "critical_layers.json")))
KORQUAD_SERVICE_CRITICAL_LAYERS_PATH = Path(os.getenv(
    "PRAG_KORQUAD_SERVICE_CRITICAL_LAYERS_PATH",
    str(BASE_DIR / "critical_layers_korquad_service.json"),
))

# The MergePRAG author config uses single_layer=9 for the public setup. Keep
# PRAG_CRITICAL_LAYER/PRAG_DEFAULT_LAYER overrides available for layer sweeps.
DEFAULT_CRITICAL_LAYER = int(os.getenv("PRAG_DEFAULT_LAYER", "9"))
NUM_KV = int(os.getenv("PRAG_NUM_KV", "16"))
HIDDEN_DIM = int(os.getenv("PRAG_HIDDEN_DIM", "1024"))
ALPHA = float(os.getenv("PRAG_ALPHA", "1.0"))
MAX_MEMORY_TOKENS = int(os.getenv("PRAG_MAX_MEMORY_TOKENS", "256"))
MAX_SEQ_LEN = int(os.getenv("PRAG_MAX_SEQ_LEN", "512"))
# Default to the paper-closer, embedding-only memory input. The question is still
# included in the memory text by QUESTION_CONDITIONED_MEMORY below, but we avoid
# concatenating full-model hidden states unless explicitly enabled.
USE_CONTEXTUAL_MEMORY = os.getenv("PRAG_USE_CONTEXTUAL_MEMORY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
QUESTION_CONDITIONED_MEMORY = os.getenv("PRAG_QUESTION_CONDITIONED_MEMORY", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RANK_MARGIN = float(os.getenv("PRAG_RANK_MARGIN", "0.5"))
RANK_WEIGHT = float(os.getenv("PRAG_RANK_WEIGHT", "1.0"))
LR = float(os.getenv("PRAG_LR", "8e-5"))
LR_MIN = float(os.getenv("PRAG_LR_MIN", "1e-6"))
EPOCHS = int(os.getenv("PRAG_EPOCHS", "4"))
LOG_EVERY = int(os.getenv("PRAG_LOG_EVERY", "25"))
SAVE_EVERY = int(os.getenv("PRAG_SAVE_EVERY", "250"))
EVAL_EVERY = int(os.getenv("PRAG_EVAL_EVERY", "250"))
EVAL_MAX_SAMPLES = int(os.getenv("PRAG_EVAL_MAX_SAMPLES", "120"))
EVAL_GENERATION_SAMPLES = int(os.getenv("PRAG_EVAL_GENERATION_SAMPLES", "4"))
EVAL_GENERATION_EVERY = int(os.getenv("PRAG_EVAL_GENERATION_EVERY", "1000"))
EVAL_GENERATION_MAX_NEW_TOKENS = int(os.getenv("PRAG_EVAL_GENERATION_MAX_NEW_TOKENS", "64"))


def contains_hangul(text: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in str(text or ""))


def model_path_tag(model_name: str | None = None) -> str:
    """Stable filename tag for model-specific scan/training artifacts."""
    name = str(model_name or MODEL_NAME).lower()
    replacements = {
        "qwen/qwen2.5-": "qwen25_",
        "qwen/qwen2-": "qwen2_",
        "qwen/": "qwen_",
    }
    for source, target in replacements.items():
        name = name.replace(source, target)
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_") or "model"


def critical_layers_path_for_run(
    *,
    model_name: str | None = None,
    korquad_service: bool = False,
    mixed_kor_service: bool = False,
    question_conditioned_memory: bool = QUESTION_CONDITIONED_MEMORY,
) -> Path:
    """Return the condition-specific layer-scan path used by train/scan."""
    if mixed_kor_service:
        stem = "critical_layers_mixed_kor_service"
    elif korquad_service:
        stem = "critical_layers_korquad_service"
    else:
        stem = "critical_layers"
    memory_tag = "qp" if question_conditioned_memory else "ponly"
    return BASE_DIR / f"{stem}_{model_path_tag(model_name)}_{memory_tag}.json"


def load_critical_layer(path: Path | None = None, *, model_name: str | None = None) -> int:
    env_layer = os.getenv("PRAG_CRITICAL_LAYER")
    if env_layer is not None:
        return int(env_layer)
    layers_path = path or CRITICAL_LAYERS_PATH
    if not layers_path.exists():
        return DEFAULT_CRITICAL_LAYER
    try:
        data = json.loads(layers_path.read_text(encoding="utf-8"))
        expected_model = model_name or MODEL_NAME
        if data.get("model") and data["model"] != expected_model:
            return DEFAULT_CRITICAL_LAYER
        layers = data.get("critical_layers") or []
        return int(layers[0]) if layers else DEFAULT_CRITICAL_LAYER
    except Exception:
        return DEFAULT_CRITICAL_LAYER
