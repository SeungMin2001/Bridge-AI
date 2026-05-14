import logging
import os

from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

_model = None
_model_error = None


def _load_model():
    """Lazy-load the embedding model; return None on failure."""
    global _model, _model_error

    if _model is not None:
        return _model
    if _model_error is not None:
        return None

    if os.getenv("EMBEDDING_DISABLED", "false").lower() == "true":
        _model_error = RuntimeError("Embedding disabled via EMBEDDING_DISABLED")
        logger.warning("[EMBED] Skipped model load: embeddings disabled")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        _model = SentenceTransformer("BAAI/bge-m3", device=device)
        logger.info("[EMBED] Model loaded on device=%s", device)
    except Exception as exc:
        _model_error = exc
        logger.warning("[EMBED] Model load failed; embeddings disabled: %s", exc)
        return None

    return _model


def get_embedding(text: str):
    model = _load_model()
    if model is None:
        return None

    try:
        # Official docs recommend dot product with normalized embeddings.
        embedding = model.encode(text, normalize_embeddings=True) #공식문서 참조하면 코사인 유사도보다 내적이 더 효과적이라고 말하고있음.
        return embedding.tolist()
    except Exception as exc:
        logger.warning("[EMBED] Encoding failed; returning None: %s", exc)
        return None
