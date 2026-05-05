"""
Paper-like MergePRAG HyperNetwork training on the current ServiceHardPair data.

This file follows the locally cloned paper code as closely as possible:
    /Users/shin/Documents/2026-1/UNIST_NLP_LAB/MhQA_hypernetwork-B31F/KV_train.py
    /Users/shin/Documents/2026-1/UNIST_NLP_LAB/MhQA_hypernetwork-B31F/MOE_model/hypernetwork.py

It intentionally avoids the experimental helpers in train.py:
- no contextual passage encoder
- no question-conditioned memory
- no query lexical focus or boosting
- no slot-wise pooling
- no pooled K/V skip or hybrid path
- no K/V RMS clamp, normalization, or alpha scaling
- no hard-negative, answer-rank, repulsion, or slot-diversity losses

Run:
    python -m llm_server.mergePRAG.train2
"""

import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    MAX_SEQ_LEN,
    MODEL_NAME,
    TRAIN_DATA_PATH,
    VALID_DATA_PATH,
    load_critical_layer,
)


BASE_DIR = Path(__file__).resolve().parent
CRITICAL_LAYER = load_critical_layer()

NUM_KV = int(os.getenv("MERGEPRAG_TRAIN2_NUM_KV", "1"))
LR = float(os.getenv("MERGEPRAG_TRAIN2_LR", "1e-4"))
LR_MIN = float(os.getenv("MERGEPRAG_TRAIN2_LR_MIN", "1e-6"))
WEIGHT_DECAY = float(os.getenv("MERGEPRAG_TRAIN2_WEIGHT_DECAY", "0.01"))
EPOCHS = int(os.getenv("MERGEPRAG_TRAIN2_EPOCHS", "1"))
HIDDEN_DIM = int(os.getenv("MERGEPRAG_TRAIN2_HIDDEN_DIM", "1024"))
LOG_EVERY = int(os.getenv("MERGEPRAG_TRAIN2_LOG_EVERY", "50"))
EVAL_EVERY = int(os.getenv("MERGEPRAG_TRAIN2_EVAL_EVERY", "500"))
SAVE_EVERY = int(os.getenv("MERGEPRAG_TRAIN2_SAVE_EVERY", "500"))
EVAL_MAX_SAMPLES = int(os.getenv("MERGEPRAG_TRAIN2_EVAL_MAX_SAMPLES", "500"))
MAX_SAMPLES = os.getenv("MERGEPRAG_TRAIN2_MAX_SAMPLES")
MAX_VAL_SAMPLES = os.getenv("MERGEPRAG_TRAIN2_MAX_VAL_SAMPLES")
RESUME = os.getenv("MERGEPRAG_TRAIN2_RESUME", "").strip().lower() in {"1", "true", "yes", "on"}
GRAD_CLIP_NORM = float(os.getenv("MERGEPRAG_TRAIN2_GRAD_CLIP_NORM", "0"))
DIAG_EVERY = int(os.getenv("MERGEPRAG_TRAIN2_DIAG_EVERY", "0"))
DIAG_MAX_SAMPLES = int(os.getenv("MERGEPRAG_TRAIN2_DIAG_MAX_SAMPLES", "3"))
DIAG_MAX_NEW_TOKENS = int(os.getenv("MERGEPRAG_TRAIN2_DIAG_MAX_NEW_TOKENS", "16"))
RUN_FINAL_DIAGNOSTIC = os.getenv("MERGEPRAG_TRAIN2_RUN_FINAL_DIAGNOSTIC", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

MAX_SAMPLES = int(MAX_SAMPLES) if MAX_SAMPLES else None
MAX_VAL_SAMPLES = int(MAX_VAL_SAMPLES) if MAX_VAL_SAMPLES else None

WEIGHTS_PATH = os.getenv(
    "MERGEPRAG_TRAIN2_WEIGHTS_PATH",
    str(BASE_DIR / "hypernet_train2_weights.pt"),
)
CHECKPOINT_PATH = os.getenv(
    "MERGEPRAG_TRAIN2_CHECKPOINT_PATH",
    str(BASE_DIR / "hypernet_train2_checkpoint.pt"),
)
LOG_PATH = os.getenv(
    "MERGEPRAG_TRAIN2_LOG_PATH",
    str(BASE_DIR / "train2_log.json"),
)
CHART_PATH = os.getenv(
    "MERGEPRAG_TRAIN2_CHART_PATH",
    str(BASE_DIR / "train2_loss_curve.png"),
)


def iter_records(dataset_path: str):
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array dataset: {dataset_path}")
        yield from data
        return

    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def format_speaker_text(speaker: str | None, text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    speaker = str(speaker or "").strip()
    return f"{speaker}: {text}" if speaker else text


def extract_hotpot_passage(sample: dict, max_sentences: int = 6) -> str:
    supporting = sample.get("supporting_facts")
    context = sample.get("context")
    if not isinstance(supporting, list) or not isinstance(context, list):
        return ""

    context_map = {}
    for item in context:
        if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str):
            title, sentences = item
            if isinstance(sentences, list):
                context_map[title] = sentences

    selected = []
    seen = set()
    for fact in supporting:
        if not (isinstance(fact, list) and len(fact) == 2):
            continue
        title, sent_idx = fact
        if title not in context_map or not isinstance(sent_idx, int):
            continue
        sentences = context_map[title]
        if 0 <= sent_idx < len(sentences):
            key = (title, sent_idx)
            if key in seen:
                continue
            seen.add(key)
            sentence = str(sentences[sent_idx]).strip()
            if sentence:
                selected.append(sentence)
        if len(selected) >= max_sentences:
            break
    return " ".join(selected).strip()


def extract_utterance_passage(sample: dict) -> str:
    utterance = sample.get("utterance")
    if isinstance(utterance, str) and utterance.strip():
        return format_speaker_text(sample.get("speaker"), utterance)

    for key in ("text", "content", "transcript", "chunk_text"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return format_speaker_text(sample.get("speaker"), value)

    for key in ("utterances", "messages", "turns"):
        value = sample.get(key)
        if not isinstance(value, list):
            continue
        lines = []
        for item in value:
            if isinstance(item, str):
                line = item.strip()
            elif isinstance(item, dict):
                line = format_speaker_text(
                    item.get("speaker") or item.get("role") or item.get("name"),
                    item.get("utterance") or item.get("text") or item.get("content"),
                )
            else:
                line = ""
            if line:
                lines.append(line)
        if lines:
            return "\n".join(lines)
    return ""


def extract_passage(sample: dict) -> str:
    passage = sample.get("passage")
    if isinstance(passage, str) and passage.strip():
        return passage.strip()

    utterance_passage = extract_utterance_passage(sample)
    if utterance_passage:
        return utterance_passage

    hotpot_passage = extract_hotpot_passage(sample)
    if hotpot_passage:
        return hotpot_passage

    facts = sample.get("facts")
    if isinstance(facts, list):
        fact_texts = [str(item).strip() for item in facts if isinstance(item, str) and str(item).strip()]
        if fact_texts:
            return "\n".join(fact_texts)

    for key in ("supporting_passage", "context", "evidence", "chunk_text"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("supporting_facts", "passages", "contexts", "evidences"):
        value = sample.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                for inner_key in ("passage", "text", "context", "chunk_text"):
                    inner_value = item.get(inner_key)
                    if isinstance(inner_value, str) and inner_value.strip():
                        return inner_value.strip()
    return ""


def extract_answer(sample: dict) -> str:
    for key in ("answer", "target", "output", "response"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    answers = sample.get("answers")
    if isinstance(answers, list):
        for item in answers:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


class MergePRAGDataset(Dataset):
    def __init__(self, dataset_path: str, max_samples: int | None = None):
        self.data = []
        for i, item in enumerate(iter_records(dataset_path)):
            if max_samples and i >= max_samples:
                break

            question = str(item.get("question", "")).strip()
            answer = extract_answer(item)
            if not (question and answer):
                continue

            hop_passages = item.get("hop_passages")
            if isinstance(hop_passages, list):
                passages = [str(p).strip() for p in hop_passages if isinstance(p, str) and p.strip()]
            else:
                passages = []

            if not passages:
                passage = extract_passage(item)
                if passage:
                    passages = [passage]

            for hop_idx, passage in enumerate(passages, start=1):
                self.data.append({
                    **item,
                    "question": question,
                    "answer": answer,
                    "passage": passage,
                    "hop_index": hop_idx,
                    "num_hops": len(passages),
                })

        print(f"[train2:data] loaded {len(self.data)} samples from {dataset_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PaperAttentivePooling(nn.Module):
    """Original HyperKVGeneratorFixed attentive pooling layer."""

    def __init__(self, d_model: int):
        super().__init__()
        self.att_pool = nn.Linear(d_model, 1)

    def forward(self, c_emb: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.att_pool(c_emb).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (c_emb * weights).sum(dim=1)


class PaperHyperNetwork(nn.Module):
    """Local paper code's HyperKVGeneratorFixed.

    Original source:
    MOE_model/hypernetwork.py::HyperKVGeneratorFixed
    """

    def __init__(self, d_model: int, num_kv: int = 1, hidden_dim: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.num_kv = num_kv
        self.pooling_layer = nn.Linear(d_model, 1)
        self.pooling = PaperAttentivePooling(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.linear_K = nn.Linear(hidden_dim, num_kv * d_model)
        self.linear_V = nn.Linear(hidden_dim, num_kv * d_model)

    def forward(self, c_emb: torch.Tensor, attention_mask: torch.Tensor | None = None):
        pooled = self.pooling(c_emb, attention_mask)
        hidden = self.mlp(pooled)
        batch = hidden.size(0)
        K = self.linear_K(hidden).view(batch, self.num_kv, self.d_model)
        V = self.linear_V(hidden).view(batch, self.num_kv, self.d_model)
        return pooled, hidden, K, V


def cross_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Local paper code's 8-head cross_attention from KV_train.py."""
    squeezed = False
    if Q.dim() == 2:
        Q = Q.unsqueeze(0)
        squeezed = True
    if K.dim() == 2:
        K = K.unsqueeze(0)
    if V.dim() == 2:
        V = V.unsqueeze(0)

    batch, query_len, d_model = Q.shape
    key_len = K.shape[1]
    num_heads = 8
    d_k = d_model // num_heads
    if d_model % num_heads != 0:
        raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

    Qh = Q.view(batch, query_len, num_heads, d_k).transpose(1, 2)
    Kh = K.view(batch, key_len, num_heads, d_k).transpose(1, 2)
    Vh = V.view(batch, key_len, num_heads, d_k).transpose(1, 2)
    att = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(d_k)
    att = att.softmax(dim=-1)
    out = att @ Vh
    out = out.transpose(1, 2).contiguous().view(batch, query_len, d_model)
    return out.squeeze(0) if squeezed else out


def make_hook(delta_K: torch.Tensor, delta_V: torch.Tensor):
    def hook_fn(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        K = delta_K.to(device=hidden.device, dtype=hidden.dtype)
        V = delta_V.to(device=hidden.device, dtype=hidden.dtype)
        delta = cross_attention(hidden, K, V)
        new_hidden = hidden + delta
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    return hook_fn


def compute_loss(logits: torch.Tensor, labels: torch.Tensor):
    """Local paper code's utils.cross_entropy for answer-token positions."""
    answer_positions = torch.where(labels != -100)
    if len(answer_positions[0]) == 0:
        return None
    return F.cross_entropy(logits[answer_positions], labels[answer_positions])


def tokenize_passage(tokenizer, passage: str, device):
    encoded = tokenizer(
        passage,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


def encode_memory(model, hypernet, tokenizer, passage: str, device):
    encoded = tokenize_passage(tokenizer, passage, device)
    with torch.no_grad():
        c_emb = model.model.embed_tokens(encoded["input_ids"]).to(dtype=torch.float32)
    pooled, hidden, delta_K, delta_V = hypernet(c_emb, encoded["attention_mask"])
    return encoded, pooled, hidden, delta_K, delta_V


def build_paper_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def build_direct_prompt(question: str, passage: str) -> str:
    return f"Passage: {passage}\nQuestion: {question}\nAnswer:"


def tokenize_qa(tokenizer, question: str, answer: str, device):
    """Match data/base.py::tok_tuples from the local paper code."""
    prompt = build_paper_prompt(question)
    model_name_lower = MODEL_NAME.lower()
    if "llama" in model_name_lower:
        answer_text = f" {answer}{tokenizer.eos_token}"
    else:
        answer_text = f"{answer}{tokenizer.eos_token}"

    tok_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    tok_answer = tokenizer(
        answer_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )

    input_ids = torch.cat(
        (tok_prompt["input_ids"], tok_answer["input_ids"][:, :-1]),
        dim=-1,
    ).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels = torch.cat(
        (
            torch.full(tok_prompt["input_ids"].shape, -100, dtype=torch.long)[:, 1:],
            tok_answer["input_ids"],
        ),
        dim=-1,
    ).to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def forward_with_memory(model, target_layer, delta_K, delta_V, tok):
    hook = target_layer.register_forward_hook(make_hook(delta_K, delta_V))
    try:
        logits = model(**tok)["logits"]
    finally:
        hook.remove()
    return logits


@torch.no_grad()
def evaluate(model, tokenizer, hypernet, target_layer, dataset, device):
    hypernet.eval()
    total_loss = 0.0
    count = 0

    for idx, sample in enumerate(dataset):
        if idx >= EVAL_MAX_SAMPLES:
            break

        _, _, _, delta_K, delta_V = encode_memory(
            model,
            hypernet,
            tokenizer,
            sample["passage"],
            device,
        )
        tok = tokenize_qa(tokenizer, sample["question"], sample["answer"], device)
        logits = forward_with_memory(model, target_layer, delta_K, delta_V, tok)
        loss = compute_loss(logits, tok["labels"])
        if loss is not None:
            total_loss += loss.item()
            count += 1

        del logits, delta_K, delta_V, tok
        if torch.cuda.is_available() and idx % 100 == 0:
            torch.cuda.empty_cache()

    hypernet.train()
    return total_loss / max(count, 1)


def extract_first_hard_negative(sample: dict) -> dict | None:
    for key in ("hard_negatives", "negative_passages", "counterfactuals", "distractors"):
        values = sample.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict):
                passage = extract_passage(item)
                answer = extract_answer(item)
                if passage and answer:
                    return {"passage": passage, "answer": answer}
            if isinstance(item, str) and item.strip():
                return {"passage": item.strip(), "answer": ""}

    for key in ("hard_negative_passage", "negative_passage", "counterfactual_passage", "distractor_passage"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return {
                "passage": value.strip(),
                "answer": str(
                    sample.get("negative_answer")
                    or sample.get("counterfactual_answer")
                    or ""
                ).strip(),
            }
        if isinstance(value, dict):
            passage = extract_passage(value)
            answer = extract_answer(value)
            if passage:
                return {"passage": passage, "answer": answer}
    return None


def decode_new_tokens(tokenizer, generated, prompt_len: int) -> str:
    text = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
    return " ".join(text.strip().split())


@torch.no_grad()
def generate_plain(model, tokenizer, prompt: str, device) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)
    generated = model.generate(
        **inputs,
        max_new_tokens=DIAG_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return decode_new_tokens(tokenizer, generated, inputs["input_ids"].shape[1])


@torch.no_grad()
def generate_with_memory(model, tokenizer, hypernet, target_layer, question: str, passage: str, device) -> str:
    _, _, _, delta_K, delta_V = encode_memory(model, hypernet, tokenizer, passage, device)
    prompt = build_paper_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)
    hook = target_layer.register_forward_hook(make_hook(delta_K, delta_V))
    try:
        generated = model.generate(
            **inputs,
            max_new_tokens=DIAG_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        hook.remove()
    return decode_new_tokens(tokenizer, generated, inputs["input_ids"].shape[1])


@torch.no_grad()
def score_answer(model, tokenizer, hypernet, target_layer, question: str, passage: str | None, answer: str, device):
    tok = tokenize_qa(tokenizer, question, answer, device)
    if passage is None:
        logits = model(**tok)["logits"]
    else:
        _, _, _, delta_K, delta_V = encode_memory(model, hypernet, tokenizer, passage, device)
        logits = forward_with_memory(model, target_layer, delta_K, delta_V, tok)
    loss = compute_loss(logits, tok["labels"])
    return float("nan") if loss is None else loss.item()


@torch.no_grad()
def run_diagnostics(model, tokenizer, hypernet, target_layer, dataset, device, limit: int = DIAG_MAX_SAMPLES):
    hypernet.eval()
    print(f"\n[train2:diagnostic] passage injection check on {min(limit, len(dataset))} validation samples")
    for idx, sample in enumerate(dataset):
        if idx >= limit:
            break
        question = sample["question"]
        passage = sample["passage"]
        answer = sample["answer"]
        negative = extract_first_hard_negative(sample)

        no_hook = generate_plain(model, tokenizer, build_paper_prompt(question), device)
        direct = generate_plain(model, tokenizer, build_direct_prompt(question, passage), device)
        hooked = generate_with_memory(model, tokenizer, hypernet, target_layer, question, passage, device)
        target_loss = score_answer(model, tokenizer, hypernet, target_layer, question, passage, answer, device)
        base_loss = score_answer(model, tokenizer, hypernet, target_layer, question, None, answer, device)

        print(f"  sample {idx + 1}")
        print(f"    q: {question}")
        print(f"    gold: {answer}")
        print(f"    no_hook: {no_hook}")
        print(f"    direct: {direct}")
        print(f"    memory: {hooked}")
        print(f"    target_loss | no_hook={base_loss:.4f}, memory={target_loss:.4f}, gain={base_loss - target_loss:+.4f}")

        if negative and negative.get("answer"):
            wrong_answer = negative["answer"]
            wrong_loss = score_answer(
                model,
                tokenizer,
                hypernet,
                target_layer,
                question,
                passage,
                wrong_answer,
                device,
            )
            neg_memory_gold = score_answer(
                model,
                tokenizer,
                hypernet,
                target_layer,
                question,
                negative["passage"],
                answer,
                device,
            )
            neg_memory_wrong = score_answer(
                model,
                tokenizer,
                hypernet,
                target_layer,
                question,
                negative["passage"],
                wrong_answer,
                device,
            )
            print(
                f"    candidate(main memory) | gold={target_loss:.4f}, negative={wrong_loss:.4f}, "
                f"prefer_gold={target_loss < wrong_loss}"
            )
            print(
                f"    candidate(negative memory) | gold={neg_memory_gold:.4f}, "
                f"negative={neg_memory_wrong:.4f}, prefer_negative={neg_memory_wrong < neg_memory_gold}"
            )
    hypernet.train()


def save_chart(log_data: dict):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[train2:chart] matplotlib unavailable; skipped chart")
        return

    steps = [item["step"] for item in log_data["step_losses"]]
    losses = [item["loss"] for item in log_data["step_losses"]]
    val_steps = [item["step"] for item in log_data["val_evals"]]
    val_losses = [item["val_loss"] for item in log_data["val_evals"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, losses, alpha=0.25, linewidth=0.6, label="train CE")
    if losses:
        window = min(50, max(1, len(losses) // 10))
        smooth = []
        for i in range(len(losses)):
            start = max(0, i - window + 1)
            smooth.append(sum(losses[start:i + 1]) / (i - start + 1))
        ax.plot(steps, smooth, linewidth=1.6, label="train CE smooth")
    if val_losses:
        ax.plot(val_steps, val_losses, "o-", linewidth=1.8, label="val CE")
    ax.set_xlabel("Step")
    ax.set_ylabel("Answer CE Loss")
    ax.set_title("train2 paper-like CE-only training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150)
    plt.close(fig)
    print(f"[train2:chart] saved {CHART_PATH}")


def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def current_config(train_dataset, val_dataset):
    return {
        "script": "train2",
        "local_paper_code": "/Users/shin/Documents/2026-1/UNIST_NLP_LAB/MhQA_hypernetwork-B31F",
        "objective": "local_kv_train_answer_ce_only",
        "model": MODEL_NAME,
        "critical_layer": CRITICAL_LAYER,
        "injection_target": f"model.model.layers[{CRITICAL_LAYER}]",
        "memory_attention": "8_head_cross_attention(layer_output, K, V)",
        "num_kv": NUM_KV,
        "hidden_dim": HIDDEN_DIM,
        "max_seq_len": MAX_SEQ_LEN,
        "passage_encoder": "token_embedding_only",
        "memory_conditioning": "passage_only",
        "pooling": "single_attentive_pooling",
        "kv_path": "mlp_to_linear_K_linear_V",
        "kv_normalization": "none",
        "auxiliary_losses": "none",
        "qa_tokenization": "paper_tok_tuples_prompt_plus_answer_minus_last",
        "lr": LR,
        "lr_min": LR_MIN,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "train_data_path": TRAIN_DATA_PATH,
        "valid_data_path": VALID_DATA_PATH,
        "train_samples": len(train_dataset),
        "valid_samples": len(val_dataset),
    }


def maybe_resume(hypernet, optimizer, scheduler, device):
    if not RESUME or not os.path.exists(CHECKPOINT_PATH):
        return 0, float("inf")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    hypernet.load_state_dict(ckpt["hypernet"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    step = int(ckpt.get("step", 0))
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    print(f"[train2:resume] resumed from step {step}")
    return step, best_val_loss


def train():
    print(f"[train2] loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    device = next(model.parameters()).device
    d_model = model.config.hidden_size
    target_layer = model.model.layers[CRITICAL_LAYER]
    print(f"[train2] d_model={d_model}, layer={CRITICAL_LAYER}, device={device}")

    hypernet = PaperHyperNetwork(d_model, num_kv=NUM_KV, hidden_dim=HIDDEN_DIM).to(device).float()
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_dataset = MergePRAGDataset(TRAIN_DATA_PATH, max_samples=MAX_SAMPLES)
    val_dataset = MergePRAGDataset(VALID_DATA_PATH, max_samples=MAX_VAL_SAMPLES)
    if len(train_dataset) == 0:
        raise RuntimeError("Empty train dataset")
    if len(val_dataset) == 0:
        raise RuntimeError("Empty validation dataset")

    total_steps = max(1, len(train_dataset) * EPOCHS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=LR_MIN,
    )

    global_step, best_val_loss = maybe_resume(hypernet, optimizer, scheduler, device)
    run_config = current_config(train_dataset, val_dataset)
    log_data = {
        "config": run_config,
        "step_losses": [],
        "val_evals": [],
        "lr_history": [],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(
        "[train2] classic config | "
        f"num_kv={NUM_KV}, lr={LR}, weight_decay={WEIGHT_DECAY}, "
        f"train={len(train_dataset)}, val={len(val_dataset)}"
    )
    print("[train2] disabled helpers | contextual=false, q_condition=false, focus=false, "
          "slotwise=false, skip=false, clamp=false, aux_losses=false")
    print(f"[train2] outputs | weights={WEIGHTS_PATH}, checkpoint={CHECKPOINT_PATH}")

    hypernet.train()
    start_time = time.time()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        count = 0
        epoch_indices = list(range(len(train_dataset)))
        random.shuffle(epoch_indices)

        for i, sample_idx in enumerate(epoch_indices):
            sample = train_dataset[sample_idx]
            expected_seen_steps = epoch * len(train_dataset) + i
            if expected_seen_steps < global_step:
                continue

            try:
                _, pooled, hidden, delta_K, delta_V = encode_memory(
                    model,
                    hypernet,
                    tokenizer,
                    sample["passage"],
                    device,
                )
                tok = tokenize_qa(tokenizer, sample["question"], sample["answer"], device)
                logits = forward_with_memory(model, target_layer, delta_K, delta_V, tok)
                loss = compute_loss(logits, tok["labels"])
                if loss is None:
                    continue

                optimizer.zero_grad()
                loss.backward()
                if GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(hypernet.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()

                loss_val = loss.item()
                total_loss += loss_val
                count += 1
                global_step += 1

                if global_step % LOG_EVERY == 0:
                    with torch.no_grad():
                        base_logits = model(**tok)["logits"]
                        base_loss = compute_loss(base_logits, tok["labels"])
                    base_loss_val = base_loss.item() if base_loss is not None else float("nan")
                    lr_now = scheduler.get_last_lr()[0]
                    avg = total_loss / max(count, 1)
                    elapsed = (time.time() - start_time) / 60
                    k_norm = delta_K.detach().norm(dim=-1).mean().item()
                    v_norm = delta_V.detach().norm(dim=-1).mean().item()
                    print(
                        f"  Step {global_step}/{total_steps} | "
                        f"loss={loss_val:.4f} | avg={avg:.4f} | lr={lr_now:.2e} | "
                        f"Knorm={k_norm:.3f} | Vnorm={v_norm:.3f} | "
                        f"base={base_loss_val:.4f} | gain={base_loss_val - loss_val:+.4f} | "
                        f"{elapsed:.1f}min"
                    )
                    log_data["lr_history"].append({"step": global_step, "lr": round(lr_now, 8)})
                    del base_logits

                log_data["step_losses"].append({
                    "step": global_step,
                    "loss": round(loss_val, 4),
                })

                if global_step % SAVE_EVERY == 0:
                    torch.save({
                        "step": global_step,
                        "hypernet": hypernet.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "config": run_config,
                    }, CHECKPOINT_PATH)
                    save_json(LOG_PATH, log_data)
                    print(f"  [train2:checkpoint] saved step {global_step}")

                if global_step % EVAL_EVERY == 0:
                    val_loss = evaluate(model, tokenizer, hypernet, target_layer, val_dataset, device)
                    elapsed = (time.time() - start_time) / 60
                    print(f"  -- [train2:val @ step {global_step}] val_loss={val_loss:.4f} | {elapsed:.1f}min")
                    log_data["val_evals"].append({
                        "step": global_step,
                        "val_loss": round(val_loss, 4),
                    })
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            "step": global_step,
                            "hypernet": hypernet.state_dict(),
                            "config": run_config,
                            "val_loss": val_loss,
                        }, WEIGHTS_PATH)
                        print("     [train2:best] saved weights")
                    if DIAG_EVERY and global_step % DIAG_EVERY == 0:
                        run_diagnostics(
                            model,
                            tokenizer,
                            hypernet,
                            target_layer,
                            val_dataset,
                            device,
                            limit=1,
                        )

                del logits, loss, tok, pooled, hidden, delta_K, delta_V
                if torch.cuda.is_available() and global_step % 100 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [train2:oom] skipped step {global_step}; clearing cache")
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

        train_loss = total_loss / max(count, 1)
        val_loss = evaluate(model, tokenizer, hypernet, target_layer, val_dataset, device)
        print(f"[train2:epoch {epoch + 1}/{EPOCHS}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        log_data["val_evals"].append({
            "step": global_step,
            "val_loss": round(val_loss, 4),
        })
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "step": global_step,
                "hypernet": hypernet.state_dict(),
                "config": run_config,
                "val_loss": val_loss,
            }, WEIGHTS_PATH)
            print("     [train2:best] saved weights")

    elapsed_total = time.time() - start_time
    log_data["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data["total_time_min"] = round(elapsed_total / 60, 1)
    log_data["best_val_loss"] = round(best_val_loss, 4)
    log_data["total_steps"] = global_step

    save_json(LOG_PATH, log_data)
    save_chart(log_data)

    if RUN_FINAL_DIAGNOSTIC:
        run_diagnostics(model, tokenizer, hypernet, target_layer, val_dataset, device)

    print(f"[train2] done in {elapsed_total / 60:.1f}min")
    print(f"[train2] best_val_loss={best_val_loss:.4f}, weights={WEIGHTS_PATH}")


if __name__ == "__main__":
    train()
