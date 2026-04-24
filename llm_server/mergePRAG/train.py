"""
MergePRAG HyperNetwork 학습 스크립트 (논문 원본 KV_train.py 기반)

- 논문: MergePRAG (ICLR 2026, Liu-Xuebing/MhQA_hypernetwork)
- HotPotQA 데이터로 학습 (1 epoch, 논문과 동일)
- Qwen freeze, HyperNetwork만 학습
- 매 step마다 hook 등록/해제 (논문 방식)
- labels=-100 마스킹으로 answer 토큰만 loss 계산
- Validation + 시각화 (발표자료용 추가)

사용법: python -m llm_server.mergePRAG.train
"""
import torch
import torch.nn.functional as F
import json
import os
import time
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import (
    ALPHA,
    ALLOW_CONFIG_MISMATCH_RESUME,
    ALLOW_LEGACY_CHECKPOINT_RESUME,
    CHART_PATH,
    CHECKPOINT_PATH,
    HIDDEN_SIM_TARGET,
    K_SIM_TARGET,
    KV_PATH_MODE,
    LOG_PATH,
    MAX_SEQ_LEN,
    MODEL_NAME,
    NEGATIVE_LOSS_WEIGHT,
    NEGATIVE_MARGIN,
    NUM_KV,
    POOLED_KV_SKIP_SCALE,
    POOLED_K_SKIP_SCALE,
    POOLED_V_SKIP_SCALE,
    QUESTION_NEGATIVE_LOSS_WEIGHT,
    QUESTION_REPULSION_LOSS_WEIGHT,
    QUERY_POOL_SCALE,
    REPULSION_LOSS_WEIGHT,
    SLOT_DIVERSITY_LOSS_WEIGHT,
    SLOT_DIVERSITY_TARGET,
    TRAIN_DATA_PATH,
    TRAIN_PROMPT_FORMAT,
    USE_POOLED_KV_SKIP,
    USE_V_RMS_CLAMP,
    USE_CONTEXTUAL_PASSAGE_ENCODER,
    USE_QUESTION_CONDITIONED_MEMORY,
    VALID_DATA_PATH,
    V_SIM_TARGET,
    V_RMS_CLAMP,
    WEIGHTS_PATH as SAVE_PATH,
    build_chat_text,
    load_critical_layer,
)
from .embedding import (
    encode_passage_states,
    tokenize_conditioned_memory,
    tokenize_passage_memory,
)
from .hypernetwork import HyperNetwork
from .cross_attention import cross_attention

# ── 설정 (논문 기본값 기반) ──
CRITICAL_LAYER = load_critical_layer()
LR = 1e-4                   # 논문 동일
LR_MIN = 1e-6               # 논문 CosineAnnealing eta_min
EPOCHS = 1                  # 논문: 1 epoch (single pass)
MAX_SAMPLES = None           # 전체 학습
MAX_VAL_SAMPLES = None       # 전체 검증
EVAL_EVERY = 500            # 500 step마다 validation
EVAL_MAX_SAMPLES = 500      # validation 시 최대 샘플 수 (전체 순회 방지)
LOG_EVERY = 50              # N step마다 터미널 출력 (논문: 49)
SAVE_EVERY = 500            # 500 step마다 체크포인트 저장
PATIENCE = 3                # 5→3으로 감소 (overfitting 방지)
TRAIN_MAX_PASSAGE_SENTENCES = 6
GRAD_CLIP_NORM = 1.0
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
        if isinstance(data, list):
            for item in data:
                yield item
            return
        raise ValueError(f"Expected a JSON array dataset: {dataset_path}")

    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def extract_hotpot_passage(sample: dict, max_sentences: int = TRAIN_MAX_PASSAGE_SENTENCES) -> str:
    supporting = sample.get("supporting_facts")
    context = sample.get("context")
    if not isinstance(supporting, list) or not isinstance(context, list):
        return ""

    context_map = {}
    for item in context:
        if (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], list)
        ):
            title, sentences = item
            context_map[title] = sentences

    selected = []
    seen = set()
    for fact in supporting:
        if not (isinstance(fact, list) and len(fact) == 2):
            continue
        title, sent_idx = fact
        if title not in context_map:
            continue
        if not isinstance(sent_idx, int):
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


def format_speaker_text(speaker: str | None, text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    speaker = str(speaker or "").strip()
    if speaker:
        return f"{speaker}: {text}"
    return text


def extract_utterance_passage(sample: dict) -> str:
    """서비스 데이터는 transcript/chunk 형태일 수 있어 speaker 발화를 passage로 정규화한다."""
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
    """서비스용 QA-passage 학습에 맞는 근거 passage를 선택한다."""
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

    for key in ("passage", "supporting_passage", "context", "evidence", "chunk_text"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("supporting_facts", "passages", "contexts", "evidences"):
        value = sample.get(key)
        if isinstance(value, list):
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


# ── 데이터셋 ──
class MergePRAGDataset(Dataset):
    def __init__(self, jsonl_path, max_samples=None):
        self.data = []
        for i, item in enumerate(iter_records(jsonl_path)):
            if max_samples and i >= max_samples:
                break
            question = item.get("question", "").strip()
            answer = extract_answer(item)
            if not (question and answer):
                continue

            hop_passages = item.get("hop_passages")
            if isinstance(hop_passages, list):
                expanded = [str(p).strip() for p in hop_passages if isinstance(p, str) and str(p).strip()]
            else:
                expanded = []

            if not expanded:
                passage = extract_passage(item)
                if passage:
                    expanded = [passage]

            for hop_idx, passage in enumerate(expanded, start=1):
                self.data.append({
                    **item,
                    "question": question,
                    "answer": answer,
                    "passage": passage,
                    "hop_index": hop_idx,
                    "num_hops": len(expanded),
                })
        self.indices_by_source = {}
        self.indices_by_passage = {}
        self.indices_by_question = {}
        self.indices_by_contrast = {}
        for idx, sample in enumerate(self.data):
            source_id = sample.get("source_id") or sample.get("id")
            if source_id is not None:
                self.indices_by_source.setdefault(str(source_id), []).append(idx)
            question_key = normalize_passage_text(sample.get("question", ""))
            if question_key:
                self.indices_by_question.setdefault(question_key, []).append(idx)
            contrast_id = sample.get("contrast_id") or sample.get("group_id")
            if contrast_id is not None:
                self.indices_by_contrast.setdefault(str(contrast_id), []).append(idx)
            passage_key = normalize_passage_text(sample.get("passage", ""))
            if passage_key:
                self.indices_by_passage.setdefault(passage_key, []).append(idx)
        print(f"[데이터] {len(self.data)}개 샘플 로드됨 ({jsonl_path})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Hook 함수 (논문 방식: 매 step 등록/해제) ──
def make_hook(delta_K, delta_V):
    """논문 원본 make_simple_cross_attn_hook 방식.
    hidden_states(Q)와 delta_K, delta_V로 cross-attention 후 residual add."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            K = delta_K.to(device=hidden.device, dtype=hidden.dtype)
            V = delta_V.to(device=hidden.device, dtype=hidden.dtype)
            delta = cross_attention(hidden, K, V)
            new_hidden = hidden + ALPHA * delta
            return (new_hidden,) + output[1:]
        else:
            hidden = output
            K = delta_K.to(device=hidden.device, dtype=hidden.dtype)
            V = delta_V.to(device=hidden.device, dtype=hidden.dtype)
            delta = cross_attention(hidden, K, V)
            return hidden + ALPHA * delta
    return hook_fn


# ── Loss 함수 (causal LM shift 적용) ──
def compute_loss(logits, labels):
    # logits[:, i]는 position i+1을 예측 → shift 필요
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    ans_indices = torch.where(shift_labels != -100)
    if len(ans_indices[0]) == 0:
        return None
    logits_flat = shift_logits[ans_indices]
    labels_flat = shift_labels[ans_indices]
    return F.cross_entropy(logits_flat, labels_flat)


def masked_mean(hidden, mask):
    weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denom


def encode_memory(model, hypernet, tokenizer, question: str, passage: str, device):
    if USE_QUESTION_CONDITIONED_MEMORY:
        encoded = tokenize_conditioned_memory(
            tokenizer,
            question,
            passage,
            device,
            max_length=MAX_SEQ_LEN,
        )
    else:
        encoded = tokenize_passage_memory(
            tokenizer,
            passage,
            device,
            max_length=MAX_SEQ_LEN,
        )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    question_mask = encoded["question_mask"]
    passage_mask = encoded["passage_mask"]
    embedded = encode_passage_states(
        model,
        input_ids,
        attention_mask=attention_mask,
        use_contextual=USE_CONTEXTUAL_PASSAGE_ENCODER,
    )
    query = masked_mean(embedded, question_mask) if USE_QUESTION_CONDITIONED_MEMORY else None
    pooled, hidden, raw_K, raw_V = hypernet.encode_embedded(
        embedded,
        attention_mask=attention_mask,
        query=query,
        focus_mask=passage_mask,
    )
    delta_K, delta_V = hypernet.normalize_kv(raw_K, raw_V)
    return input_ids, embedded, pooled, hidden, delta_K, delta_V


def forward_with_memory(model, target_layer, delta_K, delta_V, tok):
    hook = target_layer.register_forward_hook(make_hook(delta_K, delta_V))
    try:
        logits = model(input_ids=tok["input_ids"])["logits"]
    finally:
        hook.remove()
    return logits


def build_training_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def build_prompt_for_task(question: str, task: str) -> str:
    if task == "hop_fact":
        return (
            "Read the passage and output the supporting fact that is currently grounded.\n"
            f"Question: {question}\n"
            "Supporting fact:"
        )
    return build_training_prompt(question)


def tokenize_qa(tokenizer, question, answer, device, task: str = "final_qa"):
    """QA 토큰화 — compute_loss의 shift와 정확히 맞도록 정렬.
    input_ids = [prompt] + [answer+eos]  (전체)
    labels    = [-100]*prompt_len + [answer+eos]
    → compute_loss가 logits[:,:-1]과 labels[:,1:]로 shift하면
      prompt 마지막 토큰을 본 후 첫 answer 토큰을 예측."""
    task_prompt = build_prompt_for_task(question, task)
    if TRAIN_PROMPT_FORMAT == "chat":
        prompt = build_chat_text(
            tokenizer,
            question=task_prompt,
            enable_thinking=False,
        )
        answer_text = f"{answer}{tokenizer.eos_token}"
    elif TRAIN_PROMPT_FORMAT == "plain":
        prompt = task_prompt
        answer_text = f" {answer}{tokenizer.eos_token}"
    else:
        raise ValueError(f"Unsupported MERGEPRAG_TRAIN_PROMPT_FORMAT: {TRAIN_PROMPT_FORMAT}")

    tok_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    tok_answer = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN)

    prompt_len = tok_prompt["input_ids"].shape[1]
    input_ids = torch.cat((tok_prompt["input_ids"], tok_answer["input_ids"]), dim=-1).to(device)
    labels = torch.cat((
        torch.full((1, prompt_len), -100, dtype=torch.long),
        tok_answer["input_ids"]
    ), dim=-1).to(device)

    return {"input_ids": input_ids, "labels": labels}


def normalize_passage_text(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def build_inline_negative_sample(sample: dict, raw_negative, fallback_question: str) -> dict | None:
    if isinstance(raw_negative, str):
        passage = raw_negative.strip()
        if not passage:
            return None
        return {
            "question": fallback_question,
            "answer": "__hard_negative__",
            "passage": passage,
            "task": sample.get("task", "final_qa"),
            "negative_type": "explicit",
        }

    if isinstance(raw_negative, dict):
        passage = extract_passage(raw_negative)
        if not passage:
            return None
        return {
            **raw_negative,
            "question": str(raw_negative.get("question") or fallback_question).strip(),
            "answer": extract_answer(raw_negative) or str(
                raw_negative.get("negative_answer")
                or raw_negative.get("counterfactual_answer")
                or "__hard_negative__"
            ),
            "passage": passage,
            "task": raw_negative.get("task") or sample.get("task", "final_qa"),
            "negative_type": "explicit",
        }

    return None


def iter_explicit_negative_samples(sample: dict):
    question = str(sample.get("question", "")).strip()
    for key in (
        "hard_negative_passage",
        "negative_passage",
        "counterfactual_passage",
        "distractor_passage",
    ):
        value = sample.get(key)
        negative = build_inline_negative_sample(sample, value, question)
        if negative is not None:
            yield negative

    for key in ("hard_negatives", "negative_passages", "counterfactuals", "distractors"):
        values = sample.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            negative = build_inline_negative_sample(sample, item, question)
            if negative is not None:
                yield negative


def is_valid_negative(sample: dict, candidate: dict, *, require_diff_passage: bool = True) -> bool:
    if candidate is sample:
        return False
    sample_answer = sample.get("answer")
    sample_passage_key = normalize_passage_text(sample.get("passage", ""))
    candidate_passage_key = normalize_passage_text(candidate.get("passage", ""))
    if require_diff_passage and candidate_passage_key == sample_passage_key:
        return False
    return candidate.get("answer") != sample_answer


def select_negative_from_indices(dataset: MergePRAGDataset, sample: dict, index: int, indices: list[int]):
    candidates = [
        dataset.data[idx]
        for idx in indices
        if idx != index and is_valid_negative(sample, dataset.data[idx])
    ]
    if not candidates:
        return None
    return candidates[index % len(candidates)]


def get_negative_sample(dataset: MergePRAGDataset, index: int):
    sample = dataset[index]
    for explicit_negative in iter_explicit_negative_samples(sample):
        if is_valid_negative(sample, explicit_negative):
            return explicit_negative

    contrast_id = sample.get("contrast_id") or sample.get("group_id")
    if contrast_id is not None:
        contrast_negative = select_negative_from_indices(
            dataset,
            sample,
            index,
            dataset.indices_by_contrast.get(str(contrast_id), []),
        )
        if contrast_negative is not None:
            return contrast_negative

    question_key = normalize_passage_text(sample.get("question", ""))
    if question_key:
        same_question_negative = select_negative_from_indices(
            dataset,
            sample,
            index,
            dataset.indices_by_question.get(question_key, []),
        )
        if same_question_negative is not None:
            return same_question_negative

    source_id = sample.get("source_id") or sample.get("id")
    if source_id is not None:
        source_negative = select_negative_from_indices(
            dataset,
            sample,
            index,
            dataset.indices_by_source.get(str(source_id), []),
        )
        if source_negative is not None:
            return source_negative

    dataset_size = len(dataset)
    for offset in range(1, dataset_size):
        candidate = dataset[(index + offset) % dataset_size]
        if is_valid_negative(sample, candidate):
            return candidate

    # 정말로 다른 passage를 못 찾으면 그때만 기존 fallback을 허용.
    for offset in range(1, dataset_size):
        candidate = dataset[(index + offset) % dataset_size]
        if is_valid_negative(sample, candidate, require_diff_passage=False):
            return candidate

    return dataset[(index + 1) % dataset_size]


def get_same_passage_negative_sample(dataset: MergePRAGDataset, index: int):
    sample = dataset[index]
    sample_answer = sample.get("answer")
    sample_question = normalize_passage_text(sample.get("question", ""))
    sample_passage_key = normalize_passage_text(sample.get("passage", ""))
    if not sample_passage_key:
        return None

    candidates = [
        dataset.data[idx]
        for idx in dataset.indices_by_passage.get(sample_passage_key, [])
        if (
            idx != index
            and dataset.data[idx].get("answer") != sample_answer
            and normalize_passage_text(dataset.data[idx].get("question", "")) != sample_question
        )
    ]
    if not candidates:
        return None
    return candidates[index % len(candidates)]


def compute_repulsion_loss(hidden_pos, hidden_neg, delta_k_pos, delta_k_neg, delta_v_pos, delta_v_neg):
    hidden_sim = F.cosine_similarity(hidden_pos, hidden_neg).mean()
    k_sim = F.cosine_similarity(delta_k_pos.flatten(1), delta_k_neg.flatten(1)).mean()
    v_sim = F.cosine_similarity(delta_v_pos.flatten(1), delta_v_neg.flatten(1)).mean()
    loss = (
        torch.relu(hidden_sim - HIDDEN_SIM_TARGET)
        + torch.relu(k_sim - K_SIM_TARGET)
        + 2.0 * torch.relu(v_sim - V_SIM_TARGET)
    )
    return loss, hidden_sim, k_sim, v_sim


def compute_slot_diversity_loss(delta_k, delta_v):
    """num_kv>1에서 여러 slot이 같은 방향으로 복제되는 것을 약하게 막는다."""
    if delta_k.size(1) <= 1:
        zero = delta_k.new_tensor(0.0)
        nan = float("nan")
        return zero, nan, nan

    def offdiag_penalty(x):
        x = F.normalize(x, dim=-1)
        sim = torch.matmul(x, x.transpose(1, 2))
        num_slots = sim.size(-1)
        mask = ~torch.eye(num_slots, dtype=torch.bool, device=sim.device).unsqueeze(0)
        offdiag = sim.masked_select(mask)
        penalty = torch.relu(offdiag.abs() - SLOT_DIVERSITY_TARGET).mean()
        return penalty, offdiag.abs().mean()

    k_loss, k_abs_sim = offdiag_penalty(delta_k)
    v_loss, v_abs_sim = offdiag_penalty(delta_v)
    return k_loss + v_loss, k_abs_sim.item(), v_abs_sim.item()


# ── Validation ──
def evaluate(model, tokenizer, hypernet, target_layer, dataset, device):
    """Validation loss 계산 (hook per sample, no gradient)"""
    hypernet.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            if idx >= EVAL_MAX_SAMPLES:
                break

            if USE_QUESTION_CONDITIONED_MEMORY:
                encoded = tokenize_conditioned_memory(
                    tokenizer,
                    sample["question"],
                    sample["passage"],
                    device,
                    max_length=MAX_SEQ_LEN,
                )
            else:
                encoded = tokenize_passage_memory(
                    tokenizer,
                    sample["passage"],
                    device,
                    max_length=MAX_SEQ_LEN,
                )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            question_mask = encoded["question_mask"]
            passage_mask = encoded["passage_mask"]
            c_emb = encode_passage_states(
                model,
                input_ids,
                attention_mask=attention_mask,
                use_contextual=USE_CONTEXTUAL_PASSAGE_ENCODER,
            )
            query = masked_mean(c_emb, question_mask) if USE_QUESTION_CONDITIONED_MEMORY else None
            delta_K, delta_V = hypernet(
                c_emb,
                attention_mask=attention_mask,
                query=query,
                focus_mask=passage_mask,
            )

            hook = target_layer.register_forward_hook(make_hook(delta_K, delta_V))
            tok = tokenize_qa(
                tokenizer,
                sample["question"],
                sample["answer"],
                device,
                task=sample.get("task", "final_qa"),
            )
            logits = model(input_ids=tok["input_ids"])["logits"]
            hook.remove()

            loss = compute_loss(logits, tok["labels"])
            if loss is not None:
                total_loss += loss.item()
                count += 1

            # VRAM 정리
            del logits, loss
            torch.cuda.empty_cache()

    hypernet.train()
    return total_loss / max(count, 1)


# ── 차트 저장 ──
def save_chart(log_data):
    """Train/Val loss 곡선 차트 (발표자료용)"""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 좌측: Step별 Train Loss (smoothed)
    steps = [s["step"] for s in log_data["step_losses"]]
    losses = [s["loss"] for s in log_data["step_losses"]]

    window = min(50, len(losses) // 5) if len(losses) > 10 else 1
    if window > 1:
        smoothed = []
        for i in range(len(losses)):
            start = max(0, i - window + 1)
            smoothed.append(sum(losses[start:i+1]) / (i - start + 1))
    else:
        smoothed = losses

    ax1.plot(steps, losses, alpha=0.15, color="#90CAF9", linewidth=0.5)
    ax1.plot(steps, smoothed, color="#2196F3", linewidth=1.5, label="Train Loss (smoothed)")

    if log_data["val_evals"]:
        val_steps = [v["step"] for v in log_data["val_evals"]]
        val_losses = [v["val_loss"] for v in log_data["val_evals"]]
        ax1.plot(val_steps, val_losses, "o-", color="#FF5722", linewidth=2, markersize=4, label="Val Loss")

    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax1.set_title("Training Progress (per step)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 우측: LR 스케줄 곡선
    if log_data.get("lr_history"):
        lr_steps = [l["step"] for l in log_data["lr_history"]]
        lr_vals = [l["lr"] for l in log_data["lr_history"]]
        ax2.plot(lr_steps, lr_vals, color="#4CAF50", linewidth=1.5)
        ax2.set_xlabel("Step", fontsize=12)
        ax2.set_ylabel("Learning Rate", fontsize=12)
        ax2.set_title("Cosine LR Schedule", fontsize=13)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No LR data", ha="center", va="center", fontsize=14)
        ax2.set_title("LR Schedule", fontsize=13)

    plt.suptitle("MergePRAG HyperNetwork Training", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[차트] 저장: {CHART_PATH}")


def current_training_config() -> dict:
    return {
        "model": MODEL_NAME,
        "critical_layer": CRITICAL_LAYER,
        "num_kv": NUM_KV,
        "alpha": ALPHA,
        "max_seq_len": MAX_SEQ_LEN,
        "contextual": USE_CONTEXTUAL_PASSAGE_ENCODER,
        "question_conditioned": USE_QUESTION_CONDITIONED_MEMORY,
        "query_pool_scale": QUERY_POOL_SCALE,
        "kv_path_mode": KV_PATH_MODE,
        "pooled_kv_skip": USE_POOLED_KV_SKIP,
        "pooled_k_skip_scale": POOLED_K_SKIP_SCALE,
        "pooled_v_skip_scale": POOLED_V_SKIP_SCALE,
        "v_rms_clamp": USE_V_RMS_CLAMP,
        "v_rms_clamp_value": V_RMS_CLAMP,
        "train_prompt_format": TRAIN_PROMPT_FORMAT,
        "train_data_path": TRAIN_DATA_PATH,
        "valid_data_path": VALID_DATA_PATH,
        "negative_margin": NEGATIVE_MARGIN,
        "negative_loss_weight": NEGATIVE_LOSS_WEIGHT,
        "repulsion_loss_weight": REPULSION_LOSS_WEIGHT,
        "question_negative_loss_weight": QUESTION_NEGATIVE_LOSS_WEIGHT,
        "question_repulsion_loss_weight": QUESTION_REPULSION_LOSS_WEIGHT,
        "slot_diversity_loss_weight": SLOT_DIVERSITY_LOSS_WEIGHT,
        "slot_diversity_target": SLOT_DIVERSITY_TARGET,
    }


def checkpoint_config_mismatches(saved_config: dict, current_config: dict) -> list[str]:
    checked_keys = (
        "model",
        "critical_layer",
        "num_kv",
        "alpha",
        "max_seq_len",
        "contextual",
        "question_conditioned",
        "query_pool_scale",
        "kv_path_mode",
        "pooled_kv_skip",
        "pooled_k_skip_scale",
        "pooled_v_skip_scale",
        "v_rms_clamp",
        "v_rms_clamp_value",
        "train_prompt_format",
        "train_data_path",
        "valid_data_path",
    )
    mismatches = []
    for key in checked_keys:
        if saved_config.get(key) != current_config.get(key):
            mismatches.append(f"{key}: checkpoint={saved_config.get(key)!r}, current={current_config.get(key)!r}")
    return mismatches


# ── 학습 메인 ──
def train():
    print(f"[학습] 모델 로딩: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # ── Qwen freeze (논문 동일) ──
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    device = next(model.parameters()).device
    d_model = model.config.hidden_size
    target_layer = model.model.layers[CRITICAL_LAYER]
    print(f"[학습] d_model={d_model}, layer={CRITICAL_LAYER}, device={device}")

    # ── HyperNetwork 초기화 ──
    hypernet = HyperNetwork(d_model, k=NUM_KV).to(device).float()

    # ── Optimizer: AdamW (weight_decay=0 — MLP 붕괴 방지) ──
    # 기본 weight_decay=0.01은 훈련 신호가 약할 때 가중치를 0으로 끌어당겨
    # MLP 출력이 bias에 수렴 → 서로 다른 passage가 동일한 K/V로 붕괴함.
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=LR, weight_decay=0.0)

    # ── 데이터 로드 ──
    train_dataset = MergePRAGDataset(TRAIN_DATA_PATH, max_samples=MAX_SAMPLES)
    val_dataset = MergePRAGDataset(VALID_DATA_PATH, max_samples=MAX_VAL_SAMPLES)

    # ── CosineAnnealingLR (논문: T_max=len(train_loader), eta_min=1e-6) ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_dataset), eta_min=LR_MIN
    )

    # ── 체크포인트 복구 (중간에 끊긴 경우) ──
    resume_step = 0
    run_config = current_training_config()
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        try:
            checkpoint_config = ckpt.get("config")
            can_resume = True
            if checkpoint_config is None:
                can_resume = ALLOW_LEGACY_CHECKPOINT_RESUME
                print(
                    "[경고] 체크포인트에 config snapshot이 없습니다. "
                    "설정 불일치로 인한 잘못된 재개를 막기 위해 기본값으로는 새 학습을 시작합니다."
                )
                if not can_resume:
                    print("       정말 이어서 학습하려면 MERGEPRAG_ALLOW_LEGACY_CHECKPOINT_RESUME=true 를 설정하세요.")
            else:
                mismatches = checkpoint_config_mismatches(checkpoint_config, run_config)
                if mismatches and not ALLOW_CONFIG_MISMATCH_RESUME:
                    can_resume = False
                    print("[경고] 현재 설정과 체크포인트 설정이 달라 새 학습을 시작합니다.")
                    for mismatch in mismatches[:10]:
                        print(f"       - {mismatch}")
                    if len(mismatches) > 10:
                        print(f"       - ... and {len(mismatches) - 10} more")
                    print("       강제로 재개하려면 MERGEPRAG_ALLOW_CONFIG_MISMATCH_RESUME=true 를 설정하세요.")

            if can_resume:
                hypernet.load_state_dict(ckpt["hypernet"])
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                resume_step = ckpt["step"]
                best_val_loss = ckpt.get("best_val_loss", float("inf"))
                print(f"[복구] 체크포인트에서 재개: step {resume_step}")
        except (RuntimeError, KeyError, AttributeError) as e:
            print(f"[경고] 체크포인트 차원 불일치 (설정 변경됨), 처음부터 학습: {e}")
            resume_step = 0

    # ── 로그 ──
    log_data = {
        "config": {
            **run_config,
            "lr": LR,
            "lr_min": LR_MIN,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "epochs": EPOCHS,
            "eval_every": EVAL_EVERY,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        },
        "step_losses": [],
        "val_evals": [],
        "lr_history": [],
        "epochs": [],
    }

    if resume_step == 0:
        best_val_loss = float("inf")
    patience_counter = 0
    early_stopped = False

    print(f"\n[학습] 시작: {EPOCHS} epoch, train={len(train_dataset)}, val={len(val_dataset)}")
    print(f"[학습] AdamW lr={LR}, CosineAnnealing eta_min={LR_MIN}")
    print(
        f"[학습] config | num_kv={NUM_KV}, alpha={ALPHA}, "
        f"contextual={USE_CONTEXTUAL_PASSAGE_ENCODER}, kv_path_mode={KV_PATH_MODE}, "
        f"k_skip_scale={POOLED_K_SKIP_SCALE}, v_skip_scale={POOLED_V_SKIP_SCALE}, "
        f"v_rms_clamp={'on' if USE_V_RMS_CLAMP else 'off'}:{V_RMS_CLAMP}, "
        f"question_conditioned={USE_QUESTION_CONDITIONED_MEMORY}, "
        f"query_pool_scale={QUERY_POOL_SCALE}, "
        f"train_prompt_format={TRAIN_PROMPT_FORMAT}, "
        f"slot_diversity={SLOT_DIVERSITY_LOSS_WEIGHT}:{SLOT_DIVERSITY_TARGET}"
    )
    hypernet.train()
    start_time = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    global_step = resume_step  # 체크포인트 재개 시 이어서 카운트

    for epoch in range(EPOCHS):
        if early_stopped:
            break

        total_loss = 0
        count = 0

        for i, sample in enumerate(train_dataset):
            # 체크포인트 복구: 이미 학습한 step 건너뛰기
            if i < resume_step:
                continue

            try:
                # 1. passage → HyperNetwork → delta_K, delta_V (논문: CE loss만)
                passage = sample["passage"]
                input_ids, c_emb, pooled_pos, hidden_pos, delta_K, delta_V = encode_memory(
                    model, hypernet, tokenizer, sample["question"], passage, device
                )

                # 2. Q+A 토큰화 (labels=-100 마스킹)
                tok = tokenize_qa(
                    tokenizer,
                    sample["question"],
                    sample["answer"],
                    device,
                    task=sample.get("task", "final_qa"),
                )

                # 3. memory로 정답 loss — 원본 CE objective 유지
                logits = forward_with_memory(model, target_layer, delta_K, delta_V, tok)
                task_loss = compute_loss(logits, tok["labels"])
                if task_loss is None:
                    continue

                negative_sample = get_negative_sample(train_dataset, i)
                _, neg_emb, pooled_neg, hidden_neg, neg_K, neg_V = encode_memory(
                    model,
                    hypernet,
                    tokenizer,
                    sample["question"],
                    negative_sample["passage"],
                    device,
                )
                neg_logits = forward_with_memory(model, target_layer, neg_K, neg_V, tok)
                neg_task_loss = compute_loss(neg_logits, tok["labels"])
                if neg_task_loss is None:
                    neg_task_loss = task_loss.detach()

                grounding_loss = torch.relu(NEGATIVE_MARGIN + task_loss - neg_task_loss)
                repulsion_loss, hidden_sim_t, k_sim_t, v_sim_t = compute_repulsion_loss(
                    hidden_pos,
                    hidden_neg,
                    delta_K,
                    neg_K,
                    delta_V,
                    neg_V,
                )
                slot_diversity_loss, slot_k_abs_sim, slot_v_abs_sim = compute_slot_diversity_loss(
                    delta_K,
                    delta_V,
                )
                question_neg_loss = torch.tensor(0.0, device=device)
                question_repulsion_loss = torch.tensor(0.0, device=device)
                question_k_sim = float("nan")
                question_v_sim = float("nan")
                same_passage_question_neg = get_same_passage_negative_sample(train_dataset, i)
                if USE_QUESTION_CONDITIONED_MEMORY and same_passage_question_neg is not None:
                    _, same_passage_emb, _, hidden_same_passage, same_passage_K, same_passage_V = encode_memory(
                        model,
                        hypernet,
                        tokenizer,
                        same_passage_question_neg["question"],
                        same_passage_question_neg["passage"],
                        device,
                    )
                    same_passage_logits = forward_with_memory(
                        model,
                        target_layer,
                        same_passage_K,
                        same_passage_V,
                        tok,
                    )
                    same_passage_task_loss = compute_loss(same_passage_logits, tok["labels"])
                    if same_passage_task_loss is None:
                        same_passage_task_loss = task_loss.detach()
                    question_neg_loss = torch.relu(
                        NEGATIVE_MARGIN + task_loss - same_passage_task_loss
                    )
                    question_repulsion_loss, _, question_k_sim_t, question_v_sim_t = compute_repulsion_loss(
                        hidden_pos,
                        hidden_same_passage,
                        delta_K,
                        same_passage_K,
                        delta_V,
                        same_passage_V,
                    )
                    question_k_sim = question_k_sim_t.item()
                    question_v_sim = question_v_sim_t.item()
                loss = (
                    task_loss
                    + NEGATIVE_LOSS_WEIGHT * grounding_loss
                    + REPULSION_LOSS_WEIGHT * repulsion_loss
                    + QUESTION_NEGATIVE_LOSS_WEIGHT * question_neg_loss
                    + QUESTION_REPULSION_LOSS_WEIGHT * question_repulsion_loss
                    + SLOT_DIVERSITY_LOSS_WEIGHT * slot_diversity_loss
                )
                loss_val = task_loss.item()
                k_vec_norm = delta_K.detach().norm(dim=-1).mean().item()
                v_vec_norm = delta_V.detach().norm(dim=-1).mean().item()
                neg_loss_val = neg_task_loss.item()
                grounding_val = grounding_loss.item()
                repulsion_val = repulsion_loss.item()
                slot_diversity_val = slot_diversity_loss.item()
                question_neg_val = question_neg_loss.item()
                question_repulsion_val = question_repulsion_loss.item()
                same_passage_neg = (
                    normalize_passage_text(negative_sample.get("passage", ""))
                    == normalize_passage_text(sample.get("passage", ""))
                )
                pooled_cos = F.cosine_similarity(pooled_pos, pooled_neg).mean().item()
                hidden_sim = hidden_sim_t.item()
                k_sim = k_sim_t.item()
                v_sim = v_sim_t.item()

                # 진단: memory 없이 같은 샘플 평가 → memory 기여도 확인
                # global_step 증가는 아래에서 일어나므로 +1로 로깅 조건과 맞춤
                if (global_step + 1) % LOG_EVERY == 0:
                    with torch.no_grad():
                        base_logits = model(input_ids=tok["input_ids"])["logits"]
                        base_loss = compute_loss(base_logits, tok["labels"])
                    base_loss_val = base_loss.item() if base_loss is not None else float("nan")
                    del base_logits
                else:
                    base_loss_val = None

                # 4. backward → HyperNetwork만 업데이트
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(hypernet.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()

                total_loss += loss_val
                count += 1
                global_step += 1

                # VRAM 정리
                del logits, neg_logits, loss, c_emb, neg_emb, delta_K, delta_V, neg_K, neg_V, input_ids, hidden_pos, hidden_neg, pooled_neg
                if USE_QUESTION_CONDITIONED_MEMORY and same_passage_question_neg is not None:
                    del same_passage_logits, same_passage_K, same_passage_V, hidden_same_passage, same_passage_emb
                if global_step % 100 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  [OOM] step {global_step} 스킵, VRAM 정리 중...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e

            # 로그 기록 — 논문 스타일: CE loss만
            log_data["step_losses"].append({
                "step": global_step,
                "loss": round(loss_val, 4),
                "negative_loss": round(neg_loss_val, 4),
                "grounding_loss": round(grounding_val, 4),
                "repulsion_loss": round(repulsion_val, 4),
                "slot_diversity_loss": round(slot_diversity_val, 4),
                "question_negative_loss": round(question_neg_val, 4),
                "question_repulsion_loss": round(question_repulsion_val, 4),
            })

            if global_step % LOG_EVERY == 0:
                lr_now = scheduler.get_last_lr()[0]
                log_data["lr_history"].append({
                    "step": global_step,
                    "lr": round(lr_now, 8),
                })
                avg = total_loss / count
                elapsed = (time.time() - start_time) / 60
                # base_loss 대비 memory loss가 낮아야 훈련이 실제로 의미 있음
                diag = ""
                if base_loss_val is not None:
                    gain = base_loss_val - loss_val
                    diag = f" | base: {base_loss_val:.4f} (gain: {gain:+.4f})"
                print(
                    f"  Step {global_step}/{len(train_dataset)} | "
                    f"loss: {loss_val:.4f} | avg: {avg:.4f} | lr: {lr_now:.2e} | "
                    f"neg: {neg_loss_val:.4f} | rank: {grounding_val:.4f} | rep: {repulsion_val:.4f} | "
                    f"slot: {slot_diversity_val:.4f} | "
                    f"qneg: {question_neg_val:.4f} | qrep: {question_repulsion_val:.4f} | "
                    f"Knorm: {k_vec_norm:.3f} | Vnorm: {v_vec_norm:.3f} | "
                    f"Pcos: {pooled_cos:.4f} | Hcos: {hidden_sim:.4f} | Kcos: {k_sim:.4f} | Vcos: {v_sim:.4f}"
                    f" | slotK: {slot_k_abs_sim:.4f} | slotV: {slot_v_abs_sim:.4f}"
                    f" | qKcos: {question_k_sim:.4f} | qVcos: {question_v_sim:.4f}"
                    f" | neg_same_passage: {same_passage_neg}"
                    f"{diag} | "
                    f"{elapsed:.1f}min"
                )

            # 중간 체크포인트 저장
            if global_step % SAVE_EVERY == 0:
                torch.save({
                    "step": global_step,
                    "hypernet": hypernet.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": run_config,
                }, CHECKPOINT_PATH)
                # 중간 로그도 저장 (크래시 대비)
                with open(LOG_PATH, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)
                print(f"  [체크포인트] step {global_step} 저장")

            # Validation 평가
            if global_step % EVAL_EVERY == 0:
                val_loss = evaluate(model, tokenizer, hypernet, target_layer, val_dataset, device)
                elapsed = (time.time() - start_time) / 60
                print(f"  ── [Val @ step {global_step}] val_loss: {val_loss:.4f} | {elapsed:.1f}min")

                log_data["val_evals"].append({
                    "step": global_step,
                    "val_loss": round(val_loss, 4),
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(hypernet.state_dict(), SAVE_PATH)
                    print(f"     ★ Best val_loss → 가중치 저장")
                else:
                    patience_counter += 1
                    print(f"     patience: {patience_counter}/{PATIENCE}")
                    if patience_counter >= PATIENCE:
                        print(f"  [Early Stopping] 학습 중단.")
                        early_stopped = True
                        break

        # Epoch 끝
        train_loss = total_loss / max(count, 1)
        val_loss = evaluate(model, tokenizer, hypernet, target_layer, val_dataset, device)
        log_data["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
        })
        print(f"[Epoch {epoch+1}/{EPOCHS}] train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

    # ── 최종 저장 ──
    elapsed_total = time.time() - start_time
    end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data["start_time"] = start_dt
    log_data["end_time"] = end_dt
    log_data["total_time_min"] = round(elapsed_total / 60, 1)
    log_data["best_val_loss"] = round(best_val_loss, 4)
    log_data["early_stopped"] = early_stopped
    log_data["total_steps"] = global_step

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"[로그] 저장: {LOG_PATH}")

    save_chart(log_data)

    if not early_stopped:
        torch.save(hypernet.state_dict(), SAVE_PATH)

    print(f"\n[학습] 완료! 총 {elapsed_total/60:.1f}분 소요")
    print(f"[학습] best_val_loss: {best_val_loss:.4f}, 가중치: {SAVE_PATH}")


if __name__ == "__main__":
    train()
