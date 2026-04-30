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
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import (
    ALPHA,
    CHART_PATH,
    CHECKPOINT_PATH,
    LOG_PATH,
    MAX_SEQ_LEN,
    MODEL_NAME,
    NUM_KV,
    TRAIN_DATA_PATH,
    VALID_DATA_PATH,
    WEIGHTS_PATH as SAVE_PATH,
    load_critical_layer,
)
from .embedding import encode_passage_states, tokenize_conditioned_memory
from .hypernetwork import HyperNetwork
from .cross_attention import cross_attention

# ── 설정 (논문 기본값 기반) ──
CRITICAL_LAYER = load_critical_layer()
LR = 1e-4                   # 논문 동일
LR_MIN = 1e-6               # 논문 CosineAnnealing eta_min
EPOCHS = 1                  # 논문: 1 epoch (single pass)
MAX_SAMPLES = None           # 전체 학습
MAX_VAL_SAMPLES = None       # 전체 검증
EVAL_EVERY = 1000           # N step마다 validation
EVAL_MAX_SAMPLES = 500      # validation 시 최대 샘플 수 (전체 순회 방지)
LOG_EVERY = 50              # N step마다 터미널 출력 (논문: 49)
SAVE_EVERY = 500            # 중간 체크포인트 저장 주기
PATIENCE = 5                # early stopping patience
TRAIN_MAX_PASSAGE_SENTENCES = 6
TRAIN_SYSTEM_PREFIX = "Answer the question using the passage-grounded fact."
NEGATIVE_MARGIN = 0.5
NEGATIVE_LOSS_WEIGHT = 0.5
REPULSION_LOSS_WEIGHT = 0.1
USE_CONTEXTUAL_PASSAGE_ENCODER = True


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


def extract_passage(sample: dict) -> str:
    """서비스용 QA-passage 학습에 맞는 근거 passage를 선택한다."""
    passage = sample.get("passage")
    if isinstance(passage, str) and passage.strip():
        return passage.strip()

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
        for idx, sample in enumerate(self.data):
            source_id = sample.get("source_id") or sample.get("id")
            if source_id is None:
                continue
            self.indices_by_source.setdefault(str(source_id), []).append(idx)
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
    encoded = tokenize_conditioned_memory(
        tokenizer,
        question,
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
    query = masked_mean(embedded, question_mask)
    pooled, hidden, delta_K, delta_V = hypernet.encode_embedded(
        embedded,
        attention_mask=attention_mask,
        query=query,
        focus_mask=passage_mask,
    )
    return input_ids, embedded, pooled, hidden, delta_K, delta_V


def compute_repulsion_loss(hidden_pos, hidden_neg, delta_k_pos, delta_k_neg, delta_v_pos, delta_v_neg):
    hidden_sim = F.cosine_similarity(hidden_pos, hidden_neg).mean()
    k_sim = F.cosine_similarity(delta_k_pos.flatten(1), delta_k_neg.flatten(1)).mean()
    v_sim = F.cosine_similarity(delta_v_pos.flatten(1), delta_v_neg.flatten(1)).mean()
    return torch.clamp((hidden_sim + k_sim + v_sim) / 3.0, min=0.0)


def forward_with_memory(model, target_layer, delta_K, delta_V, tok):
    hook = target_layer.register_forward_hook(make_hook(delta_K, delta_V))
    try:
        logits = model(input_ids=tok["input_ids"])["logits"]
    finally:
        hook.remove()
    return logits


def build_training_prompt(question: str) -> str:
    return f"{TRAIN_SYSTEM_PREFIX}\nQuestion: {question}\nAnswer:"


def build_prompt_for_task(question: str, task: str) -> str:
    if task == "hop_fact":
        return (
            "Read the passage and output the supporting fact that is currently grounded.\n"
            f"Question: {question}\n"
            "Supporting fact:"
        )
    return build_training_prompt(question)


def tokenize_qa(tokenizer, question, answer, device, task: str = "final_qa"):
    """논문 원본과 더 가까운 단순 QA 포맷으로 토큰화."""
    prompt = build_prompt_for_task(question, task)
    answer_text = f" {answer}{tokenizer.eos_token}"

    tok_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    tok_answer = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN)

    input_ids = torch.cat((tok_prompt["input_ids"], tok_answer["input_ids"][:, :-1]), dim=-1).to(device)
    labels = torch.cat((
        torch.full((1, tok_prompt["input_ids"].shape[1] - 1), -100, dtype=torch.long),
        tok_answer["input_ids"]
    ), dim=-1).to(device)

    return {"input_ids": input_ids, "labels": labels}


def get_negative_sample(dataset: MergePRAGDataset, index: int):
    sample = dataset[index]
    source_id = sample.get("source_id") or sample.get("id")
    if source_id is not None:
        candidates = [
            dataset.data[idx]
            for idx in dataset.indices_by_source.get(str(source_id), [])
            if idx != index and dataset.data[idx].get("answer") != sample.get("answer")
        ]
        if candidates:
            return candidates[index % len(candidates)]

    dataset_size = len(dataset)
    for offset in range(1, dataset_size):
        candidate = dataset[(index + offset) % dataset_size]
        if candidate.get("answer") != sample.get("answer"):
            return candidate
    return dataset[(index + 1) % dataset_size]


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

            encoded = tokenize_conditioned_memory(
                tokenizer,
                sample["question"],
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
            query = masked_mean(c_emb, question_mask)
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
        torch_dtype=torch.float16,
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

    # ── Optimizer: AdamW (논문 동일) ──
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=LR)

    # ── 데이터 로드 ──
    train_dataset = MergePRAGDataset(TRAIN_DATA_PATH, max_samples=MAX_SAMPLES)
    val_dataset = MergePRAGDataset(VALID_DATA_PATH, max_samples=MAX_VAL_SAMPLES)

    # ── CosineAnnealingLR (논문: T_max=len(train_loader), eta_min=1e-6) ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_dataset), eta_min=LR_MIN
    )

    # ── 체크포인트 복구 (중간에 끊긴 경우) ──
    resume_step = 0
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        hypernet.load_state_dict(ckpt["hypernet"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        resume_step = ckpt["step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[복구] 체크포인트에서 재개: step {resume_step}")

    # ── 로그 ──
    log_data = {
        "config": {
            "model": MODEL_NAME,
            "critical_layer": CRITICAL_LAYER,
            "num_kv": NUM_KV,
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
                # 1. positive passage → HyperNetwork → delta_K, delta_V
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

                # 3. positive memory로 정답 loss 계산
                logits = forward_with_memory(model, target_layer, delta_K, delta_V, tok)
                task_loss = compute_loss(logits, tok["labels"])
                if task_loss is None:
                    continue

                # 4. 다른 샘플 passage를 negative memory로 사용해 grounding ranking 추가
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
                repulsion_loss = compute_repulsion_loss(
                    hidden_pos,
                    hidden_neg,
                    delta_K,
                    neg_K,
                    delta_V,
                    neg_V,
                )
                loss = (
                    task_loss
                    + NEGATIVE_LOSS_WEIGHT * grounding_loss
                    + REPULSION_LOSS_WEIGHT * repulsion_loss
                )

                # 5. backward → HyperNetwork만 업데이트
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_val = task_loss.item()
                grounding_val = grounding_loss.item()
                repulsion_val = repulsion_loss.item()
                neg_loss_val = neg_task_loss.item()
                k_vec_norm = delta_K.squeeze(0).norm(dim=-1).mean().item()
                v_vec_norm = delta_V.squeeze(0).norm(dim=-1).mean().item()
                total_loss += loss_val
                count += 1
                global_step += 1

                # VRAM 정리
                del logits, neg_logits, loss, c_emb, neg_emb, delta_K, delta_V, neg_K, neg_V, input_ids
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

            # 로그 기록
            log_data["step_losses"].append({
                "step": global_step,
                "loss": round(loss_val, 4),
                "negative_loss": round(neg_loss_val, 4),
                "grounding_loss": round(grounding_val, 4),
                "repulsion_loss": round(repulsion_val, 4),
            })

            if global_step % LOG_EVERY == 0:
                lr_now = scheduler.get_last_lr()[0]
                log_data["lr_history"].append({
                    "step": global_step,
                    "lr": round(lr_now, 8),
                })
                avg = total_loss / count
                elapsed = (time.time() - start_time) / 60
                print(
                    f"  Step {global_step}/{len(train_dataset)} | "
                    f"loss: {loss_val:.4f} | avg: {avg:.4f} | lr: {lr_now:.2e} | "
                    f"neg: {neg_loss_val:.4f} | rank: {grounding_val:.4f} | rep: {repulsion_val:.4f} | "
                    f"Knorm: {k_vec_norm:.3f} | Vnorm: {v_vec_norm:.3f} | "
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
