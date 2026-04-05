"""
MergePRAG HyperNetwork 학습 스크립트

- HotPotQA 데이터로 학습
- Qwen은 freeze, HyperNetwork만 학습
- Critical Layer 0에 K,V를 주입하고 next-token prediction loss로 학습
- Validation loss 측정 + 학습 곡선 시각화 (발표자료용)

사용법: python -m llm_server.mergePRAG.train
"""
import torch
import torch.nn.functional as F
import json
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .hypernetwork import HyperNetwork
from .embedding import embedding
from .cross_attention import cross_attention

# ── 설정 ──
MODEL_NAME = "Qwen/Qwen3.5-4B"
CRITICAL_LAYER = 0  # find_critical_layers.py에서 찾은 레이어
K_DIM = 16
ALPHA = 0.01
LR = 1e-4
EPOCHS = 3
BATCH_SIZE = 1  # VRAM 제약
MAX_SAMPLES = None  # 전체 학습
MAX_VAL_SAMPLES = None  # 전체 검증
EVAL_EVERY = 1000  # N step마다 validation 평가
PATIENCE = 3  # val loss 개선 없으면 early stopping (eval 횟수 기준)
GRAD_ACCUM_STEPS = 4  # gradient accumulation (실질 batch=4)
SAVE_PATH = "llm_server/mergePRAG/hypernet_weights.pt"
LOG_PATH = "llm_server/mergePRAG/train_log.json"
CHART_PATH = "llm_server/mergePRAG/train_loss_curve.png"
TRAIN_DATA_PATH = r"C:\Users\user\Documents\last_project\data\HotPot_train_min.jsonl"
VALID_DATA_PATH = r"C:\Users\user\Documents\last_project\data\HotPot_valid_min.jsonl"


# ── 데이터셋 ──
class MergePRAGDataset(Dataset):
    def __init__(self, jsonl_path, max_samples=None):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                item = json.loads(line.strip())
                if item["facts"]:  # facts가 있는 샘플만
                    self.data.append(item)
        print(f"[데이터] {len(self.data)}개 샘플 로드됨")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Hook 클래스 (gradient 흐름 유지) ──
class InjectHook:
    """forward hook으로 K,V를 주입. backward가 HyperNetwork로 흐르도록 함."""
    def __init__(self):
        self.K = None
        self.V = None

    def set_kv(self, K, V):
        self.K = K
        self.V = V

    def __call__(self, module, input, output):
        if self.K is None:
            return output

        if isinstance(output, tuple):
            hidden = output[0]
            K_ = self.K.to(device=hidden.device, dtype=hidden.dtype)
            V_ = self.V.to(device=hidden.device, dtype=hidden.dtype)
            modified = hidden + ALPHA * cross_attention(hidden, K_, V_)
            return (modified,) + output[1:]
        else:
            K_ = self.K.to(device=output.device, dtype=output.dtype)
            V_ = self.V.to(device=output.device, dtype=output.dtype)
            return output + ALPHA * cross_attention(output, K_, V_)


def evaluate(model, tokenizer, hypernet, inject_hook, dataset, device):
    """Validation 데이터셋에 대한 loss 계산 (no gradient)"""
    hypernet.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for sample in dataset:
            passage = " ".join(sample["facts"])
            embedded = embedding(model, tokenizer, passage)
            embedded = embedded.to(dtype=torch.float32)
            K, V = hypernet(embedded)
            inject_hook.set_kv(K, V)

            question = sample["question"]
            answer = sample["answer"]
            full_text = question + " " + answer

            question_ids = tokenizer(question, return_tensors="pt")["input_ids"]
            full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
            prompt_len = question_ids.shape[1]

            outputs = model(full_ids)
            logits = outputs.logits

            shift_logits = logits[:, prompt_len - 1:-1, :]
            shift_labels = full_ids[:, prompt_len:]

            if shift_labels.shape[1] == 0:
                inject_hook.set_kv(None, None)
                continue

            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            total_loss += loss.item()
            count += 1
            inject_hook.set_kv(None, None)

    hypernet.train()
    return total_loss / max(count, 1)


def save_chart(log_data):
    """학습/검증 loss 곡선 차트 저장 (발표자료용) - step 기반"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── 좌측: Step별 Train Loss (smoothed) ──
    steps = [s["step"] for s in log_data["step_losses"]]
    losses = [s["loss"] for s in log_data["step_losses"]]

    # 이동평균으로 스무딩 (window=50)
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

    # val loss 포인트도 좌측에 표시
    val_steps = [v["step"] for v in log_data["val_evals"]]
    val_losses = [v["val_loss"] for v in log_data["val_evals"]]
    ax1.plot(val_steps, val_losses, "o-", color="#FF5722", linewidth=2, markersize=4, label="Val Loss")

    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax1.set_title("Training Progress (per step)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── 우측: Epoch별 Train vs Val ──
    epochs = [e["epoch"] for e in log_data["epochs"]]
    train_losses_ep = [e["train_loss"] for e in log_data["epochs"]]
    val_losses_ep = [e["val_loss"] for e in log_data["epochs"]]

    ax2.plot(epochs, train_losses_ep, "o-", label="Train Loss", color="#2196F3", linewidth=2, markersize=6)
    ax2.plot(epochs, val_losses_ep, "s--", label="Validation Loss", color="#FF5722", linewidth=2, markersize=6)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax2.set_title("Epoch Summary", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("MergePRAG HyperNetwork Training", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[차트] 저장: {CHART_PATH}")


def train():
    # ── 모델 로드 ──
    print(f"[학습] 모델 로딩: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    model.eval()  # Qwen은 항상 eval (freeze)
    for param in model.parameters():
        param.requires_grad = False

    device = next(model.parameters()).device
    d_model = model.config.hidden_size
    print(f"[학습] d_model={d_model}, critical_layer={CRITICAL_LAYER}, device={device}")

    # ── HyperNetwork 초기화 ──
    hypernet = HyperNetwork(d_model, k=K_DIM).to(device).float()
    optimizer = torch.optim.Adam(hypernet.parameters(), lr=LR)

    # ── Hook 등록 ──
    inject_hook = InjectHook()
    layers = model.model.layers
    handle = layers[CRITICAL_LAYER].register_forward_hook(inject_hook)

    # ── 데이터 로드 ──
    train_dataset = MergePRAGDataset(TRAIN_DATA_PATH, max_samples=MAX_SAMPLES)
    val_dataset = MergePRAGDataset(VALID_DATA_PATH, max_samples=MAX_VAL_SAMPLES)

    # ── Cosine LR Scheduler ──
    total_steps = len(train_dataset) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── 로그 초기화 ──
    log_data = {
        "config": {
            "model": MODEL_NAME,
            "critical_layer": CRITICAL_LAYER,
            "k_dim": K_DIM,
            "alpha": ALPHA,
            "lr": LR,
            "epochs": EPOCHS,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "eval_every": EVAL_EVERY,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        },
        "epochs": [],
        "step_losses": [],  # 스텝별 loss (상세 곡선용)
        "val_evals": [],    # step별 val loss (차트용)
    }

    # ── Early Stopping 상태 ──
    best_val_loss = float("inf")
    patience_counter = 0
    early_stopped = False

    # ── 학습 루프 ──
    print(f"\n[학습] 시작: {EPOCHS} epochs, train={len(train_dataset)}, val={len(val_dataset)}")
    print(f"[학습] grad_accum={GRAD_ACCUM_STEPS}, eval_every={EVAL_EVERY} steps, patience={PATIENCE}")
    hypernet.train()
    start_time = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(EPOCHS):
        if early_stopped:
            break

        total_loss = 0
        count = 0

        for i, sample in enumerate(train_dataset):
            # 1. passage(facts) → embedding → HyperNetwork → K, V
            passage = " ".join(sample["facts"])
            embedded = embedding(model, tokenizer, passage)
            embedded = embedded.to(dtype=torch.float32)
            K, V = hypernet(embedded)

            # 2. K, V를 hook에 설정
            inject_hook.set_kv(K, V)

            # 3. question + answer → 토큰화
            question = sample["question"]
            answer = sample["answer"]
            full_text = question + " " + answer

            question_ids = tokenizer(question, return_tensors="pt")["input_ids"]
            full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
            prompt_len = question_ids.shape[1]

            # 4. forward
            outputs = model(full_ids)
            logits = outputs.logits

            # 5. answer 부분만 loss 계산
            shift_logits = logits[:, prompt_len - 1:-1, :]
            shift_labels = full_ids[:, prompt_len:]

            if shift_labels.shape[1] == 0:
                inject_hook.set_kv(None, None)
                continue

            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            # gradient accumulation: loss를 나눠서 누적
            (loss / GRAD_ACCUM_STEPS).backward()

            loss_val = loss.item()
            total_loss += loss_val
            count += 1
            global_step += 1

            # 스텝별 loss 기록
            log_data["step_losses"].append({
                "epoch": epoch + 1,
                "step": global_step,
                "loss": round(loss_val, 4),
            })

            # hook 초기화
            inject_hook.set_kv(None, None)

            # 6. gradient accumulation: N step마다 optimizer step
            if global_step % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # 7. Step별 validation 평가
            if global_step % EVAL_EVERY == 0:
                val_loss = evaluate(model, tokenizer, hypernet, inject_hook, val_dataset, device)
                elapsed = time.time() - start_time
                lr_now = scheduler.get_last_lr()[0]
                print(f"  [Step {global_step}] train_avg: {total_loss/count:.4f} | val: {val_loss:.4f} | lr: {lr_now:.2e} | {elapsed/60:.1f}min")

                log_data["val_evals"].append({
                    "step": global_step,
                    "val_loss": round(val_loss, 4),
                })

                # Early Stopping 체크
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # best 모델 저장
                    torch.save(hypernet.state_dict(), SAVE_PATH)
                    print(f"    ★ Best val_loss: {val_loss:.4f} → 가중치 저장")
                else:
                    patience_counter += 1
                    print(f"    patience: {patience_counter}/{PATIENCE}")
                    if patience_counter >= PATIENCE:
                        print(f"  [Early Stopping] val_loss 개선 없음 ({PATIENCE}회). 학습 중단.")
                        early_stopped = True
                        break

            if (i + 1) % 200 == 0:
                avg = total_loss / count
                print(f"  [Epoch {epoch+1}/{EPOCHS}] Step {i+1}/{len(train_dataset)} | avg_loss: {avg:.4f}")

        # ── Epoch 끝: validation ──
        train_loss = total_loss / max(count, 1)
        val_loss = evaluate(model, tokenizer, hypernet, inject_hook, val_dataset, device)

        log_data["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
        })

        print(f"[Epoch {epoch+1}/{EPOCHS}] train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

    # ── hook 해제 ──
    handle.remove()

    # ── 최종 저장 ──
    elapsed_total = time.time() - start_time
    end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data["start_time"] = start_dt
    log_data["end_time"] = end_dt
    log_data["total_time_min"] = round(elapsed_total / 60, 1)
    log_data["best_val_loss"] = round(best_val_loss, 4)
    log_data["early_stopped"] = early_stopped

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"[로그] 저장: {LOG_PATH}")

    save_chart(log_data)

    # early stopping이 안 됐으면 마지막 가중치 저장
    if not early_stopped:
        torch.save(hypernet.state_dict(), SAVE_PATH)

    print(f"\n[학습] 완료! 총 {elapsed_total/60:.1f}분 소요")
    print(f"[학습] best_val_loss: {best_val_loss:.4f}, 가중치: {SAVE_PATH}")


if __name__ == "__main__":
    train()
