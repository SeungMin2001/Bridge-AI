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
MAX_SAMPLES = 500  # 학습 데이터 수 제한
MAX_VAL_SAMPLES = 100  # 검증 데이터 수 제한
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
    """학습/검증 loss 곡선 차트 저장 (발표자료용)"""
    epochs = [e["epoch"] for e in log_data["epochs"]]
    train_losses = [e["train_loss"] for e in log_data["epochs"]]
    val_losses = [e["val_loss"] for e in log_data["epochs"]]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, "o-", label="Train Loss", color="#2196F3", linewidth=2)
    plt.plot(epochs, val_losses, "s--", label="Validation Loss", color="#FF5722", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.title("MergePRAG HyperNetwork Training", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
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

    # ── 로그 초기화 ──
    log_data = {
        "config": {
            "model": MODEL_NAME,
            "critical_layer": CRITICAL_LAYER,
            "k_dim": K_DIM,
            "alpha": ALPHA,
            "lr": LR,
            "epochs": EPOCHS,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        },
        "epochs": [],
        "step_losses": [],  # 스텝별 loss (상세 곡선용)
    }

    # ── 학습 루프 ──
    print(f"\n[학습] 시작: {EPOCHS} epochs, train={len(train_dataset)}, val={len(val_dataset)}, lr={LR}")
    hypernet.train()

    global_step = 0
    for epoch in range(EPOCHS):
        total_loss = 0
        count = 0

        for i, sample in enumerate(train_dataset):
            # 1. passage(facts) → embedding → HyperNetwork → K, V
            passage = " ".join(sample["facts"])
            embedded = embedding(model, tokenizer, passage)
            embedded = embedded.to(dtype=torch.float32)  # HyperNetwork는 float32
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

            # 4. forward (Qwen은 eval이지만 hook의 cross_attention을 통해 gradient 흐름)
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

            # 6. backward → HyperNetwork만 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

            if (i + 1) % 50 == 0:
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

    # ── 로그 저장 ──
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"[로그] 저장: {LOG_PATH}")

    # ── 차트 저장 ──
    save_chart(log_data)

    # ── 가중치 저장 ──
    torch.save(hypernet.state_dict(), SAVE_PATH)
    print(f"\n[학습] 완료! 가중치 저장: {SAVE_PATH}")
    print(f"[학습] 설정: critical_layer={CRITICAL_LAYER}, k={K_DIM}, alpha={ALPHA}, d_model={d_model}")


if __name__ == "__main__":
    train()
