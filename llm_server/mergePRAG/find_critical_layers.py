"""
Critical Layer Finder
- Qwen의 각 레이어에 랜덤 K,V를 주입했을 때
  answer 토큰의 loss 변화를 측정하여 critical layer를 찾는다.
- 1회 오프라인 실행 후 결과를 저장.

사용법: python -m llm_server.mergePRAG.find_critical_layers
"""
import torch
import torch.nn.functional as F
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .cross_attention import cross_attention

# ── 설정 ──
MODEL_NAME = "Qwen/Qwen3.5-4B"  # 원본 모델 (레이어 구조는 AWQ와 동일)
K_DIM = 16  # HyperNetwork의 k
ALPHA = 0.01  # injection 스케일 (main.py의 make_hook과 동일)
OUTPUT_PATH = "llm_server/mergePRAG/critical_layers.json"

# ── 테스트용 QA 샘플 (passage, question, answer) ──
TEST_SAMPLES = [
    {
        "passage": "데드락은 서로 자원을 기다리며 무한 대기하는 상태다. 조건을 하나라도 깨뜨리면 데드락을 예방할 수 있다.",
        "question": "데드락을 예방하는 방법은?",
        "answer": "조건을 하나라도 깨뜨리면 데드락을 예방할 수 있다.",
    },
    {
        "passage": "TCP는 신뢰성을 중시하고 UDP는 속도를 중시한다. OSI 7계층은 네트워크 통신 역할을 계층별로 설명하는 모델이다.",
        "question": "TCP와 UDP의 차이는?",
        "answer": "TCP는 신뢰성을 중시하고 UDP는 속도를 중시한다.",
    },
    {
        "passage": "GDP는 일정 기간 국내 최종 생산물의 시장가치 합이다. 중간재를 제외해야 GDP 이중계산을 피할 수 있다.",
        "question": "GDP란 무엇인가?",
        "answer": "GDP는 일정 기간 국내 최종 생산물의 시장가치 합이다.",
    },
    {
        "passage": "기억은 부호화, 저장, 인출의 과정으로 설명된다. 분산학습은 망각을 줄이는 데 효과적이다.",
        "question": "기억의 과정은?",
        "answer": "기억은 부호화, 저장, 인출의 과정으로 설명된다.",
    },
]


def load_model():
    print(f"[Critical Layer Finder] 모델 로딩: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
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
    model.eval()
    return model, tokenizer


def get_decoder_layers(model):
    """Qwen 모델의 디코더 레이어 리스트 반환"""
    return model.model.layers


def compute_loss(model, tokenizer, question, answer):
    """question+answer 입력에서 answer 부분의 cross-entropy loss 계산.
    logits[t]는 t+1을 예측 → logits[prompt_len-1]이 첫 answer 토큰을 예측."""
    prompt = question
    full_text = prompt + answer

    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]

    prompt_len = prompt_ids.shape[1]
    device = next(model.parameters()).device
    full_ids = full_ids.to(device)

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # LM shift: logits[t] → labels[t+1]
    # answer 예측: logits[prompt_len-1 : -1] → labels[prompt_len : ]
    shift_logits = logits[:, prompt_len - 1:-1, :]
    shift_labels = full_ids[:, prompt_len:]

    if shift_labels.numel() == 0:
        return 0.0

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    )
    return loss.item()


def compute_loss_with_hook(model, tokenizer, question, answer, layer_idx, K, V):
    """특정 레이어에 K,V hook을 걸고 loss 계산"""
    layers = get_decoder_layers(model)
    target_layer = layers[layer_idx]

    def hook_fn(module, input, output):
        # output이 tuple인 경우 첫 번째가 hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
            K_ = K.to(device=hidden.device, dtype=hidden.dtype)
            V_ = V.to(device=hidden.device, dtype=hidden.dtype)
            modified = hidden + ALPHA * cross_attention(hidden, K_, V_)
            return (modified,) + output[1:]
        else:
            K_ = K.to(device=output.device, dtype=output.dtype)
            V_ = V.to(device=output.device, dtype=output.dtype)
            return output + ALPHA * cross_attention(output, K_, V_)

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        loss = compute_loss(model, tokenizer, question, answer)
    finally:
        handle.remove()

    return loss


def find_critical_layers(model, tokenizer, top_n=5):
    """모든 레이어를 테스트하여 critical layer를 찾는다."""
    layers = get_decoder_layers(model)
    num_layers = len(layers)
    d_model = model.config.hidden_size
    device = next(model.parameters()).device

    print(f"[Critical Layer Finder] 총 {num_layers}개 레이어 분석 시작")
    print(f"[Critical Layer Finder] d_model={d_model}, k={K_DIM}")

    # 랜덤 K, V 생성 (모든 레이어에 동일하게 사용)
    torch.manual_seed(42)
    K = torch.randn(1, K_DIM, d_model, device=device)
    V = torch.randn(1, K_DIM, d_model, device=device)

    # 1. baseline loss (hook 없이)
    print("\n[1/2] Baseline loss 계산 중...")
    baseline_losses = []
    for sample in TEST_SAMPLES:
        loss = compute_loss(model, tokenizer, sample["question"], sample["answer"])
        baseline_losses.append(loss)
        print(f"  Q: {sample['question'][:30]}... → baseline loss: {loss:.4f}")
    avg_baseline = sum(baseline_losses) / len(baseline_losses)
    print(f"  평균 baseline loss: {avg_baseline:.4f}\n")

    # 2. 각 레이어별 loss 계산
    print("[2/2] 레이어별 loss 측정 중...")
    layer_results = []

    for layer_idx in range(num_layers):
        layer_losses = []
        for sample in TEST_SAMPLES:
            loss = compute_loss_with_hook(
                model, tokenizer,
                sample["question"], sample["answer"],
                layer_idx, K, V,
            )
            layer_losses.append(loss)

        avg_loss = sum(layer_losses) / len(layer_losses)
        delta = avg_baseline - avg_loss  # 양수면 loss가 줄어든 것 (좋음)
        layer_results.append({
            "layer": layer_idx,
            "avg_loss": round(avg_loss, 4),
            "delta": round(delta, 4),
        })
        marker = " ★" if abs(delta) > 0.05 else ""
        print(f"  Layer {layer_idx:2d}: loss={avg_loss:.4f}, delta={delta:+.4f}{marker}")

    # 3. delta 크기 순으로 정렬 (loss가 가장 많이 줄어든 순)
    layer_results.sort(key=lambda x: x["delta"], reverse=True)
    critical = [r["layer"] for r in layer_results[:top_n]]

    print(f"\n{'='*50}")
    print(f"Critical Layers (top-{top_n}): {critical}")
    print(f"{'='*50}")

    # 4. 결과 저장
    result = {
        "model": MODEL_NAME,
        "num_layers": num_layers,
        "d_model": d_model,
        "k": K_DIM,
        "alpha": ALPHA,
        "baseline_loss": round(avg_baseline, 4),
        "critical_layers": critical,
        "all_layers": layer_results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {OUTPUT_PATH}")

    return critical


if __name__ == "__main__":
    model, tokenizer = load_model()
    critical = find_critical_layers(model, tokenizer, top_n=5)
    print(f"\n최종 critical layers: {critical}")
    print("이 레이어 번호를 학습 및 서비스에서 사용하세요.")
