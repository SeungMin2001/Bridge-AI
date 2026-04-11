"""
MergePRAG 핵심 테스트: LLM Only vs MergePRAG 답변 비교
사용법: cd llm_server && python test_mergeprag.py
"""
import torch
import re
from run_model import run_model
from mergePRAG.hypernetwork import HyperNetwork
from mergePRAG.cross_attention import cross_attention

# ── 설정 ──
PASSAGE = "The capital of Australia is Canberra, not Sydney. Canberra was chosen as the capital in 1908 as a compromise between Sydney and Melbourne."
QUESTION = "What is the capital of Australia?"
CRITICAL_LAYER = 0
WEIGHTS = __import__('os').path.join(__import__('os').path.dirname(__file__), "mergePRAG", "hypernet_weights.pt")

# ── 모델 로드 ──
print("모델 로딩...")
model, tokenizer = run_model()
device = next(model.parameters()).device

# ── HyperNetwork 로드 ──
d_model = model.config.hidden_size
hypernet = HyperNetwork(d_model, k=16).to(device).float()
hypernet.load_state_dict(torch.load(WEIGHTS, map_location=device))
hypernet.eval()

# ── 1. passage → embedding → HyperNetwork → K, V ──
print(f"\npassage: {PASSAGE}")
with torch.no_grad():
    ids = tokenizer(PASSAGE, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
    emb = model.model.embed_tokens(ids).to(torch.float32)
    K, V = hypernet(emb)
print(f"K shape: {K.shape}, norm: {K.norm():.2f}")
print(f"V shape: {V.shape}, norm: {V.norm():.2f}")

# ── 2. hook: cross_attention으로 주입 (논문 방식) ──
def make_hook(dK, dV):
    """첫 forward pass(prefill)에서만 inject, 이후 토큰 생성에서는 무시"""
    fired = [False]
    def hook_fn(module, input, output):
        if fired[0]:
            return output  # 이미 inject 했으면 패스
        fired[0] = True
        hidden = output[0] if isinstance(output, tuple) else output
        Kd = dK.to(device=hidden.device, dtype=hidden.dtype)
        Vd = dV.to(device=hidden.device, dtype=hidden.dtype)
        delta = cross_attention(hidden, Kd, Vd)
        result = hidden + delta
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result
    return hook_fn

# ── 3. 생성 함수 ──
def generate(prompt, max_new=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return re.sub(r'<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>', '', text).strip()

prompt = f"Question: {QUESTION}\nAnswer:"

# ── [A] LLM Only ──
print(f"\n{'='*50}")
print("[A] LLM Only")
print(generate(prompt))

# ── [B] MergePRAG (hook inject) ──
print(f"\n{'='*50}")
print("[B] MergePRAG")
layer = model.model.layers[CRITICAL_LAYER]
hook = layer.register_forward_hook(make_hook(K, V))
print(generate(prompt))
hook.remove()

# ── 판정 ──
print(f"\n{'='*50}")
print("판정: MergePRAG 답변에 아래 키워드가 있으면 성공")
print("  - '1908' (passage에만 있는 연도)")
print("  - 'compromise' (passage에만 있는 표현)")
print("  - 'Sydney and Melbourne' (passage에만 있는 맥락)")
