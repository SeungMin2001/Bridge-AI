"""
MergePRAG 핵심 진단: hook이 모델 예측을 바꾸는지 확인
generate 없이 단일 forward pass로 직접 비교

사용법: cd llm_server && python test_mergeprag.py
"""
import torch
from run_model import run_model
from mergePRAG.hypernetwork import HyperNetwork
from mergePRAG.cross_attention import cross_attention
import os

PASSAGE = "shin is sunmoon university student"
QUESTION = "Who is shin?"
PROMPT = f"Question: {QUESTION}\nAnswer: /no_think"
CRITICAL_LAYER = 9
WEIGHTS = os.path.join(os.path.dirname(__file__), "mergePRAG", "hypernet_weights.pt")

# ── 모델 로드 ──
print("모델 로딩...")
model, tokenizer = run_model()
device = next(model.parameters()).device

# ── HyperNetwork → K, V ──
d_model = model.config.hidden_size
hypernet = HyperNetwork(d_model, k=16).to(device).float()
hypernet.load_state_dict(torch.load(WEIGHTS, map_location=device))
hypernet.eval()

with torch.no_grad():
    ids = tokenizer(PASSAGE, return_tensors="pt")["input_ids"].to(device)
    emb = model.model.embed_tokens(ids).to(torch.float32)
    K, V = hypernet(emb)

print(f"passage: {PASSAGE}")
print(f"K norm: {K.norm():.4f}, V norm: {V.norm():.4f}")
print(f"K per-vector norm: {K[0,0].norm():.4f}")  # L2 정규화 됐으면 ~1.0

# ── 다른 passage K,V와 비교 ──
with torch.no_grad():
    ids2 = tokenizer("Python was created by Guido van Rossum in 1991.", return_tensors="pt")["input_ids"].to(device)
    emb2 = model.model.embed_tokens(ids2).to(torch.float32)
    K2, V2 = hypernet(emb2)

sim = torch.nn.functional.cosine_similarity(K.view(1,-1), K2.view(1,-1)).item()
print(f"\n두 passage K 유사도: {sim:.4f}")
print(f"  (1.0 = 구분 못함 / 0.0~0.5 = 잘 구분)")

# ── 테스트 1: hook 없이 forward → top-5 예측 ──
print(f"\n{'='*50}")
print(f"prompt: '{PROMPT}'")
print(f"{'='*50}")

inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

def make_hook(dK, dV, alpha=1.0, diag=False):
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        Kd = dK.to(device=hidden.device, dtype=hidden.dtype)
        Vd = dV.to(device=hidden.device, dtype=hidden.dtype)
        delta = cross_attention(hidden, Kd, Vd)
        if diag:
            h_norm = hidden.norm().item()
            d_norm = delta.norm().item()
            ratio = d_norm / (h_norm + 1e-8)
            print(f"  [diag] hidden norm={h_norm:.1f}, delta norm={d_norm:.1f}, ratio={ratio:.4f}")
        result = hidden + alpha * delta
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result
    return hook_fn

layer = model.model.layers[CRITICAL_LAYER]

# ── 문장 생성 비교 ──
print(f"\n{'='*50}")
print("문장 생성 비교 (generate)")
print(f"{'='*50}")

# delta vs hidden 크기 진단 (1회만)
print("\n[진단] delta vs hidden 크기 비교:")
hook = layer.register_forward_hook(make_hook(K, V, alpha=1.0, diag=True))
with torch.no_grad():
    _ = model(**inputs)
hook.remove()

# Hook 없이 생성
with torch.no_grad():
    gen_no_hook = model.generate(
        **inputs, max_new_tokens=30, do_sample=False,
    )
answer_no = tokenizer.decode(gen_no_hook[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\n[Hook 없음] {PROMPT} {answer_no}")

# 여러 alpha로 생성 비교
for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
    hook = layer.register_forward_hook(make_hook(K, V, alpha=alpha))
    with torch.no_grad():
        gen_hook = model.generate(
            **inputs, max_new_tokens=30, do_sample=False,
        )
    hook.remove()
    answer_hook = tokenizer.decode(gen_hook[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[Hook α={alpha}] {PROMPT} {answer_hook}")
