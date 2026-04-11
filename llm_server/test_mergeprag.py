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

PASSAGE = "seungmin is man and student"
QUESTION = "Who is seungmin?"
PROMPT = f"Question: {QUESTION}\nAnswer:"
CRITICAL_LAYER = 5
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
with torch.no_grad():
    logits_no_hook = model(**inputs).logits[0, -1]  # 마지막 토큰의 예측

probs_no = torch.softmax(logits_no_hook, dim=-1)
top5_no = torch.topk(probs_no, 5)

print("\n[A] Hook 없이 (LLM Only) - 'Answer:' 다음 토큰 top-5:")
for i in range(5):
    tok = tokenizer.decode(top5_no.indices[i])
    print(f"  {i+1}. '{tok}' ({top5_no.values[i]:.4f})")

# ── 테스트 2: alpha별 hook forward → top-5 비교 ──
def make_hook(dK, dV, alpha=1.0):
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        Kd = dK.to(device=hidden.device, dtype=hidden.dtype)
        Vd = dV.to(device=hidden.device, dtype=hidden.dtype)
        delta = cross_attention(hidden, Kd, Vd)
        result = hidden + alpha * delta
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result
    return hook_fn

layer = model.model.layers[CRITICAL_LAYER]

for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    hook = layer.register_forward_hook(make_hook(K, V, alpha))
    with torch.no_grad():
        logits_hook = model(**inputs).logits[0, -1]
    hook.remove()

    probs = torch.softmax(logits_hook, dim=-1)
    top5 = torch.topk(probs, 5)
    diff = (logits_hook - logits_no_hook).norm().item()

    tokens = [tokenizer.decode(top5.indices[i]) for i in range(5)]
    probs_list = [f"{top5.values[i]:.3f}" for i in range(5)]
    print(f"\n[alpha={alpha}] logits변화={diff:.1f}")
    for i in range(5):
        print(f"  {i+1}. '{tokens[i]}' ({probs_list[i]})")
