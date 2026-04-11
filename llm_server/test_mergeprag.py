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

PASSAGE = "Mount Everest is 8849 meters tall and located in Nepal."
QUESTION = "How tall is Mount Everest?"
PROMPT = f"Question: {QUESTION}\nAnswer:"
CRITICAL_LAYER = 0
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

# ── 테스트 2: hook 있으면 forward → top-5 예측 ──
def make_hook(dK, dV):
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        Kd = dK.to(device=hidden.device, dtype=hidden.dtype)
        Vd = dV.to(device=hidden.device, dtype=hidden.dtype)
        delta = cross_attention(hidden, Kd, Vd)
        result = hidden + delta
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result
    return hook_fn

layer = model.model.layers[CRITICAL_LAYER]
hook = layer.register_forward_hook(make_hook(K, V))
with torch.no_grad():
    logits_hook = model(**inputs).logits[0, -1]
hook.remove()

probs_hook = torch.softmax(logits_hook, dim=-1)
top5_hook = torch.topk(probs_hook, 5)

print("\n[B] Hook 적용 (MergePRAG) - 'Answer:' 다음 토큰 top-5:")
for i in range(5):
    tok = tokenizer.decode(top5_hook.indices[i])
    print(f"  {i+1}. '{tok}' ({top5_hook.values[i]:.4f})")

# ── 판정 ──
print(f"\n{'='*50}")
print("판정")
print(f"{'='*50}")

# logits 변화량
diff = (logits_hook - logits_no_hook).norm().item()
print(f"logits 변화량: {diff:.4f}")
print(f"  (0에 가까우면 hook이 아무 효과 없음)")

top_no = set(top5_no.indices.tolist())
top_hook = set(top5_hook.indices.tolist())
changed = top_no != top_hook
print(f"top-5 토큰 변경: {changed}")

if diff < 0.01:
    print("\n→ K,V가 hidden에 거의 영향 없음. HyperNetwork가 유의미한 정보를 담지 못함")
elif not changed:
    print("\n→ logits는 변했지만 top 예측은 동일. K,V 효과가 약함")
else:
    print("\n→ top-5 예측이 변경됨! K,V가 모델 예측에 영향을 주고 있음")
    # 8849, Nepal 관련 토큰이 있는지
    hook_tokens = [tokenizer.decode(top5_hook.indices[i]) for i in range(5)]
    print(f"   hook top-5 토큰: {hook_tokens}")
    if any("8" in t or "Nepal" in t or "meter" in t for t in hook_tokens):
        print("   ★ passage 정보가 반영됨! MergePRAG 성공")
    else:
        print("   passage 관련 토큰은 아님. K,V가 엉뚱한 방향으로 영향")
