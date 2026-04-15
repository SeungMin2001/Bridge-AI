"""
MergePRAG 핵심 진단: hook이 모델 예측을 바꾸는지 확인
generate 없이 단일 forward pass로 직접 비교

사용법: cd llm_server && python test_mergeprag.py
"""
import torch
import os
from datetime import datetime
from run_model import run_model
from mergePRAG.config import (
    ALPHA,
    CHECKPOINT_PATH,
    NUM_KV,
    WEIGHTS_PATH,
    load_critical_layer,
    load_hypernet_state_dict,
)
from mergePRAG.hypernetwork import HyperNetwork
from mergePRAG.cross_attention import cross_attention

QUESTION = "What color is the apple?"
CRITICAL_LAYER = load_critical_layer()
PASSAGE = "The apple is blue."
ENGLISH_SYSTEM_PROMPT = "Answer in English with one short sentence."


def format_mtime(path):
    if not os.path.exists(path):
        return "missing"
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")

# ── 모델 로드 ──
print("모델 로딩...")
model, tokenizer = run_model()
device = next(model.parameters()).device

# ── HyperNetwork → K, V ──
d_model = model.config.hidden_size
hypernet = HyperNetwork(d_model, k=NUM_KV).to(device).float()
state_dict, load_info = load_hypernet_state_dict(map_location=device)
hypernet.load_state_dict(state_dict)
hypernet.eval()
step_text = f", checkpoint step={load_info['step']}" if load_info["step"] is not None else ""
print(f"HyperNetwork weights source: {load_info['source']} ({load_info['kind']}{step_text})")
print(f"weights path: {WEIGHTS_PATH} (modified_at={format_mtime(WEIGHTS_PATH)})")
print(f"checkpoint path: {CHECKPOINT_PATH} (modified_at={format_mtime(CHECKPOINT_PATH)})")

with torch.no_grad():
    ids = tokenizer(PASSAGE, return_tensors="pt")["input_ids"].to(device)
    emb = model.model.embed_tokens(ids).to(torch.float32)
    pooled = hypernet.pooling(emb)
    h = hypernet.mlp(pooled)
    K_raw, V_raw = hypernet.lp(h)
    K, V = hypernet(emb)

print(f"passage: {PASSAGE}")
print(f"pooled h norm: {h.norm():.4f}")
print(f"K raw norm: {K_raw.norm():.4f}, V raw norm: {V_raw.norm():.4f}")
print(f"K norm: {K.norm():.4f}, V norm: {V.norm():.4f}")
print(f"K per-vector norm: {K[0,0].norm():.4f}")
print(f"V per-vector norm: {V[0,0].norm():.4f}")

# ── 다른 passage K,V와 비교 ──
with torch.no_grad():
    ids2 = tokenizer("The apple is red.", return_tensors="pt")["input_ids"].to(device)
    emb2 = model.model.embed_tokens(ids2).to(torch.float32)
    pooled2 = hypernet.pooling(emb2)
    h2 = hypernet.mlp(pooled2)
    K2_raw, V2_raw = hypernet.lp(h2)
    K2, V2 = hypernet(emb2)

sim_h = torch.nn.functional.cosine_similarity(h.view(1, -1), h2.view(1, -1)).item()
print(f"\n두 passage pooled h 유사도: {sim_h:.4f}")
sim_k_raw = torch.nn.functional.cosine_similarity(K_raw.view(1, -1), K2_raw.view(1, -1)).item()
print(f"두 passage K raw 유사도: {sim_k_raw:.4f}")
sim_v_raw = torch.nn.functional.cosine_similarity(V_raw.view(1, -1), V2_raw.view(1, -1)).item()
print(f"두 passage V raw 유사도: {sim_v_raw:.4f}")
sim = torch.nn.functional.cosine_similarity(K.view(1,-1), K2.view(1,-1)).item()
print(f"\n두 passage K 유사도: {sim:.4f}")
print(f"  (높을수록 두 passage를 비슷하게 본다는 뜻)")
sim_v = torch.nn.functional.cosine_similarity(V.view(1,-1), V2.view(1,-1)).item()
print(f"두 passage V 유사도: {sim_v:.4f}")

# ── 테스트 1: hook 없이 forward → top-5 예측 ──
print(f"\n{'='*50}")
prompt_text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": ENGLISH_SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
print(f"prompt: '{prompt_text}'")
print(f"{'='*50}")

inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

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

STOP_IDS = tokenizer.encode("\nQuestion:", add_special_tokens=False)

def decode_answer(gen, input_len):
    tokens = gen[0][input_len:].tolist()
    # "Question:" 이 나오면 그 앞에서 자름
    for i in range(len(tokens) - len(STOP_IDS) + 1):
        if tokens[i:i+len(STOP_IDS)] == STOP_IDS:
            tokens = tokens[:i]
            break
    # think 태그 제거
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text

# Hook 없이 생성
with torch.no_grad():
    gen_no_hook = model.generate(
        **inputs, max_new_tokens=50, do_sample=False,
    )
answer_no = decode_answer(gen_no_hook, inputs["input_ids"].shape[1])
print(f"\n[Hook 없음] {answer_no}")

# 여러 alpha로 생성 비교
for alpha in [0.1, ALPHA, 1.0]:
    hook = layer.register_forward_hook(make_hook(K, V, alpha=alpha))
    with torch.no_grad():
        gen_hook = model.generate(
            **inputs, max_new_tokens=50, do_sample=False,
        )
    hook.remove()
    answer_hook = decode_answer(gen_hook, inputs["input_ids"].shape[1])
    print(f"[Hook α={alpha}] {answer_hook}")
