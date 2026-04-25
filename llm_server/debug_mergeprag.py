"""
MergePRAG 심층 디버깅 도구.

test_mergeprag.py 는 "문제가 있다"까지는 알려주지만, "어디가 문제인가"를
좁혀주진 않는다. 이 스크립트는 원인 층위별로 신호를 쪼개어 보여준다.

각 섹션은 독립적으로 해석 가능하며, 실패 패턴과 해석 가이드를 함께 출력한다.

사용법: cd llm_server && python debug_mergeprag.py
"""
import torch
import torch.nn.functional as F

from run_model import run_model
from mergePRAG.config import (
    ALPHA,
    KV_PATH_MODE,
    NUM_KV,
    POOLED_KV_SKIP_SCALE,
    POOLED_K_SKIP_SCALE,
    POOLED_V_SKIP_SCALE,
    SYSTEM_PROMPT,
    TRAIN_PROMPT_FORMAT,
    USE_K_RMS_CLAMP,
    USE_POOLED_KV_SKIP,
    USE_V_RMS_CLAMP,
    USE_CONTEXTUAL_PASSAGE_ENCODER,
    USE_QUESTION_CONDITIONED_MEMORY,
    K_RMS_CLAMP,
    V_RMS_CLAMP,
    build_chat_text,
    load_critical_layer,
    load_hypernet_state_dict,
)
from mergePRAG.embedding import (
    encode_passage_states,
    tokenize_conditioned_memory,
    tokenize_passage_memory,
)
from mergePRAG.hypernetwork import HyperNetwork
from mergePRAG.cross_attention import cross_attention


# ── 테스트 케이스 (원하는 대로 변경 가능) ──
QUESTION = "Who won the game?"
PASSAGE = "Manchester United won the match against Chelsea 3-1."
COMPARE_PASSAGE = "Chelsea won the match against Manchester United 3-1."
EXPECTED_ANSWER = "Manchester United"
COMPARE_EXPECTED_ANSWER = "Chelsea"
ALT_QUESTION = "Which team lost the game?"
ALT_EXPECTED_ANSWER = "Chelsea"

CRITICAL_LAYER = load_critical_layer()


def section(title: str):
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")


# ── 모델/하이퍼넷 로드 ─────────────────────────────────────────
print("모델 로딩...")
model, tokenizer = run_model()
device = next(model.parameters()).device
d_model = model.config.hidden_size

hypernet = HyperNetwork(d_model, k=NUM_KV).to(device).float()
state_dict, load_info = load_hypernet_state_dict(map_location=device)
hypernet.load_state_dict(state_dict)
hypernet.eval()
print(
    f"[load] {load_info['source']} ({load_info['kind']}, step={load_info.get('step')})"
)
print(
    f"[config] critical_layer={CRITICAL_LAYER}, alpha={ALPHA}, num_kv={NUM_KV}, "
    f"kv_path_mode={KV_PATH_MODE}, "
    f"train_prompt_format={TRAIN_PROMPT_FORMAT}, "
    f"pooled_kv_skip={USE_POOLED_KV_SKIP} "
    f"(k_scale={POOLED_K_SKIP_SCALE}, v_scale={POOLED_V_SKIP_SCALE}, default={POOLED_KV_SKIP_SCALE}), "
    f"k_rms_clamp={'on' if USE_K_RMS_CLAMP else 'off'}:{K_RMS_CLAMP}, "
    f"v_rms_clamp={'on' if USE_V_RMS_CLAMP else 'off'}:{V_RMS_CLAMP}"
)
print(f"[config] system_prompt={SYSTEM_PROMPT[:160]}")


def masked_mean(hidden, mask):
    w = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    denom = w.sum(dim=1).clamp_min(1.0)
    return (hidden * w).sum(dim=1) / denom


def get_kv(passage: str, question: str, grad: bool = False):
    """passage → (stage별 활성값) + K, V. grad=True면 hypernet 경로에 grad 유지."""
    if USE_QUESTION_CONDITIONED_MEMORY:
        encoded = tokenize_conditioned_memory(tokenizer, question, passage, device, 512)
    else:
        encoded = tokenize_passage_memory(tokenizer, passage, device, 512)

    ids = encoded["input_ids"]
    attn = encoded["attention_mask"]
    qmask = encoded["question_mask"]
    pmask = encoded["passage_mask"]

    # Qwen forward는 항상 no_grad — contextualize가 내부에서 처리.
    emb = encode_passage_states(model, ids, attn, USE_CONTEXTUAL_PASSAGE_ENCODER)

    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        query = masked_mean(emb, qmask) if USE_QUESTION_CONDITIONED_MEMORY else None
        parts = hypernet.encode_embedded_components(
            emb,
            attention_mask=attn,
            query=query,
            focus_mask=pmask,
        )
        pooled = parts["pooled"]
        hidden = parts["hidden"]
        K_raw = parts["K_raw"]
        V_raw = parts["V_raw"]
        K, V = hypernet.normalize_kv(K_raw, V_raw)
    return {
        "emb": emb,
        "pooled": pooled,
        "mlp_out": hidden,
        "K_mlp": parts["K_mlp"],
        "V_mlp": parts["V_mlp"],
        "K_skip": parts["K_skip"],
        "V_skip": parts["V_skip"],
        "K_raw": K_raw,
        "V_raw": V_raw,
        "K": K,
        "V": V,
    }


def make_hook(K, V, alpha=ALPHA):
    def hook_fn(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        Kd = K.to(device=hidden.device, dtype=hidden.dtype)
        Vd = V.to(device=hidden.device, dtype=hidden.dtype)
        delta = cross_attention(hidden, Kd, Vd)
        res = hidden + alpha * delta
        return (res,) + out[1:] if isinstance(out, tuple) else res

    return hook_fn


def build_batch(passage: str, question: str, answer_text: str, use_passage_in_prompt: bool):
    if use_passage_in_prompt:
        user_prompt = (
            "Answer the question using the passage-grounded fact.\n"
            f"Passage: {passage}\n"
            f"Question: {question}\nAnswer:"
        )
    else:
        user_prompt = (
            "Answer the question using the passage-grounded fact.\n"
            f"Question: {question}\nAnswer:"
        )
    if TRAIN_PROMPT_FORMAT == "chat":
        prompt = build_chat_text(tokenizer, question=user_prompt, enable_thinking=False)
        ans = f"{answer_text}{tokenizer.eos_token}"
    elif TRAIN_PROMPT_FORMAT == "plain":
        prompt = user_prompt
        ans = f" {answer_text}{tokenizer.eos_token}"
    else:
        raise ValueError(f"Unsupported MERGEPRAG_TRAIN_PROMPT_FORMAT: {TRAIN_PROMPT_FORMAT}")
    tok_p = tokenizer(prompt, return_tensors="pt", truncation=True)
    tok_a = tokenizer(ans, return_tensors="pt", add_special_tokens=False, truncation=True)
    plen = tok_p["input_ids"].shape[1]
    input_ids = torch.cat((tok_p["input_ids"], tok_a["input_ids"]), dim=-1).to(device)
    labels = torch.cat(
        (torch.full((1, plen), -100, dtype=torch.long), tok_a["input_ids"]),
        dim=-1,
    ).to(device)
    return {"input_ids": input_ids, "labels": labels, "prompt_len": plen}


def loss_of(input_ids, labels):
    with torch.no_grad():
        logits = model(input_ids=input_ids)["logits"]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    if valid.sum() == 0:
        return None
    return F.cross_entropy(shift_logits[valid], shift_labels[valid]).item()


def hooked_loss(batch, K=None, V=None, alpha=ALPHA):
    target = model.model.layers[CRITICAL_LAYER]
    hook = None
    if K is not None:
        hook = target.register_forward_hook(make_hook(K, V, alpha=alpha))
    try:
        with torch.no_grad():
            logits = model(input_ids=batch["input_ids"])["logits"]
    finally:
        if hook is not None:
            hook.remove()
    shift_logits = logits[:, :-1, :]
    shift_labels = batch["labels"][:, 1:]
    valid = shift_labels != -100
    if valid.sum() == 0:
        return None
    return F.cross_entropy(shift_logits[valid], shift_labels[valid]).item()


# ═══════════════════════════════════════════════════════════════
# [A] 스테이지별 cosine 전파 — collapse 진원지 탐지
# ═══════════════════════════════════════════════════════════════
section("[A] 스테이지별 passage 간 cosine — 어디서 차이가 사라지나")
main = get_kv(PASSAGE, QUESTION)
comp = get_kv(COMPARE_PASSAGE, QUESTION)


def comparable_vector(x):
    """Variable-length token sequences cannot be flattened directly.

    For sequence tensors such as [B, T, D], compare their mean feature vector so
    same-passage/different-question diagnostics do not crash when T differs.
    Fixed-size tensors such as pooled/K/V keep the original flatten behavior.
    """
    if x.dim() >= 3:
        return x.reshape(-1, x.shape[-1]).mean(dim=0)
    return x.flatten()


def vec_cos(a, b):
    va = comparable_vector(a).float()
    vb = comparable_vector(b).float()
    if va.numel() != vb.numel():
        min_len = min(va.numel(), vb.numel())
        va = va[:min_len]
        vb = vb[:min_len]
    return F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()


print(f"{'Stage':<12} {'Cosine':>10} {'|main|':>10} {'|comp|':>10}")
for name in ["emb", "pooled", "mlp_out", "K_raw", "V_raw", "K", "V"]:
    a, b = main[name], comp[name]
    print(f"{name:<12} {vec_cos(a, b):>10.4f} {a.norm().item():>10.2f} {b.norm().item():>10.2f}")

print("\n해석 가이드:")
print("  emb 단계에서 이미 >0.99 → Qwen 인코딩 자체가 한 단어 차이를 못 잡음")
print("  pooled/mlp_out에서 급상승 → pooling 또는 MLP가 정보 지움")
print("  K_raw/K에서 급상승 → LinearProjection 또는 normalize_kv 문제")
print("  건강한 상태: 각 스테이지마다 cos가 점진 증가, 최종 K/V cos < 0.95")


if USE_QUESTION_CONDITIONED_MEMORY:
    section("[A-1] same passage + different question — question-conditioned 분리력")
    alt = get_kv(PASSAGE, ALT_QUESTION)
    print(f"{'Stage':<12} {'Cosine':>10} {'|base|':>10} {'|altQ|':>10}")
    for name in ["emb", "pooled", "mlp_out", "K_raw", "V_raw", "K", "V"]:
        a, b = main[name], alt[name]
        print(f"{name:<12} {vec_cos(a, b):>10.4f} {a.norm().item():>10.2f} {b.norm().item():>10.2f}")
    print("\n해석 가이드:")
    print("  same passage인데도 질문이 다르면 pooled/K/V가 실제로 갈려야 함")
    print("  특히 V cosine이 여전히 0.98 이상이면 question-conditioned가 아직 약함")


# ═══════════════════════════════════════════════════════════════
# [A-2] MLP 경로 vs pooled->KV skip 경로 분해
# ═══════════════════════════════════════════════════════════════
section("[A-2] K/V 분해 — MLP 경로와 pooled skip 경로의 기여도")


def print_component(name: str, key: str):
    a = main.get(key)
    b = comp.get(key)
    if a is None or b is None:
        print(f"{name:<12} disabled")
        return
    print(
        f"{name:<12} cos={vec_cos(a, b):.4f} | "
        f"|main|={a.norm().item():.2f} | |comp|={b.norm().item():.2f}"
    )


print_component("K_mlp", "K_mlp")
print_component("V_mlp", "V_mlp")
print_component("K_skip", "K_skip")
print_component("V_skip", "V_skip")

if main.get("K_skip") is not None:
    main_k_ratio = main["K_skip"].norm().item() / max(main["K_mlp"].norm().item(), 1e-9)
    main_v_ratio = main["V_skip"].norm().item() / max(main["V_mlp"].norm().item(), 1e-9)
    comp_k_ratio = comp["K_skip"].norm().item() / max(comp["K_mlp"].norm().item(), 1e-9)
    comp_v_ratio = comp["V_skip"].norm().item() / max(comp["V_mlp"].norm().item(), 1e-9)
    print(
        f"\nskip/mlp norm ratio | "
        f"main K={main_k_ratio:.3f}, main V={main_v_ratio:.3f}, "
        f"comp K={comp_k_ratio:.3f}, comp V={comp_v_ratio:.3f}"
    )

print("\n해석 가이드:")
print("  K_mlp/V_mlp cos만 높고 K_skip/V_skip cos는 낮음 → MLP/head가 collapse 주범")
print("  skip norm ratio가 매우 작음(<0.1) → skip이 있어도 영향이 거의 없음")
print("  skip cos도 높음 → pooled 차이를 K/V로 옮기는 projection 자체가 못 배우는 중")


# ═══════════════════════════════════════════════════════════════
# [B] Teacher-forcing 상한선 vs no-context 하한선
# ═══════════════════════════════════════════════════════════════
section("[B] passage를 prompt에 직접 넣었을 때 vs 아예 없을 때")
tf_m = build_batch(PASSAGE, QUESTION, EXPECTED_ANSWER, True)
tf_f = build_batch(PASSAGE, QUESTION, COMPARE_EXPECTED_ANSWER, True)
nc_m = build_batch(PASSAGE, QUESTION, EXPECTED_ANSWER, False)
nc_f = build_batch(PASSAGE, QUESTION, COMPARE_EXPECTED_ANSWER, False)

l_tf_m = loss_of(tf_m["input_ids"], tf_m["labels"])
l_tf_f = loss_of(tf_f["input_ids"], tf_f["labels"])
l_nc_m = loss_of(nc_m["input_ids"], nc_m["labels"])
l_nc_f = loss_of(nc_f["input_ids"], nc_f["labels"])

print("Passage를 prompt에 포함 (상한선 — model이 도달 가능한 최적):")
print(
    f"  loss({EXPECTED_ANSWER})={l_tf_m:.3f}  "
    f"loss({COMPARE_EXPECTED_ANSWER})={l_tf_f:.3f}  Δ={l_tf_f - l_tf_m:+.3f}"
)
print("Passage 없음, hook 없음 (하한선 — 정보 0일 때):")
print(
    f"  loss({EXPECTED_ANSWER})={l_nc_m:.3f}  "
    f"loss({COMPARE_EXPECTED_ANSWER})={l_nc_f:.3f}  Δ={l_nc_f - l_nc_m:+.3f}"
)

print("\n해석 가이드:")
print("  상한 Δ > 2.0: model이 passage만 있으면 정답을 쉽게 구분 → hook의 목표가 분명")
print("  상한 Δ < 0.5: 단일 단어 차이를 model이 못 쓰고 있음 → tokenizer/프롬프트 문제")
print("  hook 실험의 loss는 이 두 값 사이에 있어야 의미가 있음")


if USE_QUESTION_CONDITIONED_MEMORY:
    section("[B-1] same passage + different question — answer supervision 분리력")
    base_q_batch = build_batch(PASSAGE, QUESTION, EXPECTED_ANSWER, False)
    alt_q_batch = build_batch(PASSAGE, ALT_QUESTION, ALT_EXPECTED_ANSWER, False)
    alt_mem = get_kv(PASSAGE, ALT_QUESTION)
    base_under_base = hooked_loss(base_q_batch, main["K"], main["V"])
    base_under_alt = hooked_loss(base_q_batch, alt_mem["K"], alt_mem["V"])
    alt_under_alt = hooked_loss(alt_q_batch, alt_mem["K"], alt_mem["V"])
    alt_under_base = hooked_loss(alt_q_batch, main["K"], main["V"])
    print(f"base question under base memory = {base_under_base:.4f}")
    print(f"base question under alt memory  = {base_under_alt:.4f}")
    print(f"alt question under alt memory   = {alt_under_alt:.4f}")
    print(f"alt question under base memory  = {alt_under_base:.4f}")
    print("\n해석 가이드:")
    print("  각 질문은 자기 memory 아래에서 loss가 더 낮아야 함")
    print("  둘이 비슷하면 same-passage question separation이 아직 약함")


# ═══════════════════════════════════════════════════════════════
# [C] Hook이 answer-position logit을 실제로 바꾸는가
# ═══════════════════════════════════════════════════════════════
section("[C] Hook 효과 — answer 첫 토큰 위치의 logit 분포")


def answer_first_dist(batch, K=None, V=None, alpha=ALPHA):
    # answer 첫 토큰을 예측하는 logit은 prompt 마지막 위치.
    ctx = batch["input_ids"][:, : batch["prompt_len"]]
    target = model.model.layers[CRITICAL_LAYER]
    hook = None
    if K is not None:
        hook = target.register_forward_hook(make_hook(K, V, alpha=alpha))
    try:
        with torch.no_grad():
            logits = model(input_ids=ctx)["logits"]
    finally:
        if hook is not None:
            hook.remove()
    return torch.softmax(logits[0, -1].float(), dim=-1)


score_batch = build_batch(PASSAGE, QUESTION, EXPECTED_ANSWER, False)
dist_no = answer_first_dist(score_batch)
dist_m = answer_first_dist(score_batch, main["K"], main["V"])
dist_c = answer_first_dist(score_batch, comp["K"], comp["V"])

answer_prefix = "" if TRAIN_PROMPT_FORMAT == "chat" else " "
target_id = tokenizer(f"{answer_prefix}{EXPECTED_ANSWER}", add_special_tokens=False)["input_ids"][0]
compare_id = tokenizer(f"{answer_prefix}{COMPARE_EXPECTED_ANSWER}", add_special_tokens=False)["input_ids"][0]
print(f"token id  {EXPECTED_ANSWER}={target_id}  {COMPARE_EXPECTED_ANSWER}={compare_id}")
print(
    f"{'case':<20}"
    f"{'p(' + EXPECTED_ANSWER + ')':>18}"
    f"{'p(' + COMPARE_EXPECTED_ANSWER + ')':>18}"
    f"{'log(target/comp)':>18}"
)
for name, d in [("no hook", dist_no), ("main memory", dist_m), ("compare memory", dist_c)]:
    pt, pc = d[target_id].item(), d[compare_id].item()
    log_ratio = torch.log(torch.tensor(pt / max(pc, 1e-30))).item()
    print(f"{name:<20}{pt:>18.6f}{pc:>18.6f}{log_ratio:>18.3f}")

eps = 1e-12
kl_m_no = (dist_m * (dist_m.clamp_min(eps).log() - dist_no.clamp_min(eps).log())).sum().item()
kl_c_no = (dist_c * (dist_c.clamp_min(eps).log() - dist_no.clamp_min(eps).log())).sum().item()
kl_m_c = (dist_m * (dist_m.clamp_min(eps).log() - dist_c.clamp_min(eps).log())).sum().item()
print(f"\nKL(main || no_hook)       = {kl_m_no:.4f}")
print(f"KL(compare || no_hook)    = {kl_c_no:.4f}")
print(f"KL(main || compare)       = {kl_m_c:.4f}")

for name, d in [("no hook", dist_no), ("main memory", dist_m), ("compare memory", dist_c)]:
    top5 = torch.topk(d, 5)
    toks = [tokenizer.decode([t.item()]) for t in top5.indices]
    probs = [round(p.item(), 4) for p in top5.values]
    print(f"{name} top5: {list(zip(toks, probs))}")

print("\n해석 가이드:")
print("  성공: log(target/comp) > 0 under main, < 0 under compare, KL(main || compare) > 0.5")
print("  collapse: KL(main || compare) < 0.01 → 두 memory가 사실상 같은 분포")
print("  무효화: KL(main || no_hook) ≈ 0 → hook이 아예 효과 없음")


# ═══════════════════════════════════════════════════════════════
# [C-2] full-answer margin 요약
# ═══════════════════════════════════════════════════════════════
section("[C-2] Full-answer margin — main vs compare memory가 실제로 답을 뒤집는가")

main_target = hooked_loss(build_batch(PASSAGE, QUESTION, EXPECTED_ANSWER, False), main["K"], main["V"])
main_compare = hooked_loss(build_batch(PASSAGE, QUESTION, COMPARE_EXPECTED_ANSWER, False), main["K"], main["V"])
comp_target = hooked_loss(build_batch(PASSAGE, QUESTION, EXPECTED_ANSWER, False), comp["K"], comp["V"])
comp_compare = hooked_loss(build_batch(PASSAGE, QUESTION, COMPARE_EXPECTED_ANSWER, False), comp["K"], comp["V"])
margin_main = main_compare - main_target
margin_compare = comp_target - comp_compare
print(f"main memory margin (compare-target)    = {margin_main:+.4f}")
print(f"compare memory margin (target-compare) = {margin_compare:+.4f}")
print(f"flip success = {margin_main > 0 and margin_compare > 0}")

print("\n해석 가이드:")
print("  둘 다 양수면 memory가 full-answer 기준으로도 passage별 분리 성공")
print("  main만 양수면 generic bias는 배웠지만 compare flip은 실패")
print("  둘 다 0 근처면 hook은 있어도 passage-specific signal이 약함")


# ═══════════════════════════════════════════════════════════════
# [D] 레이어별 신호 전파 — 주입된 delta가 살아남나
# ═══════════════════════════════════════════════════════════════
section("[D] Hook 주입 후 각 layer hidden state 변화량")

captured = {}


def make_cap(name):
    def h(_m, _i, out):
        h_val = out[0] if isinstance(out, tuple) else out
        captured[name] = h_val.detach().clone()

    return h


cap_hooks = [
    layer.register_forward_hook(make_cap(f"L{i}"))
    for i, layer in enumerate(model.model.layers)
]

with torch.no_grad():
    _ = model(input_ids=score_batch["input_ids"])
base_states = {k: v.clone() for k, v in captured.items()}

captured.clear()
inject = model.model.layers[CRITICAL_LAYER].register_forward_hook(
    make_hook(main["K"], main["V"])
)
with torch.no_grad():
    _ = model(input_ids=score_batch["input_ids"])
inject.remove()
hooked_states = {k: v.clone() for k, v in captured.items()}

for h in cap_hooks:
    h.remove()

print(f"{'Layer':<8}{'Δ norm':>12}{'Δ/ref':>10}{'cos':>12}")
n_layers = len(model.model.layers)
for i in range(n_layers):
    key = f"L{i}"
    a, b = base_states[key], hooked_states[key]
    diff = (b - a).norm().item()
    ref = a.norm().item()
    ratio = diff / max(ref, 1e-9)
    cos = F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()
    tag = " ← injected" if i == CRITICAL_LAYER else ""
    if i < 3 or abs(i - CRITICAL_LAYER) <= 2 or i >= n_layers - 2:
        print(f"L{i:<7}{diff:>12.2f}{ratio:>10.4f}{cos:>12.6f}{tag}")

print("\n해석 가이드:")
print("  injection 이전 레이어: Δ norm ≈ 0 이어야 함 (수치 오차 수준)")
print("  injection 직후: Δ/ref 가 0.01~0.3 범위면 적절, <0.001 이면 hook이 무력")
print("  마지막 레이어 cos가 0.9999 이상이면 후속 레이어가 perturbation을 지워버림")
print("    → critical layer를 더 뒤로 옮기거나 alpha를 올려야 함")


# ═══════════════════════════════════════════════════════════════
# [E] Slot ablation — 어떤 slot이 실제로 일하고 있나
# ═══════════════════════════════════════════════════════════════
section(f"[E] Slot ablation — 각 slot을 0으로 만들었을 때 p({EXPECTED_ANSWER}) 변화")

base_dist = answer_first_dist(score_batch, main["K"], main["V"])
base_pm = base_dist[target_id].item()
base_pf = base_dist[compare_id].item()
print(
    f"all slots active: p({EXPECTED_ANSWER})={base_pm:.6f}, "
    f"p({COMPARE_EXPECTED_ANSWER})={base_pf:.6f}"
)

deltas = []
for k in range(NUM_KV):
    K_mask = main["K"].clone()
    V_mask = main["V"].clone()
    K_mask[:, k, :] = 0
    V_mask[:, k, :] = 0
    d = answer_first_dist(score_batch, K_mask, V_mask)
    dpm = d[target_id].item() - base_pm
    dpf = d[compare_id].item() - base_pf
    deltas.append((k, dpm, dpf))
    print(
        f"  slot {k:2d} off: "
        f"Δp({EXPECTED_ANSWER})={dpm:+.6f}, "
        f"Δp({COMPARE_EXPECTED_ANSWER})={dpf:+.6f}"
    )

contributing = [d for d in deltas if abs(d[1]) > 1e-6 or abs(d[2]) > 1e-6]
print(f"\n유효 slot 수: {len(contributing)} / {NUM_KV}")
print("해석 가이드:")
print("  모든 Δ ≈ 0: slot이 사실상 동일한 역할 → slot 다양성 붕괴 (within-passage collapse)")
print("  특정 1~2 slot만 큰 Δ: 집중, 다른 slot들이 낭비")
print("  건강: 절반 이상 slot이 각기 다른 방향으로 기여")


# ═══════════════════════════════════════════════════════════════
# [F] 단일 샘플 overfit — 아키텍처의 학습 capacity 확인
# ═══════════════════════════════════════════════════════════════
section(f"[F] 단일 샘플 overfit — ({EXPECTED_ANSWER} passage, Q, {EXPECTED_ANSWER}) 한 쌍으로 200 step")

orig_state = {k: v.clone() for k, v in hypernet.state_dict().items()}
hypernet.train()
opt = torch.optim.AdamW(hypernet.parameters(), lr=5e-4)
target = model.model.layers[CRITICAL_LAYER]

overfit_batch = build_batch(PASSAGE, QUESTION, EXPECTED_ANSWER, False)
loss_trace = []
for step in range(200):
    kv = get_kv(PASSAGE, QUESTION, grad=True)
    hook = target.register_forward_hook(make_hook(kv["K"], kv["V"]))
    try:
        logits = model(input_ids=overfit_batch["input_ids"])["logits"]
    finally:
        hook.remove()
    shift_logits = logits[:, :-1, :]
    shift_labels = overfit_batch["labels"][:, 1:]
    valid = shift_labels != -100
    loss = F.cross_entropy(shift_logits[valid], shift_labels[valid])
    opt.zero_grad()
    loss.backward()
    opt.step()
    loss_trace.append(loss.item())
    if step % 25 == 0 or step == 199:
        print(f"  step {step:3d}: loss = {loss.item():.4f}")

final_loss = loss_trace[-1]
# 복구
hypernet.load_state_dict(orig_state)
hypernet.eval()

print(f"\n시작 loss: {loss_trace[0]:.4f}  →  최종 loss: {final_loss:.4f}")
print("해석 가이드:")
print("  최종 loss < 0.5: 아키텍처는 학습 가능 → 문제는 데이터/LR/스케줄/배치")
print("  최종 loss 2~5 에서 정체: 용량 부족 또는 gradient 병목 (hidden_dim, num_kv 상향 검토)")
print("  loss가 거의 안 떨어짐 (>10): hook → hypernet gradient 경로 끊김.")
print("    → delta_K.to(dtype=...) detach, torch.no_grad 누수, requires_grad 체크")

print(f"\n{'=' * 70}")
print("디버그 완료. 각 섹션의 '해석 가이드'를 참고해 원인 층위를 특정하세요.")
print(f"{'=' * 70}")
