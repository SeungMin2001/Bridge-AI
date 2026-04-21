"""
MergePRAG 핵심 진단: hook이 모델 예측을 바꾸는지 확인
두 passage가 서로 다른 K,V를 만들고, 그 K,V가 답변을 실제로 바꾸는지 확인

사용법: cd llm_server && python test_mergeprag.py
"""
import torch
import os
import torch.nn.functional as F
from datetime import datetime
from run_model import run_model
from mergePRAG.config import (
    ALPHA,
    CHECKPOINT_PATH,
    NUM_KV,
    USE_CONTEXTUAL_PASSAGE_ENCODER,
    USE_QUESTION_CONDITIONED_MEMORY,
    WEIGHTS_PATH,
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

QUESTION = "What is the deadline?"
CRITICAL_LAYER = load_critical_layer()
PASSAGE = "The assignment deadline is Monday."
COMPARE_PASSAGE = "The assignment deadline is Friday."
EXPECTED_ANSWER = "Monday"
COMPARE_EXPECTED_ANSWER = "Friday"
ENGLISH_SYSTEM_PROMPT = "Answer in English with one short sentence."
MAX_NEW_TOKENS = 150


def format_mtime(path):
    if not os.path.exists(path):
        return "missing"
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")


def section(title: str):
    print(f"\n[{title}]")


def short(text: str, limit: int = 72):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def fmt_score(score):
    loss, logprob = score
    return f"loss={loss:.3f}, logp={logprob:.3f}"


def verdict(label: str, ok: bool):
    return f"{label}={'OK' if ok else 'FAIL'}"

# ── 모델 로드 ──
print("모델 로딩...")
model, tokenizer = run_model()
device = next(model.parameters()).device
section("Config")
print(f"layer={CRITICAL_LAYER} | num_kv={NUM_KV} | alpha={ALPHA}")
print(
    f"contextual={USE_CONTEXTUAL_PASSAGE_ENCODER} | "
    f"question_conditioned={USE_QUESTION_CONDITIONED_MEMORY}"
)

# ── HyperNetwork → K, V ──
d_model = model.config.hidden_size
hypernet = HyperNetwork(d_model, k=NUM_KV).to(device).float()
state_dict, load_info = load_hypernet_state_dict(map_location=device)
hypernet.load_state_dict(state_dict)
hypernet.eval()
step_text = f", checkpoint step={load_info['step']}" if load_info["step"] is not None else ""
print(f"hypernet={load_info['kind']}{step_text}")
print(f"weights={format_mtime(WEIGHTS_PATH)} | checkpoint={format_mtime(CHECKPOINT_PATH)}")

def masked_mean(hidden, mask):
    weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denom


def encode_passage_stats(question: str, passage: str):
    with torch.no_grad():
        if USE_QUESTION_CONDITIONED_MEMORY:
            encoded = tokenize_conditioned_memory(
                tokenizer,
                question,
                passage,
                device,
                max_length=512,
            )
        else:
            encoded = tokenize_passage_memory(
                tokenizer,
                passage,
                device,
                max_length=512,
            )
        ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        question_mask = encoded["question_mask"]
        passage_mask = encoded["passage_mask"]
        emb = encode_passage_states(
            model,
            ids,
            attention_mask=attention_mask,
            use_contextual=USE_CONTEXTUAL_PASSAGE_ENCODER,
        )
        query = masked_mean(emb, question_mask) if USE_QUESTION_CONDITIONED_MEMORY else None
        pooled = hypernet.pooling(emb, mask=attention_mask, query=query, focus_mask=passage_mask)
        hidden = hypernet.mlp(pooled)
        K_raw, V_raw = hypernet.lp(hidden)
        K, V = hypernet(emb, attention_mask=attention_mask, query=query, focus_mask=passage_mask)
    return {
        "ids": ids,
        "attention_mask": attention_mask,
        "emb": emb,
        "pooled": pooled,
        "hidden": hidden,
        "K_raw": K_raw,
        "V_raw": V_raw,
        "K": K,
        "V": V,
    }


main_stats = encode_passage_stats(QUESTION, PASSAGE)
compare_stats = encode_passage_stats(QUESTION, COMPARE_PASSAGE)

pooled = main_stats["pooled"]
h = main_stats["hidden"]
K_raw = main_stats["K_raw"]
V_raw = main_stats["V_raw"]
K = main_stats["K"]
V = main_stats["V"]

section("Inputs")
print(f"question: {QUESTION}")
print(f"main passage: {short(PASSAGE, 96)}")
print(f"compare passage: {short(COMPARE_PASSAGE, 96)}")

# ── 다른 passage K,V와 비교 ──
h2 = compare_stats["hidden"]
pooled2 = compare_stats["pooled"]
K2_raw = compare_stats["K_raw"]
V2_raw = compare_stats["V_raw"]
K2 = compare_stats["K"]
V2 = compare_stats["V"]

sim_pooled = torch.nn.functional.cosine_similarity(pooled.view(1, -1), pooled2.view(1, -1)).item()
sim_h = torch.nn.functional.cosine_similarity(h.view(1, -1), h2.view(1, -1)).item()
sim_k_raw = torch.nn.functional.cosine_similarity(K_raw.view(1, -1), K2_raw.view(1, -1)).item()
sim_v_raw = torch.nn.functional.cosine_similarity(V_raw.view(1, -1), V2_raw.view(1, -1)).item()
sim = torch.nn.functional.cosine_similarity(K.view(1,-1), K2.view(1,-1)).item()
sim_v = torch.nn.functional.cosine_similarity(V.view(1,-1), V2.view(1,-1)).item()
section("Memory Stats")
print(
    f"cos(pooled)={sim_pooled:.4f} | cos(hidden)={sim_h:.4f} | "
    f"cos(K_raw)={sim_k_raw:.4f} | cos(V_raw)={sim_v_raw:.4f}"
)
print(
    f"cos(K)={sim:.4f} | cos(V)={sim_v:.4f} | "
    f"K_rms={K.pow(2).mean(dim=-1).sqrt().mean().item():.4f} | "
    f"V_rms={V.pow(2).mean(dim=-1).sqrt().mean().item():.4f}"
)

# ── 테스트 1: hook 없이 forward → top-5 예측 ──
prompt_text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": ENGLISH_SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = tokenizer(prompt_text, return_tensors="pt").to(device)


def build_scoring_batch(answer_text: str):
    prompt = f"Answer the question using the passage-grounded fact.\nQuestion: {QUESTION}\nAnswer:"
    answer = f" {answer_text}{tokenizer.eos_token}"
    tok_prompt = tokenizer(prompt, return_tensors="pt", truncation=True)
    tok_answer = tokenizer(answer, return_tensors="pt", add_special_tokens=False, truncation=True)
    prompt_len = tok_prompt["input_ids"].shape[1]
    input_ids = torch.cat((tok_prompt["input_ids"], tok_answer["input_ids"]), dim=-1).to(device)
    labels = torch.cat((
        torch.full((1, prompt_len), -100, dtype=torch.long),
        tok_answer["input_ids"]
    ), dim=-1).to(device)
    return {"input_ids": input_ids, "labels": labels}


def compute_answer_loss(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != -100
    if valid.sum() == 0:
        return None
    selected_logits = shift_logits[valid]
    selected_labels = shift_labels[valid]
    loss = F.cross_entropy(selected_logits, selected_labels)
    log_probs = torch.log_softmax(selected_logits, dim=-1)
    answer_logprob = log_probs.gather(-1, selected_labels.unsqueeze(-1)).sum().item()
    return loss.item(), answer_logprob

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
            print(f"delta_ratio={ratio:.4f} (hidden={h_norm:.1f}, delta={d_norm:.1f})")
        result = hidden + alpha * delta
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result
    return hook_fn

layer = model.model.layers[CRITICAL_LAYER]

# ── 문장 생성 비교 ──
section("Delta Check")

# delta vs hidden 크기 진단 (1회만)
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


def score_answer_with_memory(dK, dV, answer_text: str, alpha=ALPHA):
    score_inputs = build_scoring_batch(answer_text)
    hook = layer.register_forward_hook(make_hook(dK, dV, alpha=alpha))
    try:
        with torch.no_grad():
            logits = model(input_ids=score_inputs["input_ids"])["logits"]
    finally:
        hook.remove()
    return compute_answer_loss(logits, score_inputs["labels"])


def score_answer_without_memory(answer_text: str):
    score_inputs = build_scoring_batch(answer_text)
    with torch.no_grad():
        logits = model(input_ids=score_inputs["input_ids"])["logits"]
    return compute_answer_loss(logits, score_inputs["labels"])

# Hook 없이 생성
with torch.no_grad():
    gen_no_hook = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
    )
answer_no = decode_answer(gen_no_hook, inputs["input_ids"].shape[1])
score_no_main = score_answer_without_memory(EXPECTED_ANSWER)
score_no_compare = score_answer_without_memory(COMPARE_EXPECTED_ANSWER)
section("Generations")
print(f"no_hook      | ans={short(answer_no)}")

# 여러 alpha로 생성 비교 (main passage)
alpha_rows = []
for alpha in [0.1, ALPHA, 1.0]:
    hook = layer.register_forward_hook(make_hook(K, V, alpha=alpha))
    with torch.no_grad():
        gen_hook = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )
    hook.remove()
    answer_hook = decode_answer(gen_hook, inputs["input_ids"].shape[1])
    score_main = score_answer_with_memory(K, V, EXPECTED_ANSWER, alpha=alpha)
    score_compare = score_answer_with_memory(K, V, COMPARE_EXPECTED_ANSWER, alpha=alpha)
    alpha_rows.append((alpha, answer_hook, score_main, score_compare))

for alpha, answer_hook, score_main, score_compare in alpha_rows:
    prefers_main = score_main[1] > score_compare[1]
    print(
        f"main α={alpha:<4} | ans={short(answer_hook, 32):<32} | "
        f"{verdict('prefer_main', prefers_main)} | "
        f"target {fmt_score(score_main)} | compare {fmt_score(score_compare)}"
    )

# 비교 passage도 같은 alpha로 직접 생성
section("Passage Flip")
flip_rows = []
for alpha in [ALPHA, 1.0]:
    hook_main = layer.register_forward_hook(make_hook(K, V, alpha=alpha))
    with torch.no_grad():
        gen_main = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )
    hook_main.remove()

    hook_compare = layer.register_forward_hook(make_hook(K2, V2, alpha=alpha))
    with torch.no_grad():
        gen_compare = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )
    hook_compare.remove()

    answer_main = decode_answer(gen_main, inputs["input_ids"].shape[1])
    answer_compare = decode_answer(gen_compare, inputs["input_ids"].shape[1])
    main_target_score = score_answer_with_memory(K, V, EXPECTED_ANSWER, alpha=alpha)
    main_compare_score = score_answer_with_memory(K, V, COMPARE_EXPECTED_ANSWER, alpha=alpha)
    compare_target_score = score_answer_with_memory(K2, V2, EXPECTED_ANSWER, alpha=alpha)
    compare_compare_score = score_answer_with_memory(K2, V2, COMPARE_EXPECTED_ANSWER, alpha=alpha)
    flip_rows.append(
        (
            alpha,
            answer_main,
            answer_compare,
            main_target_score,
            main_compare_score,
            compare_target_score,
            compare_compare_score,
        )
    )

for (
    alpha,
    answer_main,
    answer_compare,
    main_target_score,
    main_compare_score,
    compare_target_score,
    compare_compare_score,
) in flip_rows:
    main_prefers_main = main_target_score[1] > main_compare_score[1]
    compare_prefers_compare = compare_compare_score[1] > compare_target_score[1]
    print(
        f"α={alpha:<4} | main='{short(answer_main, 20)}' | compare='{short(answer_compare, 20)}' | "
        f"{verdict('main->Monday', main_prefers_main)} | "
        f"{verdict('compare->Friday', compare_prefers_compare)}"
    )

# slot별 차이도 같이 확인
slot_cos_k = []
slot_cos_v = []
for idx in range(K.shape[1]):
    slot_cos_k.append(torch.nn.functional.cosine_similarity(K[0, idx].view(1, -1), K2[0, idx].view(1, -1)).item())
    slot_cos_v.append(torch.nn.functional.cosine_similarity(V[0, idx].view(1, -1), V2[0, idx].view(1, -1)).item())

section("Slot Cosine")
print(f"K avg/min/max = {sum(slot_cos_k)/len(slot_cos_k):.4f} / {min(slot_cos_k):.4f} / {max(slot_cos_k):.4f}")
print(f"V avg/min/max = {sum(slot_cos_v)/len(slot_cos_v):.4f} / {min(slot_cos_v):.4f} / {max(slot_cos_v):.4f}")
print(
    f"baseline target={fmt_score(score_no_main)} | "
    f"baseline compare={fmt_score(score_no_compare)}"
)
