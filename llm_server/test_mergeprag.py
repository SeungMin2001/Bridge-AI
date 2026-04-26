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
    KV_PATH_MODE,
    MEMORY_ENCODER_INSTRUCTION,
    NUM_KV,
    POOLED_KV_SKIP_SCALE,
    POOLED_K_SKIP_SCALE,
    POOLED_V_SKIP_SCALE,
    QUERY_LEXICAL_FOCUS_SCALE,
    QUERY_LEXICAL_FOCUS_WINDOW,
    SYSTEM_PROMPT,
    TOKEN_EMBED_SKIP_SCALE,
    TRAIN_PROMPT_FORMAT,
    USE_K_RMS_CLAMP,
    USE_POOLED_KV_SKIP,
    USE_V_RMS_CLAMP,
    USE_CONTEXTUAL_PASSAGE_ENCODER,
    USE_QUESTION_CONDITIONED_MEMORY,
    USE_QUERY_LEXICAL_FOCUS,
    USE_SLOTWISE_POOLING,
    K_RMS_CLAMP,
    V_RMS_CLAMP,
    WEIGHTS_PATH,
    build_chat_text,
    contains_hangul,
    load_critical_layer,
    load_hypernet_state_dict,
    select_memory_encoder_instruction,
    select_system_prompt,
)
from mergePRAG.embedding import (
    encode_passage_states,
    tokenize_conditioned_memory,
    tokenize_passage_memory,
)
from mergePRAG.eval_cases import get_diagnostic_case
from mergePRAG.hypernetwork import HyperNetwork
from mergePRAG.cross_attention import cross_attention

DIAGNOSTIC_CASE = get_diagnostic_case()
QUESTION = DIAGNOSTIC_CASE["question"]
ALT_QUESTION = DIAGNOSTIC_CASE["alt_question"]
CRITICAL_LAYER = load_critical_layer()
PASSAGE = DIAGNOSTIC_CASE["passage"]
COMPARE_PASSAGE = DIAGNOSTIC_CASE["compare_passage"]
EXPECTED_ANSWER = DIAGNOSTIC_CASE["answer"]
COMPARE_EXPECTED_ANSWER = DIAGNOSTIC_CASE["compare_answer"]
ALT_EXPECTED_ANSWER = DIAGNOSTIC_CASE["alt_answer"]
GENERATION_INSTRUCTION = DIAGNOSTIC_CASE["generation_instruction"]
MAX_NEW_TOKENS = 12


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
    loss, sum_logprob, avg_logprob, n_tokens = score
    return (
        f"loss={loss:.3f}, avg_logp={avg_logprob:.3f}, "
        f"sum_logp={sum_logprob:.3f}, toks={n_tokens}"
    )


def fmt_choice(label, score):
    loss, _, avg_logprob, _ = score
    return f"{label}: loss={loss:.3f}, avg_logp={avg_logprob:.3f}"


def verdict(label: str, ok: bool):
    return f"{label}={'OK' if ok else 'FAIL'}"

# ── 모델 로드 ──
print("모델 로딩...")
model, tokenizer = run_model()
device = next(model.parameters()).device
section("Config")
print(f"diagnostic_case={DIAGNOSTIC_CASE.get('case_name')}")
print(f"layer={CRITICAL_LAYER} | num_kv={NUM_KV} | alpha={ALPHA}")
print(
    f"contextual={USE_CONTEXTUAL_PASSAGE_ENCODER} | "
    f"token_embed_skip_scale={TOKEN_EMBED_SKIP_SCALE} | "
    f"kv_path_mode={KV_PATH_MODE} | "
    f"question_conditioned={USE_QUESTION_CONDITIONED_MEMORY} | "
    f"train_prompt_format={TRAIN_PROMPT_FORMAT} | "
    f"pooled_kv_skip={USE_POOLED_KV_SKIP} "
    f"(k_scale={POOLED_K_SKIP_SCALE}, v_scale={POOLED_V_SKIP_SCALE}, default={POOLED_KV_SKIP_SCALE}) | "
    f"query_lexical_focus={USE_QUERY_LEXICAL_FOCUS}:{QUERY_LEXICAL_FOCUS_SCALE}/{QUERY_LEXICAL_FOCUS_WINDOW} | "
    f"slotwise_pooling={USE_SLOTWISE_POOLING} | "
    f"k_rms_clamp={'on' if USE_K_RMS_CLAMP else 'off'}:{K_RMS_CLAMP} | "
    f"v_rms_clamp={'on' if USE_V_RMS_CLAMP else 'off'}:{V_RMS_CLAMP}"
)
print(f"system_prompt={select_system_prompt(QUESTION)[:160]}")
print(f"memory_encoder_instruction={select_memory_encoder_instruction(QUESTION, PASSAGE)[:200]}")

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
        query_focus_mask = encoded.get("query_focus_mask")
        emb = encode_passage_states(
            model,
            ids,
            attention_mask=attention_mask,
            use_contextual=USE_CONTEXTUAL_PASSAGE_ENCODER,
        )
        query = masked_mean(emb, question_mask) if USE_QUESTION_CONDITIONED_MEMORY else None
        parts = hypernet.encode_embedded_components(
            emb,
            attention_mask=attention_mask,
            query=query,
            focus_mask=passage_mask,
            query_focus_mask=query_focus_mask,
        )
        pooled = parts["pooled"]
        hidden = parts["hidden"]
        K_raw = parts["K_raw"]
        V_raw = parts["V_raw"]
        K, V = hypernet.normalize_kv(K_raw, V_raw)
    return {
        "ids": ids,
        "attention_mask": attention_mask,
        "emb": emb,
        "pooled": pooled,
        "hidden": hidden,
        "hidden_v": parts["hidden_v"],
        "K_mlp": parts["K_mlp"],
        "V_mlp": parts["V_mlp"],
        "K_skip": parts["K_skip"],
        "V_skip": parts["V_skip"],
        "K_raw": K_raw,
        "V_raw": V_raw,
        "K": K,
        "V": V,
    }


main_stats = encode_passage_stats(QUESTION, PASSAGE)
compare_stats = encode_passage_stats(QUESTION, COMPARE_PASSAGE)
alt_question_stats = encode_passage_stats(ALT_QUESTION, PASSAGE) if USE_QUESTION_CONDITIONED_MEMORY else None

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

section("Component Stats")
for name in ("hidden_v", "K_mlp", "V_mlp", "K_skip", "V_skip"):
    left = main_stats.get(name)
    right = compare_stats.get(name)
    if left is None or right is None:
        print(f"{name}: n/a")
        continue
    cos = torch.nn.functional.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item()
    left_rms = left.pow(2).mean(dim=-1).sqrt().mean().item()
    right_rms = right.pow(2).mean(dim=-1).sqrt().mean().item()
    print(f"{name}: cos={cos:.4f} | rms={left_rms:.4f}/{right_rms:.4f}")

if USE_QUESTION_CONDITIONED_MEMORY and alt_question_stats is not None:
    alt_pooled = alt_question_stats["pooled"]
    alt_hidden = alt_question_stats["hidden"]
    alt_K = alt_question_stats["K"]
    alt_V = alt_question_stats["V"]
    section("Same Passage / Different Question")
    print(
        f"question: {QUESTION} | alt_question: {ALT_QUESTION}\n"
        f"cos(pooled)={torch.nn.functional.cosine_similarity(pooled.view(1, -1), alt_pooled.view(1, -1)).item():.4f} | "
        f"cos(hidden)={torch.nn.functional.cosine_similarity(h.view(1, -1), alt_hidden.view(1, -1)).item():.4f} | "
        f"cos(K)={torch.nn.functional.cosine_similarity(K.view(1,-1), alt_K.view(1,-1)).item():.4f} | "
        f"cos(V)={torch.nn.functional.cosine_similarity(V.view(1,-1), alt_V.view(1,-1)).item():.4f}"
    )

# ── 프롬프트 포맷 ─────────────────────────────────────────────
# scoring_prompt는 훈련 형식과 일치시켜 loss 비교를 유지한다.
# generation_prompt는 자유 생성이 "the other team" 같은 일반 답으로 빠지는지 보려고
# 정답 엔티티만 요구하는 별도 진단 프롬프트를 사용한다.
def format_prompt(user_prompt: str) -> str:
    if TRAIN_PROMPT_FORMAT == "chat":
        return build_chat_text(tokenizer, question=user_prompt, enable_thinking=False)
    if TRAIN_PROMPT_FORMAT == "plain":
        return user_prompt
    raise ValueError(f"Unsupported MERGEPRAG_TRAIN_PROMPT_FORMAT: {TRAIN_PROMPT_FORMAT}")


def build_scoring_user_prompt(question: str) -> str:
    if contains_hangul(question):
        return f"질문: {question}\n답변:"
    return f"Question: {question}\nAnswer:"


def build_generation_user_prompt(question: str) -> str:
    if contains_hangul(question):
        return f"질문: {question}\n{GENERATION_INSTRUCTION}"
    return f"Question: {question}\n{GENERATION_INSTRUCTION}"


def build_direct_user_prompt(question: str, passage: str) -> str:
    if contains_hangul(f"{question}\n{passage}"):
        return (
            "본문만 사용해서 질문에 답하세요.\n"
            f"본문: {passage}\n"
            f"질문: {question}\n"
            f"{GENERATION_INSTRUCTION}"
        )
    return (
        "Answer the question using only the passage.\n"
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        f"{GENERATION_INSTRUCTION}"
    )


scoring_prompt_text = format_prompt(build_scoring_user_prompt(QUESTION))
generation_prompt_text = format_prompt(build_generation_user_prompt(QUESTION))
direct_main_prompt_text = format_prompt(build_direct_user_prompt(QUESTION, PASSAGE))
direct_compare_prompt_text = format_prompt(build_direct_user_prompt(QUESTION, COMPARE_PASSAGE))
scoring_inputs = tokenizer(scoring_prompt_text, return_tensors="pt").to(device)
generation_inputs = tokenizer(generation_prompt_text, return_tensors="pt").to(device)
direct_main_inputs = tokenizer(direct_main_prompt_text, return_tensors="pt").to(device)
direct_compare_inputs = tokenizer(direct_compare_prompt_text, return_tensors="pt").to(device)


def build_scoring_batch(answer_text: str):
    prompt = format_prompt(build_scoring_user_prompt(QUESTION))
    answer = (
        f"{answer_text}{tokenizer.eos_token}"
        if TRAIN_PROMPT_FORMAT == "chat"
        else f" {answer_text}{tokenizer.eos_token}"
    )
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
    gathered = log_probs.gather(-1, selected_labels.unsqueeze(-1)).squeeze(-1)
    answer_logprob = gathered.sum().item()
    avg_logprob = gathered.mean().item()
    return loss.item(), answer_logprob, avg_logprob, int(selected_labels.numel())

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
    _ = model(**scoring_inputs)
hook.remove()

STOP_TOKEN_SEQUENCES = [
    tokenizer.encode("\nQuestion:", add_special_tokens=False),
    tokenizer.encode("\n질문:", add_special_tokens=False),
]

def decode_answer(gen, input_len):
    tokens = gen[0][input_len:].tolist()
    # 다음 QA 라벨이 나오면 그 앞에서 자름
    for stop_ids in STOP_TOKEN_SEQUENCES:
        if not stop_ids:
            continue
        for i in range(len(tokens) - len(stop_ids) + 1):
            if tokens[i:i+len(stop_ids)] == stop_ids:
                tokens = tokens[:i]
                break
    # think 태그 제거
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = text.splitlines()[0].strip() if text else text
    return text


def prefers_target(score_target, score_compare):
    # 길이가 다른 답변 후보가 많아서 sum logprob 대신 loss/avg logprob를 기준으로 본다.
    target_loss, _, target_avg_logp, _ = score_target
    compare_loss, _, compare_avg_logp, _ = score_compare
    if abs(target_loss - compare_loss) > 1e-6:
        return target_loss < compare_loss
    return target_avg_logp > compare_avg_logp


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


def choose_candidate(score_map: dict[str, tuple]):
    return min(score_map.items(), key=lambda item: item[1][0])

# Hook 없이 생성
with torch.no_grad():
    gen_no_hook = model.generate(
        **generation_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
    )
answer_no = decode_answer(gen_no_hook, generation_inputs["input_ids"].shape[1])
with torch.no_grad():
    gen_direct_main = model.generate(
        **direct_main_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
    )
    gen_direct_compare = model.generate(
        **direct_compare_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
    )
answer_direct_main = decode_answer(gen_direct_main, direct_main_inputs["input_ids"].shape[1])
answer_direct_compare = decode_answer(gen_direct_compare, direct_compare_inputs["input_ids"].shape[1])
score_no_main = score_answer_without_memory(EXPECTED_ANSWER)
score_no_compare = score_answer_without_memory(COMPARE_EXPECTED_ANSWER)
section("Generations")
print(f"no_hook      | ans={answer_no}")
print(f"direct main  | ans={answer_direct_main}")
print(f"direct comp  | ans={answer_direct_compare}")

ALPHA_SWEEP = [round(i / 10, 1) for i in range(1, 8)]

# 여러 alpha로 생성 비교 (main passage)
alpha_rows = []
for alpha in ALPHA_SWEEP:
    hook = layer.register_forward_hook(make_hook(K, V, alpha=alpha))
    with torch.no_grad():
        gen_hook = model.generate(
            **generation_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )
    hook.remove()
    answer_hook = decode_answer(gen_hook, generation_inputs["input_ids"].shape[1])
    score_main = score_answer_with_memory(K, V, EXPECTED_ANSWER, alpha=alpha)
    score_compare = score_answer_with_memory(K, V, COMPARE_EXPECTED_ANSWER, alpha=alpha)
    alpha_rows.append((alpha, answer_hook, score_main, score_compare))

for alpha, answer_hook, score_main, score_compare in alpha_rows:
    prefers_main = prefers_target(score_main, score_compare)
    print(
        f"main α={alpha:<4} | {verdict('prefer_main', prefers_main)} | "
        f"target {fmt_score(score_main)} | compare {fmt_score(score_compare)}"
    )
    print(f"  ans: {answer_hook}")

# 비교 passage도 같은 alpha로 직접 생성
section("Passage Flip")
flip_rows = []
for alpha in ALPHA_SWEEP:
    hook_main = layer.register_forward_hook(make_hook(K, V, alpha=alpha))
    with torch.no_grad():
        gen_main = model.generate(
            **generation_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )
    hook_main.remove()

    hook_compare = layer.register_forward_hook(make_hook(K2, V2, alpha=alpha))
    with torch.no_grad():
        gen_compare = model.generate(
            **generation_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )
    hook_compare.remove()

    answer_main = decode_answer(gen_main, generation_inputs["input_ids"].shape[1])
    answer_compare = decode_answer(gen_compare, generation_inputs["input_ids"].shape[1])
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
    main_prefers_main = prefers_target(main_target_score, main_compare_score)
    compare_prefers_compare = prefers_target(compare_compare_score, compare_target_score)
    print(
        f"α={alpha:<4} | {verdict(f'main->{EXPECTED_ANSWER}', main_prefers_main)} | "
        f"{verdict(f'compare->{COMPARE_EXPECTED_ANSWER}', compare_prefers_compare)}"
    )
    print(f"  main    ans: {answer_main}")
    print(f"  compare ans: {answer_compare}")

section("Candidate Choice")
candidates = [EXPECTED_ANSWER, COMPARE_EXPECTED_ANSWER]
for alpha in ALPHA_SWEEP:
    main_scores = {
        candidate: score_answer_with_memory(K, V, candidate, alpha=alpha)
        for candidate in candidates
    }
    compare_scores = {
        candidate: score_answer_with_memory(K2, V2, candidate, alpha=alpha)
        for candidate in candidates
    }
    main_choice, main_choice_score = choose_candidate(main_scores)
    compare_choice, compare_choice_score = choose_candidate(compare_scores)
    print(
        f"α={alpha:<4} | "
        f"main_choice={main_choice} ({fmt_choice(main_choice, main_choice_score)}) | "
        f"compare_choice={compare_choice} ({fmt_choice(compare_choice, compare_choice_score)})"
    )
    print(
        "  main scores    | "
        + " | ".join(fmt_choice(candidate, score) for candidate, score in main_scores.items())
    )
    print(
        "  compare scores | "
        + " | ".join(fmt_choice(candidate, score) for candidate, score in compare_scores.items())
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
