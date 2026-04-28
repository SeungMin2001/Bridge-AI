"""
Evaluate the service-oriented lecture memory HyperNetwork.

Run:
    python -m llm_server.mergePRAG.test_service_memory
    python -m llm_server.mergePRAG.test_service_memory --case synthetic_ko
    python -m llm_server.mergePRAG.test_service_memory --dataset --max-samples 40
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_NAME, TRAIN_DATA_PATH, VALID_DATA_PATH, load_critical_layer
from .eval_cases import get_diagnostic_case
from .service_memory import (
    SERVICE_ALPHA,
    ServiceMemoryHyperNetwork,
    build_chat_prompt,
    build_direct_chat_prompt,
    compute_answer_loss,
    encode_memory,
    forward_with_memory,
    make_memory_hook,
    model_num_heads,
    tokenize_direct_qa,
    tokenize_qa,
)
from .train2 import MergePRAGDataset, extract_first_hard_negative

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = os.getenv("MERGEPRAG_SERVICE_CHECKPOINT_PATH", str(BASE_DIR / "service_memory_checkpoint.pt"))
WEIGHTS_PATH = os.getenv("MERGEPRAG_SERVICE_WEIGHTS_PATH", str(BASE_DIR / "service_memory_weights.pt"))
OVERFIT_CHECKPOINT_PATH = BASE_DIR / "service_memory_overfit_checkpoint.pt"
OVERFIT_WEIGHTS_PATH = BASE_DIR / "service_memory_overfit_weights.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate service memory weights.")
    parser.add_argument("--dataset", action="store_true", help="Evaluate dataset rows instead of one diagnostic sample.")
    parser.add_argument("--split", choices=["valid", "train"], default="valid")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--weights-path", default="")
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Evaluate the overfit checkpoint/weights instead of the normal service-memory checkpoint.",
    )
    parser.add_argument("--show-examples", type=int, default=1)
    parser.add_argument("--show-generations", type=int, default=1)
    parser.add_argument(
        "--kv-necessity",
        action="store_true",
        help="For shown examples, compare real K/V against zero and random K/V generations.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Injection strength. Defaults to the checkpoint config alpha, then code default.",
    )
    parser.add_argument("--case", default=os.getenv("MERGEPRAG_DIAGNOSTIC_CASE", "service_memory"))
    parser.add_argument("--question", default=os.getenv("MERGEPRAG_TEST_QUESTION", ""))
    parser.add_argument("--passage", default=os.getenv("MERGEPRAG_TEST_PASSAGE", ""))
    parser.add_argument("--answer", default=os.getenv("MERGEPRAG_TEST_ANSWER", ""))
    parser.add_argument("--negative-passage", default=os.getenv("MERGEPRAG_TEST_NEGATIVE_PASSAGE", ""))
    parser.add_argument("--negative-answer", default=os.getenv("MERGEPRAG_TEST_NEGATIVE_ANSWER", ""))
    parser.add_argument(
        "--simple",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a compact pass/fail-oriented report.",
    )
    return parser.parse_args()


def fmt(value: float) -> str:
    return "nan" if value != value else f"{value:.4f}"


def yn(value: bool) -> str:
    return "OK" if value else "FAIL"


def newer_than(a: Path, b: Path) -> bool:
    return a.exists() and b.exists() and a.stat().st_mtime > b.stat().st_mtime


def normalize_text(text: str) -> str:
    return "".join(str(text or "").lower().split())


def first_answer_line(text: str) -> str:
    text = str(text or "").replace("</think>", "\n").strip()
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text


def answer_hit(generated: str, expected: str) -> bool:
    return normalize_text(expected) in normalize_text(first_answer_line(generated))


def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.dim() >= 3 and b.dim() >= 3 and a.size(1) != b.size(1):
        keep = min(a.size(1), b.size(1))
        a = a[:, :keep]
        b = b[:, :keep]
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def load_state(path: Path, map_location):
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "hypernet" in state:
        return state["hypernet"], state
    return state, {"config": None, "step": None}


def build_single_sample(args):
    case = get_diagnostic_case(args.case)
    question = args.question or case["question"]
    passage = args.passage or case["passage"]
    answer = args.answer or case["answer"]
    negative_passage = args.negative_passage or case["compare_passage"]
    negative_answer = args.negative_answer or case["compare_answer"]
    return {
        "source_id": f"diagnostic:{case.get('case_name', args.case)}",
        "task": "final_qa",
        "question": question,
        "answer": answer,
        "passage": passage,
        "hard_negatives": [{"passage": negative_passage, "answer": negative_answer}],
    }


@torch.no_grad()
def score_answer(model, tokenizer, hypernet, target_layer, question, passage, answer, device, alpha, use_contextual):
    tok = tokenize_qa(tokenizer, question, answer, device)
    if passage is None:
        logits = model(**tok)["logits"]
    else:
        mem = encode_memory(model, hypernet, tokenizer, passage, device, use_contextual=use_contextual)
        logits = forward_with_memory(model, target_layer, mem["K"], mem["V"], tok, alpha=alpha)
    loss = compute_answer_loss(logits, tok["labels"])
    return float("nan") if loss is None else loss.item()


@torch.no_grad()
def score_direct_answer(model, tokenizer, question, passage, answer, device):
    tok = tokenize_direct_qa(tokenizer, question, passage, answer, device)
    logits = model(**tok)["logits"]
    loss = compute_answer_loss(logits, tok["labels"])
    return float("nan") if loss is None else loss.item()


@torch.no_grad()
def generate_from_prompt(model, tokenizer, prompt, device, max_new_tokens=24):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_with_memory(model, tokenizer, hypernet, target_layer, question, passage, device, alpha, use_contextual, max_new_tokens=24):
    mem = encode_memory(model, hypernet, tokenizer, passage, device, use_contextual=use_contextual)
    return generate_with_given_memory(
        model,
        tokenizer,
        target_layer,
        question,
        mem["K"],
        mem["V"],
        device,
        alpha,
        max_new_tokens=max_new_tokens,
    )


@torch.no_grad()
def generate_with_given_memory(model, tokenizer, target_layer, question, K, V, device, alpha, max_new_tokens=24):
    prompt = build_chat_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hook = target_layer.register_forward_hook(
        make_memory_hook(K, V, num_heads=model_num_heads(model), alpha=alpha)
    )
    try:
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        hook.remove()
    return tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def kv_necessity_generations(model, tokenizer, hypernet, target_layer, question, passage, device, alpha, use_contextual, max_new_tokens=24):
    mem = encode_memory(model, hypernet, tokenizer, passage, device, use_contextual=use_contextual)
    K_real, V_real = mem["K"], mem["V"]
    K_zero = torch.zeros_like(K_real)
    V_zero = torch.zeros_like(V_real)
    k_std = K_real.float().std().clamp_min(1e-6).to(device=K_real.device, dtype=K_real.dtype)
    v_std = V_real.float().std().clamp_min(1e-6).to(device=V_real.device, dtype=V_real.dtype)
    K_random = torch.randn_like(K_real) * k_std
    V_random = torch.randn_like(V_real) * v_std
    return {
        "real": generate_with_given_memory(model, tokenizer, target_layer, question, K_real, V_real, device, alpha, max_new_tokens),
        "zero": generate_with_given_memory(model, tokenizer, target_layer, question, K_zero, V_zero, device, alpha, max_new_tokens),
        "random": generate_with_given_memory(model, tokenizer, target_layer, question, K_random, V_random, device, alpha, max_new_tokens),
    }


def main():
    args = parse_args()
    checkpoint = Path(CHECKPOINT_PATH)
    weights = Path(WEIGHTS_PATH)
    if args.weights_path:
        weights_path = Path(args.weights_path)
        load_source = "explicit"
    elif args.overfit:
        weights_path = OVERFIT_CHECKPOINT_PATH if OVERFIT_CHECKPOINT_PATH.exists() else OVERFIT_WEIGHTS_PATH
        load_source = "overfit-checkpoint" if weights_path == OVERFIT_CHECKPOINT_PATH else "overfit-weights"
    elif checkpoint.exists():
        weights_path = checkpoint
        load_source = "checkpoint"
    else:
        weights_path = weights
        load_source = "weights"
    if not weights_path.exists():
        raise FileNotFoundError(f"No service memory weights found: {weights_path}")

    print(f"[test_service_memory] model={MODEL_NAME}")
    print(f"[test_service_memory] load_source={load_source} path={weights_path}")
    if not args.overfit and not args.weights_path and newer_than(OVERFIT_CHECKPOINT_PATH, checkpoint):
        print(
            "[test_service_memory:note] overfit checkpoint is newer than the normal checkpoint; "
            "use --overfit if you intended to evaluate it."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    device = next(model.parameters()).device
    d_model = model.config.hidden_size
    state_dict, meta = load_state(weights_path, map_location=device)
    config = meta.get("config") or {}
    alpha = float(args.alpha if args.alpha is not None else config.get("alpha", SERVICE_ALPHA))
    feature_dim = int(config.get("feature_dim", d_model * 2))
    num_kv = int(config.get("num_kv", 8))
    hidden_dim = int(config.get("hidden_dim", 1024))
    use_contextual = bool(config.get("use_contextual", True))
    hypernet = ServiceMemoryHyperNetwork(
        d_model=d_model,
        feature_dim=feature_dim,
        num_kv=num_kv,
        hidden_dim=hidden_dim,
        skip_scale=float(config.get("skip_scale", 0.5)),
        rms_clamp=float(config.get("rms_clamp", 0.5)),
        pooling_mode=str(config.get("pooling_mode", "slot")),
        max_memory_tokens=int(config.get("max_memory_tokens", 128)),
    ).to(device).float()
    hypernet.load_state_dict(state_dict)
    hypernet.eval()
    layer_idx = int(config.get("critical_layer", load_critical_layer()))
    target_layer = model.model.layers[layer_idx]
    objective = str(config.get("objective", "unknown"))
    print(
        f"[test_service_memory] step={meta.get('step')} layer={layer_idx} "
        f"num_kv={num_kv} mode={config.get('pooling_mode', 'slot')} alpha={alpha}"
    )
    print(
        "[test_service_memory] checkpoint_config | "
        f"objective={objective} | "
        f"train_phase={config.get('train_phase', 'unknown')} | "
        f"teacher_kl={config.get('teacher_kl_weight', 'n/a')} | "
        f"max_memory_tokens={config.get('max_memory_tokens', 'n/a')} | "
        f"overfit_case={config.get('overfit_case', '') or 'none'}"
    )
    if objective == "simple" or config.get("train_phase") == "phase1":
        print("[test_service_memory:note] phase1/simple checkpoint: negative_memory and flip are not trained yet.")

    if args.dataset:
        dataset_path = VALID_DATA_PATH if args.split == "valid" else TRAIN_DATA_PATH
        dataset = MergePRAGDataset(dataset_path, max_samples=args.max_samples)
        print(f"[test_service_memory] mode=dataset split={args.split} max_samples={args.max_samples}")
    else:
        dataset = [build_single_sample(args)]
        print(f"[test_service_memory] mode=single case={args.case}")

    total = 0
    base_ok = direct_ok = main_ok = neg_ok = flip_ok = 0
    gain_sum = k_cos_sum = v_cos_sum = pooled_cos_sum = 0.0
    direct_unknown = memory_generation_ok = 0
    kv_real_ok = kv_zero_ok = kv_random_ok = 0
    shown = 0
    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            negative = extract_first_hard_negative(sample)
            if not negative or not negative.get("answer") or not negative.get("passage"):
                continue
            q = sample["question"]
            passage = sample["passage"]
            gold = sample["answer"]
            neg_passage = negative["passage"]
            neg = negative["answer"]
            prompt_lang = "ko" if any("\uac00" <= ch <= "\ud7a3" for ch in q) else "en"

            base_gold = score_answer(model, tokenizer, hypernet, target_layer, q, None, gold, device, alpha, use_contextual)
            base_neg = score_answer(model, tokenizer, hypernet, target_layer, q, None, neg, device, alpha, use_contextual)
            direct_gold = score_direct_answer(model, tokenizer, q, passage, gold, device)
            direct_neg = score_direct_answer(model, tokenizer, q, passage, neg, device)
            main_gold = score_answer(model, tokenizer, hypernet, target_layer, q, passage, gold, device, alpha, use_contextual)
            main_neg = score_answer(model, tokenizer, hypernet, target_layer, q, passage, neg, device, alpha, use_contextual)
            negmem_gold = score_answer(model, tokenizer, hypernet, target_layer, q, neg_passage, gold, device, alpha, use_contextual)
            negmem_neg = score_answer(model, tokenizer, hypernet, target_layer, q, neg_passage, neg, device, alpha, use_contextual)

            main_mem = encode_memory(model, hypernet, tokenizer, passage, device, use_contextual=use_contextual)
            neg_mem = encode_memory(model, hypernet, tokenizer, neg_passage, device, use_contextual=use_contextual)
            pooled_cos = cosine_flat(main_mem["pooled"], neg_mem["pooled"])
            k_cos = cosine_flat(main_mem["K"], neg_mem["K"])
            v_cos = cosine_flat(main_mem["V"], neg_mem["V"])

            base_pref = base_gold < base_neg
            direct_pref = direct_gold < direct_neg
            main_pref = main_gold < main_neg
            neg_pref = negmem_neg < negmem_gold
            total += 1
            base_ok += int(base_pref)
            direct_ok += int(direct_pref)
            main_ok += int(main_pref)
            neg_ok += int(neg_pref)
            flip_ok += int(main_pref and neg_pref)
            gain_sum += base_gold - main_gold
            pooled_cos_sum += pooled_cos
            k_cos_sum += k_cos
            v_cos_sum += v_cos

            if shown < args.show_examples:
                if args.simple:
                    print(f"\n[{shown + 1}] {q}")
                    print(f"  prompt_lang={prompt_lang} | alpha={alpha}")
                    print(f"  expected main={gold} | expected negative={neg}")
                    print(f"  direct_prompt={yn(direct_pref)} | main_memory={yn(main_pref)} | negative_memory={yn(neg_pref)} | flip={yn(main_pref and neg_pref)}")
                    print(f"  loss main_memory: gold={fmt(main_gold)} vs neg={fmt(main_neg)}")
                    print(f"  loss neg_memory : gold={fmt(negmem_gold)} vs neg={fmt(negmem_neg)}")
                    print(f"  memory_cos pooled={pooled_cos:.4f}, K={k_cos:.4f}, V={v_cos:.4f}")
                else:
                    print(f"\n[{idx}] q={q}")
                    print(f"  gold={gold} | neg={neg}")
                    print(f"  no_hook      gold={fmt(base_gold)} neg={fmt(base_neg)} prefer_gold={base_pref}")
                    print(f"  direct       gold={fmt(direct_gold)} neg={fmt(direct_neg)} prefer_gold={direct_pref}")
                    print(f"  main_memory  gold={fmt(main_gold)} neg={fmt(main_neg)} prefer_gold={main_pref}")
                    print(f"  neg_memory   gold={fmt(negmem_gold)} neg={fmt(negmem_neg)} prefer_neg={neg_pref}")
                    print(f"  memory_cos   pooled={pooled_cos:.4f} K={k_cos:.4f} V={v_cos:.4f}")
                    print(f"  gain={fmt(base_gold - main_gold)} flip_ok={main_pref and neg_pref}")
                if shown < args.show_generations:
                    no_hook = generate_from_prompt(
                        model,
                        tokenizer,
                        build_chat_prompt(tokenizer, q),
                        device,
                        max_new_tokens=args.max_new_tokens,
                    )
                    direct = generate_from_prompt(
                        model,
                        tokenizer,
                        build_direct_chat_prompt(tokenizer, q, passage),
                        device,
                        max_new_tokens=args.max_new_tokens,
                    )
                    memory = generate_with_memory(
                        model,
                        tokenizer,
                        hypernet,
                        target_layer,
                        q,
                        passage,
                        device,
                        alpha,
                        use_contextual,
                        max_new_tokens=args.max_new_tokens,
                    )
                    direct_hit = answer_hit(direct, gold)
                    memory_hit = answer_hit(memory, gold)
                    no_hook_hit = answer_hit(no_hook, gold)
                    direct_unknown += int(not no_hook_hit and direct_hit)
                    memory_generation_ok += int(memory_hit)
                    if args.simple:
                        print(f"  gen no_hook raw={no_hook}")
                        print(f"  gen direct  raw={direct} [{yn(direct_hit)}]")
                        print(f"  gen memory  raw={memory} [{yn(memory_hit)}]")
                        if direct_pref and not direct_hit:
                            print("  note: direct loss is OK but free generation is unstable; use loss/flip as the main signal.")
                        if args.kv_necessity:
                            kv_gen = kv_necessity_generations(
                                model,
                                tokenizer,
                                hypernet,
                                target_layer,
                                q,
                                passage,
                                device,
                                alpha,
                                use_contextual,
                                max_new_tokens=args.max_new_tokens,
                            )
                            print(f"  kv real   raw={kv_gen['real']}")
                            print(f"  kv zero   raw={kv_gen['zero']}")
                            print(f"  kv random raw={kv_gen['random']}")
                            kv_real_hit = answer_hit(kv_gen["real"], gold)
                            kv_zero_hit = answer_hit(kv_gen["zero"], gold)
                            kv_random_hit = answer_hit(kv_gen["random"], gold)
                            kv_real_ok += int(kv_real_hit)
                            kv_zero_ok += int(not kv_zero_hit)
                            kv_random_ok += int(not kv_random_hit)
                    else:
                        print(f"  gen no_hook={no_hook}")
                        print(f"  gen direct ={direct}")
                        print(f"  gen memory ={memory}")
                shown += 1

            if torch.cuda.is_available() and total % 50 == 0:
                torch.cuda.empty_cache()

    denom = max(total, 1)
    if args.simple:
        print("\n[Simple Summary]")
        print(f"  evaluated: {total}/{len(dataset)} hard-pair rows")
        print(f"  no-passage answer preference    : {base_ok / denom:.3f} ({base_ok}/{denom})")
        print(f"  direct passage understands data : {direct_ok / denom:.3f} ({direct_ok}/{denom})")
        print(f"  main memory chooses main answer : {main_ok / denom:.3f} ({main_ok}/{denom})")
        print(f"  neg memory chooses neg answer   : {neg_ok / denom:.3f} ({neg_ok}/{denom})")
        print(f"  full passage flip success       : {flip_ok / denom:.3f} ({flip_ok}/{denom})")
        print(f"  avg memory gain                 : {gain_sum / denom:.4f}")
        print(f"  avg memory cosine               : pooled={pooled_cos_sum / denom:.4f}, K={k_cos_sum / denom:.4f}, V={v_cos_sum / denom:.4f}")
        if args.show_generations:
            gen_denom = max(min(args.show_examples, args.show_generations, total), 1)
            print(f"  shown generation memory hits    : {memory_generation_ok}/{gen_denom}")
            if args.kv_necessity:
                print(f"  shown kv real hits              : {kv_real_ok}/{gen_denom}")
                print(f"  shown kv zero misses            : {kv_zero_ok}/{gen_denom}")
                print(f"  shown kv random misses          : {kv_random_ok}/{gen_denom}")
        print("\n[Decision]")
        is_phase1 = objective == "simple" or config.get("train_phase") == "phase1"
        if is_phase1 and args.kv_necessity and kv_real_ok > 0 and kv_zero_ok > 0 and kv_random_ok > 0:
            print("  PASS: phase1 overfit passed; real K/V is necessary for the answer.")
        elif is_phase1:
            print("  PHASE1: judge this run with --kv-necessity; flip is expected to fail before phase2.")
        elif flip_ok / denom >= 0.70:
            print("  PASS: K/V memory is learning passage-specific answers.")
        elif flip_ok / denom >= 0.30:
            print("  PARTIAL: memory has signal, but passage flip is still unreliable.")
        else:
            print("  FAIL: model is not reliably answering from injected passage memory yet.")
    else:
        print("\n[test_service_memory:summary]")
        print(f"  evaluated hard-pair rows: {total}/{len(dataset)}")
        print(f"  avg memory gain: {gain_sum / denom:.4f}")
        print(f"  avg memory cosine: pooled={pooled_cos_sum / denom:.4f}, K={k_cos_sum / denom:.4f}, V={v_cos_sum / denom:.4f}")
        print(f"  no_hook gold preference: {base_ok}/{denom} = {base_ok / denom:.3f}")
        print(f"  direct passage gold preference: {direct_ok}/{denom} = {direct_ok / denom:.3f}")
        print(f"  main memory gold preference: {main_ok}/{denom} = {main_ok / denom:.3f}")
        print(f"  negative memory negative preference: {neg_ok}/{denom} = {neg_ok / denom:.3f}")
        print(f"  bidirectional flip success: {flip_ok}/{denom} = {flip_ok / denom:.3f}")


if __name__ == "__main__":
    main()
