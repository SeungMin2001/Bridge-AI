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
    tokenize_direct_qa,
    tokenize_qa,
)
from .train2 import MergePRAGDataset, extract_first_hard_negative
from .train_service_memory import CHECKPOINT_PATH, WEIGHTS_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate service memory weights.")
    parser.add_argument("--dataset", action="store_true", help="Evaluate dataset rows instead of one diagnostic sample.")
    parser.add_argument("--split", choices=["valid", "train"], default="valid")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--weights-path", default="")
    parser.add_argument("--show-examples", type=int, default=1)
    parser.add_argument("--show-generations", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=SERVICE_ALPHA)
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
def generate_from_prompt(model, tokenizer, prompt, device, max_new_tokens=8):
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
def generate_with_memory(model, tokenizer, hypernet, target_layer, question, passage, device, alpha, use_contextual, max_new_tokens=8):
    mem = encode_memory(model, hypernet, tokenizer, passage, device, use_contextual=use_contextual)
    prompt = build_chat_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hook = target_layer.register_forward_hook(make_memory_hook(mem["K"], mem["V"], alpha=alpha))
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


def main():
    args = parse_args()
    checkpoint = Path(CHECKPOINT_PATH)
    weights = Path(WEIGHTS_PATH)
    if args.weights_path:
        weights_path = Path(args.weights_path)
    elif checkpoint.exists():
        weights_path = checkpoint
    else:
        weights_path = weights
    if not weights_path.exists():
        raise FileNotFoundError(f"No service memory weights found: {weights_path}")

    print(f"[test_service_memory] model={MODEL_NAME}")
    print(f"[test_service_memory] weights={weights_path}")

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
    print(
        f"[test_service_memory] step={meta.get('step')} layer={layer_idx} "
        f"num_kv={num_kv} mode={config.get('pooling_mode', 'slot')} alpha={args.alpha}"
    )

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

            base_gold = score_answer(model, tokenizer, hypernet, target_layer, q, None, gold, device, args.alpha, use_contextual)
            base_neg = score_answer(model, tokenizer, hypernet, target_layer, q, None, neg, device, args.alpha, use_contextual)
            direct_gold = score_direct_answer(model, tokenizer, q, passage, gold, device)
            direct_neg = score_direct_answer(model, tokenizer, q, passage, neg, device)
            main_gold = score_answer(model, tokenizer, hypernet, target_layer, q, passage, gold, device, args.alpha, use_contextual)
            main_neg = score_answer(model, tokenizer, hypernet, target_layer, q, passage, neg, device, args.alpha, use_contextual)
            negmem_gold = score_answer(model, tokenizer, hypernet, target_layer, q, neg_passage, gold, device, args.alpha, use_contextual)
            negmem_neg = score_answer(model, tokenizer, hypernet, target_layer, q, neg_passage, neg, device, args.alpha, use_contextual)

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
                        args.alpha,
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
        print(f"  direct passage understands data : {direct_ok / denom:.3f} ({direct_ok}/{denom})")
        print(f"  main memory chooses main answer : {main_ok / denom:.3f} ({main_ok}/{denom})")
        print(f"  neg memory chooses neg answer   : {neg_ok / denom:.3f} ({neg_ok}/{denom})")
        print(f"  full passage flip success       : {flip_ok / denom:.3f} ({flip_ok}/{denom})")
        print(f"  avg memory gain                 : {gain_sum / denom:.4f}")
        print(f"  avg memory cosine               : pooled={pooled_cos_sum / denom:.4f}, K={k_cos_sum / denom:.4f}, V={v_cos_sum / denom:.4f}")
        if args.show_generations:
            gen_denom = max(min(args.show_examples, args.show_generations, total), 1)
            print(f"  shown generation memory hits    : {memory_generation_ok}/{gen_denom}")
        print("\n[Decision]")
        if flip_ok / denom >= 0.70:
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
