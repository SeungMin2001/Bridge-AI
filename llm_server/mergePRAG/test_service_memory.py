"""
Evaluate the service-oriented lecture memory HyperNetwork.

Run:
    python -m llm_server.mergePRAG.test_service_memory --split valid --max-samples 240
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_NAME, TRAIN_DATA_PATH, VALID_DATA_PATH, load_critical_layer
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
    parser.add_argument("--split", choices=["valid", "train"], default="valid")
    parser.add_argument("--max-samples", type=int, default=240)
    parser.add_argument("--weights-path", default=WEIGHTS_PATH)
    parser.add_argument("--show-examples", type=int, default=10)
    parser.add_argument("--show-generations", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=SERVICE_ALPHA)
    return parser.parse_args()


def fmt(value: float) -> str:
    return "nan" if value != value else f"{value:.4f}"


def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def load_state(path: Path, map_location):
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "hypernet" in state:
        return state["hypernet"], state
    return state, {"config": None, "step": None}


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
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_with_memory(model, tokenizer, hypernet, target_layer, question, passage, device, alpha, use_contextual):
    mem = encode_memory(model, hypernet, tokenizer, passage, device, use_contextual=use_contextual)
    prompt = build_chat_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hook = target_layer.register_forward_hook(make_memory_hook(mem["K"], mem["V"], alpha=alpha))
    try:
        generated = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        hook.remove()
    return tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    args = parse_args()
    weights_path = Path(args.weights_path)
    if not weights_path.exists():
        checkpoint = Path(CHECKPOINT_PATH)
        if checkpoint.exists():
            print(f"[test_service_memory] weights missing; using checkpoint: {checkpoint}")
            weights_path = checkpoint
        else:
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
    ).to(device).float()
    hypernet.load_state_dict(state_dict)
    hypernet.eval()
    layer_idx = int(config.get("critical_layer", load_critical_layer()))
    target_layer = model.model.layers[layer_idx]
    print(f"[test_service_memory] step={meta.get('step')} layer={layer_idx} num_kv={num_kv} alpha={args.alpha}")

    dataset_path = VALID_DATA_PATH if args.split == "valid" else TRAIN_DATA_PATH
    dataset = MergePRAGDataset(dataset_path, max_samples=args.max_samples)

    total = 0
    base_ok = direct_ok = main_ok = neg_ok = flip_ok = 0
    gain_sum = k_cos_sum = v_cos_sum = pooled_cos_sum = 0.0
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
                print(f"\n[{idx}] q={q}")
                print(f"  gold={gold} | neg={neg}")
                print(f"  no_hook      gold={fmt(base_gold)} neg={fmt(base_neg)} prefer_gold={base_pref}")
                print(f"  direct       gold={fmt(direct_gold)} neg={fmt(direct_neg)} prefer_gold={direct_pref}")
                print(f"  main_memory  gold={fmt(main_gold)} neg={fmt(main_neg)} prefer_gold={main_pref}")
                print(f"  neg_memory   gold={fmt(negmem_gold)} neg={fmt(negmem_neg)} prefer_neg={neg_pref}")
                print(f"  memory_cos   pooled={pooled_cos:.4f} K={k_cos:.4f} V={v_cos:.4f}")
                print(f"  gain={fmt(base_gold - main_gold)} flip_ok={main_pref and neg_pref}")
                if shown < args.show_generations:
                    no_hook = generate_from_prompt(model, tokenizer, build_chat_prompt(tokenizer, q), device)
                    direct = generate_from_prompt(model, tokenizer, build_direct_chat_prompt(tokenizer, q, passage), device)
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
                    )
                    print(f"  gen no_hook={no_hook}")
                    print(f"  gen direct ={direct}")
                    print(f"  gen memory ={memory}")
                shown += 1

            if torch.cuda.is_available() and total % 50 == 0:
                torch.cuda.empty_cache()

    denom = max(total, 1)
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
