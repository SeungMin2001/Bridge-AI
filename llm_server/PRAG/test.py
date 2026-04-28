"""Compact diagnostics for the clean PRAG service-memory implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ALPHA, AUGMENTED_VALID_PATH, MODEL_NAME, WEIGHTS_PATH, load_critical_layer
from .data import load_augmented_examples
from .memory import (
    HyperKVGenerator,
    build_chat_prompt,
    compute_answer_loss,
    encode_memory,
    forward_with_memory,
    make_memory_hook,
    model_num_heads,
    tokenize_qa,
)


def load_model():
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
    return model, tokenizer


@torch.no_grad()
def generate_with_kv(model, tokenizer, target_layer, question, K, V, device, max_new_tokens):
    prompt = build_chat_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hook = target_layer.register_forward_hook(make_memory_hook(K, V, model_num_heads(model), alpha=ALPHA))
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


def hit(text: str, answer: str) -> bool:
    return "".join(answer.lower().split()) in "".join(text.lower().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(AUGMENTED_VALID_PATH))
    parser.add_argument("--weights", default=str(WEIGHTS_PATH))
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--show", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    state = torch.load(Path(args.weights), map_location=device)
    config = state.get("config", {})
    layer_idx = int(config.get("critical_layer", load_critical_layer()))
    target_layer = model.model.layers[layer_idx]
    hypernet = HyperKVGenerator(
        d_model=model.config.hidden_size,
        num_kv=int(config.get("num_kv", 8)),
        hidden_dim=int(config.get("hidden_dim", 1024)),
    ).to(device).float()
    hypernet.load_state_dict(state["hypernet"])
    hypernet.eval()
    examples = load_augmented_examples(args.data, max_samples=args.max_samples)

    main_ok = neg_ok = flip_ok = gen_ok = 0
    total = 0
    with torch.no_grad():
        for idx, ex in enumerate(examples):
            if not (ex.negative_passage and ex.negative_answer):
                continue
            main_mem = encode_memory(model, tokenizer, hypernet, ex.passage, device)
            neg_mem = encode_memory(model, tokenizer, hypernet, ex.negative_passage, device)
            gold_tok = tokenize_qa(tokenizer, ex.question, ex.answer, device)
            neg_tok = tokenize_qa(tokenizer, ex.question, ex.negative_answer, device)
            main_gold = compute_answer_loss(forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], gold_tok), gold_tok["labels"]).item()
            main_neg = compute_answer_loss(forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], neg_tok), neg_tok["labels"]).item()
            neg_gold = compute_answer_loss(forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], gold_tok), gold_tok["labels"]).item()
            neg_neg = compute_answer_loss(forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], neg_tok), neg_tok["labels"]).item()
            main_pref = main_gold < main_neg
            neg_pref = neg_neg < neg_gold
            total += 1
            main_ok += int(main_pref)
            neg_ok += int(neg_pref)
            flip_ok += int(main_pref and neg_pref)
            if idx < args.show:
                real = generate_with_kv(model, tokenizer, target_layer, ex.question, main_mem["K"], main_mem["V"], device, args.max_new_tokens)
                zero = generate_with_kv(model, tokenizer, target_layer, ex.question, torch.zeros_like(main_mem["K"]), torch.zeros_like(main_mem["V"]), device, args.max_new_tokens)
                gen_ok += int(hit(real, ex.answer))
                print(f"\n[{idx + 1}] {ex.question}")
                print(f"  answer={ex.answer} neg={ex.negative_answer}")
                print(f"  main loss: gold={main_gold:.4f} neg={main_neg:.4f} pref={main_pref}")
                print(f"  neg  loss: gold={neg_gold:.4f} neg={neg_neg:.4f} pref={neg_pref}")
                print(f"  gen real={real}")
                print(f"  gen zero={zero}")
    denom = max(total, 1)
    print("\n[PRAG:test]")
    print(f"  evaluated={total}")
    print(f"  main_ok={main_ok / denom:.3f}")
    print(f"  neg_ok={neg_ok / denom:.3f}")
    print(f"  flip_ok={flip_ok / denom:.3f}")
    print(f"  shown_generation_hits={gen_ok}/{max(min(args.show, total), 1)}")


if __name__ == "__main__":
    main()
