"""Single Korean/English sanity checks for PRAG K/V passage injection.

This diagnostic uses one Korean and one English passage pair so it is easy to
see whether injected K/V memory changes the model's answer in each language.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ALPHA, MODEL_NAME, WEIGHTS_PATH, load_critical_layer
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


CASES = [
    {
        "name": "ko_quiz_day",
        "question": "신승민은 누구야?",
        "main_passage": "신승민은 선문대 학생입니다.",
        "negative_passage": "서울대 학생입니다.",
        "main_answer": "선문대 학생",
        "negative_answer": "서울대 학생",
    },
    {
        "name": "en_quiz_day",
        "question": "When is the quiz?",
        "main_passage": "The quiz is on Wednesday.",
        "negative_passage": "The quiz is on Friday.",
        "main_answer": "Wednesday",
        "negative_answer": "Friday",
    },
]


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
def generate_plain(model, tokenizer, question: str, device, max_new_tokens: int) -> str:
    prompt = build_chat_prompt(tokenizer, question)
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
    return "".join(answer.casefold().split()) in "".join(text.casefold().split())


def run_case(model, tokenizer, hypernet, target_layer, device, case: dict, max_new_tokens: int) -> bool:
    question = case["question"]
    main_passage = case["main_passage"]
    negative_passage = case["negative_passage"]
    main_answer = case["main_answer"]
    negative_answer = case["negative_answer"]

    with torch.no_grad():
        main_mem = encode_memory(model, tokenizer, hypernet, main_passage, device)
        neg_mem = encode_memory(model, tokenizer, hypernet, negative_passage, device)

        main_answer_tok = tokenize_qa(tokenizer, question, main_answer, device)
        neg_answer_tok = tokenize_qa(tokenizer, question, negative_answer, device)

        main_main_loss = compute_answer_loss(
            forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], main_answer_tok),
            main_answer_tok["labels"],
        ).item()
        main_neg_loss = compute_answer_loss(
            forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], neg_answer_tok),
            neg_answer_tok["labels"],
        ).item()
        neg_main_loss = compute_answer_loss(
            forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], main_answer_tok),
            main_answer_tok["labels"],
        ).item()
        neg_neg_loss = compute_answer_loss(
            forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], neg_answer_tok),
            neg_answer_tok["labels"],
        ).item()

        no_hook = generate_plain(model, tokenizer, question, device, max_new_tokens)
        main_gen = generate_with_kv(
            model, tokenizer, target_layer, question, main_mem["K"], main_mem["V"], device, max_new_tokens
        )
        neg_gen = generate_with_kv(
            model, tokenizer, target_layer, question, neg_mem["K"], neg_mem["V"], device, max_new_tokens
        )
        zero_gen = generate_with_kv(
            model,
            tokenizer,
            target_layer,
            question,
            torch.zeros_like(main_mem["K"]),
            torch.zeros_like(main_mem["V"]),
            device,
            max_new_tokens,
        )

    main_ok = main_main_loss < main_neg_loss
    neg_ok = neg_neg_loss < neg_main_loss
    main_gen_ok = hit(main_gen, main_answer)
    neg_gen_ok = hit(neg_gen, negative_answer)
    passed = main_ok and neg_ok and main_gen_ok and neg_gen_ok

    print(f"\n[case:{case['name']}]")
    print(f"question: {question}")
    print(f"main passage: {main_passage}")
    print(f"negative passage: {negative_passage}")
    print(
        f"main memory loss: {main_answer}={main_main_loss:.4f} "
        f"vs {negative_answer}={main_neg_loss:.4f} pref={main_ok}"
    )
    print(
        f"neg  memory loss: {main_answer}={neg_main_loss:.4f} "
        f"vs {negative_answer}={neg_neg_loss:.4f} pref={neg_ok}"
    )
    print(f"gen no_hook : {no_hook}")
    print(f"gen main_kv : {main_gen} [{'OK' if main_gen_ok else 'FAIL'}]")
    print(f"gen neg_kv  : {neg_gen} [{'OK' if neg_gen_ok else 'FAIL'}]")
    print(f"gen zero_kv : {zero_gen}")
    print(f"case decision: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(WEIGHTS_PATH))
    parser.add_argument("--max-new-tokens", type=int, default=16)
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

    print("[PRAG:single-ko-en]")
    print(f"model={MODEL_NAME}")
    print(f"layer={layer_idx} alpha={ALPHA} weights={args.weights}")
    passed = [
        run_case(model, tokenizer, hypernet, target_layer, device, case, args.max_new_tokens)
        for case in CASES
    ]
    print(f"\nfinal decision: {'PASS' if all(passed) else 'FAIL'} ({sum(passed)}/{len(passed)})")


if __name__ == "__main__":
    main()
