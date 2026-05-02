"""Single Korean sanity check for PRAG K/V passage injection.

This diagnostic uses the same chat prompt format as training/test diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ALPHA, MODEL_NAME, MULTIFACT_WEIGHTS_PATH, WEIGHTS_PATH, load_critical_layer
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
        "name": "ko_lecture_multifact",
        "question": "승민 팀장의 운영 회의에서 루바 알림 담당자는 누구야?",
        "main_passage": (
            "승민 팀장은 운영 회의에서 루바 알림 담당자는 지영이라고 설명했다."
        ),
        "negative_passage": (
            "승민 팀장은 운영 회의에서 루바 알림 담당자는 민수라고 설명했다."
        ),
        "main_answer": "지영",
        "negative_answer": "민수",
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


def run_case(model, tokenizer, hypernet, target_layer, device, case: dict, max_new_tokens: int) -> None:
    question = case["question"]
    main_passage = case["main_passage"]
    negative_passage = case["negative_passage"]
    main_answer = case["main_answer"]
    negative_answer = case["negative_answer"]

    with torch.no_grad():
        main_mem = encode_memory(model, tokenizer, hypernet, main_passage, device)
        neg_mem = encode_memory(model, tokenizer, hypernet, negative_passage, device)
        main_tok = tokenize_qa(tokenizer, question, main_answer, device)
        neg_tok = tokenize_qa(tokenizer, question, negative_answer, device)
        main_gold = compute_answer_loss(
            forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], main_tok, alpha=ALPHA),
            main_tok["labels"],
        ).item()
        main_neg = compute_answer_loss(
            forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], neg_tok, alpha=ALPHA),
            neg_tok["labels"],
        ).item()
        neg_gold = compute_answer_loss(
            forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], main_tok, alpha=ALPHA),
            main_tok["labels"],
        ).item()
        neg_neg = compute_answer_loss(
            forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], neg_tok, alpha=ALPHA),
            neg_tok["labels"],
        ).item()
        no_passage = generate_plain(model, tokenizer, question, device, max_new_tokens)
        main_gen = generate_with_kv(
            model, tokenizer, target_layer, question, main_mem["K"], main_mem["V"], device, max_new_tokens
        )
        neg_gen = generate_with_kv(
            model, tokenizer, target_layer, question, neg_mem["K"], neg_mem["V"], device, max_new_tokens
        )

    print(f"\n[case:{case['name']}]")
    print(f"question: {question}")
    print(f"passage: {main_passage}")
    print(f"negative passage: {negative_passage}")
    print(f"expected: {main_answer}")
    print(f"negative expected: {negative_answer}")
    print("\n[candidate loss]")
    print(f"main K/V: {main_answer}={main_gold:.4f} vs {negative_answer}={main_neg:.4f} pref={main_gold < main_neg}")
    print(f"neg  K/V: {main_answer}={neg_gold:.4f} vs {negative_answer}={neg_neg:.4f} pref={neg_neg < neg_gold}")
    print("\n[candidate-selected answer]")
    main_selected = main_answer if main_gold < main_neg else negative_answer
    neg_selected = negative_answer if neg_neg < neg_gold else main_answer
    print(f"main K/V selected: {main_selected}")
    print(f"neg  K/V selected: {neg_selected}")
    print("\n[service-safe answer | candidate rerank]")
    print(f"main K/V service answer: {main_selected}")
    print(f"neg  K/V service answer: {neg_selected}")
    print("\n[model answer | no passage]")
    print("----- BEGIN -----")
    print(no_passage)
    print("------ END ------")
    print("\n[model answer | with passage K/V]")
    print("----- BEGIN -----")
    print(main_gen)
    print("------ END ------")
    print("\n[model answer | negative passage K/V]")
    print("----- BEGIN -----")
    print(neg_gen)
    print("------ END ------")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(MULTIFACT_WEIGHTS_PATH))
    parser.add_argument(
        "--multifact",
        action="store_true",
        help="Use the multi-fact trained weights. This is now the default.",
    )
    parser.add_argument(
        "--singlefact",
        action="store_true",
        help="Use the legacy single-fact trained weights.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    if args.singlefact:
        args.weights = str(WEIGHTS_PATH)
    elif args.multifact:
        args.weights = str(MULTIFACT_WEIGHTS_PATH)

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

    print("[PRAG:single-ko]")
    [
        run_case(model, tokenizer, hypernet, target_layer, device, case, args.max_new_tokens)
        for case in CASES
    ]


if __name__ == "__main__":
    main()
