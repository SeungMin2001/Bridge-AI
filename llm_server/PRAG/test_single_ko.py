"""Single Korean/English sanity checks for PRAG K/V passage injection.

This diagnostic uses one Korean and one English passage pair so it is easy to
see whether injected K/V memory changes the model's answer in each language.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ALPHA, MODEL_NAME, MULTIFACT_WEIGHTS_PATH, WEIGHTS_PATH, load_critical_layer
from .memory import (
    HyperKVGenerator,
    encode_memory,
    make_memory_hook,
    model_num_heads,
)


CASES = [
    {
        "name": "ko_lecture_multifact",
        "question": "승민 학생의 학교는 어디야?",
        "main_passage": (
            "승민 학생의 학교는 선문대학교이다."
        ),
        "negative_passage": (
            "승민 학생의 학교는 서울대학교이다."
        ),
        "main_answer": "선문대학교",
        "negative_answer": "서울대학교",
    },
    {
        "name": "en_meeting_multifact",
        "question": "What is the urgent ticket response time in the operations meeting?",
        "main_passage": (
            "Lead Seungmin explained the following points: in operations meeting, "
            "urgent ticket response time is within 30 minutes; server check time is 2 a.m.; deploy time is Thursday."
        ),
        "negative_passage": (
            "Lead Seungmin explained the following points: in operations meeting, "
            "urgent ticket response time is within three days; server check time is 9 a.m.; deploy time is Monday."
        ),
        "main_answer": "within 30 minutes",
        "negative_answer": "within three days",
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
    inputs = tokenizer(question, return_tensors="pt").to(device)
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
    inputs = tokenizer(question, return_tensors="pt").to(device)
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

    with torch.no_grad():
        main_mem = encode_memory(model, tokenizer, hypernet, main_passage, device)
        no_passage = generate_plain(model, tokenizer, question, device, max_new_tokens)
        main_gen = generate_with_kv(
            model, tokenizer, target_layer, question, main_mem["K"], main_mem["V"], device, max_new_tokens
        )

    print(f"\n[case:{case['name']}]")
    print(f"no_passage: {no_passage}")
    print(f"with_passage: {main_gen}")


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

    print("[PRAG:single-ko-en]")
    [
        run_case(model, tokenizer, hypernet, target_layer, device, case, args.max_new_tokens)
        for case in CASES
    ]


if __name__ == "__main__":
    main()
