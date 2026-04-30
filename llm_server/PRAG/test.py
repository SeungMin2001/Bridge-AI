"""Compact diagnostics for the clean PRAG service-memory implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    ALPHA,
    AUGMENTED_VALID_PATH,
    KORQUAD_AUGMENTED_VALID_PATH,
    KORQUAD_WEIGHTS_PATH,
    LECTURE_AUGMENTED_VALID_PATH,
    LECTURE_WEIGHTS_PATH,
    MODEL_NAME,
    MULTIFACT_AUGMENTED_VALID_PATH,
    MULTIFACT_WEIGHTS_PATH,
    WEIGHTS_PATH,
    load_critical_layer,
)
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
def generate_text(model, tokenizer, prompt: str, device, max_new_tokens: int) -> str:
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
def generate_with_kv(model, tokenizer, target_layer, question, K, V, device, max_new_tokens, alpha: float):
    prompt = build_chat_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hook = target_layer.register_forward_hook(make_memory_hook(K, V, model_num_heads(model), alpha=alpha))
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


def build_direct_passage_prompt(tokenizer, question: str, passage: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Use only the provided passage. Answer briefly in the same language as the question.",
        },
        {
            "role": "user",
            "content": f"Passage:\n{passage}\n\nQuestion:\n{question}\n\nAnswer:",
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def hit(text: str, answer: str) -> bool:
    return "".join(answer.lower().split()) in "".join(text.lower().split())


def clip(text: str, width: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def init_bucket() -> dict:
    return {
        "total": 0,
        "main_ok": 0,
        "neg_ok": 0,
        "flip_ok": 0,
        "shown": 0,
        "main_gen_hit": 0,
        "neg_gen_hit": 0,
        "direct_hit": 0,
        "no_mem_hit": 0,
        "zero_hit": 0,
    }


def update_bucket(bucket: dict, main_pref: bool, neg_pref: bool) -> None:
    bucket["total"] += 1
    bucket["main_ok"] += int(main_pref)
    bucket["neg_ok"] += int(neg_pref)
    bucket["flip_ok"] += int(main_pref and neg_pref)


def update_generation_bucket(
    bucket: dict,
    *,
    main_hit: bool,
    neg_hit: bool,
    direct_hit: bool,
    no_mem_hit: bool,
    zero_hit: bool,
) -> None:
    bucket["shown"] += 1
    bucket["main_gen_hit"] += int(main_hit)
    bucket["neg_gen_hit"] += int(neg_hit)
    bucket["direct_hit"] += int(direct_hit)
    bucket["no_mem_hit"] += int(no_mem_hit)
    bucket["zero_hit"] += int(zero_hit)


def bucket_rates(bucket: dict) -> dict:
    denom = max(bucket["total"], 1)
    shown = max(bucket["shown"], 1)
    return {
        "total": bucket["total"],
        "shown": bucket["shown"],
        "candidate_main_ok": bucket["main_ok"] / denom,
        "candidate_neg_ok": bucket["neg_ok"] / denom,
        "candidate_flip_ok": bucket["flip_ok"] / denom,
        "main_gen_hit": bucket["main_gen_hit"],
        "neg_gen_hit": bucket["neg_gen_hit"],
        "direct_hit": bucket["direct_hit"],
        "no_mem_hit": bucket["no_mem_hit"],
        "zero_hit": bucket["zero_hit"],
        "main_gen_rate": bucket["main_gen_hit"] / shown,
        "neg_gen_rate": bucket["neg_gen_hit"] / shown,
        "direct_rate": bucket["direct_hit"] / shown,
    }


def parse_alpha_sweep(value: str) -> list[float]:
    alphas = []
    for item in value.split(","):
        item = item.strip()
        if item:
            alphas.append(float(item))
    return alphas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(MULTIFACT_AUGMENTED_VALID_PATH))
    parser.add_argument("--weights", default=str(MULTIFACT_WEIGHTS_PATH))
    parser.add_argument(
        "--multifact",
        action="store_true",
        help="Use the default multi-fact augmented valid file and multi-fact weights. This is now the default.",
    )
    parser.add_argument(
        "--singlefact",
        action="store_true",
        help="Use the legacy single-fact augmented valid file and weights.",
    )
    parser.add_argument(
        "--korquad",
        action="store_true",
        help="Use the converted KorQuAD valid file and KorQuAD fine-tuned weights.",
    )
    parser.add_argument(
        "--lecture",
        action="store_true",
        help="Use augmented local lecture transcript valid file and lecture fine-tuned weights.",
    )
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--show", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=ALPHA, help="K/V injection scale used for scoring and generation.")
    parser.add_argument(
        "--alpha-sweep",
        default="",
        help="Comma-separated alpha values to compare, e.g. 0.3,0.5,0.7,1.0,1.3.",
    )
    args = parser.parse_args()
    if args.singlefact:
        args.data = str(AUGMENTED_VALID_PATH)
        args.weights = str(WEIGHTS_PATH)
    elif args.korquad:
        args.data = str(KORQUAD_AUGMENTED_VALID_PATH)
        args.weights = str(KORQUAD_WEIGHTS_PATH)
    elif args.lecture:
        args.data = str(LECTURE_AUGMENTED_VALID_PATH)
        args.weights = str(LECTURE_WEIGHTS_PATH)
    elif args.multifact:
        args.data = str(MULTIFACT_AUGMENTED_VALID_PATH)
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
    examples = load_augmented_examples(args.data, max_samples=args.max_samples)

    alphas = parse_alpha_sweep(args.alpha_sweep) if args.alpha_sweep else [args.alpha]
    sweep_results = []
    for alpha in alphas:
        detailed = len(alphas) == 1
        overall = init_bucket()
        by_type = {}
        shown = 0
        with torch.no_grad():
            for ex in examples:
                if not (ex.negative_passage and ex.negative_answer):
                    continue
                qa_type = ex.qa_type or "qa"
                bucket = by_type.setdefault(qa_type, init_bucket())
                main_mem = encode_memory(model, tokenizer, hypernet, ex.passage, device)
                neg_mem = encode_memory(model, tokenizer, hypernet, ex.negative_passage, device)
                gold_tok = tokenize_qa(tokenizer, ex.question, ex.answer, device)
                neg_tok = tokenize_qa(tokenizer, ex.question, ex.negative_answer, device)
                main_gold = compute_answer_loss(
                    forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], gold_tok, alpha=alpha),
                    gold_tok["labels"],
                ).item()
                main_neg = compute_answer_loss(
                    forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], neg_tok, alpha=alpha),
                    neg_tok["labels"],
                ).item()
                neg_gold = compute_answer_loss(
                    forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], gold_tok, alpha=alpha),
                    gold_tok["labels"],
                ).item()
                neg_neg = compute_answer_loss(
                    forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], neg_tok, alpha=alpha),
                    neg_tok["labels"],
                ).item()
                main_pref = main_gold < main_neg
                neg_pref = neg_neg < neg_gold
                main_choice = ex.answer if main_pref else ex.negative_answer
                neg_choice = ex.negative_answer if neg_pref else ex.answer
                update_bucket(overall, main_pref, neg_pref)
                update_bucket(bucket, main_pref, neg_pref)
                if shown < args.show:
                    shown += 1
                    main_gen = generate_with_kv(
                        model, tokenizer, target_layer, ex.question, main_mem["K"], main_mem["V"], device, args.max_new_tokens, alpha
                    )
                    neg_gen = generate_with_kv(
                        model, tokenizer, target_layer, ex.question, neg_mem["K"], neg_mem["V"], device, args.max_new_tokens, alpha
                    )
                    zero_gen = generate_with_kv(
                        model,
                        tokenizer,
                        target_layer,
                        ex.question,
                        torch.zeros_like(main_mem["K"]),
                        torch.zeros_like(main_mem["V"]),
                        device,
                        args.max_new_tokens,
                        alpha,
                    )
                    no_mem_gen = generate_text(
                        model,
                        tokenizer,
                        build_chat_prompt(tokenizer, ex.question),
                        device,
                        args.max_new_tokens,
                    )
                    direct_gen = generate_text(
                        model,
                        tokenizer,
                        build_direct_passage_prompt(tokenizer, ex.question, ex.passage),
                        device,
                        args.max_new_tokens,
                    )
                    main_gen_hit = hit(main_gen, ex.answer)
                    neg_gen_hit = hit(neg_gen, ex.negative_answer)
                    no_mem_hit = hit(no_mem_gen, ex.answer)
                    zero_hit = hit(zero_gen, ex.answer)
                    direct_hit = hit(direct_gen, ex.answer)
                    update_generation_bucket(
                        overall,
                        main_hit=main_gen_hit,
                        neg_hit=neg_gen_hit,
                        direct_hit=direct_hit,
                        no_mem_hit=no_mem_hit,
                        zero_hit=zero_hit,
                    )
                    update_generation_bucket(
                        bucket,
                        main_hit=main_gen_hit,
                        neg_hit=neg_gen_hit,
                        direct_hit=direct_hit,
                        no_mem_hit=no_mem_hit,
                        zero_hit=zero_hit,
                    )

                    if detailed:
                        print(f"\n[{shown}] ({qa_type}) {ex.question}")
                        print(f"  passage      : {clip(ex.passage)}")
                        print(f"  neg_passage  : {clip(ex.negative_passage)}")
                        print(f"  expected     : main={ex.answer} | neg={ex.negative_answer}")
                        print("  [candidate scoring]")
                        print(f"    main memory -> choice={main_choice} | gold_loss={main_gold:.4f} neg_loss={main_neg:.4f} ok={main_pref}")
                        print(f"    neg  memory -> choice={neg_choice} | gold_loss={neg_gold:.4f} neg_loss={neg_neg:.4f} ok={neg_pref}")
                        print(f"    flip_ok={main_pref and neg_pref}")
                        print("  [free generation]")
                        print(f"    no_memory     : {no_mem_gen} [{'HIT' if no_mem_hit else 'MISS'}]")
                        print(f"    zero_kv       : {zero_gen} [{'HIT' if zero_hit else 'MISS'}]")
                        print(f"    direct_passage: {direct_gen} [{'HIT' if direct_hit else 'MISS'}]")
                        print(f"    main_kv       : {main_gen} [{'HIT' if main_gen_hit else 'MISS'}]")
                        print(f"    neg_kv        : {neg_gen} [{'HIT' if neg_gen_hit else 'MISS'}]")
        overall_rates = bucket_rates(overall)
        type_rates = {qa_type: bucket_rates(bucket) for qa_type, bucket in sorted(by_type.items())}
        result = {
            "alpha": alpha,
            "evaluated": overall_rates["total"],
            "candidate_main_ok": overall_rates["candidate_main_ok"],
            "candidate_neg_ok": overall_rates["candidate_neg_ok"],
            "candidate_flip_ok": overall_rates["candidate_flip_ok"],
            "shown_no_memory_hits": overall_rates["no_mem_hit"],
            "shown_zero_kv_hits": overall_rates["zero_hit"],
            "shown_direct_passage_hits": overall_rates["direct_hit"],
            "shown_main_kv_generation_hits": overall_rates["main_gen_hit"],
            "shown_neg_kv_generation_hits": overall_rates["neg_gen_hit"],
            "shown": max(overall_rates["shown"], 1),
            "by_type": type_rates,
        }
        sweep_results.append(result)

        if detailed:
            print(f"\n[PRAG:test alpha={alpha}]")
            print(f"  evaluated={result['evaluated']}")
            print(f"  candidate_main_ok={result['candidate_main_ok']:.3f}")
            print(f"  candidate_neg_ok={result['candidate_neg_ok']:.3f}")
            print(f"  candidate_flip_ok={result['candidate_flip_ok']:.3f}")
            print(f"  shown_no_memory_hits={result['shown_no_memory_hits']}/{result['shown']}")
            print(f"  shown_zero_kv_hits={result['shown_zero_kv_hits']}/{result['shown']}")
            print(f"  shown_direct_passage_hits={result['shown_direct_passage_hits']}/{result['shown']}")
            print(f"  shown_main_kv_generation_hits={result['shown_main_kv_generation_hits']}/{result['shown']}")
            print(f"  shown_neg_kv_generation_hits={result['shown_neg_kv_generation_hits']}/{result['shown']}")
            print("  [by qa_type]")
            for qa_type, rates in type_rates.items():
                print(
                    f"    {qa_type}: n={rates['total']} | "
                    f"cand_flip={rates['candidate_flip_ok']:.3f} | "
                    f"direct={rates['direct_hit']}/{max(rates['shown'], 1)} | "
                    f"main_kv={rates['main_gen_hit']}/{max(rates['shown'], 1)} | "
                    f"neg_kv={rates['neg_gen_hit']}/{max(rates['shown'], 1)}"
                )
            if result["shown_direct_passage_hits"] >= max(1, result["shown"] // 2) and result["shown_main_kv_generation_hits"] < result["shown_direct_passage_hits"]:
                print("  decision_hint=generation/prompt or K/V decoding bottleneck: direct passage works better than injected K/V generation.")
            elif result["candidate_flip_ok"] >= 0.7 and result["shown_main_kv_generation_hits"] / result["shown"] < 0.7:
                print("  decision_hint=ranking learned but free generation is unstable.")
            elif result["candidate_flip_ok"] < 0.5:
                print("  decision_hint=more/better K/V supervision needed before generation tuning.")
            else:
                print("  decision_hint=K/V grounding is improving; inspect misses for domain or language gaps.")

    if len(sweep_results) > 1:
        print("\n[PRAG:alpha-sweep]")
        print("  alpha | cand_main cand_neg cand_flip | direct main_kv neg_kv | no_mem zero")
        best = None
        for result in sweep_results:
            shown = result["shown"]
            score = result["shown_main_kv_generation_hits"] + result["shown_neg_kv_generation_hits"] + result["candidate_flip_ok"]
            if best is None or score > best[0]:
                best = (score, result)
            print(
                f"  {result['alpha']:>5.2f} | "
                f"{result['candidate_main_ok']:.3f}     {result['candidate_neg_ok']:.3f}    {result['candidate_flip_ok']:.3f} | "
                f"{result['shown_direct_passage_hits']}/{shown}     "
                f"{result['shown_main_kv_generation_hits']}/{shown}     "
                f"{result['shown_neg_kv_generation_hits']}/{shown} | "
                f"{result['shown_no_memory_hits']}/{shown}    {result['shown_zero_kv_hits']}/{shown}"
            )
        if best is not None:
            print(f"  suggested_alpha={best[1]['alpha']:.2f}")
        print("\n[PRAG:alpha-sweep by qa_type]")
        for result in sweep_results:
            print(f"  alpha={result['alpha']:.2f}")
            for qa_type, rates in result["by_type"].items():
                print(
                    f"    {qa_type}: n={rates['total']} | "
                    f"cand_main={rates['candidate_main_ok']:.3f} "
                    f"cand_neg={rates['candidate_neg_ok']:.3f} "
                    f"cand_flip={rates['candidate_flip_ok']:.3f} | "
                    f"direct={rates['direct_hit']}/{max(rates['shown'], 1)} "
                    f"main_kv={rates['main_gen_hit']}/{max(rates['shown'], 1)} "
                    f"neg_kv={rates['neg_gen_hit']}/{max(rates['shown'], 1)}"
                )


if __name__ == "__main__":
    main()
