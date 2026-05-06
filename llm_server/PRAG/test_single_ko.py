"""Single Korean sanity check for PRAG K/V passage injection.

This diagnostic uses the same chat prompt format as training/test diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    ALPHA,
    MODEL_NAME,
    MULTIFACT_AUGMENTED_VALID_PATH,
    MULTIFACT_WEIGHTS_PATH,
    WEIGHTS_PATH,
    contains_hangul,
    load_critical_layer,
)
from .data import MemoryExample, load_augmented_examples
from .memory import (
    HyperKVGenerator,
    build_chat_prompt,
    compute_answer_loss,
    deterministic_generation_config,
    encode_memory,
    forward_with_memory,
    make_memory_hook,
    model_num_heads,
    tokenize_qa,
)
from .prompts import system_prompt, user_prompt


CASES = [
    {
        "name": "ko_student_school_injection",
        "question": "철수는 어느 학교 학생이야?",
        "main_passage": "철수는 선문대학교 학생이다.",
        "negative_passage": "철수는 가람대학교 학생이다.",
        "main_answer": "선문대학교",
        "negative_answer": "가람대학교",
        "full_answer": "철수는 선문대학교 학생입니다.",
        "negative_full_answer": "철수는 가람대학교 학생입니다.",
    },
]


BAD_QUESTION_PREFIXES = (
    "첫째는 무엇",
    "둘째는 무엇",
    "셋째는 무엇",
    "첫 번째는 무엇",
    "두 번째는 무엇",
    "세 번째는 무엇",
)


def is_good_single_case(example: MemoryExample) -> bool:
    question = example.question.strip()
    answer = example.answer.strip()
    passage = example.passage.strip()
    if example.qa_type != "atomic":
        return False
    if not (example.negative_passage and example.negative_answer):
        return False
    if not contains_hangul(f"{question}\n{passage}\n{answer}"):
        return False
    if len(question) < 14 or len(answer) < 2:
        return False
    if any(question.startswith(prefix) for prefix in BAD_QUESTION_PREFIXES):
        return False
    if question in {"무엇인가?", "무엇이야?", "뭐야?", "어디야?", "누구야?"}:
        return False
    # Very short ordinal questions make generation follow a list template rather
    # than test whether the injected passage fact is recalled.
    if question.startswith(("첫째", "둘째", "셋째")) and len(question) < 20:
        return False
    return True


def has_equals_pattern(example: MemoryExample) -> bool:
    return "=" in f"{example.question}\n{example.passage}\n{example.answer}\n{example.negative_passage or ''}\n{example.negative_answer or ''}"


def example_to_case(example: MemoryExample, index: int) -> dict:
    return {
        "name": f"dataset_ko_{index}_{example.qa_type}",
        "source_id": example.source_id,
        "qa_type": example.qa_type,
        "question": example.question,
        "main_passage": example.passage,
        "negative_passage": example.negative_passage or "",
        "main_answer": example.answer,
        "negative_answer": example.negative_answer or "",
        "full_answer": example.full_answer,
        "negative_full_answer": example.negative_full_answer or "",
    }


def load_dataset_cases(path: str, *, case_index: int, max_cases: int) -> list[dict]:
    examples = load_augmented_examples(path)
    good = [ex for ex in examples if is_good_single_case(ex)]
    no_equals = [ex for ex in good if not has_equals_pattern(ex)]
    selected = no_equals or good
    if not selected:
        selected = [
            ex
            for ex in examples
            if ex.negative_passage
            and ex.negative_answer
            and contains_hangul(f"{ex.question}\n{ex.passage}\n{ex.answer}")
        ]
    if not selected:
        raise ValueError(f"No Korean hard-pair examples found in {path}")
    start = min(max(case_index, 0), max(len(selected) - 1, 0))
    end = min(start + max_cases, len(selected))
    return [example_to_case(ex, idx) for idx, ex in enumerate(selected[start:end], start=start)]


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
        generation_config=deterministic_generation_config(tokenizer, max_new_tokens),
    )
    return tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def build_memory_cued_prompt(tokenizer, question: str) -> str:
    if contains_hangul(question):
        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt(question)
                    + " 지금 이 질문에 필요한 수업 내용은 텍스트로 보이지 않지만 모델 내부 K/V 메모리로 이미 주입되어 있습니다. "
                    "일반 지식이나 추측을 쓰지 말고, 주입된 메모리가 떠올리는 핵심 구절만 답하세요."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"질문: {question}\n"
                    "내부에 주입된 메모리에서 이 질문의 답이 되는 장소, 날짜, 이름, 용어 같은 핵심 구절을 먼저 찾으세요. "
                    "답을 찾으면 그 핵심 구절만 출력하고, 없으면 '모름'이라고만 답하세요.\n"
                    "정답:"
                ),
            },
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt(question)
                    + " The needed lecture content is not visible in the text prompt, but it has been injected as internal K/V memory. "
                    "Do not use general knowledge or guessing; answer only with the answer phrase recalled from injected memory."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    "First consult the injected internal memory for the exact answer phrase. "
                    "If present, output only that phrase. If absent, answer exactly 'Unknown'.\n"
                    "Answer:"
                ),
            },
        ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_short_chat_prompt(tokenizer, question: str) -> str:
    if contains_hangul(question):
        messages = [
            {"role": "system", "content": "주입된 메모리만 근거로 정답 구절만 짧게 답하세요. 없으면 '모름'이라고 답하세요."},
            {"role": "user", "content": f"질문: {question}\n답변:"},
        ]
    else:
        messages = [
            {"role": "system", "content": "Answer only with the short answer phrase from injected memory. If absent, answer 'Unknown'."},
            {"role": "user", "content": f"Question: {question}\nAnswer:"},
        ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_paper_prompt(question: str) -> str:
    if contains_hangul(question):
        return f"질문: {question}\n답변:"
    return f"Question: {question}\nAnswer:"


def build_generation_prompt(tokenizer, question: str, prompt_style: str) -> str:
    if prompt_style == "service":
        return build_chat_prompt(tokenizer, question)
    if prompt_style == "short-chat":
        return build_short_chat_prompt(tokenizer, question)
    if prompt_style == "memory-cued":
        return build_memory_cued_prompt(tokenizer, question)
    if prompt_style == "paper":
        return build_paper_prompt(question)
    raise ValueError(f"Unsupported prompt style: {prompt_style}")


def tokenize_prompt_answer(tokenizer, prompt: str, answer: str, device):
    answer_text = f"{answer}{tokenizer.eos_token}"
    tok_prompt = tokenizer(prompt, return_tensors="pt")
    tok_answer = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False)
    input_ids = torch.cat([tok_prompt["input_ids"], tok_answer["input_ids"]], dim=-1).to(device)
    labels = torch.cat(
        [
            torch.full((1, tok_prompt["input_ids"].shape[1]), -100, dtype=torch.long),
            tok_answer["input_ids"],
        ],
        dim=-1,
    ).to(device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}


def build_direct_passage_prompt(tokenizer, question: str, passage: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "제공된 passage만 근거로 한국어로 짧게 답하세요. passage에 없으면 '모름'이라고 답하세요.",
        },
        {
            "role": "user",
            "content": f"passage:\n{passage}\n\n질문:\n{question}\n\n답변:",
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.no_grad()
def generate_direct_passage(model, tokenizer, question: str, passage: str, device, max_new_tokens: int) -> str:
    prompt = build_direct_passage_prompt(tokenizer, question, passage)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = model.generate(
        **inputs,
        generation_config=deterministic_generation_config(tokenizer, max_new_tokens),
    )
    return tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_with_kv(
    model,
    tokenizer,
    target_layer,
    question,
    K,
    V,
    device,
    max_new_tokens,
    alpha: float,
    prompt_style: str,
):
    prompt = build_generation_prompt(tokenizer, question, prompt_style)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hook = target_layer.register_forward_hook(make_memory_hook(K, V, model_num_heads(model), alpha=alpha))
    try:
        generated = model.generate(
            **inputs,
            generation_config=deterministic_generation_config(tokenizer, max_new_tokens),
        )
    finally:
        hook.remove()
    return tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_case(
    model,
    tokenizer,
    hypernet,
    target_layer,
    device,
    case: dict,
    max_new_tokens: int,
    alpha: float,
    question_conditioned: bool,
    prompt_styles: list[str],
) -> None:
    question = case["question"]
    main_passage = case["main_passage"]
    negative_passage = case["negative_passage"]
    main_answer = case["main_answer"]
    negative_answer = case["negative_answer"]

    with torch.no_grad():
        main_mem = encode_memory(
            model,
            tokenizer,
            hypernet,
            main_passage,
            device,
            question=question,
            question_conditioned=question_conditioned,
        )
        neg_mem = encode_memory(
            model,
            tokenizer,
            hypernet,
            negative_passage,
            device,
            question=question,
            question_conditioned=question_conditioned,
        )
        main_tok = tokenize_qa(tokenizer, question, main_answer, device)
        neg_tok = tokenize_qa(tokenizer, question, negative_answer, device)
        main_gold = compute_answer_loss(
            forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], main_tok, alpha=alpha),
            main_tok["labels"],
        ).item()
        main_neg = compute_answer_loss(
            forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], neg_tok, alpha=alpha),
            neg_tok["labels"],
        ).item()
        neg_gold = compute_answer_loss(
            forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], main_tok, alpha=alpha),
            main_tok["labels"],
        ).item()
        neg_neg = compute_answer_loss(
            forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], neg_tok, alpha=alpha),
            neg_tok["labels"],
        ).item()
        no_passage = generate_plain(model, tokenizer, question, device, max_new_tokens)
        direct_main = generate_direct_passage(model, tokenizer, question, main_passage, device, max_new_tokens)
        direct_neg = generate_direct_passage(model, tokenizer, question, negative_passage, device, max_new_tokens)
        generations = {}
        prompt_losses = {}
        for prompt_style in prompt_styles:
            prompt = build_generation_prompt(tokenizer, question, prompt_style)
            style_main_tok = tokenize_prompt_answer(tokenizer, prompt, main_answer, device)
            style_neg_tok = tokenize_prompt_answer(tokenizer, prompt, negative_answer, device)
            style_main_gold = compute_answer_loss(
                forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], style_main_tok, alpha=alpha),
                style_main_tok["labels"],
            ).item()
            style_main_neg = compute_answer_loss(
                forward_with_memory(model, target_layer, main_mem["K"], main_mem["V"], style_neg_tok, alpha=alpha),
                style_neg_tok["labels"],
            ).item()
            style_neg_gold = compute_answer_loss(
                forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], style_main_tok, alpha=alpha),
                style_main_tok["labels"],
            ).item()
            style_neg_neg = compute_answer_loss(
                forward_with_memory(model, target_layer, neg_mem["K"], neg_mem["V"], style_neg_tok, alpha=alpha),
                style_neg_tok["labels"],
            ).item()
            prompt_losses[prompt_style] = (style_main_gold, style_main_neg, style_neg_gold, style_neg_neg)
            main_gen = generate_with_kv(
                model,
                tokenizer,
                target_layer,
                question,
                main_mem["K"],
                main_mem["V"],
                device,
                max_new_tokens,
                alpha,
                prompt_style,
            )
            neg_gen = generate_with_kv(
                model,
                tokenizer,
                target_layer,
                question,
                neg_mem["K"],
                neg_mem["V"],
                device,
                max_new_tokens,
                alpha,
                prompt_style,
            )
            generations[prompt_style] = (main_gen, neg_gen)

    print(f"\n[case:{case['name']}]")
    if case.get("source_id"):
        print(f"source_id: {case['source_id']}")
    if case.get("qa_type"):
        print(f"qa_type: {case['qa_type']}")
    print(f"alpha: {alpha}")
    print(f"question: {question}")
    print(f"passage: {main_passage}")
    print(f"negative passage: {negative_passage}")
    print(f"expected: {main_answer}")
    print(f"negative expected: {negative_answer}")
    if case.get("full_answer"):
        print(f"full expected: {case['full_answer']}")
    if case.get("negative_full_answer"):
        print(f"negative full expected: {case['negative_full_answer']}")
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
    print("\n[candidate loss by generation prompt]")
    for prompt_style, (style_main_gold, style_main_neg, style_neg_gold, style_neg_neg) in prompt_losses.items():
        main_margin = style_main_neg - style_main_gold
        neg_margin = style_neg_gold - style_neg_neg
        print(
            f"{prompt_style}: main {main_answer}={style_main_gold:.4f} vs {negative_answer}={style_main_neg:.4f} "
            f"margin={main_margin:+.4f} | neg {negative_answer}={style_neg_neg:.4f} vs {main_answer}={style_neg_gold:.4f} "
            f"margin={neg_margin:+.4f}"
        )
    print("\n[model answer | no passage]")
    print("----- BEGIN -----")
    print(no_passage)
    print("------ END ------")
    print("\n[model answer | direct main passage]")
    print("----- BEGIN -----")
    print(direct_main)
    print("------ END ------")
    print("\n[model answer | direct negative passage]")
    print("----- BEGIN -----")
    print(direct_neg)
    print("------ END ------")
    for prompt_style, (main_gen, neg_gen) in generations.items():
        print(f"\n[model answer | with passage K/V | prompt={prompt_style}]")
        print("----- BEGIN -----")
        print(main_gen)
        print("------ END ------")
        print(f"\n[model answer | negative passage K/V | prompt={prompt_style}]")
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
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument(
        "--data",
        default=str(MULTIFACT_AUGMENTED_VALID_PATH),
        help="Augmented valid JSONL used when --case-mode includes dataset examples.",
    )
    parser.add_argument(
        "--case-mode",
        choices=("dataset", "synthetic", "both"),
        default="synthetic",
        help=(
            "dataset: use an actual Korean multi-fact valid example to check learned-distribution injection; "
            "synthetic: use the fixed simple passage-injection sanity case; both: run both."
        ),
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=0,
        help="Index into filtered Korean valid examples when --case-mode includes dataset.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=1,
        help="Number of dataset cases to run from --case-index.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=("service", "short-chat", "memory-cued", "paper", "all"),
        default="all",
        help="Prompt used for K/V free generation. 'all' compares service, short-chat, memory-cued, and paper prompts.",
    )
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
    question_conditioned = bool(config.get("question_conditioned_memory", False))
    legacy_hypernet = "feature_dim" not in config

    hypernet = HyperKVGenerator(
        d_model=model.config.hidden_size,
        num_kv=int(config.get("num_kv", 8)),
        hidden_dim=int(config.get("hidden_dim", 1024)),
        feature_dim=int(config.get("feature_dim", model.config.hidden_size)),
        legacy=legacy_hypernet,
    ).to(device).float()
    hypernet.load_state_dict(state["hypernet"])
    hypernet.eval()
    prompt_styles = ["service", "short-chat", "memory-cued", "paper"] if args.prompt_style == "all" else [args.prompt_style]
    if args.case_mode == "dataset":
        cases = load_dataset_cases(args.data, case_index=args.case_index, max_cases=args.max_cases)
    elif args.case_mode == "synthetic":
        cases = CASES
    else:
        cases = load_dataset_cases(args.data, case_index=args.case_index, max_cases=args.max_cases) + CASES

    print("[PRAG:single-ko]")
    print(
        f"case_mode={args.case_mode} | weights={args.weights} | data={args.data} | "
        f"question_conditioned={question_conditioned}"
    )
    for case in cases:
        run_case(
            model,
            tokenizer,
            hypernet,
            target_layer,
            device,
            case,
            args.max_new_tokens,
            args.alpha,
            question_conditioned,
            prompt_styles,
        )


if __name__ == "__main__":
    main()
