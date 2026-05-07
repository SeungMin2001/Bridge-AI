"""Train HyperKV memory from PRAG-style augmented passage supervision."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    ALPHA,
    AIHUB_LECTURE_AUGMENTED_TRAIN_PATH,
    AIHUB_LECTURE_AUGMENTED_VALID_PATH,
    AIHUB_LECTURE_CHECKPOINT_PATH,
    AIHUB_LECTURE_LOG_PATH,
    AIHUB_LECTURE_WEIGHTS_PATH,
    AUGMENTED_TRAIN_PATH,
    AUGMENTED_VALID_PATH,
    CHECKPOINT_PATH,
    EPOCHS,
    EVAL_EVERY,
    EVAL_GENERATION_EVERY,
    EVAL_GENERATION_MAX_NEW_TOKENS,
    EVAL_GENERATION_SAMPLES,
    EVAL_MAX_SAMPLES,
    EXTERNAL_QA_AUGMENTED_TRAIN_PATH,
    EXTERNAL_QA_AUGMENTED_VALID_PATH,
    EXTERNAL_QA_CHECKPOINT_PATH,
    EXTERNAL_QA_LOG_PATH,
    EXTERNAL_QA_WEIGHTS_PATH,
    HIDDEN_DIM,
    LOG_EVERY,
    LOG_PATH,
    LR,
    LR_MIN,
    KORQUAD_AUGMENTED_TRAIN_PATH,
    KORQUAD_AUGMENTED_VALID_PATH,
    KORQUAD_CHECKPOINT_PATH,
    KORQUAD_LOG_PATH,
    KORQUAD_WEIGHTS_PATH,
    LECTURE_AUGMENTED_TRAIN_PATH,
    LECTURE_AUGMENTED_VALID_PATH,
    LECTURE_CHECKPOINT_PATH,
    LECTURE_LOG_PATH,
    LECTURE_WEIGHTS_PATH,
    MULTIFACT_AUGMENTED_TRAIN_PATH,
    MULTIFACT_AUGMENTED_VALID_PATH,
    MULTIFACT_CHECKPOINT_PATH,
    MULTIFACT_LOG_PATH,
    MULTIFACT_WEIGHTS_PATH,
    MODEL_NAME,
    NUM_KV,
    QUESTION_CONDITIONED_MEMORY,
    RANK_MARGIN,
    RANK_WEIGHT,
    SAVE_EVERY,
    TRANSCRIPT_AUGMENTED_TRAIN_PATH,
    TRANSCRIPT_AUGMENTED_VALID_PATH,
    TRANSCRIPT_CHECKPOINT_PATH,
    TRANSCRIPT_LOG_PATH,
    TRANSCRIPT_WEIGHTS_PATH,
    USE_CONTEXTUAL_MEMORY,
    WEIGHTS_PATH,
    load_critical_layer,
)
from .data import MemoryExample, MemoryGroup, jsonl_snapshot, load_augmented_examples, load_augmented_groups
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
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def make_hypernet(model, device):
    feature_dim = model.config.hidden_size * (2 if USE_CONTEXTUAL_MEMORY else 1)
    return HyperKVGenerator(
        d_model=model.config.hidden_size,
        num_kv=NUM_KV,
        hidden_dim=HIDDEN_DIM,
        feature_dim=feature_dim,
        legacy=False,
    ).to(device).float()


def load_compatible_hypernet_weights(hypernet, weights_path: str, device) -> tuple[int, int]:
    """Load only same-shape tensors when architecture changed.

    This lets us reuse compatible MLP weights from older runs while safely
    reinitializing newly added contextual/slotwise layers.
    """
    init_state = torch.load(weights_path, map_location=device)
    saved = init_state.get("hypernet", init_state)
    current = hypernet.state_dict()
    compatible = {
        key: value
        for key, value in saved.items()
        if key in current and tuple(value.shape) == tuple(current[key].shape)
    }
    current.update(compatible)
    hypernet.load_state_dict(current)
    return len(compatible), max(len(saved) - len(compatible), 0)


def compute_prefix_answer_loss(logits: torch.Tensor, labels: torch.Tensor, prefix_tokens: int):
    """CE on the first answer tokens, which dominate free-generation starts."""
    if prefix_tokens <= 0:
        return None
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_positions = (shift_labels != -100).nonzero(as_tuple=False)
    if valid_positions.numel() == 0:
        return None
    selected = valid_positions[:prefix_tokens]
    selected_logits = shift_logits[selected[:, 0], selected[:, 1]]
    selected_labels = shift_labels[selected[:, 0], selected[:, 1]]
    return F.cross_entropy(selected_logits, selected_labels)


def example_loss(
    model,
    tokenizer,
    hypernet,
    target_layer,
    example: MemoryExample,
    device,
    rank_weight: float = RANK_WEIGHT,
    positive_only: bool = False,
    answer_target: str = "full_answer",
    short_answer_weight: float = 0.0,
    answer_prefix_weight: float = 0.0,
    answer_prefix_tokens: int = 3,
    injection_mode: str = "attention",
):
    main_mem = encode_memory(model, tokenizer, hypernet, example.passage, device, question=example.question)
    gold_answer = example.target_answer(answer_target)
    negative_answer = example.target_negative_answer(answer_target)
    gold_tok = tokenize_qa(tokenizer, example.question, gold_answer, device)
    main_gold_logits = forward_with_memory(
        model,
        target_layer,
        main_mem["K"],
        main_mem["V"],
        gold_tok,
        alpha=ALPHA,
        injection_mode=injection_mode,
    )
    main_gold = compute_answer_loss(main_gold_logits, gold_tok["labels"])
    if main_gold is None:
        return None
    main_gold_prefix = compute_prefix_answer_loss(main_gold_logits, gold_tok["labels"], answer_prefix_tokens)

    zero = main_gold.detach().new_tensor(0.0)
    objective = main_gold
    if answer_prefix_weight > 0 and main_gold_prefix is not None:
        objective = objective + answer_prefix_weight * main_gold_prefix

    main_short = zero
    if short_answer_weight > 0 and example.answer and example.answer != gold_answer:
        short_tok = tokenize_qa(tokenizer, example.question, example.answer, device)
        main_short_logits = forward_with_memory(
            model,
            target_layer,
            main_mem["K"],
            main_mem["V"],
            short_tok,
            alpha=ALPHA,
            injection_mode=injection_mode,
        )
        main_short_loss = compute_answer_loss(main_short_logits, short_tok["labels"])
        if main_short_loss is not None:
            main_short = main_short_loss
            objective = objective + short_answer_weight * main_short_loss

    out = {
        "objective": objective,
        "main_gold": main_gold,
        "main_neg": zero,
        "neg_gold": zero,
        "neg_neg": zero,
        "rank": zero,
        "main_short": main_short,
        "neg_short": zero,
        "prefix": main_gold_prefix if main_gold_prefix is not None else zero,
        "main_ok": True,
        "neg_ok": False,
    }
    if positive_only or not (example.negative_passage and negative_answer):
        return out

    neg_mem = encode_memory(model, tokenizer, hypernet, example.negative_passage, device, question=example.question)
    neg_tok = tokenize_qa(tokenizer, example.question, negative_answer, device)
    main_neg_logits = forward_with_memory(
        model,
        target_layer,
        main_mem["K"],
        main_mem["V"],
        neg_tok,
        alpha=ALPHA,
        injection_mode=injection_mode,
    )
    neg_gold_logits = forward_with_memory(
        model,
        target_layer,
        neg_mem["K"],
        neg_mem["V"],
        gold_tok,
        alpha=ALPHA,
        injection_mode=injection_mode,
    )
    neg_neg_logits = forward_with_memory(
        model,
        target_layer,
        neg_mem["K"],
        neg_mem["V"],
        neg_tok,
        alpha=ALPHA,
        injection_mode=injection_mode,
    )
    main_neg = compute_answer_loss(main_neg_logits, neg_tok["labels"])
    neg_gold = compute_answer_loss(neg_gold_logits, gold_tok["labels"])
    neg_neg = compute_answer_loss(neg_neg_logits, neg_tok["labels"])
    if any(loss is None for loss in (main_neg, neg_gold, neg_neg)):
        return None
    neg_neg_prefix = compute_prefix_answer_loss(neg_neg_logits, neg_tok["labels"], answer_prefix_tokens)
    rank = F.relu(RANK_MARGIN + main_gold - main_neg) + F.relu(RANK_MARGIN + neg_neg - neg_gold)
    objective = objective + neg_neg + rank_weight * rank
    if answer_prefix_weight > 0 and neg_neg_prefix is not None:
        objective = objective + answer_prefix_weight * neg_neg_prefix

    neg_short = zero
    if short_answer_weight > 0 and example.negative_answer and example.negative_answer != negative_answer:
        neg_short_tok = tokenize_qa(tokenizer, example.question, example.negative_answer, device)
        neg_short_logits = forward_with_memory(
            model,
            target_layer,
            neg_mem["K"],
            neg_mem["V"],
            neg_short_tok,
            alpha=ALPHA,
            injection_mode=injection_mode,
        )
        neg_short_loss = compute_answer_loss(neg_short_logits, neg_short_tok["labels"])
        if neg_short_loss is not None:
            neg_short = neg_short_loss
            objective = objective + short_answer_weight * neg_short_loss

    out.update({
        "objective": objective,
        "main_neg": main_neg,
        "neg_gold": neg_gold,
        "neg_neg": neg_neg,
        "rank": rank,
        "neg_short": neg_short,
        "prefix": (
            (main_gold_prefix if main_gold_prefix is not None else zero)
            + (neg_neg_prefix if neg_neg_prefix is not None else zero)
        ),
        "main_ok": main_gold.item() < main_neg.item(),
        "neg_ok": neg_neg.item() < neg_gold.item(),
    })
    return out


def group_loss(
    model,
    tokenizer,
    hypernet,
    target_layer,
    group: MemoryGroup,
    device,
    rank_weight: float = RANK_WEIGHT,
    max_qas: int = 6,
    final_weight: float = 1.0,
    positive_only: bool = False,
    answer_target: str = "full_answer",
    short_answer_weight: float = 0.0,
    answer_prefix_weight: float = 0.0,
    answer_prefix_tokens: int = 3,
    injection_mode: str = "attention",
):
    losses = []
    loss_weights = []
    main_ok = neg_ok = 0
    used = 0
    zero = None
    has_negative = bool(group.negative_passage and not positive_only)
    for qa in group.qas[:max_qas]:
        main_mem = encode_memory(model, tokenizer, hypernet, group.passage, device, question=qa.question)
        neg_mem = None
        if has_negative:
            neg_mem = encode_memory(model, tokenizer, hypernet, group.negative_passage, device, question=qa.question)
        gold_answer = qa.target_answer(answer_target)
        negative_answer = qa.target_negative_answer(answer_target)
        gold_tok = tokenize_qa(tokenizer, qa.question, gold_answer, device)
        main_gold_logits = forward_with_memory(
            model,
            target_layer,
            main_mem["K"],
            main_mem["V"],
            gold_tok,
            alpha=ALPHA,
            injection_mode=injection_mode,
        )
        main_gold = compute_answer_loss(main_gold_logits, gold_tok["labels"])
        if main_gold is None:
            continue
        main_gold_prefix = compute_prefix_answer_loss(main_gold_logits, gold_tok["labels"], answer_prefix_tokens)
        zero = main_gold.detach().new_tensor(0.0)
        gold_unit = main_gold
        if answer_prefix_weight > 0 and main_gold_prefix is not None:
            gold_unit = gold_unit + answer_prefix_weight * main_gold_prefix
        if short_answer_weight > 0 and qa.answer and qa.answer != gold_answer:
            short_tok = tokenize_qa(tokenizer, qa.question, qa.answer, device)
            short_logits = forward_with_memory(
                model,
                target_layer,
                main_mem["K"],
                main_mem["V"],
                short_tok,
                alpha=ALPHA,
                injection_mode=injection_mode,
            )
            short_loss = compute_answer_loss(short_logits, short_tok["labels"])
            if short_loss is not None:
                gold_unit = gold_unit + short_answer_weight * short_loss
        if neg_mem is None or not negative_answer:
            weight = final_weight if qa.qa_type == "final" else 1.0
            if weight > 0:
                losses.append(gold_unit * weight)
                loss_weights.append(main_gold.detach().new_tensor(weight))
            main_ok += 1
            used += 1
            continue

        neg_tok = tokenize_qa(tokenizer, qa.question, negative_answer, device)
        main_neg_logits = forward_with_memory(
            model,
            target_layer,
            main_mem["K"],
            main_mem["V"],
            neg_tok,
            alpha=ALPHA,
            injection_mode=injection_mode,
        )
        neg_gold_logits = forward_with_memory(
            model,
            target_layer,
            neg_mem["K"],
            neg_mem["V"],
            gold_tok,
            alpha=ALPHA,
            injection_mode=injection_mode,
        )
        neg_neg_logits = forward_with_memory(
            model,
            target_layer,
            neg_mem["K"],
            neg_mem["V"],
            neg_tok,
            alpha=ALPHA,
            injection_mode=injection_mode,
        )
        main_neg = compute_answer_loss(main_neg_logits, neg_tok["labels"])
        neg_gold = compute_answer_loss(neg_gold_logits, gold_tok["labels"])
        neg_neg = compute_answer_loss(neg_neg_logits, neg_tok["labels"])
        if any(loss is None for loss in (main_neg, neg_gold, neg_neg)):
            continue
        neg_neg_prefix = compute_prefix_answer_loss(neg_neg_logits, neg_tok["labels"], answer_prefix_tokens)
        rank = F.relu(RANK_MARGIN + main_gold - main_neg) + F.relu(RANK_MARGIN + neg_neg - neg_gold)
        unit_loss = gold_unit + neg_neg + rank_weight * rank
        if answer_prefix_weight > 0 and neg_neg_prefix is not None:
            unit_loss = unit_loss + answer_prefix_weight * neg_neg_prefix
        if short_answer_weight > 0 and qa.negative_answer and qa.negative_answer != negative_answer:
            neg_short_tok = tokenize_qa(tokenizer, qa.question, qa.negative_answer, device)
            neg_short_logits = forward_with_memory(
                model,
                target_layer,
                neg_mem["K"],
                neg_mem["V"],
                neg_short_tok,
                alpha=ALPHA,
                injection_mode=injection_mode,
            )
            neg_short_loss = compute_answer_loss(neg_short_logits, neg_short_tok["labels"])
            if neg_short_loss is not None:
                unit_loss = unit_loss + short_answer_weight * neg_short_loss
        weight = final_weight if qa.qa_type == "final" else 1.0
        if weight > 0:
            losses.append(unit_loss * weight)
            loss_weights.append(unit_loss.detach().new_tensor(weight))
        main_ok += int(main_gold.item() < main_neg.item())
        neg_ok += int(neg_neg.item() < neg_gold.item())
        used += 1

    if not losses:
        return None
    objective = torch.stack(losses).sum() / torch.stack(loss_weights).sum().clamp_min(1e-6)
    denom = max(used, 1)
    zero = zero if zero is not None else objective.detach().new_tensor(0.0)
    return {
        "objective": objective,
        "main_gold": objective.detach(),
        "neg_neg": zero,
        "rank": zero,
        "main_ok": main_ok == used,
        "neg_ok": neg_ok == used if has_negative else False,
        "group_qas": used,
        "group_main_rate": main_ok / denom,
        "group_neg_rate": neg_ok / denom,
    }


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    hypernet,
    target_layer,
    examples,
    device,
    final_weight: float = 1.0,
    positive_only: bool = False,
    answer_target: str = "full_answer",
    short_answer_weight: float = 0.0,
    answer_prefix_weight: float = 0.0,
    answer_prefix_tokens: int = 3,
    injection_mode: str = "attention",
):
    hypernet.eval()
    total = 0
    obj = main_ok = neg_ok = flip_ok = 0.0
    for example in examples:
        out = example_loss(
            model,
            tokenizer,
            hypernet,
            target_layer,
            example,
            device,
            positive_only=positive_only,
            answer_target=answer_target,
            short_answer_weight=short_answer_weight,
            answer_prefix_weight=answer_prefix_weight,
            answer_prefix_tokens=answer_prefix_tokens,
            injection_mode=injection_mode,
        )
        if out is None:
            continue
        if example.qa_type == "final":
            out["objective"] = out["objective"] * final_weight
        total += 1
        obj += out["objective"].item()
        main_ok += int(out["main_ok"])
        neg_ok += int(out["neg_ok"])
        flip_ok += int(out["main_ok"] and out["neg_ok"])
    hypernet.train()
    denom = max(total, 1)
    return {
        "count": total,
        "objective": obj / denom,
        "main_ok": main_ok / denom,
        "neg_ok": neg_ok / denom,
        "flip_ok": flip_ok / denom,
    }


@torch.no_grad()
def evaluate_groups(
    model,
    tokenizer,
    hypernet,
    target_layer,
    groups,
    device,
    rank_weight: float,
    max_qas: int,
    final_weight: float,
    positive_only: bool = False,
    answer_target: str = "full_answer",
    short_answer_weight: float = 0.0,
    answer_prefix_weight: float = 0.0,
    answer_prefix_tokens: int = 3,
    injection_mode: str = "attention",
):
    hypernet.eval()
    total = 0
    obj = main_ok = neg_ok = flip_ok = 0.0
    for group in groups:
        out = group_loss(
            model,
            tokenizer,
            hypernet,
            target_layer,
            group,
            device,
            rank_weight,
            max_qas,
            final_weight,
            positive_only,
            answer_target,
            short_answer_weight,
            answer_prefix_weight,
            answer_prefix_tokens,
            injection_mode,
        )
        if out is None:
            continue
        total += 1
        obj += out["objective"].item()
        main_ok += float(out.get("group_main_rate", int(out["main_ok"])))
        neg_ok += float(out.get("group_neg_rate", int(out["neg_ok"])))
        flip_ok += int(out["main_ok"] and out["neg_ok"])
    hypernet.train()
    denom = max(total, 1)
    return {
        "count": total,
        "objective": obj / denom,
        "main_ok": main_ok / denom,
        "neg_ok": neg_ok / denom,
        "flip_ok": flip_ok / denom,
    }


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


def text_hit(text: str, answer: str) -> bool:
    return "".join(str(answer).lower().split()) in "".join(str(text).lower().split())


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, device, max_new_tokens: int) -> str:
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
    question: str,
    K,
    V,
    device,
    max_new_tokens: int,
    alpha: float,
    injection_mode: str = "attention",
):
    prompt = build_chat_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hook = target_layer.register_forward_hook(
        make_memory_hook(K, V, model_num_heads(model), alpha=alpha, injection_mode=injection_mode)
    )
    try:
        generated = model.generate(
            **inputs,
            generation_config=deterministic_generation_config(tokenizer, max_new_tokens),
        )
    finally:
        hook.remove()
    return tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def empty_generation_bucket() -> dict:
    return {
        "count": 0,
        "main_kv_hits": 0,
        "neg_kv_hits": 0,
        "direct_passage_hits": 0,
        "no_memory_hits": 0,
        "zero_kv_hits": 0,
    }


def update_generation_bucket(bucket: dict, *, main_hit: bool, neg_hit: bool, direct_hit: bool, no_mem_hit: bool, zero_hit: bool) -> None:
    bucket["count"] += 1
    bucket["main_kv_hits"] += int(main_hit)
    bucket["neg_kv_hits"] += int(neg_hit)
    bucket["direct_passage_hits"] += int(direct_hit)
    bucket["no_memory_hits"] += int(no_mem_hit)
    bucket["zero_kv_hits"] += int(zero_hit)


def generation_rates(bucket: dict) -> dict:
    denom = max(int(bucket.get("count", 0)), 1)
    return {
        "count": int(bucket.get("count", 0)),
        "main_kv_hit_rate": bucket.get("main_kv_hits", 0) / denom,
        "neg_kv_hit_rate": bucket.get("neg_kv_hits", 0) / denom,
        "direct_passage_hit_rate": bucket.get("direct_passage_hits", 0) / denom,
        "no_memory_hit_rate": bucket.get("no_memory_hits", 0) / denom,
        "zero_kv_hit_rate": bucket.get("zero_kv_hits", 0) / denom,
        "main_kv_hits": int(bucket.get("main_kv_hits", 0)),
        "neg_kv_hits": int(bucket.get("neg_kv_hits", 0)),
        "direct_passage_hits": int(bucket.get("direct_passage_hits", 0)),
        "no_memory_hits": int(bucket.get("no_memory_hits", 0)),
        "zero_kv_hits": int(bucket.get("zero_kv_hits", 0)),
    }


@torch.no_grad()
def evaluate_generation(
    model,
    tokenizer,
    hypernet,
    target_layer,
    examples,
    device,
    *,
    max_new_tokens: int,
    alpha: float,
    injection_mode: str = "attention",
):
    """Small fixed free-generation probe for plotting service behavior.

    Candidate loss can improve while generation remains unstable. This probe is
    intentionally small and optional because generation is much slower than CE
    scoring during training.
    """
    hypernet.eval()
    overall = empty_generation_bucket()
    by_type: dict[str, dict] = {}
    for example in examples:
        if not (example.negative_passage and example.negative_answer):
            continue
        main_mem = encode_memory(model, tokenizer, hypernet, example.passage, device, question=example.question)
        neg_mem = encode_memory(model, tokenizer, hypernet, example.negative_passage, device, question=example.question)
        main_gen = generate_with_kv(
            model,
            tokenizer,
            target_layer,
            example.question,
            main_mem["K"],
            main_mem["V"],
            device,
            max_new_tokens,
            alpha,
            injection_mode,
        )
        neg_gen = generate_with_kv(
            model,
            tokenizer,
            target_layer,
            example.question,
            neg_mem["K"],
            neg_mem["V"],
            device,
            max_new_tokens,
            alpha,
            injection_mode,
        )
        zero_gen = generate_with_kv(
            model,
            tokenizer,
            target_layer,
            example.question,
            torch.zeros_like(main_mem["K"]),
            torch.zeros_like(main_mem["V"]),
            device,
            max_new_tokens,
            alpha,
            injection_mode,
        )
        no_mem_gen = generate_text(model, tokenizer, build_chat_prompt(tokenizer, example.question), device, max_new_tokens)
        direct_gen = generate_text(
            model,
            tokenizer,
            build_direct_passage_prompt(tokenizer, example.question, example.passage),
            device,
            max_new_tokens,
        )
        qa_type = example.qa_type or "qa"
        bucket = by_type.setdefault(qa_type, empty_generation_bucket())
        values = {
            "main_hit": text_hit(main_gen, example.answer),
            "neg_hit": text_hit(neg_gen, example.negative_answer),
            "direct_hit": text_hit(direct_gen, example.answer),
            "no_mem_hit": text_hit(no_mem_gen, example.answer),
            "zero_hit": text_hit(zero_gen, example.answer),
        }
        update_generation_bucket(overall, **values)
        update_generation_bucket(bucket, **values)
    hypernet.train()
    rates = generation_rates(overall)
    rates["by_type"] = {qa_type: generation_rates(bucket) for qa_type, bucket in sorted(by_type.items())}
    return rates


PATTERN_KEYS = set(["facts", "definition", "composition", "analogy", "contrast", "cause", "procedure", "example"])


def eval_stratum_key(item, fallback_kind: str) -> tuple[str, str, str, str]:
    source_id = str(getattr(item, "source_id", "") or "")
    base_id = source_id.split(":", 1)[0]
    parts = base_id.split("_")
    lang = "unknown"
    domain = "unknown"
    pattern = "unknown"
    if "ko" in parts or "en" in parts:
        lang_idx = parts.index("ko") if "ko" in parts else parts.index("en")
        lang = parts[lang_idx]
        pattern_indices = [idx for idx, part in enumerate(parts) if part in PATTERN_KEYS and idx > lang_idx]
        if pattern_indices:
            pattern_idx = pattern_indices[-1]
            pattern = parts[pattern_idx]
            domain = "_".join(parts[lang_idx + 1:pattern_idx]) or "unknown"
        elif len(parts) > lang_idx + 1:
            domain = parts[lang_idx + 1]
    qa_type = str(getattr(item, "qa_type", fallback_kind) or fallback_kind)
    return (lang, domain, pattern, qa_type)


def balanced_eval_subset(items, limit: int, seed: int, fallback_kind: str):
    if limit <= 0 or len(items) <= limit:
        return list(items)
    rng = random.Random(seed)
    by_key = {}
    for item in items:
        by_key.setdefault(eval_stratum_key(item, fallback_kind), []).append(item)
    for values in by_key.values():
        rng.shuffle(values)
    keys = list(by_key)
    rng.shuffle(keys)
    selected = []
    while len(selected) < limit and any(by_key.values()):
        for key in keys:
            if by_key[key]:
                selected.append(by_key[key].pop())
                if len(selected) >= limit:
                    break
    return selected


def print_eval_subset_summary(name: str, subset, total: int) -> None:
    counts = {}
    for item in subset:
        key = eval_stratum_key(item, name)
        counts[key] = counts.get(key, 0) + 1
    preview = ", ".join(
        f"{'/'.join(key)}:{count}"
        for key, count in sorted(counts.items())[:12]
    )
    print(
        f"[PRAG:train:eval-subset] {name} selected={len(subset)}/{total} "
        f"strata={len(counts)} preview={preview}"
    )


def save_checkpoint(path, hypernet, optimizer, scheduler, step, best_val, run_config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "hypernet": hypernet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "config": run_config,
        },
        path,
    )


def normalize_resume_config(config: dict) -> dict:
    """Normalize config keys that should not block a safe resume.

    Older checkpoints may not contain newly added dataset flags when those flags
    default to False. Treating a missing False as different would unnecessarily
    restart long runs even though the actual model/training structure is the
    same. Path separators are also normalized so Windows-style paths remain
    stable after minor code or shell changes.
    """
    normalized = dict(config or {})
    for key in (
        "multifact",
        "korquad",
        "external_qa",
        "transcript",
        "ko_content",
        "lecture",
        "aihub_lecture",
        "positive_only",
    ):
        normalized[key] = bool(normalized.get(key, False))
    for key in ("train_path", "valid_path"):
        if key in normalized:
            normalized[key] = str(normalized[key]).replace("\\", "/")
    normalized["injection_mode"] = normalized.get("injection_mode", "attention")
    return normalized


def resume_config_matches(saved_config: dict, run_config: dict) -> bool:
    return normalize_resume_config(saved_config) == normalize_resume_config(run_config)


def resume_config_diff(saved_config: dict, run_config: dict, max_items: int = 12) -> list[str]:
    saved = normalize_resume_config(saved_config)
    current = normalize_resume_config(run_config)
    keys = sorted(set(saved) | set(current))
    lines = []
    for key in keys:
        if saved.get(key) != current.get(key):
            lines.append(f"{key}: checkpoint={saved.get(key)!r} current={current.get(key)!r}")
            if len(lines) >= max_items:
                break
    return lines


def init_training_log(path, run_config, resumed_step: int):
    now = datetime.now().isoformat()
    if resumed_step > 0 and path.exists():
        try:
            log = json.loads(path.read_text(encoding="utf-8"))
            if resume_config_matches(log.get("config"), run_config):
                log.setdefault("schema_version", 2)
                log.setdefault("step_losses", [])
                log.setdefault("val_evals", [])
                log.setdefault("sessions", [])
                log["sessions"].append({"start": now, "resume_step": resumed_step})
                return log
        except Exception:
            pass
    return {
        "schema_version": 2,
        "config": run_config,
        "step_losses": [],
        "val_evals": [],
        "sessions": [{"start": now, "resume_step": resumed_step}],
        "start": now,
    }


def write_training_log(path, log):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(AUGMENTED_TRAIN_PATH))
    parser.add_argument("--valid", default=str(AUGMENTED_VALID_PATH))
    parser.add_argument(
        "--multifact",
        action="store_true",
        help="Use the default multi-fact augmented train/valid JSONL files.",
    )
    parser.add_argument(
        "--korquad",
        action="store_true",
        help="Use the converted KorQuAD Korean MRC train/valid JSONL files and separate KorQuAD output weights.",
    )
    parser.add_argument(
        "--external-qa",
        action="store_true",
        help="Use external HotpotQA/KorQuAD-style augmented train/valid files and separate output weights.",
    )
    parser.add_argument(
        "--transcript",
        action="store_true",
        help="Use transcript-style augmented train/valid files and separate transcript output weights.",
    )
    parser.add_argument(
        "--lecture",
        action="store_true",
        help="Use augmented local lecture transcript train/valid JSONL files and separate lecture output weights.",
    )
    parser.add_argument(
        "--aihub-lecture",
        action="store_true",
        help="Use augmented AI Hub university lecture train/valid JSONL files and separate output weights.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--rank-weight", type=float, default=RANK_WEIGHT)
    parser.add_argument(
        "--final-weight",
        type=float,
        default=1.0,
        help="Loss weight for final/group-summary QA examples. Lower this when broad summary QAs hurt atomic grounding.",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Ignore generated hard negatives and train only main passage -> gold answer CE. Useful for external MRC data such as KorQuAD.",
    )
    parser.add_argument(
        "--answer-target",
        choices=("answer", "full_answer"),
        default="full_answer",
        help=(
            "Target text used for generation CE. 'full_answer' trains service-style natural answers; "
            "'answer' keeps legacy short-span training."
        ),
    )
    parser.add_argument(
        "--short-answer-weight",
        type=float,
        default=0.0,
        help=(
            "Extra CE weight on the compact answer span even when --answer-target=full_answer. "
            "Use this for free-generation tuning so outputs start with the grounded answer phrase."
        ),
    )
    parser.add_argument(
        "--answer-prefix-weight",
        type=float,
        default=0.0,
        help=(
            "Extra CE weight on the first answer tokens. This directly pressures greedy generation "
            "to begin with the passage-grounded answer instead of a plausible hallucinated phrase."
        ),
    )
    parser.add_argument(
        "--answer-prefix-tokens",
        type=int,
        default=3,
        help="Number of initial answer tokens used by --answer-prefix-weight.",
    )
    parser.add_argument(
        "--group-weight",
        type=float,
        default=1.0,
        help="Weight for full-passage group tasks. Set 0 to train only expanded individual QA examples.",
    )
    parser.add_argument(
        "--group-max-qas",
        type=int,
        default=6,
        help="Maximum atomic/final QAs used from one source passage in a group-level step.",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=EVAL_MAX_SAMPLES,
        help="Fixed balanced validation subset size used during training evaluations.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Seed for the fixed balanced validation subset.",
    )
    parser.add_argument(
        "--eval-generation-samples",
        type=int,
        default=EVAL_GENERATION_SAMPLES,
        help=(
            "Run a small free-generation validation probe on this many balanced examples. "
            "Set 0 to disable because generation is much slower than loss scoring."
        ),
    )
    parser.add_argument(
        "--eval-generation-every",
        type=int,
        default=EVAL_GENERATION_EVERY,
        help="Run the optional free-generation probe every N steps. Use 0 to run at every validation.",
    )
    parser.add_argument(
        "--eval-generation-max-new-tokens",
        type=int,
        default=EVAL_GENERATION_MAX_NEW_TOKENS,
        help="Maximum generated tokens for the optional free-generation validation probe.",
    )
    parser.add_argument(
        "--injection-mode",
        choices=("attention", "add_all", "add_last", "hybrid"),
        default="attention",
        help=(
            "Memory injection operation used during training and evaluation. "
            "'attention' is the original PRAG-style hook; additive modes are ablations."
        ),
    )
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="Use only the first N expanded memory examples for a quick overfit test.",
    )
    parser.add_argument(
        "--overfit-repeat",
        type=int,
        default=1,
        help="Repeat the selected overfit examples this many times.",
    )
    parser.add_argument(
        "--init-weights",
        default="",
        help="Initialize the hypernetwork from a weights .pt file, then train with a fresh optimizer/scheduler.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    checkpoint_path = CHECKPOINT_PATH
    weights_path = WEIGHTS_PATH
    log_path = LOG_PATH
    if args.multifact:
        args.train = str(MULTIFACT_AUGMENTED_TRAIN_PATH)
        args.valid = str(MULTIFACT_AUGMENTED_VALID_PATH)
        checkpoint_path = MULTIFACT_CHECKPOINT_PATH
        weights_path = MULTIFACT_WEIGHTS_PATH
        log_path = MULTIFACT_LOG_PATH
    if args.korquad:
        args.train = str(KORQUAD_AUGMENTED_TRAIN_PATH)
        args.valid = str(KORQUAD_AUGMENTED_VALID_PATH)
        checkpoint_path = KORQUAD_CHECKPOINT_PATH
        weights_path = KORQUAD_WEIGHTS_PATH
        log_path = KORQUAD_LOG_PATH
    if args.external_qa:
        args.train = str(EXTERNAL_QA_AUGMENTED_TRAIN_PATH)
        args.valid = str(EXTERNAL_QA_AUGMENTED_VALID_PATH)
        checkpoint_path = EXTERNAL_QA_CHECKPOINT_PATH
        weights_path = EXTERNAL_QA_WEIGHTS_PATH
        log_path = EXTERNAL_QA_LOG_PATH
    if args.transcript:
        args.train = str(TRANSCRIPT_AUGMENTED_TRAIN_PATH)
        args.valid = str(TRANSCRIPT_AUGMENTED_VALID_PATH)
        checkpoint_path = TRANSCRIPT_CHECKPOINT_PATH
        weights_path = TRANSCRIPT_WEIGHTS_PATH
        log_path = TRANSCRIPT_LOG_PATH
    if args.lecture:
        args.train = str(LECTURE_AUGMENTED_TRAIN_PATH)
        args.valid = str(LECTURE_AUGMENTED_VALID_PATH)
        checkpoint_path = LECTURE_CHECKPOINT_PATH
        weights_path = LECTURE_WEIGHTS_PATH
        log_path = LECTURE_LOG_PATH
    if args.aihub_lecture:
        args.train = str(AIHUB_LECTURE_AUGMENTED_TRAIN_PATH)
        args.valid = str(AIHUB_LECTURE_AUGMENTED_VALID_PATH)
        checkpoint_path = AIHUB_LECTURE_CHECKPOINT_PATH
        weights_path = AIHUB_LECTURE_WEIGHTS_PATH
        log_path = AIHUB_LECTURE_LOG_PATH

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    layer_idx = load_critical_layer()
    target_layer = model.model.layers[layer_idx]
    train_snapshot = jsonl_snapshot(args.train)
    valid_snapshot = jsonl_snapshot(args.valid)
    print(
        "[PRAG:train:datafile] "
        f"train_rows={train_snapshot['rows']} last_train_source={train_snapshot['last_source_id']} "
        f"path={train_snapshot['path']}"
    )
    print(
        "[PRAG:train:datafile] "
        f"valid_rows={valid_snapshot['rows']} last_valid_source={valid_snapshot['last_source_id']} "
        f"path={valid_snapshot['path']}"
    )
    train_examples = load_augmented_examples(args.train, max_samples=args.max_samples or None)
    valid_examples = load_augmented_examples(args.valid, max_samples=args.max_val_samples or None)
    train_groups = load_augmented_groups(args.train, max_samples=args.max_samples or None)
    valid_groups = load_augmented_groups(args.valid, max_samples=args.max_val_samples or None)
    if args.overfit_samples > 0:
        selected = train_examples[: args.overfit_samples]
        selected_groups = train_groups[: args.overfit_samples]
        train_examples = selected * max(args.overfit_repeat, 1)
        train_groups = selected_groups * max(args.overfit_repeat, 1)
        valid_examples = selected
        valid_groups = selected_groups
        print(
            f"[PRAG:train] overfit mode: examples={len(selected)} groups={len(selected_groups)} "
            f"repeat={max(args.overfit_repeat, 1)}"
        )
    if not train_examples or not valid_examples:
        raise RuntimeError("Need non-empty augmented train and valid examples.")
    valid_eval_examples = balanced_eval_subset(valid_examples, args.eval_max_samples, args.eval_seed, "example")
    valid_eval_groups = balanced_eval_subset(valid_groups, args.eval_max_samples, args.eval_seed + 1, "group")
    valid_generation_examples = []
    if args.eval_generation_samples > 0:
        valid_generation_examples = balanced_eval_subset(
            valid_examples,
            args.eval_generation_samples,
            args.eval_seed + 2,
            "generation",
        )
    print_eval_subset_summary("example", valid_eval_examples, len(valid_examples))
    if valid_groups:
        print_eval_subset_summary("group", valid_eval_groups, len(valid_groups))
    if valid_generation_examples:
        print_eval_subset_summary("generation", valid_generation_examples, len(valid_examples))

    hypernet = make_hypernet(model, device)
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=args.lr, weight_decay=0.0)
    train_units = [("example", item) for item in train_examples]
    if args.group_weight > 0:
        train_units.extend(("group", item) for item in train_groups)
    total_steps = max(1, len(train_units) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LR_MIN)
    run_config = {
        "model": MODEL_NAME,
        "critical_layer": layer_idx,
        "num_kv": NUM_KV,
        "hidden_dim": HIDDEN_DIM,
        "feature_dim": model.config.hidden_size * (2 if USE_CONTEXTUAL_MEMORY else 1),
        "use_contextual_memory": USE_CONTEXTUAL_MEMORY,
        "question_conditioned_memory": QUESTION_CONDITIONED_MEMORY,
        "injection_mode": args.injection_mode,
        "alpha": ALPHA,
        "objective": "atomic_final_ce_plus_negative_flip",
        "multifact": args.multifact,
        "korquad": args.korquad,
        "external_qa": args.external_qa,
        "transcript": args.transcript,
        "lecture": args.lecture,
        "aihub_lecture": args.aihub_lecture,
        "rank_weight": args.rank_weight,
        "final_weight": args.final_weight,
        "positive_only": args.positive_only,
        "answer_target": args.answer_target,
        "group_weight": args.group_weight,
        "group_max_qas": args.group_max_qas,
        "eval_max_samples": args.eval_max_samples,
        "eval_seed": args.eval_seed,
        "train_path": str(args.train),
        "valid_path": str(args.valid),
        "overfit_samples": args.overfit_samples,
        "overfit_repeat": args.overfit_repeat,
    }
    if args.short_answer_weight > 0:
        run_config["short_answer_weight"] = args.short_answer_weight
    if args.answer_prefix_weight > 0:
        run_config["answer_prefix_weight"] = args.answer_prefix_weight
        run_config["answer_prefix_tokens"] = args.answer_prefix_tokens
    print(f"[PRAG:train] config: {run_config}")
    print(
        f"[PRAG:train] expanded_examples train={len(train_examples)} valid={len(valid_examples)} "
        f"from_train_rows={train_snapshot['rows']} from_valid_rows={valid_snapshot['rows']}"
    )
    print(
        f"[PRAG:train] group_examples train={len(train_groups)} valid={len(valid_groups)} "
        f"group_weight={args.group_weight} group_max_qas={args.group_max_qas}"
    )

    best_val = float("inf")
    step = 0
    print(f"[PRAG:train] outputs weights={weights_path} checkpoint={checkpoint_path} log={log_path}")

    loaded_checkpoint = False
    if args.resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        if resume_config_matches(ckpt.get("config"), run_config):
            hypernet.load_state_dict(ckpt["hypernet"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            best_val = float(ckpt.get("best_val", best_val))
            step = int(ckpt.get("step", 0))
            loaded_checkpoint = True
            print(f"[PRAG:train] resumed step={step} best_val={best_val:.4f}")
        else:
            print("[PRAG:train] checkpoint config mismatch; starting fresh.")
            for line in resume_config_diff(ckpt.get("config"), run_config):
                print(f"  [PRAG:train:config-diff] {line}")
    if not loaded_checkpoint and args.init_weights:
        loaded_keys, skipped_keys = load_compatible_hypernet_weights(hypernet, args.init_weights, device)
        print(
            f"[PRAG:train] initialized compatible hypernet tensors from weights={args.init_weights} "
            f"(loaded={loaded_keys}, skipped_shape_or_missing={skipped_keys})"
        )

    log = init_training_log(log_path, run_config, step)
    log["generation_eval_config"] = {
        "enabled": bool(valid_generation_examples),
        "samples": len(valid_generation_examples),
        "every": args.eval_generation_every,
        "max_new_tokens": args.eval_generation_max_new_tokens,
        "alpha": ALPHA,
        "injection_mode": args.injection_mode,
        "note": "Free-generation probe is intentionally outside checkpoint config so resume compatibility is stable.",
    }
    previous_runtime_sec = round(
        sum(float(session.get("elapsed_sec", 0.0)) for session in log.get("sessions", [])[:-1]),
        3,
    )
    log["previous_logged_runtime_sec"] = previous_runtime_sec
    start = time.time()
    hypernet.train()
    for _epoch in range(args.epochs):
        indices = list(range(len(train_units)))
        random.shuffle(indices)
        running = []
        running_kind = {"example": 0, "group": 0}
        for idx in indices:
            if step >= total_steps:
                break
            kind, item = train_units[idx]
            if kind == "group":
                out = group_loss(
                    model,
                    tokenizer,
                    hypernet,
                    target_layer,
                    item,
                    device,
                    rank_weight=args.rank_weight,
                    max_qas=args.group_max_qas,
                    final_weight=args.final_weight,
                    positive_only=args.positive_only,
                    answer_target=args.answer_target,
                    short_answer_weight=args.short_answer_weight,
                    answer_prefix_weight=args.answer_prefix_weight,
                    answer_prefix_tokens=args.answer_prefix_tokens,
                    injection_mode=args.injection_mode,
                )
                if out is not None:
                    out["objective"] = out["objective"] * args.group_weight
            else:
                out = example_loss(
                    model,
                    tokenizer,
                    hypernet,
                    target_layer,
                    item,
                    device,
                    rank_weight=args.rank_weight,
                    positive_only=args.positive_only,
                    answer_target=args.answer_target,
                    short_answer_weight=args.short_answer_weight,
                    answer_prefix_weight=args.answer_prefix_weight,
                    answer_prefix_tokens=args.answer_prefix_tokens,
                    injection_mode=args.injection_mode,
                )
                if out is not None and getattr(item, "qa_type", "") == "final":
                    out["objective"] = out["objective"] * args.final_weight
            if out is None:
                continue
            optimizer.zero_grad()
            out["objective"].backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            obj = out["objective"].item()
            elapsed_sec = time.time() - start
            cumulative_elapsed_sec = previous_runtime_sec + elapsed_sec
            lr_now = scheduler.get_last_lr()[0]
            running.append(obj)
            running_kind[kind] += 1
            log_entry = {
                "step": step,
                "objective": round(obj, 6),
                "main_gold": round(float(out["main_gold"].detach().item()), 6),
                "main_neg": round(float(out.get("main_neg", out["rank"]).detach().item()), 6),
                "neg_gold": round(float(out.get("neg_gold", out["rank"]).detach().item()), 6),
                "neg_neg": round(float(out["neg_neg"].detach().item()), 6),
                "rank": round(float(out["rank"].detach().item()), 6),
                "prefix": round(float(out.get("prefix", out["rank"]).detach().item()), 6),
                "main_short": round(float(out.get("main_short", out["rank"]).detach().item()), 6),
                "neg_short": round(float(out.get("neg_short", out["rank"]).detach().item()), 6),
                "main_ok": bool(out.get("main_ok", False)),
                "neg_ok": bool(out.get("neg_ok", False)),
                "kind": kind,
                "lr": lr_now,
                "elapsed_sec": round(elapsed_sec, 3),
                "elapsed_min": round(elapsed_sec / 60, 4),
                "cumulative_elapsed_sec": round(cumulative_elapsed_sec, 3),
                "cumulative_elapsed_min": round(cumulative_elapsed_sec / 60, 4),
            }
            if kind == "group":
                log_entry.update({
                    "group_qas": int(out.get("group_qas", 0)),
                    "group_main_rate": round(float(out.get("group_main_rate", 0.0)), 6),
                    "group_neg_rate": round(float(out.get("group_neg_rate", 0.0)), 6),
                })
            log["step_losses"].append(log_entry)
            if step % LOG_EVERY == 0:
                print(
                    f"  Step {step}/{total_steps} | obj={obj:.4f} | "
                    f"avg={sum(running)/len(running):.4f} | main={out['main_gold'].item():.3f} | "
                    f"neg={out['neg_neg'].item():.3f} | rank={out['rank'].item():.3f} | "
                    f"kind={kind} | mix=e{running_kind['example']}/g{running_kind['group']} | "
                    f"lr={lr_now:.2e} | {elapsed_sec / 60:.1f}min"
                )
                running = []
                running_kind = {"example": 0, "group": 0}
            if step % SAVE_EVERY == 0:
                save_checkpoint(checkpoint_path, hypernet, optimizer, scheduler, step, best_val, run_config)
                if log.get("sessions"):
                    log["sessions"][-1]["last_saved_step"] = step
                    log["sessions"][-1]["elapsed_sec"] = round(time.time() - start, 3)
                write_training_log(log_path, log)
                print(f"  [PRAG:checkpoint] saved step {step}")
            if step % EVAL_EVERY == 0:
                metrics = evaluate(
                    model,
                    tokenizer,
                    hypernet,
                    target_layer,
                    valid_eval_examples,
                    device,
                    args.final_weight,
                    args.positive_only,
                    args.answer_target,
                    args.short_answer_weight,
                    args.answer_prefix_weight,
                    args.answer_prefix_tokens,
                    args.injection_mode,
                )
                group_metrics = evaluate_groups(
                    model,
                    tokenizer,
                    hypernet,
                    target_layer,
                    valid_eval_groups,
                    device,
                    args.rank_weight,
                    args.group_max_qas,
                    args.final_weight,
                    args.positive_only,
                    args.answer_target,
                    args.short_answer_weight,
                    args.answer_prefix_weight,
                    args.answer_prefix_tokens,
                    args.injection_mode,
                ) if valid_eval_groups and args.group_weight > 0 else None
                selection_objective = metrics["objective"]
                if group_metrics is not None:
                    selection_objective = (selection_objective + args.group_weight * group_metrics["objective"]) / 2
                print(
                    f"  -- [PRAG:val @ {step}] obj={metrics['objective']:.4f} | "
                    f"main={metrics['main_ok']:.3f} neg={metrics['neg_ok']:.3f} flip={metrics['flip_ok']:.3f}"
                )
                if group_metrics is not None:
                    print(
                        f"  -- [PRAG:group-val @ {step}] obj={group_metrics['objective']:.4f} | "
                        f"main={group_metrics['main_ok']:.3f} neg={group_metrics['neg_ok']:.3f} "
                        f"flip={group_metrics['flip_ok']:.3f}"
                    )
                generation_metrics = None
                should_run_generation = (
                    bool(valid_generation_examples)
                    and (args.eval_generation_every <= 0 or step % args.eval_generation_every == 0)
                )
                if should_run_generation:
                    generation_metrics = evaluate_generation(
                        model,
                        tokenizer,
                        hypernet,
                        target_layer,
                        valid_generation_examples,
                        device,
                        max_new_tokens=args.eval_generation_max_new_tokens,
                        alpha=ALPHA,
                        injection_mode=args.injection_mode,
                    )
                    print(
                        f"  -- [PRAG:gen-val @ {step}] count={generation_metrics['count']} | "
                        f"main_kv={generation_metrics['main_kv_hit_rate']:.3f} "
                        f"neg_kv={generation_metrics['neg_kv_hit_rate']:.3f} "
                        f"direct={generation_metrics['direct_passage_hit_rate']:.3f} "
                        f"no_mem={generation_metrics['no_memory_hit_rate']:.3f} "
                        f"zero={generation_metrics['zero_kv_hit_rate']:.3f}"
                    )
                log["val_evals"].append({
                    "step": step,
                    "elapsed_sec": round(time.time() - start, 3),
                    "elapsed_min": round((time.time() - start) / 60, 4),
                    "cumulative_elapsed_sec": round(previous_runtime_sec + (time.time() - start), 3),
                    "cumulative_elapsed_min": round((previous_runtime_sec + (time.time() - start)) / 60, 4),
                    "individual": metrics,
                    "group": group_metrics,
                    "generation": generation_metrics,
                })
                if selection_objective < best_val:
                    best_val = selection_objective
                    weights_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "step": step,
                            "hypernet": hypernet.state_dict(),
                            "config": run_config,
                            "val_metrics": metrics,
                            "group_val_metrics": group_metrics,
                            "generation_val_metrics": generation_metrics,
                        },
                        weights_path,
                    )
                    print("     [PRAG:best] saved weights")
                if log.get("sessions"):
                    log["sessions"][-1]["last_eval_step"] = step
                    log["sessions"][-1]["elapsed_sec"] = round(time.time() - start, 3)
                write_training_log(log_path, log)

    metrics = evaluate(
        model,
        tokenizer,
        hypernet,
        target_layer,
        valid_eval_examples,
        device,
        args.final_weight,
        args.positive_only,
        args.answer_target,
        args.short_answer_weight,
        args.answer_prefix_weight,
        args.answer_prefix_tokens,
        args.injection_mode,
    )
    group_metrics = evaluate_groups(
        model,
        tokenizer,
        hypernet,
        target_layer,
        valid_eval_groups,
        device,
        args.rank_weight,
        args.group_max_qas,
        args.final_weight,
        args.positive_only,
        args.answer_target,
        args.short_answer_weight,
        args.answer_prefix_weight,
        args.answer_prefix_tokens,
        args.injection_mode,
    ) if valid_eval_groups and args.group_weight > 0 else None
    selection_objective = metrics["objective"]
    if group_metrics is not None:
        selection_objective = (selection_objective + args.group_weight * group_metrics["objective"]) / 2
    final_generation_metrics = None
    if valid_generation_examples:
        final_generation_metrics = evaluate_generation(
            model,
            tokenizer,
            hypernet,
            target_layer,
            valid_generation_examples,
            device,
            max_new_tokens=args.eval_generation_max_new_tokens,
            alpha=ALPHA,
            injection_mode=args.injection_mode,
        )
        print(
            f"[PRAG:gen-final] count={final_generation_metrics['count']} | "
            f"main_kv={final_generation_metrics['main_kv_hit_rate']:.3f} "
            f"neg_kv={final_generation_metrics['neg_kv_hit_rate']:.3f} "
            f"direct={final_generation_metrics['direct_passage_hit_rate']:.3f} "
            f"no_mem={final_generation_metrics['no_memory_hit_rate']:.3f} "
            f"zero={final_generation_metrics['zero_kv_hit_rate']:.3f}"
        )
    if selection_objective < best_val:
        torch.save(
            {
                "step": step,
                "hypernet": hypernet.state_dict(),
                "config": run_config,
                "val_metrics": metrics,
                "group_val_metrics": group_metrics,
                "generation_val_metrics": final_generation_metrics,
            },
            weights_path,
        )
    save_checkpoint(checkpoint_path, hypernet, optimizer, scheduler, step, min(best_val, selection_objective), run_config)
    if log.get("sessions"):
        log["sessions"][-1]["end"] = datetime.now().isoformat()
        log["sessions"][-1]["elapsed_sec"] = round(time.time() - start, 3)
        log["sessions"][-1]["elapsed_min"] = round((time.time() - start) / 60, 4)
        log["sessions"][-1]["final_step"] = step
    log["end"] = datetime.now().isoformat()
    log["final_step"] = step
    log["final_val"] = metrics
    log["final_group_val"] = group_metrics
    log["final_generation_val"] = final_generation_metrics
    log["total_logged_runtime_sec"] = round(sum(float(session.get("elapsed_sec", 0.0)) for session in log.get("sessions", [])), 3)
    write_training_log(log_path, log)
    print(f"[PRAG:train] done step={step} final_val={metrics} final_group_val={group_metrics}")


if __name__ == "__main__":
    main()
