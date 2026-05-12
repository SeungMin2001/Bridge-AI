"""Compare question+passage and passage-only PRAG memories on the same cases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import (
    ALPHA,
    KORQUAD_SERVICE_AUGMENTED_TRAIN_PATH,
    KORQUAD_SERVICE_AUGMENTED_VALID_PATH,
    MODEL_NAME,
    MULTIFACT_AUGMENTED_TRAIN_PATH,
    MULTIFACT_AUGMENTED_VALID_PATH,
    load_critical_layer,
)
from .memory import HyperKVGenerator, compute_answer_loss, encode_merged_memory, forward_with_memory, tokenize_qa
from .test_single_ko import (
    answer_hit,
    build_direct_passage_prompt,
    compute_prefix_answer_loss,
    generate_direct_passage,
    generate_plain,
    generate_with_kv,
    load_dataset_cases,
    load_model,
    tokenize_prompt_answer,
)
from .train import answer_phrase_candidates, compute_answer_phrase_loss


DEFAULT_QP_WEIGHTS = Path(__file__).resolve().parent / "prag_mixed_kor_service_orthomerge_qwen25_3b_qp_memory_checkpoint.pt"
DEFAULT_PONLY_WEIGHTS = (
    Path(__file__).resolve().parent / "prag_mixed_kor_service_orthomerge_qwen25_3b_ponly_memory_checkpoint.pt"
)


@dataclass
class Variant:
    label: str
    path: Path
    state: dict
    config: dict
    hypernet: HyperKVGenerator
    question_conditioned: bool
    layer_idx: int
    step: int | str
    best_val: float | None


def default_mixed_data(split: str) -> str:
    multifact_path = MULTIFACT_AUGMENTED_TRAIN_PATH if split == "train" else MULTIFACT_AUGMENTED_VALID_PATH
    korquad_path = KORQUAD_SERVICE_AUGMENTED_TRAIN_PATH if split == "train" else KORQUAD_SERVICE_AUGMENTED_VALID_PATH
    return f"{multifact_path};{korquad_path}"


def load_variant(label: str, path: str | Path, model, device) -> Variant:
    weight_path = Path(path)
    state = torch.load(weight_path, map_location="cpu")
    config = state.get("config", {}) or {}
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
    return Variant(
        label=label,
        path=weight_path,
        state=state,
        config=config,
        hypernet=hypernet,
        question_conditioned=bool(config.get("question_conditioned_memory", False)),
        layer_idx=int(config.get("critical_layer", load_critical_layer())),
        step=state.get("step", "unknown"),
        best_val=state.get("best_val"),
    )


def ensure_same_base_model(qp_state: dict, ponly_state: dict) -> str:
    qp_model = str((qp_state.get("config") or {}).get("model") or MODEL_NAME)
    ponly_model = str((ponly_state.get("config") or {}).get("model") or MODEL_NAME)
    if qp_model != ponly_model:
        raise ValueError(f"Cannot compare variants with different base models: qp={qp_model!r}, ponly={ponly_model!r}")
    return qp_model


def fmt(value) -> str:
    if value is None:
        return "n/a"
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


@torch.no_grad()
def evaluate_variant(
    *,
    variant: Variant,
    model,
    tokenizer,
    target_layer,
    device,
    case: dict,
    max_new_tokens: int,
    alpha: float,
    prompt_style: str,
    injection_mode: str,
    answer_prefix_tokens: int,
) -> dict:
    question = case["question"]
    main_answer = case["main_answer"]
    full_answer = case.get("full_answer") or main_answer
    main_passages = [case["main_passage"]] + list(case.get("merge_passages") or [])
    memory = encode_merged_memory(
        model,
        tokenizer,
        variant.hypernet,
        main_passages,
        device,
        question=question,
        question_conditioned=variant.question_conditioned,
    )
    target = tokenize_qa(tokenizer, question, full_answer, device)
    logits = forward_with_memory(
        model,
        target_layer,
        memory["K"],
        memory["V"],
        target,
        alpha=alpha,
        injection_mode=injection_mode,
    )
    kv_loss = compute_answer_loss(logits, target["labels"])
    prefix_loss = compute_prefix_answer_loss(logits, target["labels"], answer_prefix_tokens)
    phrase_loss = compute_answer_phrase_loss(logits, target["labels"], tokenizer, full_answer, main_answer)
    generated = generate_with_kv(
        model,
        tokenizer,
        target_layer,
        question,
        memory["K"],
        memory["V"],
        device,
        max_new_tokens,
        alpha,
        prompt_style,
        injection_mode,
    )
    return {
        "loss": None if kv_loss is None else float(kv_loss.item()),
        "prefix_loss": None if prefix_loss is None else float(prefix_loss.item()),
        "phrase_loss": None if phrase_loss is None else float(phrase_loss.item()),
        "answer": generated,
        "hit": answer_hit(generated, case),
        "merged_count": len(main_passages),
    }


def print_case_report(case: dict, direct: str, no_memory: str, qp: dict, ponly: dict) -> None:
    main_passages = [case["main_passage"]] + list(case.get("merge_passages") or [])
    print("\n" + "=" * 96)
    print(f"[case:{case['name']}]")
    if case.get("source_id"):
        print(f"source_id: {case['source_id']}")
    print(f"question: {case['question']}")
    print(f"expected: {case.get('full_answer') or case['main_answer']}")
    print(f"phrase_targets: {' | '.join(answer_phrase_candidates(case['main_answer']))}")
    print(f"merged_passages: {len(main_passages)}")
    for idx, passage in enumerate(main_passages, start=1):
        print(f"  passage[{idx}]: {passage}")

    print("\n[baseline]")
    print(f"direct_RAG_hit={answer_hit(direct, case)} | direct_RAG_answer={direct}")
    print(f"no_memory_hit={answer_hit(no_memory, case)} | no_memory_answer={no_memory}")

    print("\n[variant comparison]")
    print(
        "question+passage | "
        f"hit={qp['hit']} | loss={fmt(qp['loss'])} | phrase={fmt(qp['phrase_loss'])} | "
        f"prefix={fmt(qp['prefix_loss'])}"
    )
    print(f"  answer: {qp['answer']}")
    print(
        "passage-only     | "
        f"hit={ponly['hit']} | loss={fmt(ponly['loss'])} | phrase={fmt(ponly['phrase_loss'])} | "
        f"prefix={fmt(ponly['prefix_loss'])}"
    )
    print(f"  answer: {ponly['answer']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qp-weights", default=str(DEFAULT_QP_WEIGHTS), help="question+passage checkpoint/weights path.")
    parser.add_argument("--ponly-weights", default=str(DEFAULT_PONLY_WEIGHTS), help="passage-only checkpoint/weights path.")
    parser.add_argument("--data", default=None, help="Augmented JSONL path. Use ';' to compare over mixed datasets.")
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--dataset-merge-max-passages", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--prompt-style", choices=("service", "short-chat", "memory-cued", "paper"), default="service")
    parser.add_argument("--injection-mode", choices=("attention", "add_all", "add_last", "hybrid"), default="attention")
    parser.add_argument("--answer-prefix-tokens", type=int, default=3)
    args = parser.parse_args()

    qp_state = torch.load(Path(args.qp_weights), map_location="cpu")
    ponly_state = torch.load(Path(args.ponly_weights), map_location="cpu")
    model_name = ensure_same_base_model(qp_state, ponly_state)

    model, tokenizer = load_model(model_name)
    device = next(model.parameters()).device
    qp = load_variant("question+passage", args.qp_weights, model, device)
    ponly = load_variant("passage-only", args.ponly_weights, model, device)
    if qp.layer_idx != ponly.layer_idx:
        raise ValueError(f"Critical layer mismatch: qp={qp.layer_idx}, ponly={ponly.layer_idx}")
    target_layer = model.model.layers[qp.layer_idx]

    data_path = args.data or default_mixed_data(args.split)
    cases = load_dataset_cases(
        data_path,
        case_index=args.case_index,
        max_cases=args.max_cases,
        merge_max_passages=args.dataset_merge_max_passages,
    )

    print("[PRAG:compare-memory-variants]")
    print(f"model={model_name} layer={qp.layer_idx} data={data_path}")
    print(
        f"question+passage weights={qp.path} step={qp.step} best_val={fmt(qp.best_val)} "
        f"q_cond={qp.question_conditioned}"
    )
    print(
        f"passage-only     weights={ponly.path} step={ponly.step} best_val={fmt(ponly.best_val)} "
        f"q_cond={ponly.question_conditioned}"
    )

    totals = {
        "count": 0,
        "direct_hits": 0,
        "no_memory_hits": 0,
        "qp_hits": 0,
        "ponly_hits": 0,
        "qp_loss": [],
        "ponly_loss": [],
        "qp_phrase": [],
        "ponly_phrase": [],
    }
    for case in cases:
        question = case["question"]
        main_passages = [case["main_passage"]] + list(case.get("merge_passages") or [])
        direct_passage = "\n\n".join(main_passages)
        direct = generate_direct_passage(model, tokenizer, question, direct_passage, device, args.max_new_tokens)
        no_memory = generate_plain(model, tokenizer, question, device, args.max_new_tokens)
        qp_result = evaluate_variant(
            variant=qp,
            model=model,
            tokenizer=tokenizer,
            target_layer=target_layer,
            device=device,
            case=case,
            max_new_tokens=args.max_new_tokens,
            alpha=args.alpha,
            prompt_style=args.prompt_style,
            injection_mode=args.injection_mode,
            answer_prefix_tokens=args.answer_prefix_tokens,
        )
        ponly_result = evaluate_variant(
            variant=ponly,
            model=model,
            tokenizer=tokenizer,
            target_layer=target_layer,
            device=device,
            case=case,
            max_new_tokens=args.max_new_tokens,
            alpha=args.alpha,
            prompt_style=args.prompt_style,
            injection_mode=args.injection_mode,
            answer_prefix_tokens=args.answer_prefix_tokens,
        )
        print_case_report(case, direct, no_memory, qp_result, ponly_result)

        totals["count"] += 1
        totals["direct_hits"] += int(answer_hit(direct, case))
        totals["no_memory_hits"] += int(answer_hit(no_memory, case))
        totals["qp_hits"] += int(qp_result["hit"])
        totals["ponly_hits"] += int(ponly_result["hit"])
        for key, result_key, result in (
            ("qp_loss", "loss", qp_result),
            ("ponly_loss", "loss", ponly_result),
            ("qp_phrase", "phrase_loss", qp_result),
            ("ponly_phrase", "phrase_loss", ponly_result),
        ):
            if result[result_key] is not None:
                totals[key].append(result[result_key])

    denom = max(totals["count"], 1)

    def avg(values: list[float]) -> str:
        return "n/a" if not values else f"{sum(values) / len(values):.4f}"

    print("\n" + "=" * 96)
    print("[summary]")
    print(f"cases={totals['count']}")
    print(f"direct_RAG_hit_rate={totals['direct_hits'] / denom:.3f}")
    print(f"no_memory_hit_rate={totals['no_memory_hits'] / denom:.3f}")
    print(f"question+passage_hit_rate={totals['qp_hits'] / denom:.3f} avg_loss={avg(totals['qp_loss'])} avg_phrase={avg(totals['qp_phrase'])}")
    print(
        f"passage-only_hit_rate={totals['ponly_hits'] / denom:.3f} "
        f"avg_loss={avg(totals['ponly_loss'])} avg_phrase={avg(totals['ponly_phrase'])}"
    )


if __name__ == "__main__":
    main()
