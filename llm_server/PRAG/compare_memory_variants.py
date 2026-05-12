"""Compare question+passage and passage-only PRAG memories on the same cases."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
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


EVAL_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
QUESTION_STOPWORDS = {
    "무엇인가",
    "무엇이야",
    "무엇",
    "뭐야",
    "뭐라고",
    "뭐에",
    "뭐",
    "누구인가",
    "누구야",
    "누구",
    "언제인가",
    "언제야",
    "언제",
    "어디인가",
    "어디야",
    "어디에",
    "어디",
    "얼마인가",
    "얼마야",
    "얼마",
    "왜",
    "어떤",
    "무슨",
    "몇",
    "질문",
    "답변",
    "설명",
    "설명하셨어",
    "설명했어",
    "하셨어",
    "했어",
    "인가",
    "이야",
}
KOREAN_PARTICLES = (
    "으로부터",
    "에게서",
    "으로서",
    "에서",
    "에게",
    "처럼",
    "으로",
    "보다",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "와",
    "과",
    "도",
    "만",
    "로",
)


def normalize_eval_text(text: str) -> str:
    return "".join(EVAL_TOKEN_RE.findall(str(text or "").casefold()))


def eval_tokens(text: str) -> list[str]:
    tokens = EVAL_TOKEN_RE.findall(str(text or "").casefold())
    cleaned: list[str] = []
    for token in tokens:
        if re.search(r"[가-힣]", token):
            for suffix in KOREAN_PARTICLES:
                if token.endswith(suffix) and len(token) > len(suffix) + 1:
                    token = token[: -len(suffix)]
                    break
        if token in QUESTION_STOPWORDS:
            continue
        if len(token) <= 1 and not token.isdigit():
            continue
        cleaned.append(token)
    return cleaned


def char_f1(prediction: str, reference: str) -> float:
    pred = normalize_eval_text(prediction)
    ref = normalize_eval_text(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / max(len(pred), 1)
    recall = overlap / max(len(ref), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def relation_terms(case: dict) -> list[str]:
    """Terms that define what relation the question is asking about.

    We keep terms shared by the question and reference answer, then remove
    compact answer phrases. This catches differences such as "이벤트 보상" vs
    generic "보상" when both generated answers contain the same value.
    """

    question_terms = eval_tokens(case.get("question", ""))
    reference_terms = set(eval_tokens(case.get("full_answer") or case.get("main_answer") or ""))
    phrase_terms = set()
    for phrase in answer_phrase_candidates(case.get("main_answer", "")):
        phrase_terms.update(eval_tokens(phrase))

    terms: list[str] = []
    seen: set[str] = set()
    for term in question_terms:
        if term in seen or term in phrase_terms:
            continue
        if term in reference_terms:
            terms.append(term)
            seen.add(term)

    if terms:
        return terms

    # Fallback for terse references where the expected answer omits the
    # question relation. Keep non-answer question terms as a softer proxy.
    for term in question_terms:
        if term not in seen and term not in phrase_terms:
            terms.append(term)
            seen.add(term)
    return terms


def relation_coverage(answer: str, case: dict) -> float:
    terms = relation_terms(case)
    if not terms:
        return 1.0
    answer_key = normalize_eval_text(answer)
    matched = sum(1 for term in terms if normalize_eval_text(term) in answer_key)
    return matched / len(terms)


def score_answer(answer: str, case: dict) -> dict:
    full_answer = case.get("full_answer") or case.get("main_answer") or ""
    phrase_hit = answer_hit(answer, case)
    full_f1 = char_f1(answer, full_answer)
    relation = relation_coverage(answer, case)
    # Primary service quality: value correctness first, then relation fidelity
    # and full-answer overlap. Loss remains a diagnostic, not the selector.
    quality = 0.50 * float(phrase_hit) + 0.30 * relation + 0.20 * full_f1
    return {
        "phrase_hit": phrase_hit,
        "answer_char_f1": full_f1,
        "relation_coverage": relation,
        "quality_score": quality,
        "relation_terms": relation_terms(case),
    }


def compare_quality(qp_score: float, ponly_score: float, tie_threshold: float) -> str:
    delta = qp_score - ponly_score
    if abs(delta) <= tie_threshold:
        return "tie"
    return "question+passage" if delta > 0 else "passage-only"


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
    scores = score_answer(generated, case)
    return {
        "loss": None if kv_loss is None else float(kv_loss.item()),
        "prefix_loss": None if prefix_loss is None else float(prefix_loss.item()),
        "phrase_loss": None if phrase_loss is None else float(phrase_loss.item()),
        "answer": generated,
        "hit": answer_hit(generated, case),
        "merged_count": len(main_passages),
        **scores,
    }


def print_case_report(case: dict, direct: str, no_memory: str, qp: dict, ponly: dict, winner: str) -> None:
    main_passages = [case["main_passage"]] + list(case.get("merge_passages") or [])
    print("\n" + "=" * 96)
    print(f"[case:{case['name']}]")
    if case.get("source_id"):
        print(f"source_id: {case['source_id']}")
    print(f"question: {case['question']}")
    print(f"expected: {case.get('full_answer') or case['main_answer']}")
    print(f"phrase_targets: {' | '.join(answer_phrase_candidates(case['main_answer']))}")
    print(f"relation_terms: {' | '.join(relation_terms(case)) or 'n/a'}")
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
        f"prefix={fmt(qp['prefix_loss'])} | rel={fmt(qp['relation_coverage'])} | "
        f"f1={fmt(qp['answer_char_f1'])} | quality={fmt(qp['quality_score'])}"
    )
    print(f"  answer: {qp['answer']}")
    print(
        "passage-only     | "
        f"hit={ponly['hit']} | loss={fmt(ponly['loss'])} | phrase={fmt(ponly['phrase_loss'])} | "
        f"prefix={fmt(ponly['prefix_loss'])} | rel={fmt(ponly['relation_coverage'])} | "
        f"f1={fmt(ponly['answer_char_f1'])} | quality={fmt(ponly['quality_score'])}"
    )
    print(f"  answer: {ponly['answer']}")
    print(f"pairwise_winner: {winner}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_report(path: Path, summary: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[PRAG:compare] matplotlib unavailable; skipped plot ({exc})")
        return

    variants = ["question+passage", "passage-only"]
    metric_specs = [
        ("hit_rate", "Hit"),
        ("avg_relation_coverage", "Relation"),
        ("avg_answer_char_f1", "Answer F1"),
        ("avg_quality_score", "Quality"),
    ]
    values = {
        "question+passage": [summary["question+passage"][key] for key, _ in metric_specs],
        "passage-only": [summary["passage-only"][key] for key, _ in metric_specs],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = list(range(len(metric_specs)))
    width = 0.36
    axes[0].bar([i - width / 2 for i in x], values["question+passage"], width, label="question+passage", color="#2563eb")
    axes[0].bar([i + width / 2 for i in x], values["passage-only"], width, label="passage-only", color="#f97316")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label for _, label in metric_specs], rotation=15)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Answer Quality Metrics")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    outcomes = summary["pairwise"]
    outcome_labels = ["QP win", "P-only win", "Tie"]
    outcome_values = [
        outcomes.get("question+passage", 0),
        outcomes.get("passage-only", 0),
        outcomes.get("tie", 0),
    ]
    axes[1].bar(outcome_labels, outcome_values, color=["#2563eb", "#f97316", "#94a3b8"])
    axes[1].set_title("Pairwise Wins")
    axes[1].set_ylabel("Cases")
    axes[1].grid(axis="y", alpha=0.25)
    for idx, value in enumerate(outcome_values):
        axes[1].text(idx, value, str(value), ha="center", va="bottom")

    fig.suptitle(f"PRAG Memory Variant Comparison (n={summary['cases']})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[PRAG:compare] plot saved: {path}")


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
    parser.add_argument("--quality-tie-threshold", type=float, default=0.025)
    parser.add_argument("--output-json", default=None, help="Optional path to write detailed JSON metrics.")
    parser.add_argument("--output-csv", default=None, help="Optional path to write per-case CSV metrics.")
    parser.add_argument("--plot-path", default=None, help="Optional path to write a PNG comparison figure.")
    parser.add_argument("--report-dir", default=None, help="Write JSON, CSV, and PNG report files into this directory.")
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
        "qp_relation": [],
        "ponly_relation": [],
        "qp_f1": [],
        "ponly_f1": [],
        "qp_quality": [],
        "ponly_quality": [],
    }
    pairwise_counts = {"question+passage": 0, "passage-only": 0, "tie": 0}
    records: list[dict] = []
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
        winner = compare_quality(qp_result["quality_score"], ponly_result["quality_score"], args.quality_tie_threshold)
        pairwise_counts[winner] += 1
        print_case_report(case, direct, no_memory, qp_result, ponly_result, winner)

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
        totals["qp_relation"].append(qp_result["relation_coverage"])
        totals["ponly_relation"].append(ponly_result["relation_coverage"])
        totals["qp_f1"].append(qp_result["answer_char_f1"])
        totals["ponly_f1"].append(ponly_result["answer_char_f1"])
        totals["qp_quality"].append(qp_result["quality_score"])
        totals["ponly_quality"].append(ponly_result["quality_score"])
        records.append(
            {
                "case": case["name"],
                "source_id": case.get("source_id") or "",
                "question": case["question"],
                "expected": case.get("full_answer") or case["main_answer"],
                "phrase_targets": " | ".join(answer_phrase_candidates(case["main_answer"])),
                "relation_terms": " | ".join(relation_terms(case)),
                "direct_hit": answer_hit(direct, case),
                "no_memory_hit": answer_hit(no_memory, case),
                "qp_hit": qp_result["hit"],
                "ponly_hit": ponly_result["hit"],
                "qp_relation": qp_result["relation_coverage"],
                "ponly_relation": ponly_result["relation_coverage"],
                "qp_answer_char_f1": qp_result["answer_char_f1"],
                "ponly_answer_char_f1": ponly_result["answer_char_f1"],
                "qp_quality": qp_result["quality_score"],
                "ponly_quality": ponly_result["quality_score"],
                "qp_loss": qp_result["loss"],
                "ponly_loss": ponly_result["loss"],
                "qp_phrase_loss": qp_result["phrase_loss"],
                "ponly_phrase_loss": ponly_result["phrase_loss"],
                "pairwise_winner": winner,
                "qp_answer": qp_result["answer"],
                "ponly_answer": ponly_result["answer"],
            }
        )

    denom = max(totals["count"], 1)

    def avg(values: list[float]) -> str:
        return "n/a" if not values else f"{sum(values) / len(values):.4f}"

    print("\n" + "=" * 96)
    print("[summary]")
    print(f"cases={totals['count']}")
    print(f"direct_RAG_hit_rate={totals['direct_hits'] / denom:.3f}")
    print(f"no_memory_hit_rate={totals['no_memory_hits'] / denom:.3f}")
    print(
        f"question+passage_hit_rate={totals['qp_hits'] / denom:.3f} "
        f"avg_relation={avg(totals['qp_relation'])} avg_f1={avg(totals['qp_f1'])} "
        f"avg_quality={avg(totals['qp_quality'])} avg_loss={avg(totals['qp_loss'])} "
        f"avg_phrase={avg(totals['qp_phrase'])}"
    )
    print(
        f"passage-only_hit_rate={totals['ponly_hits'] / denom:.3f} "
        f"avg_relation={avg(totals['ponly_relation'])} avg_f1={avg(totals['ponly_f1'])} "
        f"avg_quality={avg(totals['ponly_quality'])} avg_loss={avg(totals['ponly_loss'])} "
        f"avg_phrase={avg(totals['ponly_phrase'])}"
    )
    print(
        "pairwise="
        f"question+passage:{pairwise_counts['question+passage']} "
        f"passage-only:{pairwise_counts['passage-only']} tie:{pairwise_counts['tie']}"
    )

    def avg_float(values: list[float]) -> float:
        return 0.0 if not values else float(sum(values) / len(values))

    summary = {
        "cases": totals["count"],
        "direct_RAG_hit_rate": totals["direct_hits"] / denom,
        "no_memory_hit_rate": totals["no_memory_hits"] / denom,
        "question+passage": {
            "hit_rate": totals["qp_hits"] / denom,
            "avg_relation_coverage": avg_float(totals["qp_relation"]),
            "avg_answer_char_f1": avg_float(totals["qp_f1"]),
            "avg_quality_score": avg_float(totals["qp_quality"]),
            "avg_loss": avg_float(totals["qp_loss"]),
            "avg_phrase_loss": avg_float(totals["qp_phrase"]),
        },
        "passage-only": {
            "hit_rate": totals["ponly_hits"] / denom,
            "avg_relation_coverage": avg_float(totals["ponly_relation"]),
            "avg_answer_char_f1": avg_float(totals["ponly_f1"]),
            "avg_quality_score": avg_float(totals["ponly_quality"]),
            "avg_loss": avg_float(totals["ponly_loss"]),
            "avg_phrase_loss": avg_float(totals["ponly_phrase"]),
        },
        "pairwise": pairwise_counts,
        "quality_tie_threshold": args.quality_tie_threshold,
    }
    payload = {
        "summary": summary,
        "records": records,
    }

    report_dir = Path(args.report_dir) if args.report_dir else None
    json_path = Path(args.output_json) if args.output_json else (report_dir / "memory_variant_comparison.json" if report_dir else None)
    csv_path = Path(args.output_csv) if args.output_csv else (report_dir / "memory_variant_comparison.csv" if report_dir else None)
    plot_path = Path(args.plot_path) if args.plot_path else (report_dir / "memory_variant_comparison.png" if report_dir else None)
    if json_path:
        write_json(json_path, payload)
        print(f"[PRAG:compare] json saved: {json_path}")
    if csv_path:
        write_csv(csv_path, records)
        print(f"[PRAG:compare] csv saved: {csv_path}")
    if plot_path:
        plot_report(plot_path, summary)


if __name__ == "__main__":
    main()
