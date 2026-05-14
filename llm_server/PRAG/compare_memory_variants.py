"""Compare question+passage and passage-only PRAG memories on the same cases."""

from __future__ import annotations

import argparse
import csv
import json
import string
import re
import time
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
DEFAULT_NATURAL_TEST_PATH = Path(__file__).resolve().parents[2] / "data" / "PRAG_natural_service_test_450.jsonl"


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
    if split == "test":
        return str(DEFAULT_NATURAL_TEST_PATH)
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
ENGLISH_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
PUNCT_TABLE = str.maketrans("", "", string.punctuation + "“”‘’·…，。！？、；：「」『』（）《》〈〉【】")
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


def normalize_qa_answer(text: str) -> str:
    """SQuAD/FiD-style answer normalization for EM/F1.

    The original QA literature lowercases, removes punctuation/articles, and
    collapses whitespace. We additionally use casefold for Unicode text.
    """

    normalized = str(text or "").casefold().translate(PUNCT_TABLE)
    normalized = ENGLISH_ARTICLES_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def compact_qa_answer(text: str) -> str:
    """Whitespace-insensitive form used only for Korean alias matching."""

    return normalize_qa_answer(text).replace(" ", "")


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


def accepted_answers(case: dict) -> list[str]:
    """Reference aliases, matching FiD's accepted-answer-list evaluation style."""

    candidates: list[str] = []
    for value in (
        case.get("full_answer"),
        case.get("main_answer"),
        *(case.get("hit_phrases") or []),
        *answer_phrase_candidates(case.get("main_answer", "")),
    ):
        if value:
            candidates.append(str(value))

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = compact_qa_answer(candidate)
        if key and key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def normalized_exact_match(prediction: str, reference: str) -> bool:
    pred = normalize_qa_answer(prediction)
    ref = normalize_qa_answer(reference)
    if pred == ref:
        return True
    # Korean service answers often differ only in spacing.
    return compact_qa_answer(prediction) == compact_qa_answer(reference)


def max_normalized_em(prediction: str, references: list[str]) -> bool:
    return any(normalized_exact_match(prediction, reference) for reference in references)


def answer_value_hit(prediction: str, references: list[str]) -> bool:
    pred_key = compact_qa_answer(prediction)
    return any((ref_key := compact_qa_answer(reference)) and ref_key in pred_key for reference in references)


def token_f1_against_reference(prediction: str, reference: str) -> float:
    pred_tokens = eval_tokens(normalize_qa_answer(prediction))
    ref_tokens = eval_tokens(normalize_qa_answer(reference))
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / max(len(ref_tokens), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def max_token_f1(prediction: str, references: list[str]) -> float:
    if not references:
        return 0.0
    return max(token_f1_against_reference(prediction, reference) for reference in references)


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


PROMPT_LEAK_MARKERS = (
    "질문:",
    "답변:",
    "question:",
    "answer:",
    "passage:",
    "options:",
    "expected:",
    "-----",
)


def has_prompt_leak(answer: str) -> bool:
    lowered = str(answer or "").casefold()
    return any(marker in lowered for marker in PROMPT_LEAK_MARKERS)


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
    references = accepted_answers(case)
    normalized_em = max_normalized_em(answer, references)
    qa_token_f1 = max_token_f1(answer, references)
    # Literature-aligned QA score: common PRAG/RAG papers report EM and/or F1.
    qa_score = 0.50 * float(normalized_em) + 0.50 * qa_token_f1
    return {
        "normalized_em": normalized_em,
        "answer_token_f1": qa_token_f1,
        "qa_score": qa_score,
    }


def recall_references(case: dict) -> list[str]:
    """Gold answer values for Recall@K over merged passages.

    Unlike generation scoring, retrieval recall should check whether the
    evidence passages contain the answer value, not necessarily the full
    natural-language answer sentence.
    """

    candidates: list[str] = []
    for value in (
        *(case.get("hit_phrases") or []),
        *answer_phrase_candidates(case.get("main_answer", "")),
        case.get("main_answer"),
    ):
        if value:
            candidates.append(str(value))

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = compact_qa_answer(candidate)
        if key and key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def recall_at_k(case: dict, passages: list[str]) -> bool:
    """Whether the selected/merged top-K passages contain the gold answer."""

    return answer_value_hit("\n".join(passages), recall_references(case))


def compare_quality(qp_score: float, ponly_score: float, tie_threshold: float) -> str:
    delta = qp_score - ponly_score
    if abs(delta) <= tie_threshold:
        return "tie"
    return "question+passage" if delta > 0 else "passage-only"


def sync_if_cuda(device) -> None:
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def merged_passages_for_case(case: dict) -> list[str]:
    candidates = [case.get("main_passage", "")] + list(case.get("merge_passages") or [])
    passages: list[str] = []
    seen: set[str] = set()
    for passage in candidates:
        text = str(passage or "").strip()
        if not text:
            continue
        key = normalize_eval_text(text)
        if key in seen:
            continue
        seen.add(key)
        passages.append(text)
    return passages


@torch.no_grad()
def evaluate_variant(
    *,
    variant: Variant,
    model,
    tokenizer,
    device,
    case: dict,
    max_new_tokens: int,
    alpha: float,
    prompt_style: str,
    injection_mode: str,
    answer_prefix_tokens: int,
    include_loss: bool,
) -> dict:
    question = case["question"]
    main_answer = case["main_answer"]
    full_answer = case.get("full_answer") or main_answer
    main_passages = merged_passages_for_case(case)
    sync_if_cuda(device)
    started_at = time.perf_counter()
    # Each variant may have been trained with a different critical layer.
    target_layer = model.model.layers[variant.layer_idx]
    memory = encode_merged_memory(
        model,
        tokenizer,
        variant.hypernet,
        main_passages,
        device,
        question=question,
        question_conditioned=variant.question_conditioned,
    )
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
    sync_if_cuda(device)
    inference_time_s = time.perf_counter() - started_at

    kv_loss = prefix_loss = phrase_loss = None
    if include_loss:
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
    scores = score_answer(generated, case)
    return {
        "loss": None if kv_loss is None else float(kv_loss.item()),
        "prefix_loss": None if prefix_loss is None else float(prefix_loss.item()),
        "phrase_loss": None if phrase_loss is None else float(phrase_loss.item()),
        "answer": generated,
        "merged_count": len(main_passages),
        "inference_time_s": inference_time_s,
        **scores,
    }


def print_case_report(
    case: dict,
    direct: str | None,
    no_memory: str | None,
    qp: dict,
    ponly: dict,
    winner: str,
    recall_hit: bool,
    recall_k: int,
) -> None:
    main_passages = merged_passages_for_case(case)
    print("\n" + "=" * 96)
    print(f"[case:{case['name']}]")
    if case.get("source_id"):
        print(f"source_id: {case['source_id']}")
    print(f"question: {case['question']}")
    print(f"expected: {case.get('full_answer') or case['main_answer']}")
    if case.get("root_source_id"):
        print(f"root_source_id: {case['root_source_id']}")
    if case.get("merge_group_source_id"):
        print(f"merge_group_source_id: {case['merge_group_source_id']}")
    if case.get("merge_policy"):
        print(f"merge_policy: {case['merge_policy']}")
    print(f"same_source_merge: {bool(case.get('same_source_merge'))}")
    print(f"merged_passages: {len(main_passages)}")
    print(f"recall@{recall_k}: {recall_hit}")
    for idx, passage in enumerate(main_passages, start=1):
        print(f"  passage[{idx}]: {passage}")

    if direct is not None and no_memory is not None:
        direct_scores = score_answer(direct, case)
        no_memory_scores = score_answer(no_memory, case)
        print("\n[baseline]")
        print(f"direct_RAG | em={direct_scores['normalized_em']} | f1={fmt(direct_scores['answer_token_f1'])}")
        print(f"  answer: {direct}")
        print(f"no_memory  | em={no_memory_scores['normalized_em']} | f1={fmt(no_memory_scores['answer_token_f1'])}")
        print(f"  answer: {no_memory}")

    print("\n[variant comparison]")
    print(
        "question+passage | "
        f"em={qp['normalized_em']} | f1={fmt(qp['answer_token_f1'])} | "
        f"qa_score={fmt(qp['qa_score'])} | time={fmt(qp['inference_time_s'])}s"
    )
    print(f"  answer: {qp['answer']}")
    print(
        "passage-only     | "
        f"em={ponly['normalized_em']} | f1={fmt(ponly['answer_token_f1'])} | "
        f"qa_score={fmt(ponly['qa_score'])} | time={fmt(ponly['inference_time_s'])}s"
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
        from matplotlib.ticker import FormatStrFormatter
    except Exception as exc:
        print(f"[PRAG:compare] matplotlib unavailable; skipped plot ({exc})")
        return

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    qp_color = "#2f5f9f"
    ponly_color = "#c66a2e"
    grid_color = "#d7dce2"

    metric_specs = [
        ("normalized_em_rate", "EM"),
        ("avg_answer_token_f1", "Token F1"),
        ("recall_at_k", "Recall@K"),
    ]

    def values_for(specs: list[tuple[str, str]], label: str) -> list[float]:
        return [float(summary[label][key]) for key, _ in specs]

    def style_axis(ax) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9aa3af")
        ax.spines["bottom"].set_color("#9aa3af")
        ax.grid(axis="y", color=grid_color, linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)

    def annotate_bars(ax, bars, offset: float = 0.012) -> None:
        upper = ax.get_ylim()[1]
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + upper * offset,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#111827",
            )

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), gridspec_kw={"width_ratios": [2.1, 1.0]})
    fig.patch.set_facecolor("white")
    ax = axes[0]

    width = 0.34
    x = list(range(len(metric_specs)))
    qp_values = values_for(metric_specs, "question+passage")
    ponly_values = values_for(metric_specs, "passage-only")
    bars_qp = ax.bar(
        [i - width / 2 for i in x],
        qp_values,
        width,
        label="Question + Passage",
        color=qp_color,
        edgecolor="#1f2937",
        linewidth=0.35,
    )
    bars_ponly = ax.bar(
        [i + width / 2 for i in x],
        ponly_values,
        width,
        label="Passage Only",
        color=ponly_color,
        edgecolor="#1f2937",
        linewidth=0.35,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_specs])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Accuracy and Retrieval")
    annotate_bars(ax, bars_qp)
    annotate_bars(ax, bars_ponly)
    style_axis(ax)

    ax_time = axes[1]
    time_specs = [("avg_inference_time_s", "Avg. Time")]
    time_x = list(range(len(time_specs)))
    time_qp = values_for(time_specs, "question+passage")
    time_ponly = values_for(time_specs, "passage-only")
    bars_time_qp = ax_time.bar(
        [i - width / 2 for i in time_x],
        time_qp,
        width,
        label="Question + Passage",
        color=qp_color,
        edgecolor="#1f2937",
        linewidth=0.35,
    )
    bars_time_ponly = ax_time.bar(
        [i + width / 2 for i in time_x],
        time_ponly,
        width,
        label="Passage Only",
        color=ponly_color,
        edgecolor="#1f2937",
        linewidth=0.35,
    )
    ax_time.set_xticks(time_x)
    ax_time.set_xticklabels([label for _, label in time_specs])
    ax_time.set_ylabel("Seconds (lower is better)")
    ax_time.set_title("Inference Time")
    annotate_bars(ax_time, bars_time_qp, offset=0.018)
    annotate_bars(ax_time, bars_time_ponly, offset=0.018)
    style_axis(ax_time)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"Question+Passage vs Passage-Only (n={summary['cases']})", y=1.12, fontweight="bold")
    fig.tight_layout(pad=1.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[PRAG:compare] plot saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qp-weights", default=str(DEFAULT_QP_WEIGHTS), help="question+passage checkpoint/weights path.")
    parser.add_argument("--ponly-weights", default=str(DEFAULT_PONLY_WEIGHTS), help="passage-only checkpoint/weights path.")
    parser.add_argument("--data", default=None, help="Augmented JSONL path. Use ';' to compare over mixed datasets.")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--dataset-merge-max-passages", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--prompt-style", choices=("service", "short-chat", "memory-cued", "paper"), default="service")
    parser.add_argument("--injection-mode", choices=("attention", "add_all", "add_last", "hybrid"), default="attention")
    parser.add_argument("--answer-prefix-tokens", type=int, default=3)
    parser.add_argument("--quality-tie-threshold", type=float, default=0.025)
    parser.add_argument("--include-baselines", action="store_true", help="Also generate direct-RAG and no-memory baselines.")
    parser.add_argument("--include-loss", action="store_true", help="Also compute CE diagnostics. Slower; off by default.")
    parser.add_argument("--quiet-cases", action="store_true", help="Suppress per-case text output and write only summary/report files.")
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

    data_path = args.data or default_mixed_data(args.split)
    cases = load_dataset_cases(
        data_path,
        case_index=args.case_index,
        max_cases=args.max_cases,
        merge_max_passages=args.dataset_merge_max_passages,
    )

    print("[PRAG:compare-memory-variants]")
    print(f"model={model_name} qp_layer={qp.layer_idx} ponly_layer={ponly.layer_idx} data={data_path}")
    print(
        f"question+passage weights={qp.path} step={qp.step} best_val={fmt(qp.best_val)} "
        f"layer={qp.layer_idx} q_cond={qp.question_conditioned}"
    )
    print(
        f"passage-only     weights={ponly.path} step={ponly.step} best_val={fmt(ponly.best_val)} "
        f"layer={ponly.layer_idx} q_cond={ponly.question_conditioned}"
    )

    totals = {
        "count": 0,
        "direct_em": 0,
        "no_memory_em": 0,
        "direct_token_f1": [],
        "no_memory_token_f1": [],
        "qp_em": 0,
        "ponly_em": 0,
        "qp_loss": [],
        "ponly_loss": [],
        "qp_phrase": [],
        "ponly_phrase": [],
        "qp_token_f1": [],
        "ponly_token_f1": [],
        "qp_qa_score": [],
        "ponly_qa_score": [],
        "qp_inference_time_s": [],
        "ponly_inference_time_s": [],
        "recall_hits": 0,
        "same_source_merges": 0,
    }
    pairwise_counts = {"question+passage": 0, "passage-only": 0, "tie": 0}
    records: list[dict] = []
    for case in cases:
        question = case["question"]
        main_passages = merged_passages_for_case(case)
        recall_hit = recall_at_k(case, main_passages)
        direct = no_memory = None
        direct_result = no_memory_result = None
        if args.include_baselines:
            direct_passage = "\n\n".join(main_passages)
            direct = generate_direct_passage(model, tokenizer, question, direct_passage, device, args.max_new_tokens)
            no_memory = generate_plain(model, tokenizer, question, device, args.max_new_tokens)
            direct_result = score_answer(direct, case)
            no_memory_result = score_answer(no_memory, case)
        qp_result = evaluate_variant(
            variant=qp,
            model=model,
            tokenizer=tokenizer,
            device=device,
            case=case,
            max_new_tokens=args.max_new_tokens,
            alpha=args.alpha,
            prompt_style=args.prompt_style,
            injection_mode=args.injection_mode,
            answer_prefix_tokens=args.answer_prefix_tokens,
            include_loss=args.include_loss,
        )
        ponly_result = evaluate_variant(
            variant=ponly,
            model=model,
            tokenizer=tokenizer,
            device=device,
            case=case,
            max_new_tokens=args.max_new_tokens,
            alpha=args.alpha,
            prompt_style=args.prompt_style,
            injection_mode=args.injection_mode,
            answer_prefix_tokens=args.answer_prefix_tokens,
            include_loss=args.include_loss,
        )
        winner = compare_quality(qp_result["qa_score"], ponly_result["qa_score"], args.quality_tie_threshold)
        pairwise_counts[winner] += 1
        if not args.quiet_cases:
            print_case_report(
                case,
                direct,
                no_memory,
                qp_result,
                ponly_result,
                winner,
                recall_hit,
                args.dataset_merge_max_passages,
            )

        totals["count"] += 1
        totals["recall_hits"] += int(recall_hit)
        totals["same_source_merges"] += int(bool(case.get("same_source_merge")))
        if args.include_baselines and direct_result is not None and no_memory_result is not None:
            totals["direct_em"] += int(direct_result["normalized_em"])
            totals["no_memory_em"] += int(no_memory_result["normalized_em"])
            totals["direct_token_f1"].append(direct_result["answer_token_f1"])
            totals["no_memory_token_f1"].append(no_memory_result["answer_token_f1"])
        totals["qp_em"] += int(qp_result["normalized_em"])
        totals["ponly_em"] += int(ponly_result["normalized_em"])
        if args.include_loss:
            for key, result_key, result in (
                ("qp_loss", "loss", qp_result),
                ("ponly_loss", "loss", ponly_result),
                ("qp_phrase", "phrase_loss", qp_result),
                ("ponly_phrase", "phrase_loss", ponly_result),
            ):
                if result[result_key] is not None:
                    totals[key].append(result[result_key])
        totals["qp_token_f1"].append(qp_result["answer_token_f1"])
        totals["ponly_token_f1"].append(ponly_result["answer_token_f1"])
        totals["qp_qa_score"].append(qp_result["qa_score"])
        totals["ponly_qa_score"].append(ponly_result["qa_score"])
        totals["qp_inference_time_s"].append(qp_result["inference_time_s"])
        totals["ponly_inference_time_s"].append(ponly_result["inference_time_s"])
        records.append(
            {
                "case": case["name"],
                "source_id": case.get("source_id") or "",
                "root_source_id": case.get("root_source_id") or "",
                "merge_group_source_id": case.get("merge_group_source_id") or "",
                "same_source_merge": bool(case.get("same_source_merge")),
                "merge_policy": case.get("merge_policy") or "",
                "merge_group_qas": case.get("merge_group_qas") or "",
                "merged_passage_count": len(main_passages),
                "qp_layer": qp.layer_idx,
                "ponly_layer": ponly.layer_idx,
                "question": case["question"],
                "expected": case.get("full_answer") or case["main_answer"],
                "recall_at_k": recall_hit,
                "direct_normalized_em": "" if direct_result is None else direct_result["normalized_em"],
                "no_memory_normalized_em": "" if no_memory_result is None else no_memory_result["normalized_em"],
                "direct_answer_token_f1": "" if direct_result is None else direct_result["answer_token_f1"],
                "no_memory_answer_token_f1": "" if no_memory_result is None else no_memory_result["answer_token_f1"],
                "qp_normalized_em": qp_result["normalized_em"],
                "ponly_normalized_em": ponly_result["normalized_em"],
                "qp_answer_token_f1": qp_result["answer_token_f1"],
                "ponly_answer_token_f1": ponly_result["answer_token_f1"],
                "qp_qa_score": qp_result["qa_score"],
                "ponly_qa_score": ponly_result["qa_score"],
                "qp_inference_time_s": qp_result["inference_time_s"],
                "ponly_inference_time_s": ponly_result["inference_time_s"],
                "qp_loss": "" if qp_result["loss"] is None else qp_result["loss"],
                "ponly_loss": "" if ponly_result["loss"] is None else ponly_result["loss"],
                "qp_phrase_loss": "" if qp_result["phrase_loss"] is None else qp_result["phrase_loss"],
                "ponly_phrase_loss": "" if ponly_result["phrase_loss"] is None else ponly_result["phrase_loss"],
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
    print(f"same_source_merge_rate={totals['same_source_merges'] / denom:.3f}")
    print(f"recall_at_{args.dataset_merge_max_passages}={totals['recall_hits'] / denom:.3f}")
    if args.include_baselines:
        print(f"direct_RAG_em_rate={totals['direct_em'] / denom:.3f} avg_token_f1={avg(totals['direct_token_f1'])}")
        print(f"no_memory_em_rate={totals['no_memory_em'] / denom:.3f} avg_token_f1={avg(totals['no_memory_token_f1'])}")
    print(
        f"question+passage_em_rate={totals['qp_em'] / denom:.3f} "
        f"avg_token_f1={avg(totals['qp_token_f1'])} "
        f"avg_qa_score={avg(totals['qp_qa_score'])} "
        f"avg_inference_time_s={avg(totals['qp_inference_time_s'])}"
    )
    print(
        f"passage-only_em_rate={totals['ponly_em'] / denom:.3f} "
        f"avg_token_f1={avg(totals['ponly_token_f1'])} "
        f"avg_qa_score={avg(totals['ponly_qa_score'])} "
        f"avg_inference_time_s={avg(totals['ponly_inference_time_s'])}"
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
        "same_source_merge_rate": totals["same_source_merges"] / denom,
        "recall_k": args.dataset_merge_max_passages,
        "recall_at_k": totals["recall_hits"] / denom,
        "question+passage": {
            "normalized_em_rate": totals["qp_em"] / denom,
            "avg_answer_token_f1": avg_float(totals["qp_token_f1"]),
            "avg_qa_score": avg_float(totals["qp_qa_score"]),
            "recall_at_k": totals["recall_hits"] / denom,
            "avg_inference_time_s": avg_float(totals["qp_inference_time_s"]),
        },
        "passage-only": {
            "normalized_em_rate": totals["ponly_em"] / denom,
            "avg_answer_token_f1": avg_float(totals["ponly_token_f1"]),
            "avg_qa_score": avg_float(totals["ponly_qa_score"]),
            "recall_at_k": totals["recall_hits"] / denom,
            "avg_inference_time_s": avg_float(totals["ponly_inference_time_s"]),
        },
        "pairwise": pairwise_counts,
        "quality_tie_threshold": args.quality_tie_threshold,
    }
    if args.include_baselines:
        summary["direct_RAG"] = {
            "normalized_em_rate": totals["direct_em"] / denom,
            "avg_answer_token_f1": avg_float(totals["direct_token_f1"]),
        }
        summary["no_memory"] = {
            "normalized_em_rate": totals["no_memory_em"] / denom,
            "avg_answer_token_f1": avg_float(totals["no_memory_token_f1"]),
        }
    if args.include_loss:
        summary["question+passage"].update({
            "avg_loss": avg_float(totals["qp_loss"]),
            "avg_phrase_loss": avg_float(totals["qp_phrase"]),
        })
        summary["passage-only"].update({
            "avg_loss": avg_float(totals["ponly_loss"]),
            "avg_phrase_loss": avg_float(totals["ponly_phrase"]),
        })
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
