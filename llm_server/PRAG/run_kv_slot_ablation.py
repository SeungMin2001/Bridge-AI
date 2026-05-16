"""Run num_kv ablations and summarize them like a paper table."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def orthomerge_output_path(path: Path) -> Path:
    name = path.name
    if "_memory_" in name:
        name = name.replace("_memory_", "_orthomerge_memory_", 1)
    else:
        name = f"{path.stem}_orthomerge{path.suffix}"
    return path.with_name(name)


def tagged_output_path(path: Path, suffix: str) -> Path:
    suffix = str(suffix or "").strip().strip("_")
    if not suffix:
        return path
    name = path.name
    if "_memory_" in name:
        name = name.replace("_memory_", f"_{suffix}_memory_", 1)
    else:
        name = f"{path.stem}_{suffix}{path.suffix}"
    return path.with_name(name)


def checkpoint_path_for_suffix(output_suffix: str) -> Path:
    base = Path("llm_server/PRAG/prag_memory_checkpoint.pt")
    return tagged_output_path(orthomerge_output_path(base), output_suffix)


def run_command(cmd: list[str], *, dry_run: bool = False) -> None:
    print(f"\n[PRAG:kv-ablation] $ {' '.join(cmd)}", flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def parse_k_values(text: str) -> list[int]:
    values = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start <= 0 or end <= 0 or end < start:
                raise ValueError(f"Invalid num_kv range: {chunk}")
            values.extend(range(start, end + 1))
            continue
        value = int(chunk)
        if value <= 0:
            raise ValueError(f"num_kv must be positive: {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one --k-values entry is required.")
    return values


def read_summary(report_dir: Path) -> dict:
    path = report_dir / "memory_variant_comparison.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def read_best_layer(path: str | Path, *, label: str) -> int:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    layers = data.get("critical_layers") or []
    if not layers:
        raise RuntimeError(f"No critical_layers found in {path} for {label}")
    layer = int(layers[0])
    print(f"[PRAG:kv-ablation] {label} critical_layer={layer} from {path}", flush=True)
    return layer


def scan_variant(args: argparse.Namespace, *, output: Path, num_kv: int, question_conditioned: bool, question_fusion: str) -> int:
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.find_critical_layers",
        "--model",
        args.model,
        "--train",
        args.train,
        "--valid",
        args.valid,
        "--layers",
        args.scan_layers,
        "--steps",
        str(args.scan_steps),
        "--top-n",
        "5",
        "--lr",
        str(args.scan_lr),
        "--answer-target",
        "full_answer",
        "--final-weight",
        "1.0",
        "--injection-mode",
        args.injection_mode,
        "--output",
        str(output),
        "--num-kv",
        str(num_kv),
        "--question-fusion",
        question_fusion,
    ]
    cmd.append("--question-conditioned-memory" if question_conditioned else "--no-question-conditioned-memory")
    cmd.append("--resume" if args.resume else "--no-resume")
    run_command(cmd, dry_run=args.dry_run)
    return 0 if args.dry_run else read_best_layer(output, label="question+passage" if question_conditioned else "passage-only")


def train_variant(
    args: argparse.Namespace,
    *,
    output_suffix: str,
    num_kv: int,
    critical_layer: int,
    question_conditioned: bool,
    question_fusion: str,
) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.train",
        "--model",
        args.model,
        "--output-suffix",
        output_suffix,
        "--num-kv",
        str(num_kv),
        "--train",
        args.train,
        "--valid",
        args.valid,
        "--merge-aware",
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--positive-only",
        "--answer-target",
        "full_answer",
        "--final-weight",
        "1.0",
        "--example-weight",
        str(args.example_weight),
        "--group-weight",
        str(args.group_weight),
        "--merge-weight",
        str(args.merge_weight),
        "--merge-max-passages",
        str(args.merge_max_passages),
        "--answer-phrase-weight",
        "7.0",
        "--answer-prefix-weight",
        "0.0",
        "--short-answer-weight",
        "0.0",
        "--eval-generation-samples",
        "20",
        "--eval-generation-every",
        "1000",
        "--eval-generation-max-new-tokens",
        "64",
        "--injection-mode",
        args.injection_mode,
        "--critical-layer",
        str(critical_layer),
        "--question-fusion",
        question_fusion,
    ]
    cmd.append("--question-conditioned-memory" if question_conditioned else "--no-question-conditioned-memory")
    cmd.append("--resume" if args.resume else "--no-resume")
    run_command(cmd, dry_run=args.dry_run)
    return checkpoint_path_for_suffix(output_suffix)


def compare_against_fixed_baseline(args: argparse.Namespace, *, qp_weights: Path, ponly_weights: Path, report_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.compare_memory_variants",
        "--qp-weights",
        str(qp_weights),
        "--ponly-weights",
        str(ponly_weights),
        "--data",
        args.test_data,
        "--case-index",
        "0",
        "--max-cases",
        str(args.test_max_cases),
        "--dataset-merge-max-passages",
        str(args.merge_max_passages),
        "--max-new-tokens",
        str(args.test_max_new_tokens),
        "--prompt-style",
        args.test_prompt_style,
        "--injection-mode",
        args.injection_mode,
        "--alpha",
        "1.0",
        "--report-dir",
        str(report_dir),
    ]
    if args.include_loss:
        cmd.append("--include-loss")
    if args.quiet_eval:
        cmd.append("--quiet-cases")
    cmd.append("--resume" if args.resume else "--no-resume")
    run_command(cmd, dry_run=args.dry_run)


def prepare_fixed_ponly_baseline(args: argparse.Namespace) -> Path:
    if args.fixed_ponly_weights:
        path = Path(args.fixed_ponly_weights)
        if not args.dry_run and not path.exists():
            raise FileNotFoundError(f"Fixed passage-only baseline checkpoint not found: {path}")
        print(f"[PRAG:kv-ablation] fixed passage-only baseline={path}", flush=True)
        return path

    k = args.ponly_baseline_num_kv
    suffix = f"{args.suffix_prefix}_ponly_baseline_kv{k}_ep{args.epochs}"
    scan_path = Path(args.scan_root) / f"critical_layers_{suffix}.json"
    layer = scan_variant(
        args,
        output=scan_path,
        num_kv=k,
        question_conditioned=False,
        question_fusion=args.ponly_question_fusion,
    )
    return train_variant(
        args,
        output_suffix=suffix,
        num_kv=k,
        critical_layer=layer,
        question_conditioned=False,
        question_fusion=args.ponly_question_fusion,
    )


def run_qp_only(args: argparse.Namespace, k: int, ponly_weights: Path, report_dir: Path) -> None:
    qp_suffix = f"{args.suffix_prefix}_qp_kvadapt_kv{k}_ep{args.epochs}"
    qp_scan = Path(args.scan_root) / f"critical_layers_{qp_suffix}.json"
    qp_layer = scan_variant(
        args,
        output=qp_scan,
        num_kv=k,
        question_conditioned=True,
        question_fusion=args.qp_question_fusion,
    )
    qp_weights = train_variant(
        args,
        output_suffix=qp_suffix,
        num_kv=k,
        critical_layer=qp_layer,
        question_conditioned=True,
        question_fusion=args.qp_question_fusion,
    )
    compare_against_fixed_baseline(args, qp_weights=qp_weights, ponly_weights=ponly_weights, report_dir=report_dir)


def run_both(args: argparse.Namespace, k: int, report_dir: Path) -> None:
    qp_suffix = f"{args.suffix_prefix}_qp_kvadapt_kv{k}_ep{args.epochs}"
    ponly_suffix = f"{args.suffix_prefix}_ponly_kv{k}_ep{args.epochs}"
    qp_scan = Path(args.scan_root) / f"critical_layers_{qp_suffix}.json"
    ponly_scan = Path(args.scan_root) / f"critical_layers_{ponly_suffix}.json"
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.run_related_merge_experiment",
        "--num-kv",
        str(k),
        "--train",
        args.train,
        "--valid",
        args.valid,
        "--test-data",
        args.test_data,
        "--qp-question-fusion",
        args.qp_question_fusion,
        "--ponly-question-fusion",
        args.ponly_question_fusion,
        "--qp-output-suffix",
        qp_suffix,
        "--ponly-output-suffix",
        ponly_suffix,
        "--qp-scan-output",
        str(qp_scan),
        "--ponly-scan-output",
        str(ponly_scan),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--scan-steps",
        str(args.scan_steps),
        "--scan-layers",
        args.scan_layers,
        "--example-weight",
        str(args.example_weight),
        "--group-weight",
        str(args.group_weight),
        "--merge-weight",
        str(args.merge_weight),
        "--merge-max-passages",
        str(args.merge_max_passages),
        "--test-max-cases",
        str(args.test_max_cases),
        "--test-max-new-tokens",
        str(args.test_max_new_tokens),
        "--test-prompt-style",
        args.test_prompt_style,
        "--report-dir",
        str(report_dir),
    ]
    if args.include_loss:
        cmd.append("--include-loss")
    if args.quiet_eval:
        cmd.append("--quiet-eval")
    if args.resume:
        cmd.append("--resume")
    if args.dry_run:
        cmd.append("--dry-run")
    print(f"\n[PRAG:kv-ablation] k={k} $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def collect_row(k: int, report_dir: Path) -> dict:
    summary = read_summary(report_dir)
    qp = summary["question+passage"]
    ponly = summary["passage-only"]
    return {
        "num_kv": k,
        "cases": summary.get("cases", 0),
        "qp_mergeprag_hit": float(qp.get("mergeprag_hit_rate", 0.0)),
        "ponly_mergeprag_hit": float(ponly.get("mergeprag_hit_rate", 0.0)),
        "delta_mergeprag_hit": float(qp.get("mergeprag_hit_rate", 0.0)) - float(ponly.get("mergeprag_hit_rate", 0.0)),
        "qp_token_f1": float(qp.get("avg_answer_token_f1", 0.0)),
        "ponly_token_f1": float(ponly.get("avg_answer_token_f1", 0.0)),
        "delta_token_f1": float(qp.get("avg_answer_token_f1", 0.0)) - float(ponly.get("avg_answer_token_f1", 0.0)),
        "qp_qa_score": float(qp.get("avg_qa_score", 0.0)),
        "ponly_qa_score": float(ponly.get("avg_qa_score", 0.0)),
        "delta_qa_score": float(qp.get("avg_qa_score", 0.0)) - float(ponly.get("avg_qa_score", 0.0)),
        "qp_time_s": float(qp.get("avg_inference_time_s", 0.0)),
        "ponly_time_s": float(ponly.get("avg_inference_time_s", 0.0)),
        "report_dir": str(report_dir),
    }


def write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "num_kv",
        "cases",
        "qp_mergeprag_hit",
        "ponly_mergeprag_hit",
        "delta_mergeprag_hit",
        "qp_token_f1",
        "ponly_token_f1",
        "delta_token_f1",
        "qp_qa_score",
        "ponly_qa_score",
        "delta_qa_score",
        "qp_time_s",
        "ponly_time_s",
        "report_dir",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_ablation(path: Path, rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency.
        print(f"[PRAG:kv-ablation] matplotlib unavailable; skipped plot ({exc})")
        return

    rows = sorted(rows, key=lambda row: row["num_kv"])
    x = [row["num_kv"] for row in rows]
    qp_hit = [100 * row["qp_mergeprag_hit"] for row in rows]
    ponly_hit = [100 * row["ponly_mergeprag_hit"] for row in rows]
    qp_f1 = [100 * row["qp_token_f1"] for row in rows]
    ponly_f1 = [100 * row["ponly_token_f1"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.suptitle("Ablation on Number of K/V Memory Slots", fontsize=15, fontweight="bold")
    colors = {"qp": "#2f5f9f", "ponly": "#c66a2e"}

    axes[0].plot(x, qp_hit, marker="o", linewidth=2.4, color=colors["qp"], label="Question+Passage")
    axes[0].plot(x, ponly_hit, marker="s", linewidth=2.4, color=colors["ponly"], label="Passage-only")
    axes[0].set_title("(a) MergePRAG Hit")
    axes[0].set_xlabel("# K/V slots")
    axes[0].set_ylabel("Hit Rate (%)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(v) for v in x])
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(x, qp_f1, marker="o", linewidth=2.4, color=colors["qp"], label="Question+Passage")
    axes[1].plot(x, ponly_f1, marker="s", linewidth=2.4, color=colors["ponly"], label="Passage-only")
    axes[1].set_title("(b) Token F1")
    axes[1].set_xlabel("# K/V slots")
    axes[1].set_ylabel("F1 (%)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(v) for v in x])
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PRAG num_kv ablation and build a Table-7-style summary.")
    parser.add_argument("--k-values", default="1,2,4,8,16,32")
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--suffix-prefix", default="entity_simple4000")
    parser.add_argument("--report-root", default="llm_server/PRAG/reports/kv_slot_ablation")
    parser.add_argument("--scan-root", default="llm_server/PRAG")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--injection-mode", choices=("attention", "add_all", "add_last", "hybrid"), default="attention")
    parser.add_argument(
        "--ablate-target",
        choices=("qp", "both"),
        default="qp",
        help="qp: vary num_kv only for question+passage and use a fixed passage-only baseline. both: vary both variants.",
    )
    parser.add_argument(
        "--fixed-ponly-weights",
        default="",
        help="Existing passage-only checkpoint used as the fixed baseline for --ablate-target qp.",
    )
    parser.add_argument(
        "--ponly-baseline-num-kv",
        type=int,
        default=32,
        help="If --fixed-ponly-weights is omitted, train one fixed passage-only baseline with this num_kv.",
    )
    parser.add_argument("--qp-question-fusion", choices=("auto", "none", "text_concat", "feature_concat", "kv_adapter"), default="kv_adapter")
    parser.add_argument("--ponly-question-fusion", choices=("auto", "none", "text_concat", "feature_concat", "kv_adapter"), default="none")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--scan-lr", type=float, default=1e-4)
    parser.add_argument("--scan-steps", type=int, default=60)
    parser.add_argument("--scan-layers", default="all")
    parser.add_argument("--example-weight", type=float, default=0.0)
    parser.add_argument("--group-weight", type=float, default=0.0)
    parser.add_argument("--merge-weight", type=float, default=1.0)
    parser.add_argument("--merge-max-passages", type=int, default=4)
    parser.add_argument("--test-max-cases", type=int, default=400)
    parser.add_argument("--test-max-new-tokens", type=int, default=128)
    parser.add_argument("--test-prompt-style", choices=("service", "short-chat", "memory-cued", "paper"), default="service")
    parser.add_argument("--include-loss", action="store_true")
    parser.add_argument("--quiet-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = []
    report_root = Path(args.report_root)
    fixed_ponly_weights = None
    if args.ablate_target == "qp":
        fixed_ponly_weights = prepare_fixed_ponly_baseline(args)
    for k in parse_k_values(args.k_values):
        report_dir = report_root / f"kv{k}"
        if args.ablate_target == "qp":
            run_qp_only(args, k, fixed_ponly_weights, report_dir)
        else:
            run_both(args, k, report_dir)
        if not args.dry_run:
            rows.append(collect_row(k, report_dir))

    if args.dry_run:
        print("[PRAG:kv-ablation] dry-run complete")
        return

    table_path = report_root / "kv_slot_ablation_table.csv"
    json_path = report_root / "kv_slot_ablation_table.json"
    plot_path = report_root / "kv_slot_ablation_curves.png"
    write_table(table_path, rows)
    json_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_ablation(plot_path, rows)
    best = max(rows, key=lambda row: (row["qp_mergeprag_hit"], row["qp_token_f1"], row["qp_qa_score"]))
    print(f"[PRAG:kv-ablation] best_qp_num_kv={best['num_kv']} hit={best['qp_mergeprag_hit']:.4f} f1={best['qp_token_f1']:.4f}")
    print(f"[PRAG:kv-ablation] table saved: {table_path}")
    print(f"[PRAG:kv-ablation] json saved: {json_path}")
    print(f"[PRAG:kv-ablation] plot saved: {plot_path}")


if __name__ == "__main__":
    main()
