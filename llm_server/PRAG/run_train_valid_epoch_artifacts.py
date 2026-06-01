"""Run train/validation epoch evaluation and create model-specific figures.

This is a thin orchestration script around ``plot_epoch_performance.py``.  It
evaluates the saved Question+Passage and Passage-only epoch checkpoints on the
train and validation datasets, then writes four paper-friendly accuracy plots:

* Question+Passage train accuracy
* Question+Passage validation accuracy
* Passage-only train accuracy
* Passage-only validation accuracy

For the paper figures, ``Accuracy`` is computed from token-level F1 because
this project uses free-form generated answers rather than closed-set labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_RUN_NAME = "entity_simple4000_step3200_lr2em5_kv64_ep10_test400"
DEFAULT_OUTPUT_NAME = "entity_simple4000_step3200_lr2em5_kv64_ep10_train_valid400"

COLORS = {
    "qp": "#2563eb",
    "ponly": "#ea580c",
}
MARKERS = {
    "qp": "o",
    "ponly": "s",
}
LABELS = {
    "qp": "Question+Passage",
    "ponly": "Passage-only",
}


def run_command(cmd: list[str]) -> None:
    print(f"\n[PRAG:train-valid] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, 0)))
    except (TypeError, ValueError):
        return 0


def summary_float(block: dict, *keys: str) -> float:
    for key in keys:
        value = block.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def eval_epoch_performance(args: argparse.Namespace, *, split: str, data: str, max_cases: int, report_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.plot_epoch_performance",
        "--qp-epoch-dir",
        args.qp_epoch_dir,
        "--ponly-epoch-dir",
        args.ponly_epoch_dir,
        "--data",
        data,
        "--case-index",
        str(args.case_index),
        "--max-cases",
        str(max_cases),
        "--dataset-merge-max-passages",
        str(args.dataset_merge_max_passages),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--prompt-style",
        args.prompt_style,
        "--injection-mode",
        args.injection_mode,
        "--alpha",
        str(args.alpha),
        "--report-dir",
        str(report_dir),
        "--scoring-style",
        args.scoring_style,
        "--qp-log",
        args.qp_log,
        "--ponly-log",
        args.ponly_log,
    ]
    if args.quiet_cases:
        cmd.append("--quiet-cases")
    else:
        cmd.append("--no-quiet-cases")
    cmd.append("--resume" if args.resume else "--no-resume")
    print(f"[PRAG:train-valid] split={split} data={data} max_cases={max_cases}", flush=True)
    run_command(cmd)


def eval_epoch_zero(args: argparse.Namespace, *, split: str, data: str, max_cases: int, report_dir: Path) -> None:
    if not args.qp_step0_weights or not args.ponly_step0_weights:
        raise ValueError("--include-epoch0 requires --qp-step0-weights and --ponly-step0-weights.")
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.compare_memory_variants",
        "--qp-weights",
        args.qp_step0_weights,
        "--ponly-weights",
        args.ponly_step0_weights,
        "--data",
        data,
        "--case-index",
        str(args.case_index),
        "--max-cases",
        str(max_cases),
        "--dataset-merge-max-passages",
        str(args.dataset_merge_max_passages),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--prompt-style",
        args.prompt_style,
        "--injection-mode",
        args.injection_mode,
        "--alpha",
        str(args.alpha),
        "--report-dir",
        str(report_dir),
        "--scoring-style",
        args.scoring_style,
    ]
    if args.quiet_cases:
        cmd.append("--quiet-cases")
    cmd.append("--resume" if args.resume else "--no-resume")
    print(f"[PRAG:train-valid] split={split} epoch=0 data={data} max_cases={max_cases}", flush=True)
    run_command(cmd)


def row_from_epoch0_report(path: Path) -> dict[str, str] | None:
    report_path = path / "memory_variant_comparison.json"
    if not report_path.exists():
        return None
    data = json.loads(report_path.read_text(encoding="utf-8"))
    summary = data.get("summary", data)
    qp = summary.get("question+passage", {})
    ponly = summary.get("passage-only", {})
    row = {
        "epoch": "0",
        "qp_mergeprag_hit": summary_float(qp, "mergeprag_hit_rate"),
        "ponly_mergeprag_hit": summary_float(ponly, "mergeprag_hit_rate"),
        "qp_token_precision": summary_float(qp, "avg_token_precision", "avg_answer_token_precision"),
        "ponly_token_precision": summary_float(ponly, "avg_token_precision", "avg_answer_token_precision"),
        "qp_token_recall": summary_float(qp, "avg_token_recall", "avg_answer_token_recall"),
        "ponly_token_recall": summary_float(ponly, "avg_token_recall", "avg_answer_token_recall"),
        "qp_token_f1": summary_float(qp, "avg_token_f1", "avg_answer_token_f1"),
        "ponly_token_f1": summary_float(ponly, "avg_token_f1", "avg_answer_token_f1"),
        "qp_qa_score": summary_float(qp, "avg_qa_score"),
        "ponly_qa_score": summary_float(ponly, "avg_qa_score"),
        "qp_time_s": summary_float(qp, "avg_inference_time_s"),
        "ponly_time_s": summary_float(ponly, "avg_inference_time_s"),
    }
    for metric in ("mergeprag_hit", "token_precision", "token_recall", "token_f1", "qa_score"):
        row[f"delta_{metric}"] = float(row[f"qp_{metric}"]) - float(row[f"ponly_{metric}"])
    return {key: str(value) for key, value in row.items()}


def with_epoch0_rows(rows: list[dict[str, str]], epoch0_report_dir: Path) -> list[dict[str, str]]:
    epoch0 = row_from_epoch0_report(epoch0_report_dir)
    if epoch0 is None:
        return rows
    fieldnames = list(rows[0].keys()) if rows else list(epoch0.keys())
    normalized_epoch0 = {key: epoch0.get(key, "") for key in fieldnames}
    return [normalized_epoch0, *rows]


def plot_single_accuracy(path: Path, rows: list[dict[str, str]], *, model: str, split: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    column = "qp_token_f1" if model == "qp" else "ponly_token_f1"
    epochs = [as_int(row, "epoch") for row in rows]
    values = [100.0 * as_float(row, column) for row in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(
        epochs,
        values,
        color=COLORS[model],
        marker=MARKERS[model],
        linewidth=2.4,
        markersize=5.4,
        label=f"{LABELS[model]} ({split})",
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xticks(epochs)
    ax.grid(True, which="major", color="#e5e7eb", linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=10.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PRAG:train-valid] plot saved: {path}")


def plot_combined_accuracy(path: Path, rows: list[dict[str, str]], *, split: str, max_epoch: int | None = None) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if max_epoch is not None:
        rows = [row for row in rows if as_int(row, "epoch") <= max_epoch]
    epochs = [as_int(row, "epoch") for row in rows]
    qp = [100.0 * as_float(row, "qp_token_f1") for row in rows]
    ponly = [100.0 * as_float(row, "ponly_token_f1") for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, qp, color=COLORS["qp"], marker=MARKERS["qp"], linewidth=2.6, markersize=5.6, label=LABELS["qp"])
    ax.plot(epochs, ponly, color=COLORS["ponly"], marker=MARKERS["ponly"], linewidth=2.6, markersize=5.6, label=LABELS["ponly"])
    ax.text(0.03, 0.94, split.title(), transform=ax.transAxes, fontsize=12, fontweight="bold", color="#111827")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xlim(min(epochs) - 0.15, max(epochs) + 0.15)
    ax.set_xticks(epochs)
    ax.minorticks_on()
    ax.grid(True, which="major", color="#e5e7eb", linewidth=1.0)
    ax.grid(True, which="minor", color="#f3f4f6", linewidth=0.6, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=10.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[PRAG:train-valid] plot saved: {path}")


def build_paper_artifacts(output_root: Path, *, include_epoch0: bool = False, combined_max_epoch: int | None = None) -> None:
    train_csv = output_root / "train_epoch_performance" / "epoch_performance.csv"
    valid_csv = output_root / "validation_epoch_performance" / "epoch_performance.csv"
    train_rows = read_csv_rows(train_csv)
    valid_rows = read_csv_rows(valid_csv)
    if include_epoch0:
        train_rows = with_epoch0_rows(train_rows, output_root / "train_epoch_performance" / "epoch_000")
        valid_rows = with_epoch0_rows(valid_rows, output_root / "validation_epoch_performance" / "epoch_000")

    artifact_dir = output_root / "paper_artifacts"
    specs = [
        ("qp", "train", train_rows, "question_passage_train_accuracy.png"),
        ("qp", "validation", valid_rows, "question_passage_validation_accuracy.png"),
        ("ponly", "train", train_rows, "passage_only_train_accuracy.png"),
        ("ponly", "validation", valid_rows, "passage_only_validation_accuracy.png"),
    ]
    for model, split, rows, filename in specs:
        plot_single_accuracy(artifact_dir / filename, rows, model=model, split=split)
    plot_combined_accuracy(artifact_dir / "train_accuracy_combined.png", train_rows, split="train")
    plot_combined_accuracy(artifact_dir / "validation_accuracy_combined.png", valid_rows, split="validation")
    if combined_max_epoch is not None:
        suffix = f"epoch0_{combined_max_epoch}" if include_epoch0 else f"epoch1_{combined_max_epoch}"
        plot_combined_accuracy(
            artifact_dir / f"train_accuracy_{suffix}_combined.png",
            train_rows,
            split="train",
            max_epoch=combined_max_epoch,
        )
        plot_combined_accuracy(
            artifact_dir / f"validation_accuracy_{suffix}_combined.png",
            valid_rows,
            split="validation",
            max_epoch=combined_max_epoch,
        )

    summary_rows: list[dict] = []
    for split, rows in (("train", train_rows), ("validation", valid_rows)):
        for row in rows:
            summary_rows.append(
                {
                    "split": split,
                    "epoch": as_int(row, "epoch"),
                    "question_passage_accuracy": f"{100.0 * as_float(row, 'qp_token_f1'):.4f}",
                    "passage_only_accuracy": f"{100.0 * as_float(row, 'ponly_token_f1'):.4f}",
                    "question_passage_hit_accuracy": f"{100.0 * as_float(row, 'qp_mergeprag_hit'):.4f}",
                    "passage_only_hit_accuracy": f"{100.0 * as_float(row, 'ponly_mergeprag_hit'):.4f}",
                    "question_passage_token_f1": f"{100.0 * as_float(row, 'qp_token_f1'):.4f}",
                    "passage_only_token_f1": f"{100.0 * as_float(row, 'ponly_token_f1'):.4f}",
                    "question_passage_qa_score": f"{100.0 * as_float(row, 'qp_qa_score'):.4f}",
                    "passage_only_qa_score": f"{100.0 * as_float(row, 'ponly_qa_score'):.4f}",
                }
            )
    write_csv(
        artifact_dir / "train_validation_epoch_metrics.csv",
        summary_rows,
        [
            "split",
            "epoch",
            "question_passage_accuracy",
            "passage_only_accuracy",
            "question_passage_hit_accuracy",
            "passage_only_hit_accuracy",
            "question_passage_token_f1",
            "passage_only_token_f1",
            "question_passage_qa_score",
            "passage_only_qa_score",
        ],
    )
    print(f"[PRAG:train-valid] csv saved: {artifact_dir / 'train_validation_epoch_metrics.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PRAG epoch checkpoints on train/validation sets and create four paper figures."
    )
    parser.add_argument(
        "--qp-epoch-dir",
        default=f"llm_server/PRAG/epoch_checkpoints/{DEFAULT_RUN_NAME}_qp_kvadapt",
    )
    parser.add_argument(
        "--ponly-epoch-dir",
        default=f"llm_server/PRAG/epoch_checkpoints/{DEFAULT_RUN_NAME}_ponly",
    )
    parser.add_argument("--train", default="data/PRAG_entity_multifact_simple_train4000.jsonl")
    parser.add_argument("--valid", default="data/PRAG_entity_multifact_simple_valid4000.jsonl")
    parser.add_argument("--train-max-cases", type=int, default=400)
    parser.add_argument("--valid-max-cases", type=int, default=400)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--dataset-merge-max-passages", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--prompt-style",
        choices=("service", "short-chat", "memory-cued", "paper", "mergeprag"),
        default="service",
    )
    parser.add_argument(
        "--injection-mode",
        choices=("attention", "add_all", "add_last", "hybrid"),
        default="attention",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--scoring-style", choices=("prag", "mergeprag"), default="prag")
    parser.add_argument(
        "--qp-log",
        default=f"llm_server/PRAG/prag_train_log_orthomerge_{DEFAULT_RUN_NAME}_qp_kvadapt.json",
    )
    parser.add_argument(
        "--ponly-log",
        default=f"llm_server/PRAG/prag_train_log_orthomerge_{DEFAULT_RUN_NAME}_ponly.json",
    )
    parser.add_argument(
        "--output-root",
        default=f"llm_server/PRAG/reports/{DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument("--quiet-cases", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-epoch0", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qp-step0-weights", default="")
    parser.add_argument("--ponly-step0-weights", default="")
    parser.add_argument(
        "--combined-max-epoch",
        type=int,
        default=None,
        help="Also write combined train/validation plots truncated at this epoch.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.include_epoch0:
        eval_epoch_zero(
            args,
            split="train",
            data=args.train,
            max_cases=args.train_max_cases,
            report_dir=output_root / "train_epoch_performance" / "epoch_000",
        )
        eval_epoch_zero(
            args,
            split="validation",
            data=args.valid,
            max_cases=args.valid_max_cases,
            report_dir=output_root / "validation_epoch_performance" / "epoch_000",
        )
    eval_epoch_performance(
        args,
        split="train",
        data=args.train,
        max_cases=args.train_max_cases,
        report_dir=output_root / "train_epoch_performance",
    )
    eval_epoch_performance(
        args,
        split="validation",
        data=args.valid,
        max_cases=args.valid_max_cases,
        report_dir=output_root / "validation_epoch_performance",
    )
    build_paper_artifacts(output_root, include_epoch0=args.include_epoch0, combined_max_epoch=args.combined_max_epoch)
    print("[PRAG:train-valid] done")


if __name__ == "__main__":
    main()
