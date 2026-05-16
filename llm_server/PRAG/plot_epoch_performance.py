"""Evaluate epoch checkpoints and draw paper-style PRAG performance curves."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


def epoch_from_path(path: Path) -> int | None:
    match = re.search(r"epoch[_-](\d+)", path.stem)
    return int(match.group(1)) if match else None


def epoch_checkpoints(path: str | Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    root = Path(path)
    for ckpt in sorted(root.glob("epoch_*.pt")):
        epoch = epoch_from_path(ckpt)
        if epoch is not None:
            out[epoch] = ckpt
    return out


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_compare(args: argparse.Namespace, epoch: int, qp_path: Path, ponly_path: Path, report_dir: Path) -> dict:
    epoch_report_dir = report_dir / f"epoch_{epoch:03d}"
    summary_path = epoch_report_dir / "memory_variant_comparison.json"
    if args.resume and summary_path.exists():
        return read_json(summary_path)["summary"]

    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.compare_memory_variants",
        "--qp-weights",
        str(qp_path),
        "--ponly-weights",
        str(ponly_path),
        "--data",
        args.data,
        "--case-index",
        str(args.case_index),
        "--max-cases",
        str(args.max_cases),
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
        str(epoch_report_dir),
        "--scoring-style",
        args.scoring_style,
    ]
    if args.include_loss:
        cmd.append("--include-loss")
    if args.quiet_cases:
        cmd.append("--quiet-cases")
    cmd.append("--resume" if args.resume else "--no-resume")
    print(f"[PRAG:epoch-performance] epoch={epoch} $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return read_json(summary_path)["summary"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
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
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def read_loss_curve(path: str | Path) -> tuple[list[float], list[float]]:
    log_path = Path(path)
    if not str(path) or not log_path.exists():
        return [], []
    data = read_json(log_path)
    steps_per_epoch = (
        (data.get("training_schedule") or {}).get("steps_per_epoch")
        or max(1, int(data.get("final_step") or 1))
    )
    xs: list[float] = []
    ys: list[float] = []
    for row in data.get("step_losses") or []:
        step = row.get("step")
        objective = row.get("objective")
        if step is None or objective is None:
            continue
        xs.append(float(step) / float(steps_per_epoch))
        ys.append(float(objective))
    return xs, moving_average(ys, 25)


def plot_curves(path: Path, rows: list[dict], *, qp_log: str, ponly_log: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency.
        print(f"[PRAG:epoch-performance] matplotlib unavailable; skipped plot ({exc})")
        return

    epochs = [row["epoch"] for row in rows]
    qp_hit = [100 * row["qp_mergeprag_hit"] for row in rows]
    ponly_hit = [100 * row["ponly_mergeprag_hit"] for row in rows]
    qp_f1 = [100 * row["qp_token_f1"] for row in rows]
    ponly_f1 = [100 * row["ponly_token_f1"] for row in rows]
    qp_qa = [100 * row["qp_qa_score"] for row in rows]
    ponly_qa = [100 * row["ponly_qa_score"] for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
    fig.suptitle("Epoch-wise PRAG Performance", fontsize=16, fontweight="bold")
    colors = {"qp": "#2f5f9f", "ponly": "#c66a2e"}

    ax = axes[0, 0]
    qp_x, qp_loss = read_loss_curve(qp_log)
    ponly_x, ponly_loss = read_loss_curve(ponly_log)
    if qp_x and qp_loss:
        ax.plot(qp_x, qp_loss, color=colors["qp"], linewidth=2.2, label="Question+Passage")
    if ponly_x and ponly_loss:
        ax.plot(ponly_x, ponly_loss, color=colors["ponly"], linewidth=2.2, label="Passage-only")
    ax.set_title("(a) Training Objective")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    panels = [
        (axes[0, 1], "(b) MergePRAG Hit", qp_hit, ponly_hit, "Hit Rate (%)"),
        (axes[1, 0], "(c) Token F1", qp_f1, ponly_f1, "F1 (%)"),
        (axes[1, 1], "(d) QA Score", qp_qa, ponly_qa, "QA Score (%)"),
    ]
    for ax, title, qp_values, ponly_values, ylabel in panels:
        ax.plot(epochs, qp_values, marker="o", linewidth=2.4, color=colors["qp"], label="Question+Passage")
        ax.plot(epochs, ponly_values, marker="s", linewidth=2.4, color=colors["ponly"], label="Passage-only")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QP/P-only epoch checkpoints and plot performance curves.")
    parser.add_argument("--qp-epoch-dir", required=True)
    parser.add_argument("--ponly-epoch-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=400)
    parser.add_argument("--dataset-merge-max-passages", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt-style", choices=("service", "short-chat", "memory-cued", "paper", "mergeprag"), default="service")
    parser.add_argument("--injection-mode", choices=("attention", "add_all", "add_last", "hybrid"), default="attention")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--scoring-style", choices=("prag", "mergeprag"), default="prag")
    parser.add_argument("--include-loss", action="store_true")
    parser.add_argument("--quiet-cases", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--qp-log", default="")
    parser.add_argument("--ponly-log", default="")
    args = parser.parse_args()

    qp_epochs = epoch_checkpoints(args.qp_epoch_dir)
    ponly_epochs = epoch_checkpoints(args.ponly_epoch_dir)
    epochs = sorted(set(qp_epochs) & set(ponly_epochs))
    if not epochs:
        raise RuntimeError(f"No matching epoch checkpoints: qp={args.qp_epoch_dir}, ponly={args.ponly_epoch_dir}")

    report_dir = Path(args.report_dir)
    rows: list[dict] = []
    for epoch in epochs:
        summary = run_compare(args, epoch, qp_epochs[epoch], ponly_epochs[epoch], report_dir)
        qp = summary["question+passage"]
        ponly = summary["passage-only"]
        rows.append(
            {
                "epoch": epoch,
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
            }
        )

    csv_path = report_dir / "epoch_performance.csv"
    json_path = report_dir / "epoch_performance.json"
    png_path = report_dir / "epoch_performance_curves.png"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_curves(png_path, rows, qp_log=args.qp_log, ponly_log=args.ponly_log)
    print(f"[PRAG:epoch-performance] csv saved: {csv_path}")
    print(f"[PRAG:epoch-performance] json saved: {json_path}")
    print(f"[PRAG:epoch-performance] plot saved: {png_path}")


if __name__ == "__main__":
    main()
