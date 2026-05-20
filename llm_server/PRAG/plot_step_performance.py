"""Evaluate step checkpoints and draw early-stage PRAG learning curves."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


def step_from_path(path: Path) -> int | None:
    match = re.search(r"step[_-](\d+)", path.stem)
    return int(match.group(1)) if match else None


def step_checkpoints(path: str | Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    root = Path(path)
    for ckpt in sorted(root.glob("step_*.pt")):
        step = step_from_path(ckpt)
        if step is not None:
            out[step] = ckpt
    return out


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_compare(args: argparse.Namespace, step: int, qp_path: Path, ponly_path: Path, report_dir: Path) -> dict:
    step_report_dir = report_dir / f"step_{step:06d}"
    summary_path = step_report_dir / "memory_variant_comparison.json"
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
        str(step_report_dir),
        "--scoring-style",
        args.scoring_style,
    ]
    if args.include_loss:
        cmd.append("--include-loss")
    if args.quiet_cases:
        cmd.append("--quiet-cases")
    cmd.append("--resume" if args.resume else "--no-resume")
    print(f"[PRAG:step-performance] step={step} $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return read_json(summary_path)["summary"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
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


def style_axis(ax) -> None:
    ax.grid(True, alpha=0.28, linewidth=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def step_tick_label(step: int) -> str:
    if step == 0:
        return "0"
    if step >= 1000:
        return f"{step / 1000:g}k"
    return str(step)


def configure_step_axis(ax, steps: list[int], *, log_x: bool) -> None:
    plot_steps = [max(step, 1) if log_x else step for step in steps]
    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(max(1, min(plot_steps) * 0.78), max(plot_steps) * 1.16)
    else:
        span = max(plot_steps) - min(plot_steps)
        pad = max(20, span * 0.04)
        ax.set_xlim(min(plot_steps) - pad, max(plot_steps) + pad)
    ax.set_xticks(plot_steps)
    ax.set_xticklabels([step_tick_label(step) for step in steps])
    ax.set_xlabel("Training Step" + (" (log scale)" if log_x else ""))
    if len(steps) >= 4:
        early_end = max(steps[min(3, len(steps) - 1)], 1)
        ax.axvspan(max(1, min(plot_steps) * 0.78) if log_x else min(plot_steps), early_end, color="#F3F7FF", alpha=0.75, zorder=0)


def plot_metric_axis(
    ax,
    steps: list[int],
    qp_values: list[float],
    ponly_values: list[float],
    *,
    title: str,
    ylabel: str,
    colors: dict[str, str],
    log_x: bool,
) -> None:
    x = [max(step, 1) if log_x else step for step in steps]
    ax.plot(x, qp_values, marker="o", linewidth=2.45, markersize=5.6, color=colors["qp"], label="Question+Passage")
    ax.plot(x, ponly_values, marker="s", linewidth=2.45, markersize=5.4, color=colors["ponly"], label="Passage-only")
    lo = min(qp_values + ponly_values)
    hi = max(qp_values + ponly_values)
    pad = max(1.0, (hi - lo) * 0.18)
    ax.set_ylim(max(0, lo - pad), min(100, hi + pad))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    configure_step_axis(ax, steps, log_x=log_x)
    style_axis(ax)


def save_single(path: Path, draw_fn) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency.
        print(f"[PRAG:step-performance] matplotlib unavailable; skipped single plot ({exc})")
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    draw_fn(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"[PRAG:step-performance] plot saved: {path}")


def plot_curves(path: Path, rows: list[dict], *, log_x: bool) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency.
        print(f"[PRAG:step-performance] matplotlib unavailable; skipped plot ({exc})")
        return

    steps = [row["step"] for row in rows]
    qp_hit = [100 * row["qp_mergeprag_hit"] for row in rows]
    ponly_hit = [100 * row["ponly_mergeprag_hit"] for row in rows]
    qp_f1 = [100 * row["qp_token_f1"] for row in rows]
    ponly_f1 = [100 * row["ponly_token_f1"] for row in rows]
    qp_qa = [100 * row["qp_qa_score"] for row in rows]
    ponly_qa = [100 * row["ponly_qa_score"] for row in rows]
    delta_hit = [100 * row["delta_mergeprag_hit"] for row in rows]
    delta_f1 = [100 * row["delta_token_f1"] for row in rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    colors = {"qp": "#245A9A", "ponly": "#C96B2C", "gain": "#2B7A5B"}
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.5))
    fig.suptitle("Early Step-wise PRAG Performance", fontsize=16, fontweight="bold")
    panels = [
        (axes[0, 0], "(a) Token F1", qp_f1, ponly_f1, "F1 (%)"),
        (axes[0, 1], "(b) Hit Rate", qp_hit, ponly_hit, "Hit Rate (%)"),
        (axes[1, 0], "(c) QA Score", qp_qa, ponly_qa, "QA Score (%)"),
    ]
    for ax, title, qp_values, ponly_values, ylabel in panels:
        plot_metric_axis(ax, steps, qp_values, ponly_values, title=title, ylabel=ylabel, colors=colors, log_x=log_x)
    x = [max(step, 1) if log_x else step for step in steps]
    axes[1, 1].plot(x, delta_f1, marker="o", linewidth=2.45, markersize=5.6, color=colors["qp"], label="F1 gain")
    axes[1, 1].plot(x, delta_hit, marker="s", linewidth=2.45, markersize=5.4, color=colors["gain"], label="Hit gain")
    axes[1, 1].axhline(0, color="#606060", linewidth=1.0, alpha=0.75)
    lo = min(delta_f1 + delta_hit)
    hi = max(delta_f1 + delta_hit)
    pad = max(1.0, (hi - lo) * 0.18)
    axes[1, 1].set_ylim(lo - pad, hi + pad)
    axes[1, 1].set_title("(d) Q+P Gain over Passage-only")
    axes[1, 1].set_ylabel("Gain (percentage points)")
    configure_step_axis(axes[1, 1], steps, log_x=log_x)
    style_axis(axes[1, 1])
    axes[0, 0].legend(frameon=False, loc="lower right")
    axes[1, 1].legend(frameon=False, loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[PRAG:step-performance] plot saved: {path}")

    single_dir = path.parent / "single_panels"
    save_single(
        single_dir / "step_token_f1.png",
        lambda ax: plot_metric_axis(ax, steps, qp_f1, ponly_f1, title="Token F1 by Training Step", ylabel="F1 (%)", colors=colors, log_x=log_x),
    )
    save_single(
        single_dir / "step_hit_rate.png",
        lambda ax: plot_metric_axis(ax, steps, qp_hit, ponly_hit, title="Hit Rate by Training Step", ylabel="Hit Rate (%)", colors=colors, log_x=log_x),
    )
    save_single(
        single_dir / "step_qa_score.png",
        lambda ax: plot_metric_axis(ax, steps, qp_qa, ponly_qa, title="QA Score by Training Step", ylabel="QA Score (%)", colors=colors, log_x=log_x),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QP/P-only step checkpoints and plot early performance curves.")
    parser.add_argument("--qp-step-dir", required=True)
    parser.add_argument("--ponly-step-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=100)
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
    parser.add_argument("--log-x", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    qp_steps = step_checkpoints(args.qp_step_dir)
    ponly_steps = step_checkpoints(args.ponly_step_dir)
    steps = sorted(set(qp_steps) & set(ponly_steps))
    if not steps:
        raise RuntimeError(f"No matching step checkpoints: qp={args.qp_step_dir}, ponly={args.ponly_step_dir}")

    report_dir = Path(args.report_dir)
    rows: list[dict] = []
    for step in steps:
        summary = run_compare(args, step, qp_steps[step], ponly_steps[step], report_dir)
        qp = summary["question+passage"]
        ponly = summary["passage-only"]
        rows.append(
            {
                "step": step,
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

    csv_path = report_dir / "step_performance.csv"
    json_path = report_dir / "step_performance.json"
    png_path = report_dir / "step_performance_curves.png"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_curves(png_path, rows, log_x=args.log_x)
    print(f"[PRAG:step-performance] csv saved: {csv_path}")
    print(f"[PRAG:step-performance] json saved: {json_path}")
    print(f"[PRAG:step-performance] plot saved: {png_path}")


if __name__ == "__main__":
    main()
