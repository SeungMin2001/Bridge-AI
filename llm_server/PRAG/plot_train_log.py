"""Summarize and plot PRAG training logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .config import LOG_PATH, MULTIFACT_LOG_PATH


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out = []
    total = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        out.append(total / len(queue))
    return out


def write_step_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = [
        "step",
        "kind",
        "objective",
        "main_gold",
        "main_neg",
        "neg_gold",
        "neg_neg",
        "rank",
        "main_ok",
        "neg_ok",
        "lr",
        "elapsed_min",
        "cumulative_elapsed_min",
        "group_qas",
        "group_main_rate",
        "group_neg_rate",
        "merge_passages_avg",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_val_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "step",
        "elapsed_min",
        "cumulative_elapsed_min",
        "individual_objective",
        "individual_main_ok",
        "individual_neg_ok",
        "individual_flip_ok",
        "group_objective",
        "group_main_ok",
        "group_neg_ok",
        "group_flip_ok",
        "merge_objective",
        "merge_main_ok",
        "merge_neg_ok",
        "merge_flip_ok",
        "merge_passages_avg",
        "generation_count",
        "generation_main_kv_hit_rate",
        "generation_neg_kv_hit_rate",
        "generation_direct_passage_hit_rate",
        "generation_no_memory_hit_rate",
        "generation_zero_kv_hit_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            individual = row.get("individual") or {}
            group = row.get("group") or {}
            merge = row.get("merge") or {}
            generation = row.get("generation") or {}
            writer.writerow({
                "step": row.get("step", ""),
                "elapsed_min": row.get("elapsed_min", ""),
                "cumulative_elapsed_min": row.get("cumulative_elapsed_min", ""),
                "individual_objective": individual.get("objective", ""),
                "individual_main_ok": individual.get("main_ok", ""),
                "individual_neg_ok": individual.get("neg_ok", ""),
                "individual_flip_ok": individual.get("flip_ok", ""),
                "group_objective": group.get("objective", ""),
                "group_main_ok": group.get("main_ok", ""),
                "group_neg_ok": group.get("neg_ok", ""),
                "group_flip_ok": group.get("flip_ok", ""),
                "merge_objective": merge.get("objective", ""),
                "merge_main_ok": merge.get("main_ok", ""),
                "merge_neg_ok": merge.get("neg_ok", ""),
                "merge_flip_ok": merge.get("flip_ok", ""),
                "merge_passages_avg": merge.get("merge_passages_avg", ""),
                "generation_count": generation.get("count", ""),
                "generation_main_kv_hit_rate": generation.get("main_kv_hit_rate", ""),
                "generation_neg_kv_hit_rate": generation.get("neg_kv_hit_rate", ""),
                "generation_direct_passage_hit_rate": generation.get("direct_passage_hit_rate", ""),
                "generation_no_memory_hit_rate": generation.get("no_memory_hit_rate", ""),
                "generation_zero_kv_hit_rate": generation.get("zero_kv_hit_rate", ""),
            })


def plot_png(output: Path, step_rows: list[dict], val_rows: list[dict], smooth: int) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[PRAG:plot] matplotlib unavailable; skipped PNG plot ({exc})")
        return False

    steps = [int(row["step"]) for row in step_rows]
    objectives = [float(row["objective"]) for row in step_rows]
    ranks = [float(row.get("rank") or 0.0) for row in step_rows]
    elapsed = [float(row.get("cumulative_elapsed_min") or row.get("elapsed_min") or 0.0) for row in step_rows]

    val_steps = [int(row["step"]) for row in val_rows]
    val_obj = [float((row.get("individual") or {}).get("objective", 0.0)) for row in val_rows]
    group_obj = [float((row.get("group") or {}).get("objective", 0.0)) for row in val_rows if row.get("group")]
    group_steps = [int(row["step"]) for row in val_rows if row.get("group")]
    merge_obj = [float((row.get("merge") or {}).get("objective", 0.0)) for row in val_rows if row.get("merge")]
    merge_steps = [int(row["step"]) for row in val_rows if row.get("merge")]
    val_flip = [float((row.get("individual") or {}).get("flip_ok", 0.0)) for row in val_rows]
    group_flip = [float((row.get("group") or {}).get("flip_ok", 0.0)) for row in val_rows if row.get("group")]
    merge_flip = [float((row.get("merge") or {}).get("flip_ok", 0.0)) for row in val_rows if row.get("merge")]
    generation_rows = [row for row in val_rows if row.get("generation")]
    generation_steps = [int(row["step"]) for row in generation_rows]
    generation_main = [float((row.get("generation") or {}).get("main_kv_hit_rate", 0.0)) for row in generation_rows]
    generation_neg = [float((row.get("generation") or {}).get("neg_kv_hit_rate", 0.0)) for row in generation_rows]
    generation_direct = [
        float((row.get("generation") or {}).get("direct_passage_hit_rate", 0.0)) for row in generation_rows
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("PRAG Training Log")

    axes[0, 0].plot(steps, moving_average(objectives, smooth), label=f"train objective ma{smooth}")
    axes[0, 0].scatter(val_steps, val_obj, s=24, label="val objective")
    if group_obj:
        axes[0, 0].scatter(group_steps, group_obj, s=24, label="group val objective")
    if merge_obj:
        axes[0, 0].scatter(merge_steps, merge_obj, s=24, label="merge val objective")
    axes[0, 0].set_title("Objective")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, moving_average(ranks, smooth), color="tab:orange", label=f"rank ma{smooth}")
    axes[0, 1].set_title("Rank Loss")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(val_steps, val_flip, marker="o", label="val flip")
    if group_flip:
        axes[1, 0].plot(group_steps, group_flip, marker="o", label="group val flip")
    if merge_flip:
        axes[1, 0].plot(merge_steps, merge_flip, marker="o", label="merge val flip")
    if generation_rows:
        axes[1, 0].plot(generation_steps, generation_main, marker="x", linestyle="--", label="gen main_kv")
        axes[1, 0].plot(generation_steps, generation_neg, marker="x", linestyle="--", label="gen neg_kv")
        axes[1, 0].plot(generation_steps, generation_direct, marker="x", linestyle="--", label="gen direct")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].set_title("Candidate And Generation Accuracy")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, elapsed, color="tab:green", label="elapsed min")
    axes[1, 1].set_title("Elapsed Time")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("minutes")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True


def plot_objective_time_png(
    output: Path,
    step_rows: list[dict],
    val_rows: list[dict],
    smooth: int,
    *,
    hide_val_labels: bool = False,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[PRAG:plot] matplotlib unavailable; skipped PNG plot ({exc})")
        return False

    steps = [int(row["step"]) for row in step_rows]
    objectives = [float(row["objective"]) for row in step_rows]
    elapsed = [float(row.get("cumulative_elapsed_min") or row.get("elapsed_min") or 0.0) for row in step_rows]
    val_steps = [int(row["step"]) for row in val_rows]
    val_obj = [float((row.get("individual") or {}).get("objective", 0.0)) for row in val_rows]
    group_steps = [int(row["step"]) for row in val_rows if row.get("group")]
    group_obj = [float((row.get("group") or {}).get("objective", 0.0)) for row in val_rows if row.get("group")]
    merge_steps = [int(row["step"]) for row in val_rows if row.get("merge")]
    merge_obj = [float((row.get("merge") or {}).get("objective", 0.0)) for row in val_rows if row.get("merge")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PRAG Objective And Time")

    axes[0].plot(steps, moving_average(objectives, smooth), label=f"train objective ma{smooth}")
    if val_obj:
        axes[0].scatter(val_steps, val_obj, s=28, label=None if hide_val_labels else "val objective")
    if group_obj:
        axes[0].scatter(group_steps, group_obj, s=28, label=None if hide_val_labels else "group val objective")
    if merge_obj:
        axes[0].scatter(merge_steps, merge_obj, s=28, label=None if hide_val_labels else "merge val objective")
    axes[0].set_title("Objective")
    axes[0].set_xlabel("step")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, elapsed, color="tab:green", label="elapsed min")
    axes[1].set_title("Elapsed Time")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("minutes")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True


def parse_labeled_log(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip() or Path(path).stem, Path(path.strip())
    path = Path(value.strip())
    return path.stem, path


def preferred_val_objective(row: dict) -> float | None:
    for key in ("merge", "group", "individual"):
        metrics = row.get(key) or {}
        if "objective" in metrics:
            return float(metrics["objective"])
    return None


def plot_compare_objective_time_png(
    output: Path,
    series: list[tuple[str, Path, dict]],
    smooth: int,
    *,
    stitch: bool = False,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[PRAG:plot] matplotlib unavailable; skipped compare PNG plot ({exc})")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("PRAG Objective And Time Comparison")

    step_offset = 0
    elapsed_offset = 0.0
    summary = []
    for label, path, data in series:
        step_rows = data.get("step_losses") or []
        val_rows = data.get("val_evals") or []
        if not step_rows:
            continue
        raw_steps = [int(row["step"]) for row in step_rows]
        raw_elapsed = [float(row.get("cumulative_elapsed_min") or row.get("elapsed_min") or 0.0) for row in step_rows]
        steps = [step + step_offset for step in raw_steps]
        elapsed = [minute + elapsed_offset for minute in raw_elapsed]
        objectives = [float(row["objective"]) for row in step_rows]
        line = axes[0].plot(steps, moving_average(objectives, smooth), label=f"{label} train ma{smooth}")[0]
        color = line.get_color()

        val_points = [(int(row["step"]) + step_offset, preferred_val_objective(row)) for row in val_rows]
        val_points = [(step, obj) for step, obj in val_points if obj is not None]
        if val_points:
            axes[0].scatter(
                [step for step, _obj in val_points],
                [obj for _step, obj in val_points],
                s=22,
                marker="x",
                color=color,
                label=f"{label} val",
            )

        axes[1].plot(steps, elapsed, color=color, label=label)
        summary.append({
            "label": label,
            "log": str(path),
            "steps_logged": len(step_rows),
            "val_evals": len(val_rows),
            "first_step": raw_steps[0] if raw_steps else None,
            "last_step": raw_steps[-1] if raw_steps else None,
            "step_offset": step_offset,
            "elapsed_offset_min": elapsed_offset,
        })
        if stitch:
            step_offset += max(raw_steps) if raw_steps else 0
            elapsed_offset += max(raw_elapsed) if raw_elapsed else 0.0

    axes[0].set_title("Objective")
    axes[0].set_xlabel("step" if not stitch else "stitched step")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Elapsed Time")
    axes[1].set_xlabel("step" if not stitch else "stitched step")
    axes[1].set_ylabel("minutes" if not stitch else "stitched minutes")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=str(MULTIFACT_LOG_PATH))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--smooth", type=int, default=25)
    parser.add_argument(
        "--objective-time-only",
        action="store_true",
        help="Plot only objective curves and elapsed time.",
    )
    parser.add_argument(
        "--hide-val-labels",
        action="store_true",
        help="Hide validation labels in the objective/time-only plot legend.",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Label for the primary log when plotting with --compare-log.",
    )
    parser.add_argument(
        "--compare-log",
        action="append",
        default=[],
        help="Additional log to plot together. Use LABEL=path or just path. Can be repeated.",
    )
    parser.add_argument(
        "--stitch-logs",
        action="store_true",
        help="Offset later logs by previous logs' last step/runtime so separate runs appear as one continuation.",
    )
    parser.add_argument("--generic", action="store_true", help="Use the non-multifact default log path.")
    args = parser.parse_args()

    log_path = Path(args.log)
    if args.generic and args.log == str(MULTIFACT_LOG_PATH):
        log_path = LOG_PATH
    output_dir = Path(args.output_dir) if args.output_dir else log_path.with_suffix("").with_name(log_path.stem + "_plots")
    data = json.loads(log_path.read_text(encoding="utf-8"))
    step_rows = data.get("step_losses") or []
    val_rows = data.get("val_evals") or []
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_log:
        primary_label = args.label or log_path.stem
        series = [(primary_label, log_path, data)]
        for item in args.compare_log:
            label, compare_path = parse_labeled_log(item)
            series.append((label, compare_path, json.loads(compare_path.read_text(encoding="utf-8"))))
        compare_png = output_dir / "combined_objective_time_curves.png"
        plotted = plot_compare_objective_time_png(
            compare_png,
            series,
            args.smooth,
            stitch=args.stitch_logs,
        )
        print(f"[PRAG:plot] compare_logs={len(series)} stitch={args.stitch_logs}")
        print(f"[PRAG:plot] compare_png={compare_png if plotted else 'not_created'}")
        print(f"[PRAG:plot] compare_summary={compare_png.with_suffix('.summary.json') if plotted else 'not_created'}")

    step_csv = output_dir / "step_losses.csv"
    val_csv = output_dir / "val_evals.csv"
    summary_json = output_dir / "summary.json"
    png_path = output_dir / ("objective_time_curves.png" if args.objective_time_only else "training_curves.png")
    write_step_csv(step_csv, step_rows)
    write_val_csv(val_csv, val_rows)

    final_val = data.get("final_val") or {}
    final_group = data.get("final_group_val") or {}
    final_generation = data.get("final_generation_val") or {}
    sessions = data.get("sessions") or []
    generation_evals = [row for row in val_rows if row.get("generation")]
    summary = {
        "log": str(log_path),
        "steps_logged": len(step_rows),
        "val_evals": len(val_rows),
        "generation_evals": len(generation_evals),
        "final_step": data.get("final_step"),
        "total_logged_runtime_sec": data.get("total_logged_runtime_sec"),
        "total_logged_runtime_min": round(float(data.get("total_logged_runtime_sec") or 0.0) / 60, 4),
        "sessions": sessions,
        "final_val": final_val,
        "final_group_val": final_group,
        "final_generation_val": final_generation,
        "last_generation_eval": generation_evals[-1].get("generation") if generation_evals else {},
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if step_rows and args.objective_time_only:
        plotted = plot_objective_time_png(
            png_path,
            step_rows,
            val_rows,
            args.smooth,
            hide_val_labels=args.hide_val_labels,
        )
    else:
        plotted = plot_png(png_path, step_rows, val_rows, args.smooth) if step_rows else False

    print(f"[PRAG:plot] log={log_path}")
    print(f"[PRAG:plot] steps={len(step_rows)} val_evals={len(val_rows)} final_step={data.get('final_step')}")
    print(f"[PRAG:plot] runtime_min={summary['total_logged_runtime_min']}")
    print(f"[PRAG:plot] final_val={final_val}")
    print(f"[PRAG:plot] final_group_val={final_group}")
    if final_generation:
        print(f"[PRAG:plot] final_generation_val={final_generation}")
    print(f"[PRAG:plot] step_csv={step_csv}")
    print(f"[PRAG:plot] val_csv={val_csv}")
    print(f"[PRAG:plot] summary={summary_json}")
    print(f"[PRAG:plot] png={png_path if plotted else 'not_created'}")


if __name__ == "__main__":
    main()
