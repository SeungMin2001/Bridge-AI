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
        "group_qas",
        "group_main_rate",
        "group_neg_rate",
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
        "individual_objective",
        "individual_main_ok",
        "individual_neg_ok",
        "individual_flip_ok",
        "group_objective",
        "group_main_ok",
        "group_neg_ok",
        "group_flip_ok",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            individual = row.get("individual") or {}
            group = row.get("group") or {}
            writer.writerow({
                "step": row.get("step", ""),
                "elapsed_min": row.get("elapsed_min", ""),
                "individual_objective": individual.get("objective", ""),
                "individual_main_ok": individual.get("main_ok", ""),
                "individual_neg_ok": individual.get("neg_ok", ""),
                "individual_flip_ok": individual.get("flip_ok", ""),
                "group_objective": group.get("objective", ""),
                "group_main_ok": group.get("main_ok", ""),
                "group_neg_ok": group.get("neg_ok", ""),
                "group_flip_ok": group.get("flip_ok", ""),
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
    elapsed = [float(row.get("elapsed_min") or 0.0) for row in step_rows]

    val_steps = [int(row["step"]) for row in val_rows]
    val_obj = [float((row.get("individual") or {}).get("objective", 0.0)) for row in val_rows]
    group_obj = [float((row.get("group") or {}).get("objective", 0.0)) for row in val_rows if row.get("group")]
    group_steps = [int(row["step"]) for row in val_rows if row.get("group")]
    val_flip = [float((row.get("individual") or {}).get("flip_ok", 0.0)) for row in val_rows]
    group_flip = [float((row.get("group") or {}).get("flip_ok", 0.0)) for row in val_rows if row.get("group")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("PRAG Training Log")

    axes[0, 0].plot(steps, moving_average(objectives, smooth), label=f"train objective ma{smooth}")
    axes[0, 0].scatter(val_steps, val_obj, s=24, label="val objective")
    if group_obj:
        axes[0, 0].scatter(group_steps, group_obj, s=24, label="group val objective")
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
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].set_title("Flip Accuracy")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=str(MULTIFACT_LOG_PATH))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--smooth", type=int, default=25)
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

    step_csv = output_dir / "step_losses.csv"
    val_csv = output_dir / "val_evals.csv"
    summary_json = output_dir / "summary.json"
    png_path = output_dir / "training_curves.png"
    write_step_csv(step_csv, step_rows)
    write_val_csv(val_csv, val_rows)

    final_val = data.get("final_val") or {}
    final_group = data.get("final_group_val") or {}
    sessions = data.get("sessions") or []
    summary = {
        "log": str(log_path),
        "steps_logged": len(step_rows),
        "val_evals": len(val_rows),
        "final_step": data.get("final_step"),
        "total_logged_runtime_sec": data.get("total_logged_runtime_sec"),
        "total_logged_runtime_min": round(float(data.get("total_logged_runtime_sec") or 0.0) / 60, 4),
        "sessions": sessions,
        "final_val": final_val,
        "final_group_val": final_group,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plotted = plot_png(png_path, step_rows, val_rows, args.smooth) if step_rows else False

    print(f"[PRAG:plot] log={log_path}")
    print(f"[PRAG:plot] steps={len(step_rows)} val_evals={len(val_rows)} final_step={data.get('final_step')}")
    print(f"[PRAG:plot] runtime_min={summary['total_logged_runtime_min']}")
    print(f"[PRAG:plot] final_val={final_val}")
    print(f"[PRAG:plot] final_group_val={final_group}")
    print(f"[PRAG:plot] step_csv={step_csv}")
    print(f"[PRAG:plot] val_csv={val_csv}")
    print(f"[PRAG:plot] summary={summary_json}")
    print(f"[PRAG:plot] png={png_path if plotted else 'not_created'}")


if __name__ == "__main__":
    main()
