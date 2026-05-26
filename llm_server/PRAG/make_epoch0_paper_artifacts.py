"""Prepend a measured epoch-0 evaluation to paper artifact figures.

This utility is intentionally small and file-based so it can be run after an
epoch-0 `compare_memory_variants` report is produced on the GPU machine.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("summary", data)


def plot_accuracy(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    qp = [float(row["qp_accuracy"]) * 100.0 for row in rows]
    ponly = [float(row["ponly_accuracy"]) * 100.0 for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=220)
    ax.plot(epochs, qp, color="#2563A9", marker="o", markersize=4.2, linewidth=2.1, label="Question+Passage")
    ax.plot(epochs, ponly, color="#D36B2D", marker="s", markersize=4.0, linewidth=2.1, label="Passage-only")
    ax.set_title("Epoch-wise Accuracy", pad=10, weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-0.25, max(epochs) + 0.25)
    ax.set_ylim(0, 100)
    ax.set_xticks(epochs)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(True, axis="both", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_loss(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    epochs = [int(row["epoch"]) for row in rows]
    qp = [float(row["qp_loss"]) for row in rows]
    ponly = [float(row["ponly_loss"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=220)
    ax.plot(epochs, qp, color="#2563A9", marker="o", markersize=4.2, linewidth=2.1, label="Question+Passage")
    ax.plot(epochs, ponly, color="#D36B2D", marker="s", markersize=4.0, linewidth=2.1, label="Passage-only")
    ax.set_title("Epoch-wise Training Loss", pad=10, weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_xlim(-0.25, max(epochs) + 0.25)
    ax.set_yscale("log")
    ax.set_xticks(epochs)
    ax.grid(True, which="major", axis="both", alpha=0.22, linewidth=0.8)
    ax.grid(True, which="minor", axis="y", alpha=0.08, linewidth=0.5)
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-metrics", required=True, help="Existing paper_epoch_metrics.csv for epochs 1..N.")
    parser.add_argument("--epoch-loss", required=True, help="Existing epoch_loss.csv for epochs 1..N.")
    parser.add_argument("--epoch0-report", required=True, help="Epoch-0 memory_variant_comparison.json.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paper_rows = read_csv(Path(args.paper_metrics))
    loss_rows = read_csv(Path(args.epoch_loss))
    summary = load_summary(Path(args.epoch0_report))
    output_dir = Path(args.output_dir)

    qp0 = summary["question+passage"]
    ponly0 = summary["passage-only"]
    epoch0_accuracy = {
        "epoch": "0",
        "qp_accuracy": str(float(qp0.get("mergeprag_hit_rate", qp0.get("avg_qa_score", 0.0)))),
        "ponly_accuracy": str(float(ponly0.get("mergeprag_hit_rate", ponly0.get("avg_qa_score", 0.0)))),
        "delta_accuracy": str(
            float(qp0.get("mergeprag_hit_rate", qp0.get("avg_qa_score", 0.0)))
            - float(ponly0.get("mergeprag_hit_rate", ponly0.get("avg_qa_score", 0.0)))
        ),
    }

    metric_fieldnames = list(paper_rows[0].keys())
    full_metrics = []
    full_metrics.append({key: epoch0_accuracy.get(key, "") for key in metric_fieldnames})
    full_metrics.extend(paper_rows)
    write_csv(output_dir / "paper_epoch_metrics_with_epoch0.csv", metric_fieldnames, full_metrics)
    plot_accuracy(full_metrics, output_dir / "epoch_accuracy_measured_epoch0_y100.png")

    # Loss is available only when compare_memory_variants was run with --include-loss.
    if "avg_loss" in qp0 and "avg_loss" in ponly0:
        loss_fieldnames = list(loss_rows[0].keys())
        epoch0_loss = {
            "epoch": "0",
            "qp_loss": str(float(qp0["avg_loss"])),
            "ponly_loss": str(float(ponly0["avg_loss"])),
            "qp_count": str(summary.get("cases", "")),
            "ponly_count": str(summary.get("cases", "")),
        }
        full_loss = [{key: epoch0_loss.get(key, "") for key in loss_fieldnames}]
        full_loss.extend(loss_rows)
        write_csv(output_dir / "epoch_loss_with_measured_epoch0.csv", loss_fieldnames, full_loss)
        plot_loss(full_loss, output_dir / "epoch_loss_measured_epoch0.png")
    else:
        print("[PRAG:epoch0-artifacts] epoch0 report has no avg_loss; skipped epoch0 loss plot.")

    print(f"[PRAG:epoch0-artifacts] saved: {output_dir / 'epoch_accuracy_measured_epoch0_y100.png'}")


if __name__ == "__main__":
    main()
