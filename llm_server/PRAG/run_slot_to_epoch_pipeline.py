"""Run slot search, then train/evaluate epoch curves with the best slot.

This script is intentionally a thin orchestrator over the existing PRAG
commands so checkpoints, logs, and reports keep the same format as manual runs.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], *, dry_run: bool = False) -> None:
    print(f"\n[PRAG:slot-to-epoch] $ {' '.join(cmd)}", flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def read_best_slot(table_path: Path, metric: str) -> int:
    if not table_path.exists():
        raise FileNotFoundError(f"Missing slot ablation table: {table_path}")

    metric_key = {
        "hit": "qp_mergeprag_hit",
        "f1": "qp_token_f1",
        "qa": "qp_qa_score",
    }[metric]

    rows: list[dict] = []
    with table_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise RuntimeError(f"Empty slot ablation table: {table_path}")

    def score(row: dict) -> tuple[float, float, float, int]:
        return (
            float(row.get(metric_key) or 0.0),
            float(row.get("qp_token_f1") or 0.0),
            float(row.get("qp_qa_score") or 0.0),
            int(row["num_kv"]),
        )

    best = max(rows, key=score)
    best_slot = int(best["num_kv"])
    print(
        "[PRAG:slot-to-epoch] best_slot="
        f"{best_slot} metric={metric} "
        f"hit={float(best.get('qp_mergeprag_hit') or 0.0):.4f} "
        f"f1={float(best.get('qp_token_f1') or 0.0):.4f} "
        f"qa={float(best.get('qp_qa_score') or 0.0):.4f}",
        flush=True,
    )
    return best_slot


def slot_search_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.run_kv_slot_ablation",
        "--k-values",
        args.k_values,
        "--ablate-target",
        "qp",
        "--qp-critical-layer",
        str(args.qp_critical_layer),
        "--fixed-ponly-weights",
        args.fixed_ponly_weights,
        "--train",
        args.train,
        "--valid",
        args.valid,
        "--test-data",
        args.test_data,
        "--suffix-prefix",
        args.slot_suffix_prefix,
        "--epochs",
        str(args.slot_epochs),
        "--test-max-cases",
        str(args.slot_test_max_cases),
        "--test-max-new-tokens",
        str(args.test_max_new_tokens),
        "--report-root",
        args.slot_report_root,
    ]
    if args.resume:
        cmd.append("--resume")
    return cmd


def epoch_experiment_command(args: argparse.Namespace, best_slot: int) -> list[str]:
    suffix_tag = f"{args.epoch_suffix_prefix}_kv{best_slot}_ep{args.epoch_epochs}"
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.run_related_merge_experiment",
        "--num-kv",
        str(best_slot),
        "--train",
        args.train,
        "--valid",
        args.valid,
        "--test-data",
        args.test_data,
        "--qp-question-fusion",
        "kv_adapter",
        "--ponly-question-fusion",
        "none",
        "--qp-output-suffix",
        f"{suffix_tag}_qp_kvadapt",
        "--ponly-output-suffix",
        f"{suffix_tag}_ponly",
        "--epochs",
        str(args.epoch_epochs),
        "--lr",
        str(args.lr),
        "--test-max-cases",
        str(args.epoch_test_max_cases),
        "--test-max-new-tokens",
        str(args.test_max_new_tokens),
        "--report-dir",
        str(Path(args.epoch_report_root) / f"kv{best_slot}"),
        "--epoch-checkpoint-root",
        args.epoch_checkpoint_root,
        "--eval-epoch-performance",
        "--qp-critical-layer",
        str(args.qp_critical_layer),
        "--ponly-critical-layer",
        str(args.ponly_critical_layer),
    ]
    if args.quiet_eval:
        cmd.append("--quiet-eval")
    if args.resume:
        cmd.append("--resume")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the best BridgePRAG slot, then run epoch-wise performance training/evaluation."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--fixed-ponly-weights", required=True)
    parser.add_argument("--k-values", default="1,2,4,8,16,32,48,64,80,96")
    parser.add_argument("--slot-suffix-prefix", default="entity_simple4000_qp_latency400_fixedlayer")
    parser.add_argument("--slot-report-root", default="llm_server/PRAG/reports/kv_slot_latency_entity_simple4000_qp_400cases")
    parser.add_argument("--slot-epochs", type=int, default=5)
    parser.add_argument("--slot-test-max-cases", type=int, default=400)
    parser.add_argument("--best-metric", choices=("hit", "f1", "qa"), default="hit")
    parser.add_argument("--epoch-suffix-prefix", default="entity_simple4000_epoch_bestslot_fixedlayer")
    parser.add_argument("--epoch-report-root", default="llm_server/PRAG/reports/entity_simple4000_epoch_bestslot")
    parser.add_argument("--epoch-checkpoint-root", default="llm_server/PRAG/epoch_checkpoints")
    parser.add_argument("--epoch-epochs", type=int, default=10)
    parser.add_argument("--epoch-test-max-cases", type=int, default=400)
    parser.add_argument("--test-max-new-tokens", type=int, default=128)
    parser.add_argument("--qp-critical-layer", type=int, default=23)
    parser.add_argument("--ponly-critical-layer", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--quiet-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-slot-search", action="store_true")
    parser.add_argument("--best-slot", type=int, default=0, help="Use this slot directly when --skip-slot-search is set.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.skip_slot_search:
        run_command(slot_search_command(args), dry_run=args.dry_run)

    if args.skip_slot_search and args.best_slot > 0:
        best_slot = int(args.best_slot)
        print(f"[PRAG:slot-to-epoch] use provided best_slot={best_slot}", flush=True)
    elif args.dry_run:
        best_slot = int(args.best_slot or 32)
        print(f"[PRAG:slot-to-epoch] dry-run placeholder best_slot={best_slot}", flush=True)
    else:
        best_slot = read_best_slot(Path(args.slot_report_root) / "kv_slot_ablation_table.csv", args.best_metric)

    run_command(epoch_experiment_command(args, best_slot), dry_run=args.dry_run)
    print("\n[PRAG:slot-to-epoch] done", flush=True)


if __name__ == "__main__":
    main()
