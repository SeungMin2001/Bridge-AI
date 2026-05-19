"""One-command KorQuAD-based BridgePRAG pipeline.

Pipeline stages:
1. Build KorQuAD-derived simple multi-fact seed rows.
2. LLM-augment the seed rows with the existing entity-multifact augmenter.
3. Split the augmented rows into train/valid/test without root leakage.
4. Run K/V-slot search with critical-layer scans.
5. Train/evaluate final QP and passage-only models with epoch-wise figures.

Every stage is subprocess-based so the produced checkpoints, logs, reports,
and figures stay compatible with the manual commands already used in this repo.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], *, dry_run: bool = False) -> None:
    print(f"\n[PRAG:korquad-bridge] $ {' '.join(cmd)}", flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def model_tag(model: str) -> str:
    text = str(model or "model").lower()
    text = text.replace("qwen/qwen", "qwen")
    text = text.replace("2.5", "25").replace("3.5", "35")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "model"


def read_best_slot(table_path: Path, metric: str) -> int:
    metric_key = {
        "hit": "qp_mergeprag_hit",
        "f1": "qp_token_f1",
        "qa": "qp_qa_score",
    }[metric]
    if not table_path.exists():
        raise FileNotFoundError(f"Missing slot table: {table_path}")
    with table_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty slot table: {table_path}")

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
        "[PRAG:korquad-bridge] best_slot="
        f"{best_slot} metric={metric} "
        f"hit={float(best.get('qp_mergeprag_hit') or 0.0):.4f} "
        f"f1={float(best.get('qp_token_f1') or 0.0):.4f} "
        f"qa={float(best.get('qp_qa_score') or 0.0):.4f}",
        flush=True,
    )
    return best_slot


def build_seed_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.build_korquad_multifact_seed",
        "--dataset",
        args.dataset,
        "--split",
        args.source_split,
        "--output",
        str(args.seed_output),
        "--count",
        str(args.total_roots),
        "--facts-per-row",
        str(args.facts_per_row),
        "--min-facts-per-row",
        str(args.min_facts_per_row),
        "--evidence-chars",
        str(args.evidence_chars),
        "--source-prefix",
        args.source_prefix,
        "--seed",
        str(args.seed),
    ]
    if args.input:
        cmd.extend(["--input", args.input])
    if args.max_source_records > 0:
        cmd.extend(["--max-source-records", str(args.max_source_records)])
    return cmd


def augment_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.augment_entity_multifact",
        "--input",
        str(args.seed_output),
        "--output",
        str(args.augmented_output),
        "--model",
        args.augment_model or args.model,
        "--vllm-url",
        args.vllm_url,
        "--api-mode",
        args.api_mode,
        "--max-new-tokens",
        str(args.augment_max_new_tokens),
        "--timeout",
        str(args.augment_timeout),
        "--variants-per-row",
        str(args.variants_per_row),
    ]
    if args.augment_limit > 0:
        cmd.extend(["--limit", str(args.augment_limit)])
    cmd.append("--resume" if args.resume else "--no-resume")
    return cmd


def split_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "llm_server.PRAG.split_entity_multifact_dataset",
        "--input",
        str(args.augmented_output),
        "--train-output",
        str(args.train_output),
        "--valid-output",
        str(args.valid_output),
        "--test-output",
        str(args.test_output),
        "--train-roots",
        str(args.train_roots),
        "--valid-roots",
        str(args.valid_roots),
        "--test-roots",
        str(args.test_roots),
        "--seed",
        str(args.split_seed),
    ]


def slot_search_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.run_kv_slot_ablation",
        "--k-values",
        args.k_values,
        "--ablate-target",
        args.ablate_target,
        "--train",
        str(args.train_output),
        "--valid",
        str(args.valid_output),
        "--test-data",
        str(args.test_output),
        "--suffix-prefix",
        args.slot_suffix_prefix,
        "--epochs",
        str(args.slot_epochs),
        "--test-max-cases",
        str(args.slot_test_max_cases),
        "--test-max-new-tokens",
        str(args.test_max_new_tokens),
        "--report-root",
        str(args.slot_report_root),
        "--model",
        args.model,
        "--scan-root",
        args.scan_root,
        "--ponly-baseline-num-kv",
        str(args.ponly_baseline_num_kv),
        "--scan-steps",
        str(args.scan_steps),
        "--scan-layers",
        args.scan_layers,
        "--lr",
        str(args.lr),
    ]
    if args.qp_critical_layer is not None:
        cmd.extend(["--qp-critical-layer", str(args.qp_critical_layer)])
    if args.ponly_critical_layer is not None:
        cmd.extend(["--ponly-critical-layer", str(args.ponly_critical_layer)])
    if args.reuse_critical_layers:
        cmd.append("--reuse-critical-layers")
    if args.resume:
        cmd.append("--resume")
    if args.quiet_eval:
        cmd.append("--quiet-eval")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def final_experiment_command(args: argparse.Namespace, best_slot: int) -> list[str]:
    suffix_tag = f"{args.epoch_suffix_prefix}_kv{best_slot}_ep{args.epoch_epochs}"
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.run_related_merge_experiment",
        "--model",
        args.model,
        "--num-kv",
        str(best_slot),
        "--train",
        str(args.train_output),
        "--valid",
        str(args.valid_output),
        "--test-data",
        str(args.test_output),
        "--qp-question-fusion",
        "kv_adapter",
        "--ponly-question-fusion",
        "none",
        "--qp-output-suffix",
        f"{suffix_tag}_qp_kvadapt",
        "--ponly-output-suffix",
        f"{suffix_tag}_ponly",
        "--qp-scan-output",
        str(Path(args.scan_root) / f"critical_layers_{suffix_tag}_qp_kvadapt.json"),
        "--ponly-scan-output",
        str(Path(args.scan_root) / f"critical_layers_{suffix_tag}_ponly.json"),
        "--epochs",
        str(args.epoch_epochs),
        "--lr",
        str(args.lr),
        "--scan-steps",
        str(args.scan_steps),
        "--scan-layers",
        args.scan_layers,
        "--test-max-cases",
        str(args.epoch_test_max_cases),
        "--test-max-new-tokens",
        str(args.test_max_new_tokens),
        "--report-dir",
        str(Path(args.epoch_report_root) / f"kv{best_slot}"),
        "--epoch-checkpoint-root",
        args.epoch_checkpoint_root,
        "--eval-epoch-performance",
    ]
    if args.qp_critical_layer is not None:
        cmd.extend(["--qp-critical-layer", str(args.qp_critical_layer)])
    if args.ponly_critical_layer is not None:
        cmd.extend(["--ponly-critical-layer", str(args.ponly_critical_layer)])
    if args.reuse_critical_layers:
        cmd.append("--reuse-critical-layers")
    if args.quiet_eval:
        cmd.append("--quiet-eval")
    if args.resume:
        cmd.append("--resume")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a separate KorQuAD-based BridgePRAG experiment end to end.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--augment-model", default="", help="Defaults to --model when omitted.")
    parser.add_argument("--dataset", default="KorQuAD/squad_kor_v1")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--input", default="", help="Optional local KorQuAD/SQuAD-style source file.")
    parser.add_argument("--experiment-name", default="", help="Output namespace. Defaults to korquad_multifact_<model_tag>.")
    parser.add_argument("--train-roots", type=int, default=3200)
    parser.add_argument("--valid-roots", type=int, default=400)
    parser.add_argument("--test-roots", type=int, default=400)
    parser.add_argument("--facts-per-row", type=int, default=3)
    parser.add_argument("--min-facts-per-row", type=int, default=3)
    parser.add_argument("--max-source-records", type=int, default=0)
    parser.add_argument("--evidence-chars", type=int, default=110)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--split-seed", type=int, default=20260519)
    parser.add_argument("--source-prefix", default="korquad_multifact_seed")
    parser.add_argument("--vllm-url", default="http://localhost:8001/v1/completions")
    parser.add_argument("--api-mode", choices=("auto", "completion", "chat"), default="auto")
    parser.add_argument("--augment-max-new-tokens", type=int, default=2400)
    parser.add_argument("--augment-timeout", type=float, default=180.0)
    parser.add_argument("--augment-limit", type=int, default=0)
    parser.add_argument("--variants-per-row", type=int, default=1)
    parser.add_argument("--k-values", default="1,2,4,8,16,32,48,64,80,96")
    parser.add_argument("--ablate-target", choices=("qp", "both"), default="qp")
    parser.add_argument("--slot-epochs", type=int, default=5)
    parser.add_argument("--slot-test-max-cases", type=int, default=400)
    parser.add_argument("--best-metric", choices=("hit", "f1", "qa"), default="hit")
    parser.add_argument("--ponly-baseline-num-kv", type=int, default=32)
    parser.add_argument("--epoch-epochs", type=int, default=10)
    parser.add_argument("--epoch-test-max-cases", type=int, default=400)
    parser.add_argument("--test-max-new-tokens", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--scan-steps", type=int, default=60)
    parser.add_argument("--scan-layers", default="all")
    parser.add_argument("--qp-critical-layer", type=int, default=None)
    parser.add_argument("--ponly-critical-layer", type=int, default=None)
    parser.add_argument("--reuse-critical-layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-augment", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-slot-search", action="store_true")
    parser.add_argument("--best-slot", type=int, default=0, help="Use directly when --skip-slot-search is set.")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    exp = args.experiment_name or f"korquad_multifact_{model_tag(args.model)}"
    total_roots = args.train_roots + args.valid_roots + args.test_roots
    args.total_roots = total_roots
    args.seed_output = Path(f"data/PRAG_{exp}_seed.jsonl")
    args.augmented_output = Path(f"data/PRAG_{exp}_augmented_all.jsonl")
    args.train_output = Path(f"data/PRAG_{exp}_train.jsonl")
    args.valid_output = Path(f"data/PRAG_{exp}_valid.jsonl")
    args.test_output = Path(f"data/PRAG_{exp}_test.jsonl")
    args.slot_suffix_prefix = f"{exp}_slot"
    args.epoch_suffix_prefix = f"{exp}_epoch"
    args.slot_report_root = Path(f"llm_server/PRAG/reports/{exp}/slot_ablation")
    args.epoch_report_root = Path(f"llm_server/PRAG/reports/{exp}/epoch_bestslot")
    args.epoch_checkpoint_root = f"llm_server/PRAG/epoch_checkpoints/{exp}"
    args.scan_root = f"llm_server/PRAG/{exp}_critical_layers"
    Path(args.scan_root).mkdir(parents=True, exist_ok=True)

    print(
        f"[PRAG:korquad-bridge] experiment={exp} model={args.model} "
        f"roots train/valid/test={args.train_roots}/{args.valid_roots}/{args.test_roots}",
        flush=True,
    )

    if not args.skip_build:
        seed_rows = count_jsonl(args.seed_output)
        if args.resume and seed_rows >= total_roots:
            print(f"[PRAG:korquad-bridge] reuse seed rows={seed_rows}: {args.seed_output}", flush=True)
        else:
            run_command(build_seed_command(args), dry_run=args.dry_run)

    if not args.skip_augment:
        run_command(augment_command(args), dry_run=args.dry_run)

    if not args.skip_split:
        run_command(split_command(args), dry_run=args.dry_run)

    if args.skip_final and args.skip_slot_search:
        print("[PRAG:korquad-bridge] done data stages only", flush=True)
        return

    if args.skip_slot_search:
        if args.best_slot <= 0:
            raise SystemExit("--best-slot is required when --skip-slot-search is set.")
        best_slot = int(args.best_slot)
        print(f"[PRAG:korquad-bridge] use provided best_slot={best_slot}", flush=True)
    else:
        run_command(slot_search_command(args), dry_run=args.dry_run)
        best_slot = int(args.best_slot or 32) if args.dry_run else read_best_slot(args.slot_report_root / "kv_slot_ablation_table.csv", args.best_metric)

    if not args.skip_final:
        run_command(final_experiment_command(args, best_slot), dry_run=args.dry_run)

    print("\n[PRAG:korquad-bridge] done", flush=True)


if __name__ == "__main__":
    main()
