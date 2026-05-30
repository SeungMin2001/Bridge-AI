"""Run the step-300 BridgePRAG experiment and train/validation figures.

This wrapper exists to keep the Windows demo/paper workflow to one short
command.  It creates the 300-row training subset if needed, runs the paired
Question+Passage vs Passage-only experiment, then evaluates epoch checkpoints
on train/validation splits and writes the four paper figures.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RUN_NAME = "entity_simple300_lr1em5_kv64_ep10"


def run_command(cmd: list[str]) -> None:
    print(f"\n[PRAG:step300-full] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def ensure_train_subset(source: Path, target: Path, count: int) -> None:
    if target.exists():
        rows = target.read_text(encoding="utf-8").splitlines()
        if len(rows) == count:
            print(f"[PRAG:step300-full] train subset exists: {target} rows={count}")
            return
        print(f"[PRAG:step300-full] rewriting train subset: {target} rows={len(rows)} -> {count}")

    source_rows = source.read_text(encoding="utf-8").splitlines()
    if len(source_rows) < count:
        raise RuntimeError(f"Not enough rows in {source}: required={count}, found={len(source_rows)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(source_rows[:count]) + "\n", encoding="utf-8")
    print(f"[PRAG:step300-full] train subset saved: {target} rows={count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run step-300 BridgePRAG training, test evaluation, and train/validation paper figures."
    )
    parser.add_argument("--source-train", default="data/PRAG_entity_multifact_simple_train4000.jsonl")
    parser.add_argument("--train", default="data/PRAG_entity_multifact_simple_train300_seed42.jsonl")
    parser.add_argument("--valid", default="data/PRAG_entity_multifact_simple_valid4000.jsonl")
    parser.add_argument("--test-data", default="data/PRAG_entity_multifact_simple_hard_test400.jsonl")
    parser.add_argument("--train-subset-size", type=int, default=300)
    parser.add_argument("--eval-cases", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", default="1e-5")
    parser.add_argument("--num-kv", type=int, default=64)
    parser.add_argument("--qp-critical-layer", type=int, default=23)
    parser.add_argument("--ponly-critical-layer", type=int, default=20)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    ensure_train_subset(Path(args.source_train), Path(args.train), args.train_subset_size)

    qp_suffix = f"{RUN_NAME}_qp_kvadapt"
    ponly_suffix = f"{RUN_NAME}_ponly"
    test_report_dir = f"llm_server/PRAG/reports/{RUN_NAME}_test{args.eval_cases}"
    train_valid_report_dir = f"llm_server/PRAG/reports/{RUN_NAME}_train_valid{args.eval_cases}"

    train_cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.run_related_merge_experiment",
        "--num-kv",
        str(args.num_kv),
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
        qp_suffix,
        "--ponly-output-suffix",
        ponly_suffix,
        "--epochs",
        str(args.epochs),
        "--lr",
        args.lr,
        "--test-max-cases",
        str(args.eval_cases),
        "--test-max-new-tokens",
        "128",
        "--report-dir",
        test_report_dir,
        "--epoch-checkpoint-root",
        "llm_server/PRAG/epoch_checkpoints",
        "--eval-epoch-performance",
        "--make-paper-artifacts",
        "--qp-critical-layer",
        str(args.qp_critical_layer),
        "--ponly-critical-layer",
        str(args.ponly_critical_layer),
        "--eval-generation-samples",
        "0",
        "--quiet-eval",
    ]
    if args.resume:
        train_cmd.append("--resume")
    else:
        train_cmd.extend(
            [
                "--no-resume-train",
                "--no-resume-eval",
                "--no-resume-epoch-eval",
                "--no-resume-step-eval",
                "--no-resume-scan",
            ]
        )
    run_command(train_cmd)

    train_valid_cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.run_train_valid_epoch_artifacts",
        "--qp-epoch-dir",
        f"llm_server/PRAG/epoch_checkpoints/{qp_suffix}",
        "--ponly-epoch-dir",
        f"llm_server/PRAG/epoch_checkpoints/{ponly_suffix}",
        "--train",
        args.train,
        "--valid",
        args.valid,
        "--train-max-cases",
        str(args.eval_cases),
        "--valid-max-cases",
        str(args.eval_cases),
        "--qp-log",
        f"llm_server/PRAG/prag_train_log_orthomerge_{qp_suffix}.json",
        "--ponly-log",
        f"llm_server/PRAG/prag_train_log_orthomerge_{ponly_suffix}.json",
        "--output-root",
        train_valid_report_dir,
    ]
    train_valid_cmd.append("--resume" if args.resume else "--no-resume")
    run_command(train_valid_cmd)

    print("[PRAG:step300-full] done")
    print(f"[PRAG:step300-full] test artifacts: {test_report_dir}")
    print(f"[PRAG:step300-full] train/valid artifacts: {train_valid_report_dir}/paper_artifacts")


if __name__ == "__main__":
    main()
