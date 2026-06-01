"""Run a subset-sized BridgePRAG experiment plus epoch-0 train/valid figures.

This wrapper keeps the Windows paper workflow to one command. It creates a
fixed-size training subset, trains Question+Passage and Passage-only with the
same selected critical layer, saves the step-0 checkpoints, and then writes
train/validation figures that include epoch 0.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> None:
    print(f"\n[PRAG:subset-full] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def normalize_lr_tag(lr: str) -> str:
    return str(lr).replace("e-", "em").replace("e+", "ep").replace(".", "p").replace("-", "m")


def ensure_train_subset(source: Path, target: Path, count: int) -> None:
    if target.exists():
        rows = target.read_text(encoding="utf-8").splitlines()
        if len(rows) == count:
            print(f"[PRAG:subset-full] train subset exists: {target} rows={count}")
            return
        print(f"[PRAG:subset-full] rewriting train subset: {target} rows={len(rows)} -> {count}")

    source_rows = source.read_text(encoding="utf-8").splitlines()
    if len(source_rows) < count:
        raise RuntimeError(f"Not enough rows in {source}: required={count}, found={len(source_rows)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(source_rows[:count]) + "\n", encoding="utf-8")
    print(f"[PRAG:subset-full] train subset saved: {target} rows={count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run subset BridgePRAG training, test evaluation, and epoch-0 train/validation figures."
    )
    parser.add_argument("--source-train", default="data/PRAG_entity_multifact_simple_train4000.jsonl")
    parser.add_argument("--valid", default="data/PRAG_entity_multifact_simple_valid4000.jsonl")
    parser.add_argument("--test-data", default="data/PRAG_entity_multifact_simple_hard_test400.jsonl")
    parser.add_argument("--train-subset-size", type=int, default=900)
    parser.add_argument("--eval-cases", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", default="1e-5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-kv", type=int, default=64)
    parser.add_argument("--critical-layer", type=int, default=23)
    parser.add_argument("--combined-max-epoch", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    train_path = Path(f"data/PRAG_entity_multifact_simple_train{args.train_subset_size}_seed{args.seed}.jsonl")
    ensure_train_subset(Path(args.source_train), train_path, args.train_subset_size)

    lr_tag = normalize_lr_tag(args.lr)
    run_name = (
        f"entity_simple{args.train_subset_size}_lr{lr_tag}_seed{args.seed}"
        f"_layer{args.critical_layer}_kv{args.num_kv}_ep{args.epochs}"
    )
    qp_suffix = f"{run_name}_qp_kvadapt"
    ponly_suffix = f"{run_name}_ponly"
    test_report_dir = f"llm_server/PRAG/reports/{run_name}_test{args.eval_cases}"
    train_valid_report_dir = f"llm_server/PRAG/reports/{run_name}_train_valid{args.eval_cases}"
    step_checkpoint_root = "llm_server/PRAG/step_checkpoints"
    epoch_checkpoint_root = "llm_server/PRAG/epoch_checkpoints"

    train_cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.run_related_merge_experiment",
        "--num-kv",
        str(args.num_kv),
        "--train",
        str(train_path),
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
        "--seed",
        str(args.seed),
        "--test-max-cases",
        str(args.eval_cases),
        "--test-max-new-tokens",
        "128",
        "--report-dir",
        test_report_dir,
        "--epoch-checkpoint-root",
        epoch_checkpoint_root,
        "--save-step-checkpoints",
        "--step-checkpoint-root",
        step_checkpoint_root,
        "--step-checkpoint-steps",
        "0",
        "--eval-epoch-performance",
        "--make-paper-artifacts",
        "--qp-critical-layer",
        str(args.critical_layer),
        "--ponly-critical-layer",
        str(args.critical_layer),
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
        f"{epoch_checkpoint_root}/{qp_suffix}",
        "--ponly-epoch-dir",
        f"{epoch_checkpoint_root}/{ponly_suffix}",
        "--qp-step0-weights",
        f"{step_checkpoint_root}/{qp_suffix}/step_000000.pt",
        "--ponly-step0-weights",
        f"{step_checkpoint_root}/{ponly_suffix}/step_000000.pt",
        "--include-epoch0",
        "--train",
        str(train_path),
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
        "--combined-max-epoch",
        str(args.combined_max_epoch),
    ]
    train_valid_cmd.append("--resume" if args.resume else "--no-resume")
    run_command(train_valid_cmd)

    print("[PRAG:subset-full] done")
    print(f"[PRAG:subset-full] test artifacts: {test_report_dir}")
    print(f"[PRAG:subset-full] train/valid artifacts: {train_valid_report_dir}/paper_artifacts")


if __name__ == "__main__":
    main()
