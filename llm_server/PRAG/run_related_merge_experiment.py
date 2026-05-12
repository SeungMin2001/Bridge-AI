"""Run the related-merge PRAG comparison experiment end to end.

One command performs:
1. critical-layer scan for question+passage memory,
2. critical-layer scan for passage-only memory,
3. question+passage orthogonal-merge training,
4. passage-only orthogonal-merge training.

The script intentionally launches the existing modules as subprocesses instead
of duplicating training logic, so the saved logs/checkpoints remain identical to
normal manual runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_TRAIN = "data/PRAG_related_merge_augmented_train.jsonl"
DEFAULT_VALID = "data/PRAG_related_merge_augmented_valid.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B"
DEFAULT_QP_SUFFIX = "related_qwen25_3b_qp"
DEFAULT_PONLY_SUFFIX = "related_qwen25_3b_ponly"
DEFAULT_QP_SCAN = "llm_server/PRAG/critical_layers_related_merge_qwen25_3b_qp.json"
DEFAULT_PONLY_SCAN = "llm_server/PRAG/critical_layers_related_merge_qwen25_3b_ponly.json"


def run_command(cmd: list[str], *, dry_run: bool = False) -> None:
    printable = " ".join(cmd)
    print(f"\n[PRAG:related-experiment] $ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def read_best_layer(path: str | Path, *, label: str) -> int:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    layers = data.get("critical_layers") or []
    if not layers:
        raise RuntimeError(f"No critical_layers found in {path} for {label}")
    layer = int(layers[0])
    print(f"[PRAG:related-experiment] {label} critical_layer={layer} from {path}", flush=True)
    return layer


def scan_command(args: argparse.Namespace, *, output: str, question_conditioned: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.find_critical_layers",
        "--model",
        args.model,
        "--train",
        args.train,
        "--valid",
        args.valid,
        "--layers",
        args.scan_layers,
        "--steps",
        str(args.scan_steps),
        "--top-n",
        str(args.top_n),
        "--lr",
        str(args.scan_lr),
        "--answer-target",
        args.answer_target,
        "--final-weight",
        str(args.final_weight),
        "--injection-mode",
        args.injection_mode,
        "--output",
        output,
    ]
    cmd.append("--question-conditioned-memory" if question_conditioned else "--no-question-conditioned-memory")
    cmd.append("--resume" if args.resume_scan else "--no-resume")
    return cmd


def train_command(
    args: argparse.Namespace,
    *,
    output_suffix: str,
    critical_layer: int,
    question_conditioned: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.train",
        "--model",
        args.model,
        "--output-suffix",
        output_suffix,
        "--train",
        args.train,
        "--valid",
        args.valid,
        "--merge-aware",
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--positive-only",
        "--answer-target",
        args.answer_target,
        "--final-weight",
        str(args.final_weight),
        "--example-weight",
        str(args.example_weight),
        "--group-weight",
        str(args.group_weight),
        "--merge-weight",
        str(args.merge_weight),
        "--merge-max-passages",
        str(args.merge_max_passages),
        "--answer-phrase-weight",
        str(args.answer_phrase_weight),
        "--answer-prefix-weight",
        str(args.answer_prefix_weight),
        "--short-answer-weight",
        str(args.short_answer_weight),
        "--eval-generation-samples",
        str(args.eval_generation_samples),
        "--eval-generation-every",
        str(args.eval_generation_every),
        "--eval-generation-max-new-tokens",
        str(args.eval_generation_max_new_tokens),
        "--injection-mode",
        args.injection_mode,
        "--critical-layer",
        str(critical_layer),
    ]
    cmd.append("--question-conditioned-memory" if question_conditioned else "--no-question-conditioned-memory")
    cmd.append("--resume" if args.resume_train else "--no-resume")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find QP/P-only critical layers, then train both related-merge PRAG variants."
    )
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--valid", default=DEFAULT_VALID)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--scan-lr", type=float, default=1e-4)
    parser.add_argument("--scan-steps", type=int, default=60)
    parser.add_argument("--scan-layers", default="all")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--answer-target", choices=("answer", "full_answer"), default="full_answer")
    parser.add_argument("--final-weight", type=float, default=1.0)
    parser.add_argument("--example-weight", type=float, default=0.0)
    parser.add_argument("--group-weight", type=float, default=0.0)
    parser.add_argument("--merge-weight", type=float, default=1.0)
    parser.add_argument("--merge-max-passages", type=int, default=4)
    parser.add_argument("--answer-phrase-weight", type=float, default=7.0)
    parser.add_argument("--answer-prefix-weight", type=float, default=0.0)
    parser.add_argument("--short-answer-weight", type=float, default=0.0)
    parser.add_argument("--eval-generation-samples", type=int, default=20)
    parser.add_argument("--eval-generation-every", type=int, default=1000)
    parser.add_argument("--eval-generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--injection-mode", choices=("attention", "add_all", "add_last", "hybrid"), default="attention")
    parser.add_argument("--qp-output-suffix", default=DEFAULT_QP_SUFFIX)
    parser.add_argument("--ponly-output-suffix", default=DEFAULT_PONLY_SUFFIX)
    parser.add_argument("--qp-scan-output", default=DEFAULT_QP_SCAN)
    parser.add_argument("--ponly-scan-output", default=DEFAULT_PONLY_SCAN)
    parser.add_argument("--resume-scan", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_path = Path(args.train)
    valid_path = Path(args.valid)
    if not args.dry_run:
        if not train_path.exists():
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not valid_path.exists():
            raise FileNotFoundError(f"Valid file not found: {valid_path}")

    run_command(scan_command(args, output=args.qp_scan_output, question_conditioned=True), dry_run=args.dry_run)
    run_command(scan_command(args, output=args.ponly_scan_output, question_conditioned=False), dry_run=args.dry_run)

    qp_layer = 0 if args.dry_run else read_best_layer(args.qp_scan_output, label="question+passage")
    ponly_layer = 0 if args.dry_run else read_best_layer(args.ponly_scan_output, label="passage-only")

    run_command(
        train_command(
            args,
            output_suffix=args.qp_output_suffix,
            critical_layer=qp_layer,
            question_conditioned=True,
        ),
        dry_run=args.dry_run,
    )
    run_command(
        train_command(
            args,
            output_suffix=args.ponly_output_suffix,
            critical_layer=ponly_layer,
            question_conditioned=False,
        ),
        dry_run=args.dry_run,
    )
    print("\n[PRAG:related-experiment] done", flush=True)


if __name__ == "__main__":
    main()
