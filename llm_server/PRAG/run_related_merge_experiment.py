"""Run the related-merge PRAG comparison experiment end to end.

One command performs:
1. critical-layer scan for question+passage memory,
2. critical-layer scan for passage-only memory,
3. question+passage orthogonal-merge training,
4. passage-only orthogonal-merge training,
5. comparison evaluation of the two trained memories.

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
DEFAULT_TEST = "data/PRAG_related_merge_augmented_hard_test400_v2.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B"
DEFAULT_QP_SUFFIX = "related_qwen25_3b_qp"
DEFAULT_PONLY_SUFFIX = "related_qwen25_3b_ponly"
DEFAULT_QP_SCAN = "llm_server/PRAG/critical_layers_related_merge_qwen25_3b_qp.json"
DEFAULT_PONLY_SCAN = "llm_server/PRAG/critical_layers_related_merge_qwen25_3b_ponly.json"
DEFAULT_REPORT_DIR = "llm_server/PRAG/reports/related_qp_vs_ponly_hard400_v2_500_tok32"


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


def maybe_run_scan(args: argparse.Namespace, *, output: str, question_conditioned: bool, label: str) -> None:
    output_path = Path(output)
    if args.reuse_critical_layers and output_path.exists():
        print(
            f"[PRAG:related-experiment] reuse {label} critical layers: {output_path}",
            flush=True,
        )
        return
    if args.reuse_critical_layers and not output_path.exists():
        print(
            f"[PRAG:related-experiment] missing {label} critical layers, scanning once: {output_path}",
            flush=True,
        )
    run_command(scan_command(args, output=output, question_conditioned=question_conditioned), dry_run=args.dry_run)


def orthomerge_output_path(path: Path) -> Path:
    name = path.name
    if "_memory_" in name:
        name = name.replace("_memory_", "_orthomerge_memory_", 1)
    else:
        name = f"{path.stem}_orthomerge{path.suffix}"
    return path.with_name(name)


def tagged_output_path(path: Path, suffix: str) -> Path:
    suffix = str(suffix or "").strip().strip("_")
    if not suffix:
        return path
    name = path.name
    if "_memory_" in name:
        name = name.replace("_memory_", f"_{suffix}_memory_", 1)
    else:
        name = f"{path.stem}_{suffix}{path.suffix}"
    return path.with_name(name)


def checkpoint_path_for_suffix(output_suffix: str) -> Path:
    base = Path("llm_server/PRAG/prag_memory_checkpoint.pt")
    return tagged_output_path(orthomerge_output_path(base), output_suffix)


def train_log_path_for_suffix(output_suffix: str) -> Path:
    base = Path("llm_server/PRAG/prag_train_log.json")
    return tagged_output_path(orthomerge_output_path(base), output_suffix)


def epoch_dir_for_suffix(output_suffix: str, root: str) -> Path:
    return Path(root) / output_suffix


def scan_command(args: argparse.Namespace, *, output: str, question_conditioned: bool) -> list[str]:
    question_fusion = args.qp_question_fusion if question_conditioned else args.ponly_question_fusion
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
        "--num-kv",
        str(args.num_kv),
        "--question-fusion",
        question_fusion,
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
    question_fusion = args.qp_question_fusion if question_conditioned else args.ponly_question_fusion
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.train",
        "--model",
        args.model,
        "--output-suffix",
        output_suffix,
        "--num-kv",
        str(args.num_kv),
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
        "--question-fusion",
        question_fusion,
    ]
    if args.save_epoch_checkpoints:
        cmd.extend(
            [
                "--save-epoch-checkpoints",
                "--epoch-checkpoint-dir",
                str(epoch_dir_for_suffix(output_suffix, args.epoch_checkpoint_root)),
            ]
        )
    cmd.append("--question-conditioned-memory" if question_conditioned else "--no-question-conditioned-memory")
    cmd.append("--resume" if args.resume_train else "--no-resume")
    return cmd


def eval_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.compare_memory_variants",
        "--qp-weights",
        str(checkpoint_path_for_suffix(args.qp_output_suffix)),
        "--ponly-weights",
        str(checkpoint_path_for_suffix(args.ponly_output_suffix)),
        "--data",
        args.test_data,
        "--case-index",
        str(args.test_case_index),
        "--max-cases",
        str(args.test_max_cases),
        "--dataset-merge-max-passages",
        str(args.merge_max_passages),
        "--max-new-tokens",
        str(args.test_max_new_tokens),
        "--prompt-style",
        args.test_prompt_style,
        "--injection-mode",
        args.injection_mode,
        "--alpha",
        str(args.test_alpha),
        "--report-dir",
        args.report_dir,
    ]
    if args.include_loss:
        cmd.append("--include-loss")
    if args.include_baselines:
        cmd.append("--include-baselines")
    if args.quiet_eval:
        cmd.append("--quiet-cases")
    cmd.append("--resume" if args.resume_eval else "--no-resume")
    return cmd


def epoch_eval_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "llm_server.PRAG.plot_epoch_performance",
        "--qp-epoch-dir",
        str(epoch_dir_for_suffix(args.qp_output_suffix, args.epoch_checkpoint_root)),
        "--ponly-epoch-dir",
        str(epoch_dir_for_suffix(args.ponly_output_suffix, args.epoch_checkpoint_root)),
        "--data",
        args.test_data,
        "--case-index",
        str(args.test_case_index),
        "--max-cases",
        str(args.test_max_cases),
        "--dataset-merge-max-passages",
        str(args.merge_max_passages),
        "--max-new-tokens",
        str(args.test_max_new_tokens),
        "--prompt-style",
        args.test_prompt_style,
        "--injection-mode",
        args.injection_mode,
        "--alpha",
        str(args.test_alpha),
        "--report-dir",
        str(Path(args.report_dir) / "epoch_performance"),
        "--qp-log",
        str(train_log_path_for_suffix(args.qp_output_suffix)),
        "--ponly-log",
        str(train_log_path_for_suffix(args.ponly_output_suffix)),
    ]
    if args.include_loss:
        cmd.append("--include-loss")
    if args.quiet_eval:
        cmd.append("--quiet-cases")
    cmd.append("--resume" if args.resume_epoch_eval else "--no-resume")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find QP/P-only critical layers, then train both related-merge PRAG variants."
    )
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--valid", default=DEFAULT_VALID)
    parser.add_argument("--test-data", default=DEFAULT_TEST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-kv", type=int, default=16)
    parser.add_argument("--qp-question-fusion", choices=("auto", "none", "text_concat", "feature_concat", "kv_adapter"), default="kv_adapter")
    parser.add_argument("--ponly-question-fusion", choices=("auto", "none", "text_concat", "feature_concat", "kv_adapter"), default="none")
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
    parser.add_argument("--test-case-index", type=int, default=0)
    parser.add_argument("--test-max-cases", type=int, default=500)
    parser.add_argument("--test-max-new-tokens", type=int, default=32)
    parser.add_argument("--test-prompt-style", choices=("service", "short-chat", "memory-cued", "paper"), default="service")
    parser.add_argument("--test-alpha", type=float, default=1.0)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--include-loss", action="store_true")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--quiet-eval", action="store_true")
    parser.add_argument(
        "--save-epoch-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-epoch checkpoints during both QP and passage-only training.",
    )
    parser.add_argument(
        "--epoch-checkpoint-root",
        default="llm_server/PRAG/epoch_checkpoints",
        help="Root directory for per-epoch checkpoints.",
    )
    parser.add_argument(
        "--eval-epoch-performance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After final evaluation, evaluate each epoch checkpoint pair and plot paper-style epoch performance.",
    )
    parser.add_argument("--resume-epoch-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qp-output-suffix", default=DEFAULT_QP_SUFFIX)
    parser.add_argument("--ponly-output-suffix", default=DEFAULT_PONLY_SUFFIX)
    parser.add_argument("--qp-scan-output", default=DEFAULT_QP_SCAN)
    parser.add_argument("--ponly-scan-output", default=DEFAULT_PONLY_SCAN)
    parser.add_argument(
        "--qp-critical-layer",
        type=int,
        default=None,
        help="Use this fixed question+passage critical layer and skip the QP scan.",
    )
    parser.add_argument(
        "--ponly-critical-layer",
        type=int,
        default=None,
        help="Use this fixed passage-only critical layer and skip the passage-only scan.",
    )
    parser.add_argument(
        "--reuse-critical-layers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip layer scan when the configured critical-layer JSON already exists.",
    )
    parser.add_argument("--resume-scan", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true", help="Resume scan, train, and evaluation stages with one flag.")
    parser.add_argument("--skip-eval", action="store_true", help="Run only scan and training stages.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.resume:
        args.resume_scan = True
        args.resume_train = True
        args.resume_eval = True
        args.resume_epoch_eval = True
    if args.eval_epoch_performance:
        args.save_epoch_checkpoints = True

    default_qp_suffix_requested = args.qp_output_suffix == DEFAULT_QP_SUFFIX
    default_qp_scan_requested = args.qp_scan_output == DEFAULT_QP_SCAN
    default_report_requested = args.report_dir == DEFAULT_REPORT_DIR
    if args.num_kv != 16:
        kv_tag = f"kv{args.num_kv}"
        if default_qp_suffix_requested:
            args.qp_output_suffix = f"{DEFAULT_QP_SUFFIX}_{kv_tag}"
        if args.ponly_output_suffix == DEFAULT_PONLY_SUFFIX:
            args.ponly_output_suffix = f"{DEFAULT_PONLY_SUFFIX}_{kv_tag}"
        if default_qp_scan_requested:
            args.qp_scan_output = f"llm_server/PRAG/critical_layers_related_merge_qwen25_3b_qp_{kv_tag}.json"
        if args.ponly_scan_output == DEFAULT_PONLY_SCAN:
            args.ponly_scan_output = f"llm_server/PRAG/critical_layers_related_merge_qwen25_3b_ponly_{kv_tag}.json"
        if default_report_requested:
            args.report_dir = f"{DEFAULT_REPORT_DIR}_{kv_tag}"
    if args.qp_question_fusion == "feature_concat":
        if default_qp_suffix_requested and not args.qp_output_suffix.endswith("_fconcat"):
            args.qp_output_suffix = f"{args.qp_output_suffix}_fconcat"
        if default_qp_scan_requested:
            scan_path = Path(args.qp_scan_output)
            args.qp_scan_output = str(scan_path.with_name(f"{scan_path.stem}_fconcat{scan_path.suffix}"))
        if default_report_requested and not args.report_dir.endswith("_fconcat"):
            args.report_dir = f"{args.report_dir}_fconcat"
    if args.qp_question_fusion == "kv_adapter":
        if default_qp_suffix_requested and not args.qp_output_suffix.endswith("_kvadapt"):
            args.qp_output_suffix = f"{args.qp_output_suffix}_kvadapt"
        if default_qp_scan_requested:
            scan_path = Path(args.qp_scan_output)
            args.qp_scan_output = str(scan_path.with_name(f"{scan_path.stem}_kvadapt{scan_path.suffix}"))
        if default_report_requested and not args.report_dir.endswith("_kvadapt"):
            args.report_dir = f"{args.report_dir}_kvadapt"

    train_path = Path(args.train)
    valid_path = Path(args.valid)
    test_path = Path(args.test_data)
    if not args.dry_run:
        if not train_path.exists():
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not valid_path.exists():
            raise FileNotFoundError(f"Valid file not found: {valid_path}")
        if not args.skip_eval and not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")

    if args.qp_critical_layer is None:
        maybe_run_scan(args, output=args.qp_scan_output, question_conditioned=True, label="question+passage")
        qp_layer = 0 if args.dry_run else read_best_layer(args.qp_scan_output, label="question+passage")
    else:
        qp_layer = int(args.qp_critical_layer)
        print(f"[PRAG:related-experiment] fixed question+passage critical_layer={qp_layer}", flush=True)

    if args.ponly_critical_layer is None:
        maybe_run_scan(args, output=args.ponly_scan_output, question_conditioned=False, label="passage-only")
        ponly_layer = 0 if args.dry_run else read_best_layer(args.ponly_scan_output, label="passage-only")
    else:
        ponly_layer = int(args.ponly_critical_layer)
        print(f"[PRAG:related-experiment] fixed passage-only critical_layer={ponly_layer}", flush=True)

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
    if not args.skip_eval:
        run_command(eval_command(args), dry_run=args.dry_run)
        if args.eval_epoch_performance:
            run_command(epoch_eval_command(args), dry_run=args.dry_run)
    print("\n[PRAG:related-experiment] done", flush=True)


if __name__ == "__main__":
    main()
