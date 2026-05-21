"""Run the final BridgePRAG entity-multifact experiment end to end.

The defaults follow the professor-feedback revision plan:

* larger synthetic entity-multifact dataset (8000 roots),
* lower learning rate (2e-5),
* equal K/V slot count for both variants (64),
* fixed critical layers from the previous verified scans (QP=23, P-only=20),
* epoch-wise evaluation and final paper artifacts focused on Accuracy.

The script is a thin orchestration layer over the existing PRAG modules, so the
actual data generation, training, checkpointing, and comparison logic remains
identical to the normal manual commands.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen2.5-3B"
DEFAULT_AUGMENT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_EXPERIMENT = "entity_simple8000_lr2e5_kv64_ep15"


def run_command(cmd: list[str], *, dry_run: bool = False) -> None:
    print(f"\n[PRAG:entity-final] $ {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def maybe_skip_file(path: Path, *, expected_lines: int | None, resume: bool, label: str) -> bool:
    if not resume or not path.exists():
        return False
    if expected_lines is None:
        print(f"[PRAG:entity-final] resume skip {label}: {path}", flush=True)
        return True
    actual = count_lines(path)
    if actual >= expected_lines:
        print(f"[PRAG:entity-final] resume skip {label}: {path} rows={actual}", flush=True)
        return True
    print(f"[PRAG:entity-final] resume continue {label}: {path} rows={actual}/{expected_lines}", flush=True)
    return False


def paths_for(args: argparse.Namespace) -> dict[str, Path]:
    data_root = Path(args.data_root)
    name = args.experiment_name
    return {
        "seed": data_root / f"PRAG_{name}_seed.jsonl",
        "augmented": data_root / f"PRAG_{name}_augmented_all.jsonl",
        "normalized": data_root / f"PRAG_{name}_normalized_all.jsonl",
        "train": data_root / f"PRAG_{name}_train.jsonl",
        "valid": data_root / f"PRAG_{name}_valid.jsonl",
        "test": data_root / f"PRAG_{name}_test.jsonl",
    }


def build_seed_command(args: argparse.Namespace, seed_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "llm_server.PRAG.build_entity_multifact_seed",
        "--count",
        str(args.count),
        "--output",
        str(seed_path),
        "--facts-per-row",
        str(args.facts_per_row),
        "--final-variants",
        str(args.final_variants),
        "--source-prefix",
        args.source_prefix,
    ]


def augment_command(args: argparse.Namespace, seed_path: Path, augmented_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "llm_server.PRAG.augment_entity_multifact",
        "--input",
        str(seed_path),
        "--output",
        str(augmented_path),
        "--model",
        args.augment_model,
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
        "--resume",
    ]


def normalize_command(input_path: Path, output_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "llm_server.PRAG.normalize_entity_multifact_simple",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]


def split_command(args: argparse.Namespace, input_path: Path, train: Path, valid: Path, test: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "llm_server.PRAG.split_entity_multifact_dataset",
        "--input",
        str(input_path),
        "--train-output",
        str(train),
        "--valid-output",
        str(valid),
        "--test-output",
        str(test),
        "--train-roots",
        str(args.train_roots),
        "--valid-roots",
        str(args.valid_roots),
        "--test-roots",
        str(args.test_roots),
        "--seed",
        str(args.split_seed),
    ]


def related_experiment_command(args: argparse.Namespace, paths: dict[str, Path], report_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "llm_server.PRAG.run_related_merge_experiment",
        "--model",
        args.model,
        "--num-kv",
        str(args.num_kv),
        "--train",
        str(paths["train"]),
        "--valid",
        str(paths["valid"]),
        "--test-data",
        str(paths["test"]),
        "--qp-question-fusion",
        "kv_adapter",
        "--ponly-question-fusion",
        "none",
        "--qp-output-suffix",
        f"{args.experiment_name}_qp_kvadapt",
        "--ponly-output-suffix",
        f"{args.experiment_name}_ponly",
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--example-weight",
        "0.0",
        "--group-weight",
        "0.0",
        "--merge-weight",
        "1.0",
        "--merge-max-passages",
        str(args.merge_max_passages),
        "--answer-phrase-weight",
        str(args.answer_phrase_weight),
        "--answer-prefix-weight",
        "0.0",
        "--short-answer-weight",
        "0.0",
        "--eval-generation-samples",
        str(args.eval_generation_samples),
        "--eval-generation-every",
        str(args.eval_generation_every),
        "--eval-generation-max-new-tokens",
        str(args.eval_generation_max_new_tokens),
        "--test-max-cases",
        str(args.test_max_cases),
        "--test-max-new-tokens",
        str(args.test_max_new_tokens),
        "--test-prompt-style",
        args.test_prompt_style,
        "--scoring-style",
        args.scoring_style,
        "--report-dir",
        str(report_dir),
        "--epoch-checkpoint-root",
        args.epoch_checkpoint_root,
        "--eval-epoch-performance",
        "--qp-critical-layer",
        str(args.qp_critical_layer),
        "--ponly-critical-layer",
        str(args.ponly_critical_layer),
        "--quiet-eval",
        "--resume",
    ]


def train_log_path(output_suffix: str) -> Path:
    return Path("llm_server/PRAG") / f"prag_train_log_orthomerge_{output_suffix}.json"


def paper_artifact_command(args: argparse.Namespace, report_dir: Path, artifact_dir: Path) -> list[str]:
    qp_suffix = f"{args.experiment_name}_qp_kvadapt"
    ponly_suffix = f"{args.experiment_name}_ponly"
    return [
        sys.executable,
        "-m",
        "llm_server.PRAG.make_final_paper_artifacts",
        "--epoch-report-dir",
        str(report_dir / "epoch_performance"),
        "--qp-log",
        str(train_log_path(qp_suffix)),
        "--ponly-log",
        str(train_log_path(ponly_suffix)),
        "--output-dir",
        str(artifact_dir),
        "--max-epochs",
        str(args.epochs),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the revised final BridgePRAG experiment with one command.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--augment-model",
        default=DEFAULT_AUGMENT_MODEL,
        help="Model id served by the local augmentation vLLM endpoint. This can differ from --model.",
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--count", type=int, default=8000)
    parser.add_argument("--train-roots", type=int, default=6400)
    parser.add_argument("--valid-roots", type=int, default=800)
    parser.add_argument("--test-roots", type=int, default=800)
    parser.add_argument("--facts-per-row", type=int, default=3)
    parser.add_argument("--final-variants", type=int, default=1)
    parser.add_argument("--source-prefix", default="entity_multifact_final_seed")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--vllm-url", default="http://localhost:8001/v1/completions")
    parser.add_argument("--api-mode", choices=("auto", "completion", "chat"), default="auto")
    parser.add_argument("--augment-max-new-tokens", type=int, default=2400)
    parser.add_argument("--augment-timeout", type=float, default=180.0)
    parser.add_argument("--variants-per-row", type=int, default=1)
    parser.add_argument("--num-kv", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--qp-critical-layer", type=int, default=23)
    parser.add_argument("--ponly-critical-layer", type=int, default=20)
    parser.add_argument("--merge-max-passages", type=int, default=4)
    parser.add_argument("--answer-phrase-weight", type=float, default=7.0)
    parser.add_argument("--eval-generation-samples", type=int, default=20)
    parser.add_argument("--eval-generation-every", type=int, default=1000)
    parser.add_argument("--eval-generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--test-max-cases", type=int, default=800)
    parser.add_argument("--test-max-new-tokens", type=int, default=128)
    parser.add_argument("--test-prompt-style", choices=("service", "short-chat", "memory-cued", "paper"), default="service")
    parser.add_argument("--scoring-style", choices=("prag", "mergeprag"), default="prag")
    parser.add_argument("--report-root", default="llm_server/PRAG/reports")
    parser.add_argument("--epoch-checkpoint-root", default="llm_server/PRAG/epoch_checkpoints")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-augment", action="store_true")
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    expected_total = args.train_roots + args.valid_roots + args.test_roots
    if args.count < expected_total:
        raise SystemExit(f"--count must be >= split roots ({args.count} < {expected_total})")

    paths = paths_for(args)
    report_dir = Path(args.report_root) / args.experiment_name
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else report_dir / "paper_artifacts"

    if not args.skip_build and not maybe_skip_file(paths["seed"], expected_lines=args.count, resume=args.resume, label="seed build"):
        run_command(build_seed_command(args, paths["seed"]), dry_run=args.dry_run)

    if not args.skip_augment and not maybe_skip_file(paths["augmented"], expected_lines=args.count, resume=args.resume, label="augmentation"):
        run_command(augment_command(args, paths["seed"], paths["augmented"]), dry_run=args.dry_run)

    normalize_input = paths["seed"] if args.skip_augment else paths["augmented"]
    if not args.skip_normalize and not maybe_skip_file(paths["normalized"], expected_lines=args.count, resume=args.resume, label="normalization"):
        run_command(normalize_command(normalize_input, paths["normalized"]), dry_run=args.dry_run)

    split_input = paths["normalized"] if not args.skip_normalize else normalize_input
    split_ready = (
        maybe_skip_file(paths["train"], expected_lines=args.train_roots, resume=args.resume, label="train split")
        and maybe_skip_file(paths["valid"], expected_lines=args.valid_roots, resume=args.resume, label="valid split")
        and maybe_skip_file(paths["test"], expected_lines=args.test_roots, resume=args.resume, label="test split")
    )
    if not args.skip_split and not split_ready:
        run_command(split_command(args, split_input, paths["train"], paths["valid"], paths["test"]), dry_run=args.dry_run)

    if not args.skip_train:
        run_command(related_experiment_command(args, paths, report_dir), dry_run=args.dry_run)

    if not args.skip_artifacts:
        run_command(paper_artifact_command(args, report_dir, artifact_dir), dry_run=args.dry_run)

    print(f"\n[PRAG:entity-final] done", flush=True)
    print(f"[PRAG:entity-final] report_dir={report_dir}", flush=True)
    print(f"[PRAG:entity-final] paper_artifacts={artifact_dir}", flush=True)


if __name__ == "__main__":
    main()
