"""Config-driven Kaggle runner for SFT/DPO method comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HELDOUT_TASKS = (
    "U-TR-A-C1-HARD",
    "U-TR-B-CHECK-BASE",
    "U-TR-B-C1-BASE",
    "U-TR-B-CTX-HARD",
    "U-TR-B-TOOL-HARD",
    "U-TR-B-CONFLICT-WEAK",
)


def _default_repo() -> Path:
    mounted = Path("/kaggle/input/lite-agent-train-v4/repo")
    if mounted.exists():
        return mounted
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kaggle SFT/DPO method runner")
    parser.add_argument("--repo", default=str(_default_repo()))
    parser.add_argument("--mode", choices=("sft", "dpo"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", default="/kaggle/input/qwen2.5-7b-instruct")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loss-mode", choices=("full", "assistant_only"), default="full")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--eval-after-train", action="store_true")
    parser.add_argument("--heldout-rollouts", type=int, default=10)
    parser.add_argument("--heldout-out-root", required=True)
    parser.add_argument("--heldout-trace-root", required=True)
    parser.add_argument(
        "--heldout-tasks",
        default=",".join(HELDOUT_TASKS),
        help="comma-separated full-loop held-out tasks",
    )
    parser.add_argument("--train-version", default="train-v2")
    parser.add_argument(
        "--heldout-provider", choices=("offline", "transformers"), default="offline"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help="SFT adapter used to initialise DPO; also the DPO baseline adapter",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--expected-train-sha256", default=None)
    parser.add_argument("--expected-val-sha256", default=None)
    parser.add_argument("--expected-init-adapter-sha256", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    config = Path(args.config)
    if not config.is_absolute():
        config = repo / config
    model_path = Path(args.model_path)
    train_file = Path(args.train_file)
    val_file = Path(args.val_file)
    output_dir = Path(args.output_dir)
    init_adapter = Path(args.init_adapter) if args.init_adapter else None
    heldout_tasks = [item.strip() for item in args.heldout_tasks.split(",") if item.strip()]
    if not heldout_tasks:
        raise ValueError("--heldout-tasks must contain at least one task id")
    for path, label in (
        (repo, "repo"),
        (config, "config"),
        (model_path, "model"),
        (train_file, "train"),
        (val_file, "val"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} path does not exist: {path}")
    if init_adapter is not None and not (init_adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"init adapter missing adapter_config.json: {init_adapter}")
    if args.mode == "dpo" and init_adapter is None:
        raise ValueError("DPO method comparison requires --init-adapter from the SFT baseline")

    if args.mode == "sft":
        from training.train_sft import run_sft

        run_manifest: dict[str, Any] = run_sft(
            config_path=config,
            model_path=model_path,
            train_file=train_file,
            val_file=val_file,
            output_dir=output_dir,
            dry_run=bool(args.dry_run),
            loss_mode=args.loss_mode,
            seed=args.seed,
        )
    else:
        from training.train_dpo import run_dpo

        run_manifest = run_dpo(
            config_path=config,
            model_path=model_path,
            train_file=train_file,
            val_file=val_file,
            output_dir=output_dir,
            dry_run=bool(args.dry_run),
            init_adapter=init_adapter,
            max_steps=args.max_steps,
            expected_train_sha256=args.expected_train_sha256,
            expected_val_sha256=args.expected_val_sha256,
            expected_init_adapter_sha256=args.expected_init_adapter_sha256,
        )

    result: dict[str, Any] = {
        "mode": args.mode,
        "loss_mode": args.loss_mode,
        "seed": args.seed,
        "config": str(config),
        "train_file": str(train_file),
        "val_file": str(val_file),
        "heldout_tasks": heldout_tasks,
        "init_adapter": str(init_adapter) if init_adapter is not None else None,
        "max_steps": args.max_steps,
        "expected_sha256": {
            "train": args.expected_train_sha256,
            "val": args.expected_val_sha256,
            "init_adapter": args.expected_init_adapter_sha256,
        },
        "output_dir": str(output_dir),
        "run_manifest": run_manifest,
    }
    if args.eval_after_train and not args.dry_run:
        from training.kaggle.heldout_compare import run_compare

        comparison = run_compare(
            repo=repo,
            model=str(model_path),
            val_file=val_file,
            adapter_path=output_dir,
            out_root=Path(args.heldout_out_root),
            trace_root=Path(args.heldout_trace_root),
            rollouts=args.heldout_rollouts,
            train_version=args.train_version,
            provider=args.heldout_provider,
            tasks=heldout_tasks,
            baseline_adapter_path=init_adapter if args.mode == "dpo" else None,
        )
        result["comparison"] = comparison
    result_path = output_dir / "method_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
