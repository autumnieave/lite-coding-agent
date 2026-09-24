"""Kaggle T4 QLoRA SFT runner for Qwen2.5-1.5B-Instruct.

The runner is intentionally thin: model loading, 4-bit quantization and TRL wiring live in
`training/train_sft.py`. This file only resolves Kaggle mount paths and records the run.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path


def _default_repo() -> Path:
    env = os.environ.get("LITE_AGENT_REPO")
    if env:
        return Path(env)
    mounted = Path("/kaggle/input/lite-coding-agent")
    if mounted.exists():
        return mounted
    return Path(__file__).resolve().parents[2]


def _preflight(paths: Sequence[tuple[str, Path]]) -> None:
    missing = [f"{label}: {path}" for label, path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Kaggle 挂载路径缺失：\n" + "\n".join(missing))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kaggle T4 QLoRA SFT")
    parser.add_argument("--repo", default=str(_default_repo()))
    parser.add_argument(
        "--config", default="training/configs/qlora_1_5b_t4.json", help="相对 repo 或绝对路径"
    )
    parser.add_argument("--model-path", default="/kaggle/input/qwen2.5-1.5b-instruct")
    parser.add_argument("--train-file", default="/kaggle/input/lite-agent-training/train.jsonl")
    parser.add_argument("--val-file", default="/kaggle/input/lite-agent-training/val.jsonl")
    parser.add_argument("--output-dir", default="/kaggle/working/sft_adapter")
    parser.add_argument(
        "--eval-after-train",
        action="store_true",
        help="训练后在同一 kernel 内跑 baseline vs adapter 的全链路 held-out 对比",
    )
    parser.add_argument("--heldout-out-root", default="/kaggle/working/heldout")
    parser.add_argument("--heldout-trace-root", default="/kaggle/working/heldout_traces")
    parser.add_argument("--heldout-rollouts", type=int, default=10)
    parser.add_argument("--heldout-max-model-len", type=int, default=8192)
    parser.add_argument("--train-version", default="train-v2")
    parser.add_argument(
        "--install-heldout-runtime",
        action="store_true",
        help="训练后安装 vLLM；只影响 held-out，不进入 core 依赖",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from training.train_sft import run_sft

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo / config_path
    model_path = Path(args.model_path)
    train_file = Path(args.train_file)
    val_file = Path(args.val_file)
    output_dir = Path(args.output_dir)
    _preflight(
        [
            ("repo", repo),
            ("config", config_path),
            ("model", model_path),
            ("train", train_file),
            ("val", val_file),
        ]
    )
    manifest = run_sft(
        config_path=config_path,
        model_path=model_path,
        train_file=train_file,
        val_file=val_file,
        output_dir=output_dir,
        dry_run=bool(args.dry_run),
    )
    if args.eval_after_train and not args.dry_run:
        import gc

        import torch

        from training.adapter_artifacts import validate_saved_adapter
        from training.kaggle.heldout_compare import run_compare

        validate_saved_adapter(output_dir)
        if args.install_heldout_runtime:
            import subprocess

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", "vllm==0.12.0"],
                check=True,
                timeout=1800,
            )
        gc.collect()
        torch.cuda.empty_cache()
        comparison = run_compare(
            repo=repo,
            model=str(model_path),
            val_file=val_file,
            adapter_path=output_dir,
            out_root=Path(args.heldout_out_root),
            trace_root=Path(args.heldout_trace_root),
            rollouts=args.heldout_rollouts,
            train_version=args.train_version,
            max_model_len=args.heldout_max_model_len,
        )
        print(comparison)
    print("SFT runner finished; only the LoRA adapter should be written to /kaggle/working.")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
