"""Train the v14 step-3.5 light SFT start policy."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.train_sft import run_sft  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v14 light SFT start policy")
    parser.add_argument("--config", type=Path, default=Path("training/configs/sft_v14_light.json"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("private/v14/step3_5_light_sft/data/train.jsonl"),
    )
    parser.add_argument(
        "--val-file", type=Path, default=Path("private/v14/step3_5_light_sft/data/val.jsonl")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("private/v14/step3_5_light_sft/adapter")
    )
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_sft(
        config_path=args.config,
        model_path=args.model_path,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
