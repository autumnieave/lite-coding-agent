"""Run the v10 step-5 DPO arms (equal vs severity-weighted) from one adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.train_dpo import run_dpo  # noqa: E402

ARMS: tuple[tuple[str, str], ...] = (("equal", "train.jsonl"), ("weighted", "train_weighted.jsonl"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_arms(data_dir: Path) -> list[tuple[str, Path]]:
    if not (data_dir / "val.jsonl").is_file():
        raise FileNotFoundError(f"missing val.jsonl in {data_dir}")
    planned: list[tuple[str, Path]] = []
    for name, filename in ARMS:
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing arm data: {path}")
        planned.append((name, path))
    return planned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v10 step5 severity-weighted DPO")
    parser.add_argument(
        "--config", type=Path, default=Path("training/configs/dpo_v10_conflict.json")
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path("/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    )
    parser.add_argument("--init-adapter", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("private/v10/train_v10_conflict"))
    parser.add_argument("--out-dir", type=Path, default=Path("private/v10/dpo_severity_v10"))
    parser.add_argument("--expected-init-adapter-sha256", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    planned = plan_arms(args.data_dir)
    val_file = args.data_dir / "val.jsonl"
    results: dict[str, Any] = {"arms": {}, "data_dir": str(args.data_dir)}
    if args.init_adapter is not None:
        weight = args.init_adapter / "adapter_model.safetensors"
        results["init_adapter"] = str(args.init_adapter)
        results["init_adapter_sha256"] = sha256_file(weight) if weight.is_file() else None
    for name, train_file in planned:
        manifest = run_dpo(
            config_path=args.config,
            model_path=args.model_path,
            train_file=train_file,
            val_file=val_file,
            output_dir=args.out_dir / name,
            dry_run=args.dry_run,
            init_adapter=args.init_adapter,
            max_steps=args.max_steps,
            expected_train_sha256=sha256_file(train_file),
            expected_val_sha256=sha256_file(val_file),
            expected_init_adapter_sha256=args.expected_init_adapter_sha256,
        )
        results["arms"][name] = {
            "train_file": str(train_file),
            "train_sha256": manifest.get("train_sha256"),
            "optimizer_steps": manifest.get("optimizer_steps"),
            "status": manifest.get("status"),
            "output_dir": str(args.out_dir / name),
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "arms_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
