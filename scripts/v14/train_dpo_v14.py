"""Run the v14 step-4 DPO arms: beta grid x equal / severity-weighted."""

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

from training.severity_dpo import SeverityWeightedDPOTrainer  # noqa: E402
from training.train_dpo import run_dpo  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v14 step4 DPO beta grid")
    parser.add_argument("--config", type=Path, default=Path("training/configs/dpo_v14.json"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    )
    parser.add_argument("--init-adapter", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("private/v14/step4_dpo/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("private/v14/step4_dpo/arms"))
    parser.add_argument("--betas", default="0.1,0.2,0.5")
    parser.add_argument("--arms", default="equal,weighted")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--expected-init-adapter-sha256", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_file = args.data_dir / "train.jsonl"
    val_file = args.data_dir / "val.jsonl"
    weights_file = args.data_dir / "severity_weights.json"
    for path in (train_file, val_file, weights_file):
        if not path.is_file():
            raise FileNotFoundError(f"missing DPO data file: {path}")
    weights = json.loads(weights_file.read_text(encoding="utf-8"))

    results: dict[str, Any] = {
        "config": str(args.config),
        "data_dir": str(args.data_dir),
        "train_sha256": sha256_file(train_file),
        "val_sha256": sha256_file(val_file),
        "train_pairs": sum(
            1 for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ),
        "val_pairs": sum(
            1 for line in val_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ),
        "arms": {},
    }
    if args.init_adapter is not None:
        weight_path = args.init_adapter / "adapter_model.safetensors"
        results["init_adapter"] = str(args.init_adapter)
        results["init_adapter_sha256"] = sha256_file(weight_path) if weight_path.is_file() else None

    for beta in [float(item) for item in args.betas.split(",") if item.strip()]:
        for arm in [item.strip() for item in args.arms.split(",") if item.strip()]:
            if arm not in {"equal", "weighted"}:
                raise ValueError(f"unknown arm: {arm}")
            name = f"{arm}_beta{beta:g}"
            manifest = run_dpo(
                config_path=args.config,
                model_path=args.model_path,
                train_file=train_file,
                val_file=val_file,
                output_dir=args.out_dir / name,
                dry_run=args.dry_run,
                init_adapter=args.init_adapter,
                max_steps=args.max_steps,
                expected_train_sha256=results["train_sha256"],
                expected_val_sha256=results["val_sha256"],
                expected_init_adapter_sha256=args.expected_init_adapter_sha256,
                trainer_cls=SeverityWeightedDPOTrainer if arm == "weighted" else None,
                severity_weights=weights if arm == "weighted" else None,
                overrides={
                    "beta": beta,
                    "num_train_epochs": args.epochs,
                    "label_smoothing": args.label_smoothing,
                },
            )
            results["arms"][name] = {
                "arm": arm,
                "beta": beta,
                "status": manifest.get("status"),
                "optimizer_steps": manifest.get("optimizer_steps"),
                "output_dir": str(args.out_dir / name),
                "train_loss": manifest.get("train_loss"),
                "severity_weighting": manifest.get("severity_weighting"),
            }
            print(json.dumps(results["arms"][name], ensure_ascii=False))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "arms_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
