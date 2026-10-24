"""Build the private Kaggle dataset bundle for method experiments."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections.abc import Iterable
from pathlib import Path

REPO_FILES = (
    "pyproject.toml",
    "training/requirements-kaggle.txt",
    "training/configs/qlora_7b_t4.json",
    "training/configs/dpo_7b_t4.json",
    "private/sft_data_v4/train.jsonl",
    "private/sft_data_v4/val.jsonl",
    "private/sft_data_v4/train_mix15.jsonl",
    "private/sft_data_v4/general_dolly.jsonl",
    "private/sft_data_v4/manifest.json",
    "private/sft_data_v4/mix_manifest.json",
    "private/sft_data_v4/TRAIN_V4_FROZEN.json",
    "private/sft_data_v4/TRAIN_V4_FROZEN.txt",
    "private/dpo_v1/dpo_train.jsonl",
    "private/dpo_v1/dpo_val.jsonl",
    "private/dpo_v1/dpo_pairs.jsonl",
    "private/dpo_v1/manifest.json",
)
MINIMAL_REPO_FILES = (
    "pyproject.toml",
    "training/requirements-kaggle.txt",
    "training/configs/qlora_7b_t4.json",
    "training/configs/dpo_7b_t4.json",
    "private/sft_data_v4/train.jsonl",
    "private/sft_data_v4/val.jsonl",
    "private/sft_data_v4/general_dolly.jsonl",
    "private/sft_data_v4/manifest.json",
    "private/sft_data_v4/mix_manifest.json",
    "private/sft_data_v4/TRAIN_V4_FROZEN.json",
    "private/sft_data_v4/TRAIN_V4_FROZEN.txt",
    "private/dpo_v1/dpo_train.jsonl",
    "private/dpo_v1/dpo_val.jsonl",
    "private/dpo_v1/manifest.json",
)
V5_MINIMAL_REPO_FILES = (
    "pyproject.toml",
    "training/requirements-kaggle.txt",
    "training/configs/qlora_7b_t4.json",
    "private/sft_data_v5/train.jsonl",
    "private/sft_data_v5/val.jsonl",
    "private/sft_data_v5/manifest.json",
    "private/sft_data_v5/TRAIN_V5_FROZEN.json",
    "private/sft_data_v5/TRAIN_V5_FROZEN.txt",
)
V5_REPO_FILES = (
    *V5_MINIMAL_REPO_FILES,
    "training/configs/dpo_7b_t4.json",
    "private/dpo_v1/dpo_train.jsonl",
    "private/dpo_v1/dpo_val.jsonl",
    "private/dpo_v1/manifest.json",
)
PY_ROOTS = ("src", "scripts", "training")


def _python_files(root: Path) -> Iterable[Path]:
    for relative in PY_ROOTS:
        for path in sorted((root / relative).rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield path


def _bundle_files(
    root: Path, *, minimal: bool = False, train_version: str = "train-v4"
) -> list[Path]:
    files = list(_python_files(root))
    if train_version == "train-v5":
        selected = V5_MINIMAL_REPO_FILES if minimal else V5_REPO_FILES
    else:
        selected = MINIMAL_REPO_FILES if minimal else REPO_FILES
    files.extend(root / relative for relative in selected)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"method bundle inputs missing: {missing}")
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def build_bundle(
    *,
    root: Path,
    out_dir: Path,
    dataset_id: str,
    minimal: bool = False,
    train_version: str = "train-v4",
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_zip = out_dir / "repo.zip"
    files = _bundle_files(root, minimal=minimal, train_version=train_version)
    with zipfile.ZipFile(
        repo_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    metadata = {
        "title": f"Lite Agent {train_version.title()} Method Bundle",
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (out_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "dataset_id": dataset_id,
        "train_version": train_version,
        "repo_zip": str(repo_zip),
        "repo_zip_bytes": repo_zip.stat().st_size,
        "repo_files": len(files),
        "files": [path.relative_to(root).as_posix() for path in files],
    }
    (out_dir / "bundle_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build the Kaggle method bundle")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("private/kaggle_method_bundle"))
    parser.add_argument("--dataset-id", default="autumnsq/lite-agent-method-v4")
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--train-version", choices=("train-v4", "train-v5"), default="train-v4")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_bundle(
        root=args.root.resolve(),
        out_dir=args.out_dir.resolve(),
        dataset_id=args.dataset_id,
        minimal=args.minimal,
        train_version=args.train_version,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
