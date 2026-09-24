"""Validate saved and downloaded LoRA adapter artifacts.

Kaggle's output download has produced zero-byte adapter files in earlier runs.
These checks make that failure loud before a held-out evaluation is started.
"""

from __future__ import annotations

from pathlib import Path

WEIGHT_SUFFIXES = (".safetensors", ".bin")


def adapter_files(adapter_dir: Path) -> dict[str, int]:
    """Return ``{filename: size}`` for regular files in an adapter directory."""

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter_dir}")
    files = {
        path.name: path.stat().st_size for path in sorted(adapter_dir.iterdir()) if path.is_file()
    }
    if not files:
        raise ValueError(f"adapter directory is empty: {adapter_dir}")
    return files


def validate_saved_adapter(adapter_dir: Path) -> dict[str, object]:
    """Require a non-empty adapter config plus at least one non-empty weight file."""

    files = adapter_files(adapter_dir)
    weights = sorted(name for name in files if name.endswith(WEIGHT_SUFFIXES))
    if not weights:
        raise ValueError(f"no adapter weight file in {adapter_dir}")
    zero_bytes = sorted(name for name, size in files.items() if size <= 0)
    if zero_bytes:
        raise ValueError(f"zero-byte adapter files in {adapter_dir}: {zero_bytes}")
    return {
        "files": files,
        "total_bytes": sum(files.values()),
        "weights": weights,
    }


def compare_adapter_artifacts(source_dir: Path, downloaded_dir: Path) -> dict[str, object]:
    """Compare a saved adapter with a downloaded copy by filename and size."""

    source = adapter_files(source_dir)
    downloaded = adapter_files(downloaded_dir)
    missing = sorted(set(source) - set(downloaded))
    extra = sorted(set(downloaded) - set(source))
    mismatched = {
        name: {"source": source[name], "downloaded": downloaded[name]}
        for name in sorted(set(source) & set(downloaded))
        if source[name] != downloaded[name]
    }
    if missing or extra or mismatched:
        raise ValueError(
            "adapter download mismatch: "
            f"missing={missing}, extra={extra}, size_mismatch={mismatched}"
        )
    return {
        "matched": True,
        "files": len(source),
        "total_bytes": sum(source.values()),
    }
