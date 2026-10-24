"""Adapter save/download validation tests; no model and no GPU."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.adapter_artifacts import (
    compare_adapter_artifacts,
    validate_saved_adapter,
)


def test_validate_saved_adapter_accepts_nonempty_weights(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    result = validate_saved_adapter(tmp_path)
    assert result["total_bytes"] == len(b"weights") + len("{}")
    assert result["weights"] == ["adapter_model.safetensors"]


def test_validate_saved_adapter_rejects_zero_byte_weight(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="zero-byte"):
        validate_saved_adapter(tmp_path)


def test_compare_adapter_artifacts_accepts_matching_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    for directory in (source, target):
        (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        (directory / "adapter_model.safetensors").write_bytes(b"weights")
    result = compare_adapter_artifacts(source, target)
    assert result["matched"] is True
    assert result["files"] == 2


def test_compare_adapter_artifacts_detects_size_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    (target / "adapter_model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="size_mismatch"):
        compare_adapter_artifacts(source, target)
