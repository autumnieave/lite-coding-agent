"""Final-test manifest freezing tests; no model or rollout is used."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from scripts.unified_eval import build_fixture
from training.difficulty_tuning import TUNING_TASK_SPECS
from training.final_test import (
    _validate_prompt_references,
    build_final_specs,
    canonical_payload,
    freeze_final_test,
)
from training.unified_dataset import TASK_SPECS


def test_final_specs_are_disjoint_from_pilot_and_tuning_tasks() -> None:
    specs = build_final_specs()
    existing_ids = {spec.task_id for spec in (*TASK_SPECS, *TUNING_TASK_SPECS)}
    existing_seeds = {spec.fixture_seed for spec in (*TASK_SPECS, *TUNING_TASK_SPECS)}
    assert len(specs) == 11
    assert not ({spec.task_id for spec in specs} & existing_ids)
    assert not ({spec.fixture_seed for spec in specs} & existing_seeds)


def test_final_specs_cover_five_dimensions_and_three_levels() -> None:
    specs = build_final_specs()
    assert {spec.tuning_dimension for spec in specs} == {
        "constraint_count",
        "context_length",
        "tool_parameters",
        "conflict",
        "check_depth",
    }
    assert {spec.difficulty for spec in specs} == {"weak", "medium", "strong"}
    assert {spec.context_tier for spec in specs} == {"short", "medium", "long"}


def test_freeze_writes_deterministic_manifest_and_metadata(tmp_path: Path) -> None:
    specs = build_final_specs()
    expected = hashlib.sha256(canonical_payload(specs).encode("utf-8")).hexdigest()
    result = freeze_final_test(tmp_path, root=Path.cwd())
    assert result["sha256"] == expected
    assert result["no_rollout_yet"] is True
    assert (tmp_path / "final_test.jsonl").read_text(encoding="utf-8") == canonical_payload(specs)
    assert hashlib.sha256((tmp_path / "final_test.jsonl").read_bytes()).hexdigest() == expected
    assert (tmp_path / "FINAL_TEST_FROZEN.json").is_file()
    assert (tmp_path / "FINAL_TEST_FROZEN.txt").is_file()


def test_check_prompts_reference_test_fixture() -> None:
    specs = {spec.task_id: spec for spec in build_final_specs()}
    for task_id in ("U-FT10-BASE", "U-FT11-HARD"):
        prompt = "\n".join(specs[task_id].user_turns)
        assert "project/notes/part07.txt" in prompt
        assert "第 53 行" in prompt
        assert "FIX-T03" in prompt
        assert "part02.txt" not in prompt
        assert "FIX-D03" not in prompt


def test_validator_rejects_dev_prompt_against_test_fixture() -> None:
    source = next(spec for spec in TUNING_TASK_SPECS if spec.task_id == "U-TUNE-CHECK-BASE")
    test_spec = type(source)(
        **{
            **source.to_dict(),
            "task_id": "U-FT10-BASE",
            "split": "test",
            "fixture_seed": 95010,
        }
    )
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        fixture = build_fixture(test_spec, root)
        with pytest.raises(ValueError, match="missing test fixture path"):
            _validate_prompt_references(test_spec, fixture, root)
