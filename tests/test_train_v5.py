from __future__ import annotations

import tempfile
from pathlib import Path

from scripts.unified_eval import build_fixture
from training.build_train_v5 import _select_train, build_v5_specs
from training.unified_dataset import TASK_SPECS, UnifiedTaskSpec


def test_v5_specs_are_disjoint_and_have_profiled_fixtures() -> None:
    specs = build_v5_specs()
    ids = {spec.task_id for spec in specs}
    dev_ids = {spec.task_id for spec in TASK_SPECS}
    assert len(specs) == 212
    assert ids.isdisjoint(dev_ids)
    assert {spec.split for spec in specs} == {"train", "val"}
    supplement = [spec for spec in specs if spec.task_id.startswith("U-TR5-EXTRA-")]
    assert len(supplement) == 50
    assert all(spec.split == "train" for spec in supplement)
    assert len({spec.fixture_seed for spec in supplement}) == 50

    ctx = next(spec for spec in specs if spec.task_id == "U-TR5-00-CTX-HARD")
    with tempfile.TemporaryDirectory() as temporary:
        fixture = build_fixture(ctx, Path(temporary))
        assert len(fixture.context_paths) == 8
        first = (Path(temporary) / fixture.context_paths[0]).read_text(encoding="utf-8")
        assert len(first.splitlines()) == 4

    tool = next(spec for spec in specs if spec.task_id == "U-TR5-00-TOOL-HARD")
    with tempfile.TemporaryDirectory() as temporary:
        fixture = build_fixture(tool, Path(temporary))
        assert len(fixture.expected_read_paths) == 2
        assert all((Path(temporary) / path).is_file() for path in fixture.expected_read_paths)


def test_optional_profile_defaults_do_not_change_legacy_manifest() -> None:
    legacy = TASK_SPECS[0]
    payload = legacy.to_dict()
    assert "context_file_count" not in payload
    assert "path_targets" not in payload

    ctx = next(spec for spec in build_v5_specs() if spec.family == "context_recall")
    ctx_payload = ctx.to_dict()
    assert ctx_payload["context_file_count"] == 8
    assert ctx_payload["context_lines_per_file"] == 4


def test_unified_task_spec_keeps_optional_fields_typed() -> None:
    spec = UnifiedTaskSpec(
        task_id="synthetic",
        split="train",
        difficulty="weak",
        context_tier="short",
        fixture_seed=1,
        constraints=(),
        user_turns=("x",),
        required_tools=(),
        allowed_tools=(),
        min_tool_calls=0,
        steps_limit=1,
        checks=(),
    )
    assert spec.to_dict()["task_id"] == "synthetic"


def test_ctx_hard_cap_does_not_drop_other_context_recall_rows() -> None:
    v4_rows = [{"id": "v4-context", "metadata": {"category": "CTX", "task_id": "U-TR-B-CTX-WEAK"}}]
    v5_rows = [
        {
            "id": f"v5-ctx-hard-{index}",
            "metadata": {"category": "CTX", "task_id": f"U-TR5-{index}-CTX-HARD"},
        }
        for index in range(45)
    ]
    selected = _select_train(v4_rows, v5_rows)
    assert len(selected) == 41
    assert selected[0]["id"] == "v4-context"
    assert sum("CTX-HARD" in row["metadata"]["task_id"] for row in selected) == 40


def test_base_cap_stops_at_fifty() -> None:
    v4_rows = [
        {"id": f"v4-base-{index}", "metadata": {"category": "BASE", "task_id": "U-TR-B-C1-BASE"}}
        for index in range(20)
    ]
    v5_rows = [
        {
            "id": f"v5-base-{index}",
            "metadata": {"category": "BASE", "task_id": f"U-TR5-{index}-C1-BASE"},
        }
        for index in range(40)
    ]
    selected = _select_train(v4_rows, v5_rows)
    assert len(selected) == 50
