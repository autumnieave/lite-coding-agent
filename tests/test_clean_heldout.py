from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.kaggle import clean_heldout


def _adapter(root: Path) -> Path:
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    return adapter


def test_read_task_ids_accepts_json_and_text(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '# comment\nA\n{"task_id": "B"}\n{"metadata": {"task_id": "C"}}\n',
        encoding="utf-8",
    )

    assert clean_heldout.read_task_ids(path) == ["A", "B", "C"]


def test_read_task_ids_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text('{"task_id": "A"}\n{"task_id": "A"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        clean_heldout.read_task_ids(path)


def test_select_specs_preserves_requested_order() -> None:
    pool = SimpleNamespace(
        tasks=(
            SimpleNamespace(spec=SimpleNamespace(task_id="A")),
            SimpleNamespace(spec=SimpleNamespace(task_id="B")),
        )
    )

    selected = clean_heldout.select_specs(pool, ["B", "A"])

    assert [spec.task_id for spec in selected] == ["B", "A"]


def test_select_specs_reports_missing_ids() -> None:
    pool = SimpleNamespace(tasks=(SimpleNamespace(spec=SimpleNamespace(task_id="A")),))

    with pytest.raises(ValueError, match="missing from pool"):
        clean_heldout.select_specs(pool, ["A", "Z"])


def test_build_namespace_keeps_sampling_fields(tmp_path: Path) -> None:
    args = clean_heldout.build_namespace(
        model_path=tmp_path / "model",
        adapter_path=tmp_path / "adapter",
        out_dir=tmp_path / "out",
        trace_dir=tmp_path / "trace",
        rollouts=10,
        limit=0,
        resume=False,
        max_model_len=8192,
        max_tokens=512,
        load_timeout=240.0,
        rollout_timeout=300.0,
        temperature=0.7,
        top_p=0.9,
        seed=20261019,
        device_map=None,
    )

    assert isinstance(args, Namespace)
    assert args.rollouts == 10
    assert args.temperature == 0.7
    assert args.top_p == 0.9


def test_run_clean_heldout_selects_tasks_and_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(tmp_path)
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text('{"task_id": "B"}\n{"task_id": "A"}\n', encoding="utf-8")
    train_file = tmp_path / "train.jsonl"
    train_file.write_text("{}\n", encoding="utf-8")

    specs = {task_id: SimpleNamespace(task_id=task_id) for task_id in ("A", "B")}
    pool = SimpleNamespace(tasks=tuple(SimpleNamespace(spec=spec) for spec in specs.values()))
    captured: dict[str, object] = {}

    async def fake_run_condition(condition: str, *, specs, args):
        captured["condition"] = condition
        captured["task_ids"] = [spec.task_id for spec in specs]
        captured["out_dir"] = args.out_dir
        return {"condition": condition, "status": "completed", "rollouts": 2}

    monkeypatch.setattr(clean_heldout, "build_heldout_pool", lambda _: pool)
    from training.kaggle import stats_pilot

    monkeypatch.setattr(stats_pilot, "run_condition", fake_run_condition)

    result = asyncio.run(
        clean_heldout.run_clean_heldout(
            train_file=train_file,
            task_ids_file=tasks_file,
            model_path=tmp_path / "model",
            adapter_path=adapter,
            out_dir=tmp_path / "out",
            trace_dir=tmp_path / "trace",
            rollouts=1,
        )
    )

    assert captured["condition"] == "sft"
    assert captured["task_ids"] == ["B", "A"]
    assert result["task_ids"] == ["B", "A"]
    assert result["adapter_model_sha256"]
    written = json.loads((tmp_path / "out/summary.json").read_text(encoding="utf-8"))
    assert written["condition"]["status"] == "completed"
