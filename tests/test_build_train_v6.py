from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_train_v6 import freeze, read_task_ids


def _row(task_id: str, rollout: int = 0, **metadata: object) -> dict[str, object]:
    return {
        "id": f"row-{task_id}-{rollout}",
        "messages": [{"role": "user", "content": "x"}],
        "completion": "{}",
        "metadata": {"task_id": task_id, "rollout_index": rollout, **metadata},
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _source(tmp_path: Path, train: list[dict[str, object]], val: list[dict[str, object]]) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    _write_jsonl(root / "train.jsonl", train)
    _write_jsonl(root / "val.jsonl", val)
    return root


def test_freeze_removes_every_row_for_heldout_task(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        [_row("A", 0), _row("A", 1), _row("B", 0), _row("C", 0)],
        [_row("D", 0)],
    )
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"task_id": "A"}])

    manifest = freeze(
        source_dir=source,
        tasks_file=tasks,
        out_dir=tmp_path / "out",
        created_at="2026-10-25T00:00:00Z",
    )

    kept = [
        json.loads(line)
        for line in (tmp_path / "out/train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["metadata"]["task_id"] for row in kept] == ["B", "C"]
    assert manifest["train_examples"] == 2
    assert manifest["val_examples"] == 1
    assert manifest["heldout"]["removed_train_rows"] == 2
    assert manifest["heldout"]["removed_rows_by_task"] == {"A": 2}
    assert manifest["checks"]["heldout_tasks_remaining_in_train"] == []
    assert manifest["train_sha256"]


def test_freeze_refuses_val_overlap(tmp_path: Path) -> None:
    source = _source(tmp_path, [_row("A"), _row("B")], [_row("A")])
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"task_id": "A"}])

    with pytest.raises(ValueError, match="unexpectedly appear in val"):
        freeze(source_dir=source, tasks_file=tasks, out_dir=tmp_path / "out")


def test_freeze_rejects_missing_heldout_task(tmp_path: Path) -> None:
    source = _source(tmp_path, [_row("B")], [])
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"task_id": "A"}])

    with pytest.raises(ValueError, match="missing from train file"):
        freeze(source_dir=source, tasks_file=tasks, out_dir=tmp_path / "out")


def test_read_task_ids_accepts_text_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text(
        '# comment\nA\n{"metadata": {"task_id": "B"}}\n{"task_id": "C"}\n',
        encoding="utf-8",
    )

    assert read_task_ids(path) == ["A", "B", "C"]


def test_freeze_is_deterministic_for_same_inputs(tmp_path: Path) -> None:
    source = _source(tmp_path, [_row("A"), _row("B")], [])
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks, [{"task_id": "A"}])

    first = freeze(
        source_dir=source,
        tasks_file=tasks,
        out_dir=tmp_path / "out1",
        created_at="fixed",
    )
    second = freeze(
        source_dir=source,
        tasks_file=tasks,
        out_dir=tmp_path / "out2",
        created_at="fixed",
    )

    assert first["train_sha256"] == second["train_sha256"]
    assert first["combined_sha256"] == second["combined_sha256"]
