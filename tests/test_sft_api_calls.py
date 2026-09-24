"""Teacher API call counting; no network."""

from __future__ import annotations

import json
from pathlib import Path

from training.build_sft_dataset import count_teacher_api_calls


def _trace(path: Path, task_id: str, turns: int) -> None:
    events = [{"event": "run_start", "task_id": task_id}]
    events += [{"event": "llm_turn"}] * turns
    events.append({"event": "run_end"})
    path.write_text("\n".join(json.dumps(item) for item in events) + "\n", encoding="utf-8")


def test_count_teacher_api_calls_filters_tasks(tmp_path: Path) -> None:
    _trace(tmp_path / "a.jsonl", "keep", 3)
    _trace(tmp_path / "b.jsonl", "skip", 5)
    assert count_teacher_api_calls(tmp_path, {"keep"}) == 3
