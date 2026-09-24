"""训练数据构造的单测：分 split、压缩工具证据、无 trace 泄漏。"""

from __future__ import annotations

import json
from pathlib import Path

from training.data import (
    build_example,
    collect_examples,
    load_episode,
    split_for,
    write_dataset,
)


def _write_trace(path: Path, *, task_id: str, run_id: int, checks: dict[str, bool]) -> None:
    events = [
        {
            "event": "run_start",
            "experiment": "unit",
            "task_id": task_id,
            "group": "on",
            "run_id": run_id,
            "rollout_index": 0,
            "constraints": [{"id": "C1", "content": "输出必须是合法 JSON"}],
        },
        {
            "event": "llm_turn",
            "step_label": "probe_json",
            "messages": [
                {"role": "system", "content": "base system"},
                {"role": "user", "content": "输出 JSON"},
            ],
            "response": {"content": '{"a_b": 1}'},
            "tool_calls": [{"tool_name": "read_file", "path_escape": False}],
            "tool_results": [{"tool_name": "read_file", "ok": True, "error_kind": None}],
        },
        {"event": "run_end", "checks": checks, "step_replies": {"probe_json": '{"a_b": 1}'}},
    ]
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events), encoding="utf-8")


def test_build_example_compacts_tool_results_and_keeps_constraints(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace, task_id="stress", run_id=2, checks={"json_valid": True})
    episode = load_episode(trace)
    assert episode is not None
    example = build_example(episode)
    assert example is not None
    assert example["messages"][0]["content"].startswith("base system")
    assert "[C1]" in example["messages"][0]["content"]
    assert example["metadata"]["constraint_ids"] == ["C1"]
    assert example["reward_context"]["tool_calls"][0]["tool_name"] == "read_file"
    assert example["metadata"]["reward"]["hard_gate"] is True


def test_train_val_split_has_no_run_key_overlap(tmp_path: Path) -> None:
    examples = []
    for task_id, run_id in (("stress", 2), ("stress", 10), ("C4", 1), ("C4", 4)):
        path = tmp_path / f"{task_id}_{run_id}.jsonl"
        checks = (
            {"lowercase_name": True, "code_in_new_file": True}
            if task_id == "C4"
            else {"json_valid": True}
        )
        _write_trace(path, task_id=task_id, run_id=run_id, checks=checks)
        episode = load_episode(path)
        assert episode is not None
        example = build_example(episode)
        assert example is not None
        examples.append(example)
    manifest = write_dataset(examples, tmp_path / "out")
    assert manifest["train"] == 2
    assert manifest["val"] == 2
    assert split_for("stress", 2) == "train"
    assert split_for("stress", 10) == "val"
    assert split_for("C4", 1) == "train"
    assert split_for("C4", 4) == "val"
    assert manifest["train_run_keys"] and manifest["val_run_keys"]


def test_collect_examples_deduplicates_same_run_and_rollout(tmp_path: Path) -> None:
    for name in ("formal", "manual_variant"):
        path = tmp_path / name / "trace.jsonl"
        path.parent.mkdir()
        _write_trace(path, task_id="stress", run_id=2, checks={"json_valid": True})
    examples = collect_examples(tmp_path, stress_runs={2}, c4_runs=set())
    assert len(examples) == 1
