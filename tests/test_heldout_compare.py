"""Held-out comparison tests; no model, no GPU, no subprocess."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from training.kaggle.heldout_compare import (
    build_eval_command,
    load_val_tasks,
    run_compare,
    summarize_condition,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_val_tasks_reads_metadata(tmp_path: Path) -> None:
    val = tmp_path / "val.jsonl"
    _write_jsonl(
        val,
        [
            {"metadata": {"task_id": "U-TR-B-TOOL-HARD"}},
            {"metadata": {"task_id": "U-TR-B-C1-BASE"}},
            {"metadata": {"task_id": "U-TR-B-TOOL-HARD"}},
        ],
    )
    assert load_val_tasks(val) == ["U-TR-B-C1-BASE", "U-TR-B-TOOL-HARD"]


def test_build_eval_command_adds_adapter_only_for_sft(tmp_path: Path) -> None:
    common = dict(
        repo=tmp_path,
        model="Qwen/Qwen2.5-1.5B-Instruct",
        out_dir=tmp_path / "out",
        trace_dir=tmp_path / "traces",
        tasks=["U-TR-B-C1-BASE"],
        rollouts=10,
    )
    baseline = build_eval_command(**common)
    sft = build_eval_command(**common, adapter_path="/kaggle/working/adapter")
    assert "--adapter-path" not in baseline
    assert sft[sft.index("--adapter-path") + 1] == "/kaggle/working/adapter"
    assert "--train-version" in sft
    assert sft[sft.index("--train-version") + 1] == "train-v2"


def test_build_eval_command_wires_provider_quantization(tmp_path: Path) -> None:
    common = dict(
        repo=tmp_path,
        model="/kaggle/temp/qwen2.5-7b-instruct",
        out_dir=tmp_path / "out",
        trace_dir=tmp_path / "traces",
        tasks=["U-TR-A-C1-HARD"],
        rollouts=10,
    )
    nf4 = build_eval_command(**common, provider="transformers", quantization="nf4")
    assert nf4[nf4.index("--provider") + 1] == "transformers"
    assert nf4[nf4.index("--quantization") + 1] == "nf4"
    assert "--offline-quantization" not in nf4

    awq = build_eval_command(**common, provider="offline", quantization="awq")
    assert awq[awq.index("--provider") + 1] == "offline"
    assert awq[awq.index("--offline-quantization") + 1] == "awq"
    assert "--quantization" not in awq


def test_summarize_condition_marks_stable_and_counts_tool_results(tmp_path: Path) -> None:
    task_id = "U-TR-B-C1-BASE"
    rows = [
        {
            "task_id": task_id,
            "rollout_index": index,
            "checks": {"json_valid": index >= 6},
            "task_success": index < 4,
            "within_steps": True,
        }
        for index in range(10)
    ]
    trace_dir = tmp_path / "traces" / "train"
    _write_jsonl(
        trace_dir / f"{task_id}_rollout00.jsonl",
        [
            {
                "event": "llm_turn",
                "tool_results": [
                    {"ok": True, "is_error": False},
                    {"ok": False, "is_error": True},
                ],
            }
        ],
    )
    spec = SimpleNamespace(
        task_id=task_id,
        difficulty="weak",
        checks=[SimpleNamespace(name="json_valid")],
    )
    summary = summarize_condition(rows, [spec], trace_dir)
    assert summary["task_success"] == {"hits": 4, "total": 10, "rate": 0.4}
    assert summary["stable_checks"] == 1
    assert summary["tool_results"] == {"ok": 1, "total": 2, "rate": 0.5}
    assert summary["tasks"][0]["kind"] == "base"


def test_summarize_condition_separates_step_failures_from_valid_stable(tmp_path: Path) -> None:
    task_id = "U-TR-B-C1-BASE"
    rows = [
        {
            "task_id": task_id,
            "rollout_index": index,
            "checks": {"json_valid": index >= 16},
            "task_success": index >= 18,
            "within_steps": index >= 10,
        }
        for index in range(20)
    ]
    spec = SimpleNamespace(
        task_id=task_id,
        difficulty="weak",
        checks=[SimpleNamespace(name="json_valid")],
    )
    summary = summarize_condition(rows, [spec], tmp_path / "missing-traces")
    task = summary["tasks"][0]
    check = task["checks"][0]

    assert summary["stable_checks"] == 1
    assert summary["valid_stable_checks"] == 1
    assert summary["within_steps_failures"] == 10
    assert task["rollouts"] == 20
    assert task["valid_rollouts"] == 10
    assert task["within_steps_failures"] == 10
    assert check["hits"] == 16
    assert check["valid_hits"] == 6
    assert check["valid_total"] == 10
    assert check["valid_stability"] == "stable"


def test_run_compare_accepts_explicit_tasks_and_reference_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "dpo_adapter"
    reference = tmp_path / "sft_adapter"
    adapter.mkdir()
    reference.mkdir()
    val_file = tmp_path / "val.jsonl"
    _write_jsonl(val_file, [{"metadata": {"task_id": "U-TR4-NOT-A-BEHAVIOUR-TASK"}}])

    result = run_compare(
        repo=tmp_path,
        model="Qwen/Qwen2.5-7B-Instruct",
        val_file=val_file,
        adapter_path=adapter,
        out_root=tmp_path / "out",
        trace_root=tmp_path / "traces",
        rollouts=10,
        train_version="train-v2",
        provider="transformers",
        quantization="nf4",
        dry_run=True,
        tasks=["U-TR-B-C1-BASE"],
        baseline_adapter_path=reference,
    )

    assert result["baseline_adapter_path"] == str(reference)
    baseline_cmd = result["conditions"]["baseline"]["command"]
    candidate_cmd = result["conditions"]["sft"]["command"]
    assert baseline_cmd[baseline_cmd.index("--adapter-path") + 1] == str(reference)
    assert candidate_cmd[candidate_cmd.index("--adapter-path") + 1] == str(adapter)
    assert candidate_cmd[candidate_cmd.index("--tasks") + 1] == "U-TR-B-C1-BASE"
