"""Kaggle submission contract helpers; no GPU, model or network."""

from __future__ import annotations

import json
import signal
import subprocess
import time
from pathlib import Path

import pytest

from training.kaggle.contracts import (
    ContractError,
    append_jsonl,
    classify_exception,
    completed_keys,
    latest_records,
    require_nonempty_file,
    time_limit,
    validate_summary,
)


def test_require_nonempty_file_rejects_missing_and_empty(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(ContractError, match="不存在") as missing_error:
        require_nonempty_file(missing, "prompts")
    assert missing_error.value.error_kind == "input_missing"

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ContractError, match="为空") as empty_error:
        require_nonempty_file(empty, "prompts")
    assert empty_error.value.error_kind == "input_empty"


def test_append_jsonl_flushes_and_keeps_existing_rows(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    append_jsonl(path, {"run_id": 2, "rollout_index": 0})
    append_jsonl(path, {"run_id": 2, "rollout_index": 1})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"run_id": 2, "rollout_index": 0},
        {"run_id": 2, "rollout_index": 1},
    ]


def test_completed_keys_reads_composite_keys(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"run_id": 2, "rollout_index": 0},
                {"run_id": 10, "rollout_index": 3},
                {"run_id": 10},
            )
        ),
        encoding="utf-8",
    )
    assert completed_keys(path, ("run_id", "rollout_index")) == {(2, 0), (10, 3)}


def test_completed_keys_retries_rows_with_error_kind(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"run_id": 2, "rollout_index": 0, "output": "ok"},
                {"run_id": 2, "rollout_index": 1, "error_kind": "inference_timeout"},
            )
        ),
        encoding="utf-8",
    )
    assert completed_keys(path, ("run_id", "rollout_index")) == {(2, 0)}


def test_latest_records_keeps_the_last_success_after_a_retry() -> None:
    rows = [
        {"run_id": 2, "rollout_index": 0, "error_kind": "inference_timeout"},
        {"run_id": 2, "rollout_index": 0, "output": "ok"},
        {"run_id": 10, "rollout_index": 0, "output": "ok"},
    ]
    merged = latest_records(rows, ("run_id", "rollout_index"))
    assert merged == [
        {"run_id": 2, "rollout_index": 0, "output": "ok"},
        {"run_id": 10, "rollout_index": 0, "output": "ok"},
    ]


def test_classify_exception_uses_stable_error_kinds() -> None:
    assert classify_exception(ContractError("model_load_timeout", "x")) == "model_load_timeout"
    assert classify_exception(subprocess.CalledProcessError(1, ["x"])) == "child_process_failed"
    assert classify_exception(FileNotFoundError("x")) == "model_path_missing"
    assert classify_exception(MemoryError("x")) == "oom"
    assert classify_exception(ImportError("x")) == "dependency_missing"
    assert classify_exception(RuntimeError("x")) == "unknown"


def test_validate_summary_rejects_missing_required_keys() -> None:
    with pytest.raises(ContractError) as caught:
        validate_summary({"status": "ok"}, ("status", "records"))
    assert caught.value.error_kind == "summary_schema_invalid"


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM is POSIX-only")
def test_time_limit_raises_categorized_timeout() -> None:
    with pytest.raises(ContractError) as caught, time_limit(0.01, "inference_timeout"):
        time.sleep(0.2)
    assert caught.value.error_kind == "inference_timeout"


def test_time_limit_zero_is_noop() -> None:
    with time_limit(0, "model_load_timeout"):
        pass
