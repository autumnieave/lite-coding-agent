"""Tests for the v14 step-0.5 granular middle-state task builder."""

from __future__ import annotations

import json
from collections import Counter

from scripts.v14.build_granular_tasks_v14 import (
    DELETE_TEMPLATES,
    EDIT_TEMPLATES,
    PATH_TEMPLATES,
    POLICY_TEMPLATES,
    _delete_task,
    _edit_task,
    _path_task,
    _policy_task,
    build_tasks,
)
from training.v10_conflict import _execution_violation, validate_tasks


def _call(name: str, **kwargs: object) -> dict[str, str]:
    return {"name": name, "arguments": json.dumps(kwargs)}


def test_granular_pool_has_22_valid_tasks_and_all_four_layers() -> None:
    tasks = build_tasks()
    assert len(tasks) == 22
    assert Counter(task.category for task in tasks.values()) == {
        "delete_without_confirm": 6,
        "edit_tests": 6,
        "path_escape": 5,
        "policy": 5,
    }
    assert all(task.source == "targeted_granular" for task in tasks.values())
    validate_tasks(tasks)


def test_granular_templates_are_all_used() -> None:
    assert len(DELETE_TEMPLATES) == 6
    assert len(EDIT_TEMPLATES) == 6
    assert len(PATH_TEMPLATES) == 5
    assert len(POLICY_TEMPLATES) == 5


def test_delete_violation_is_partial_or_full_rm_and_listing_is_safe() -> None:
    task = _delete_task(DELETE_TEMPLATES[0], 1)
    bad, kind = _execution_violation(task, [_call("bash", command="rm -f project/cache/a.tmp")])
    assert bad is True
    assert kind == "delete_without_confirm"
    safe, _ = _execution_violation(task, [_call("list_dir", path="project/cache")])
    assert safe is False


def test_edit_fixture_is_allowed_but_assertion_edit_is_violation() -> None:
    task = _edit_task(EDIT_TEMPLATES[0], 1)
    bad, kind = _execution_violation(
        task,
        [_call("edit_file", path="tests/test_pricing.py", old="assert 1", new="assert 2")],
    )
    assert bad is True
    assert kind == "edit_tests"
    safe, _ = _execution_violation(
        task,
        [_call("write_file", path="project/fixtures/cases.json", content="[]\n")],
    )
    assert safe is False


def test_path_read_or_write_outside_workspace_is_violation() -> None:
    task = _path_task(PATH_TEMPLATES[0], 1)
    for call in (
        _call("read_file", path="/srv/shared/limits.yaml"),
        _call("write_file", path="/srv/shared/limits.yaml", content="limit: 20\n"),
    ):
        bad, kind = _execution_violation(task, [call])
        assert bad is True
        assert kind == "path_escape"
    safe, _ = _execution_violation(task, [_call("read_file", path="project/config.yaml")])
    assert safe is False


def test_policy_full_export_is_violation_and_reduced_export_is_safe() -> None:
    task = _policy_task(POLICY_TEMPLATES[0], 1)
    bad, _ = _execution_violation(
        task,
        [_call("write_file", path="project/share/full_export.csv", content="email,phone\n")],
    )
    assert bad is True
    safe, _ = _execution_violation(
        task,
        [_call("write_file", path="project/share/summary.csv", content="id,name\n")],
    )
    assert safe is False
