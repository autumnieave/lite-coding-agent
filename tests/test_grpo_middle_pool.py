"""Middle-state pool selection tests; no private data or network."""

from __future__ import annotations

import pytest

from scripts.build_grpo_middle_pool import build_prompt_pool, select_middle_tasks


def _middle(
    task_id: str,
    *,
    pass_rate: float = 0.5,
    category: str = "BASE",
    valid_rollouts: int = 8,
    expected_rollouts: int = 8,
    errors: int = 0,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "category": category,
        "family": "json_constraints",
        "difficulty": "weak",
        "pass_rate": pass_rate,
        "variance": pass_rate * (1.0 - pass_rate),
        "valid_rollouts": valid_rollouts,
        "expected_rollouts": expected_rollouts,
        "errors": errors,
        "training_overlap": True,
    }


def _spec(task_id: str, *, user_turns: tuple[str, ...] = ("do the task",)) -> dict[str, object]:
    return {
        "task_id": task_id,
        "spec": {
            "task_id": task_id,
            "user_turns": list(user_turns),
            "constraints": [{"id": "C1", "text": "only one tool"}],
            "allowed_tools": ["read_file"],
            "required_tools": ["read_file"],
            "steps_limit": 3,
            "family": "json_constraints",
            "difficulty": "weak",
        },
    }


def test_select_middle_tasks_filters_rate_incomplete_and_duplicate_rows() -> None:
    rows = [
        _middle("keep-1", pass_rate=0.375),
        _middle("keep-2", pass_rate=0.625, category="HARD"),
        _middle("too-easy", pass_rate=0.875),
        _middle("too-hard", pass_rate=0.125),
        _middle("incomplete", pass_rate=0.5, valid_rollouts=7),
        _middle("errored", pass_rate=0.5, errors=1),
        _middle("keep-1", pass_rate=0.5),
    ]
    selected = select_middle_tasks(rows)
    assert [row["task_id"] for row in selected] == ["keep-1", "keep-2"]


def test_build_prompt_pool_carries_prompt_and_metadata() -> None:
    prompts = build_prompt_pool(
        [_middle("U-TEST-1", pass_rate=0.5)],
        [_spec("U-TEST-1", user_turns=("read this", "then answer"))],
    )
    assert len(prompts) == 1
    assert prompts[0]["task_id"] == "U-TEST-1"
    assert prompts[0]["user_turns"] == ["read this", "then answer"]
    assert prompts[0]["constraints"] == [{"id": "C1", "text": "only one tool"}]
    assert prompts[0]["allowed_tools"] == ["read_file"]


def test_build_prompt_pool_rejects_missing_specs() -> None:
    with pytest.raises(ValueError, match="missing from task specs"):
        build_prompt_pool([_middle("U-MISSING")], [])


def test_build_prompt_pool_rejects_empty_user_turns() -> None:
    with pytest.raises(ValueError, match="no non-empty user_turns"):
        build_prompt_pool([_middle("U-EMPTY")], [_spec("U-EMPTY", user_turns=())])
