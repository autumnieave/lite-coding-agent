"""Prompt-pool selection and final-turn scoring tests; no model or network."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from scripts.grpo_prompt_pool import (
    _variance,
    _variance_status,
    completion_matches,
    run_pool,
    score_completion,
    select_prompt_pool,
)


def _row(index: int, category: str, *, completion: str = "CTX-1", expects_json: bool = False):
    return {
        "id": f"row-{index}",
        "messages": [{"role": "user", "content": f"prompt {index}"}],
        "completion": completion,
        "reward_context": {
            "expects_json": expects_json,
            "expects_file": False,
            "within_steps": True,
            "tools_allowed": True,
            "tool_calls": [],
            "tool_results": [],
            "file_observations": None,
        },
        "metadata": {"task_id": f"T-{index}", "category": category},
    }


def test_prompt_pool_covers_all_five_categories() -> None:
    categories = ["BASE", "HARD", "CTX", "TOOL", "CONFLICT"] * 4
    rows = [_row(index, category) for index, category in enumerate(categories)]
    selected = select_prompt_pool(rows, per_category=3, seed=1)
    assert len(selected) == 15
    assert {row["metadata"]["category"] for row in selected} == {
        "BASE",
        "HARD",
        "CTX",
        "TOOL",
        "CONFLICT",
    }


def test_prompt_pool_is_deterministic_and_deduplicates_messages() -> None:
    rows = [_row(index, "BASE") for index in range(6)]
    rows.append({**rows[0], "id": "duplicate"})
    first = [row["id"] for row in select_prompt_pool(rows, per_category=4, seed=7)]
    second = [row["id"] for row in select_prompt_pool(rows, per_category=4, seed=7)]
    assert first == second
    assert len(first) == 4


def test_json_match_is_semantic_and_rejects_prose() -> None:
    assert completion_matches('{"a_b": 1}', expected='{"a_b":1}', expects_json=True)
    assert not completion_matches('{"a_b": 1}\n说明', expected='{"a_b":1}', expects_json=True)
    assert not completion_matches('{"a_b": 2}', expected='{"a_b":1}', expects_json=True)


def test_token_match_is_exact_after_strip() -> None:
    assert completion_matches(" CTX-1\n", expected="CTX-1", expects_json=False)
    assert not completion_matches("CODE=CTX-1", expected="CTX-1", expects_json=False)


def test_wrong_token_gets_zero_pool_reward_even_if_reward_shape_is_valid() -> None:
    row = _row(0, "CTX", completion="CTX-1", expects_json=False)
    scored = score_completion(row, "CTX-2")
    assert scored["task_success"] is False
    assert scored["pool_reward"] == 0.0


def test_empty_rollout_list_is_not_measured_not_zero_variance() -> None:
    assert _variance([]) is None
    assert _variance_status([]) == "not_measured"


def test_measured_all_zero_rewards_report_zero_variance() -> None:
    assert _variance([0.0, 0.0, 0.0]) == 0.0
    assert _variance_status([0.0, 0.0, 0.0]) == "measured_zero"


def test_measured_mixed_rewards_report_positive_variance() -> None:
    variance = _variance([0.0, 1.0, 1.0, 1.0])
    assert variance is not None
    assert variance > 0.0
    assert _variance_status([0.0, 1.0, 1.0, 1.0]) == "measured_positive"


def test_run_pool_rejects_zero_rollouts_before_reading_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="zero rollouts is not variance evidence"):
        asyncio.run(
            run_pool(
                train_file=tmp_path / "missing.jsonl",
                out_dir=tmp_path / "out",
                provider=object(),
                model_label="test",
                rollouts=0,
                per_category=1,
                temperature=0.8,
                seed=1,
                max_new_tokens=1,
            )
        )
