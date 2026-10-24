"""Prompt-pool selection and final-turn scoring tests; no model or network."""

from __future__ import annotations

from scripts.grpo_prompt_pool import completion_matches, score_completion, select_prompt_pool


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
