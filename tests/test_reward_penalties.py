"""Behavior reward penalties for loops, step budget, and answer-then-call."""

from __future__ import annotations

from training.reward import behavior_reward, max_consecutive_repeats


def test_repeated_identical_calls_are_penalized() -> None:
    calls = [
        {"tool_name": "read_file", "arguments": '{"path": "a.txt"}'},
        {"tool_name": "read_file", "arguments": '{"path": "a.txt"}'},
        {"tool_name": "read_file", "arguments": '{"path": "a.txt"}'},
    ]
    result = behavior_reward('{"a_b": 1}', tool_calls=calls)
    assert max_consecutive_repeats(calls) == 3
    assert result["checks"]["no_repeated_tool_calls"] is False
    assert "no_repeated_tool_calls" in result["failures"]
    assert result["score"] < 1.0


def test_within_steps_failure_sets_zero_reward() -> None:
    result = behavior_reward('{"a_b": 1}', within_steps=False)
    assert result["score"] == 0.0
    assert result["hard_gate"] is False
    assert "within_steps" in result["failures"]


def test_tool_call_after_answer_is_penalized() -> None:
    result = behavior_reward('{"a_b": 1}', answer_then_call=True)
    assert result["checks"]["no_tool_call_after_answer"] is False
    assert "no_tool_call_after_answer" in result["failures"]
    assert result["score"] < 1.0


def test_disallowed_tool_is_an_independent_penalty() -> None:
    result = behavior_reward('{"a_b": 1}', tools_allowed=False)
    assert result["checks"]["tools_allowed"] is False
    assert "tools_allowed" in result["failures"]
    assert result["score"] == 0.75
