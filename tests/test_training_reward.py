"""训练奖励函数的本地单测，不加载模型、不联网。"""

from __future__ import annotations

from training.reward import (
    behavior_reward,
    file_written_correctly,
    json_parseable,
    path_escape_free,
    snake_case_keys,
)


def test_json_signals_are_scripted() -> None:
    assert json_parseable('{"a_b": 1}') is True
    assert json_parseable('{"a_b": 1}\n说明') is False
    assert snake_case_keys('{"a_b": 1}') is True
    assert snake_case_keys('{"aB": 1}') is False


def test_path_escape_uses_tool_evidence_not_reply_text() -> None:
    assert path_escape_free() is True
    assert path_escape_free([{"path_escape": True}]) is False
    assert path_escape_free([], [{"error_kind": "path_escape"}]) is False


def test_file_signal_requires_real_observation() -> None:
    assert file_written_correctly(None) is None
    assert file_written_correctly({"file_written_correctly": True}) is True
    assert (
        file_written_correctly({"file_name_lower": True, "content_has_required_marker": False})
        is False
    )


def test_file_task_reward_does_not_use_json_claim() -> None:
    result = behavior_reward(
        "我已经把文件写好了",
        file_observations={"file_written_correctly": False},
        expects_json=False,
        expects_file=True,
    )
    assert result["hard_gate"] is False
    assert "file_written_correctly" in result["failures"]


def test_json_task_hard_gate_and_score() -> None:
    result = behavior_reward('{"a_b": 1}')
    assert result["hard_gate"] is True
    assert result["score"] == 1.0

    bad = behavior_reward('{"aB": 1}')
    assert bad["hard_gate"] is False
    assert "snake_case_keys" in bad["failures"]
