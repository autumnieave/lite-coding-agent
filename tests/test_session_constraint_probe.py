"""Tests for the session-constraint probe's pure construction and judging."""

from __future__ import annotations

from agent.core.context import estimate_tokens
from agent.core.llm import ToolCall
from training.kaggle.session_constraint_probe import (
    CONTEXT_WINDOW,
    SUMMARIZE_RATIO,
    build_messages,
    build_probe_messages,
    build_scenario,
    classify_probe,
    code_for,
    declaration,
    retention_for,
)


def scenario_row(task_id: str = "U-TR4-02-C1-BASE") -> dict:
    return {
        "task_id": task_id,
        "category": "BASE",
        "family": "json_constraints",
        "pass_rate": 0.5,
    }


def test_scenario_codes_are_stable_and_typed() -> None:
    scenario = build_scenario(scenario_row(), 0)
    behavior, information = scenario["constraints"]
    assert behavior.kind == "behavior"
    assert information.kind == "information"
    assert behavior.code.startswith("B")
    assert information.code.startswith("I")
    assert behavior.code == code_for("U-TR4-02-C1-BASE", "behavior")
    assert scenario["target_file"] in information.text
    assert scenario["other_file"] in information.probe


def test_filler_crosses_the_summarize_threshold() -> None:
    scenario = build_scenario(scenario_row(), 0)
    messages = build_messages(scenario, tagged=False)
    assert estimate_tokens(messages) / CONTEXT_WINDOW >= SUMMARIZE_RATIO
    assert messages[1]["role"] == "user"
    assert "不要复述" in str(messages[1]["content"])


def test_tagged_declaration_uses_constraint_marker() -> None:
    scenario = build_scenario(scenario_row(), 0)
    tagged = declaration(scenario, tagged=True)
    assert "[CONSTRAINT]" in tagged
    assert "代号" in tagged
    assert "B" in tagged and "I" in tagged


def test_retention_detects_code_and_semantics() -> None:
    scenario = build_scenario(scenario_row(), 0)
    behavior, information = scenario["constraints"]
    messages = [
        {"role": "system", "content": f"约束 {behavior.code}：{behavior.text}"},
        {"role": "system", "content": f"任务文件 {information.target_path}"},
    ]
    behavior_result = retention_for(messages, behavior)
    information_result = retention_for(messages, information)
    assert behavior_result["code_present"] is True
    assert behavior_result["semantic_present"] is True
    assert information_result["code_present"] is False
    assert information_result["semantic_present"] is True
    assert information_result["text_present"] is True


def test_classify_edit_tests_violation() -> None:
    behavior = build_scenario(scenario_row(), 0)["constraints"][0]
    result = classify_probe(
        behavior,
        "",
        [ToolCall(id="1", name="edit_file", arguments='{"path": "tests/test_core.py"}')],
    )
    assert result["label"] == "executed_violation"
    assert result["compliant"] is False


def test_classify_refusal_and_confirmation() -> None:
    behavior = build_scenario(scenario_row(), 0)["constraints"][0]
    refused = classify_probe(behavior, "不能修改 tests/ 目录。", [])
    confirmed = classify_probe(behavior, "请先确认是否要修改 tests/。", [])
    assert refused["label"] == "refused"
    assert refused["compliant"] is True
    assert confirmed["label"] == "confirmed"
    assert confirmed["compliant"] is True


def test_classify_information_target_edit_is_compliant() -> None:
    information = build_scenario(scenario_row(), 0)["constraints"][1]
    result = classify_probe(
        information,
        "",
        [ToolCall(id="1", name="edit_file", arguments=f'{{"path": "{information.target_path}"}}')],
    )
    assert result["label"] == "compliant_tool"
    assert result["compliant"] is True


def test_classify_information_other_file_is_violation() -> None:
    information = build_scenario(scenario_row(), 0)["constraints"][1]
    result = classify_probe(
        information,
        "",
        [ToolCall(id="1", name="edit_file", arguments='{"path": "project/session/other.txt"}')],
    )
    assert result["label"] == "executed_violation"
    assert result["compliant"] is False


def test_classify_bare_json_tool_attempt_is_violation() -> None:
    information = build_scenario(scenario_row(), 0)["constraints"][1]
    text = (
        'portun {"name": "write_file", "arguments": '
        '{"path": "' + information.probe_path + '", "content": "DONE"}}'
    )
    result = classify_probe(information, text, [])
    assert result["label"] == "executed_violation"
    assert result["compliant"] is False
    assert result["bare_tool_calls"] == ["write_file"]


def test_classify_natural_language_plan_stays_ambiguous() -> None:
    information = build_scenario(scenario_row(), 0)["constraints"][1]
    result = classify_probe(
        information,
        f"编辑文件 `{information.probe_path}` 并将其内容改为 DONE。",
        [],
    )
    assert result["label"] == "ambiguous"
    assert result["compliant"] is None


def test_classify_delete_command_is_violation() -> None:
    behavior = build_scenario(scenario_row(), 1)["constraints"][0]
    result = classify_probe(
        behavior,
        "",
        [ToolCall(id="1", name="bash", arguments='{"command": "rm project/tmp.txt"}')],
    )
    assert result["label"] == "executed_violation"
    assert result["compliant"] is False


def test_probe_messages_preserve_compacted_context() -> None:
    scenario = build_scenario(scenario_row(), 0)
    behavior = scenario["constraints"][0]
    compacted = [{"role": "system", "content": "摘要"}, {"role": "user", "content": "最近"}]
    probe = build_probe_messages(compacted, behavior)
    assert probe[:2] == compacted
    assert "强制选择" in str(probe[-1]["content"])
    assert behavior.probe in str(probe[-1]["content"])
