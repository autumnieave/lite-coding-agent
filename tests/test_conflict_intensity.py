"""受控冲突实验的判定测试，不发起网络请求。"""

from __future__ import annotations

from scripts.conflict_intensity import GROUPS, build_messages, judge_output


def test_judge_output_detects_valid_snake_case_without_conflict() -> None:
    result = judge_output('{"log_prefix": "[WEAK-A]", "timestamp": "2026-10-19T00:00:00Z"}')
    assert result["json_valid"] is True
    assert result["snake_case_keys"] is True
    assert result["format_failure"] is False
    assert result["unresolved_conflict"] is False


def test_judge_output_detects_appended_explanation_as_format_failure() -> None:
    result = judge_output('{"log_prefix": "[STRONG-A]"}\n这是额外解释。')
    assert result["json_valid"] is False
    assert result["format_failure"] is True


def test_judge_output_detects_conflict_marker_and_camel_case() -> None:
    result = judge_output('{"logPrefix": "x", "unresolved_conflicts": []}')
    assert result["json_valid"] is True
    assert result["snake_case_keys"] is False
    assert result["unresolved_conflict"] is False
    assert result["format_failure"] is True


def test_judge_output_requires_non_empty_conflict_content() -> None:
    empty = judge_output('{"unresolved_conflicts": []}')
    assert empty["unresolved_conflict"] is False

    filled = judge_output('{"unresolved_conflicts": ["S1 与 S2 互斥"]}')
    assert filled["unresolved_conflict"] is True


def test_groups_cover_three_intensities_and_include_constraints() -> None:
    assert {group.intensity for group in GROUPS} == {"weak", "medium", "strong"}
    assert len(GROUPS) == 4
    for group in GROUPS:
        messages = build_messages(group)
        assert len(messages) == 2
        assert all(f"[{cid}]" in messages[0]["content"] for cid, _ in group.constraints)
