"""reward 与 trace 检查对齐的本地测试。"""

from __future__ import annotations

from training.evaluate_reward import compare_to_checks


def test_compare_to_checks_matches_json_and_snake_case() -> None:
    result = {"checks": {"json_parseable": True, "snake_case_keys": True}}
    comparisons = compare_to_checks(result, {"json_valid": True, "snake_case_keys": True})
    assert comparisons == {"json_parseable": True, "snake_case_keys": True}


def test_compare_to_checks_detects_disagreement() -> None:
    result = {"checks": {"json_parseable": True, "snake_case_keys": False}}
    comparisons = compare_to_checks(result, {"json_valid": False, "snake_case_keys": True})
    assert comparisons == {"json_parseable": False, "snake_case_keys": False}
