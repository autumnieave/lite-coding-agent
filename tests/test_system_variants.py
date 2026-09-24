"""System-side variant registry tests; no model and no rollout."""

from __future__ import annotations

import sys

import pytest

from scripts import four_group_eval
from training.system_variants import (
    JSON_SELF_CHECK_SUFFIX,
    PLAIN_SELF_CHECK_SUFFIX,
    SELF_CHECK_SUFFIX,
    get_variant,
    render_prompt_suffix,
)


class _Check:
    def __init__(self, name: str) -> None:
        self.name = name


class _Spec:
    def __init__(self, *checks: str) -> None:
        self.checks = tuple(_Check(name) for name in checks)


def test_s0_has_no_prompt_suffix() -> None:
    variant = get_variant("s0")
    assert variant.prompt_suffix == ""
    assert variant.output_self_check is False
    assert variant.constrained_decoding is False


def test_s1_is_prompt_layer_self_check() -> None:
    variant = get_variant("s1")
    assert variant.output_self_check is True
    assert variant.prompt_suffix == SELF_CHECK_SUFFIX
    assert variant.availability == "available"


def test_s1t_is_task_aware_and_keeps_generic_s1_intact() -> None:
    json_variant = get_variant("s1t")
    plain_variant = get_variant("s1t")
    json_suffix = render_prompt_suffix(json_variant, _Spec("json_valid", "context_read"))
    plain_suffix = render_prompt_suffix(plain_variant, _Spec("answer_exact", "context_read"))

    assert JSON_SELF_CHECK_SUFFIX in json_suffix
    assert PLAIN_SELF_CHECK_SUFFIX in plain_suffix
    assert "JSON" not in plain_suffix
    assert get_variant("s1").prompt_suffix == SELF_CHECK_SUFFIX
    assert "s1t" not in get_variant("s1").id


def test_s2_is_preregistered_but_not_available() -> None:
    variant = get_variant("s2")
    assert variant.constrained_decoding is True
    assert variant.task_aware_self_check is True
    assert variant.availability == "kaggle_not_run"


def test_four_group_s1_maps_to_task_aware_variant() -> None:
    assert four_group_eval.GROUP_TO_VARIANT["S1"] == "s1t"


def test_unknown_variant_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown system variant"):
        get_variant("s9")


def test_training_groups_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["four_group_eval.py", "--group", "SFT"])
    assert four_group_eval.main() == 2
