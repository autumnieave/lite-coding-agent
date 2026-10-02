from __future__ import annotations

from training.severity_weights import normalised_weights, weight_for_prompt

WEIGHTS = {"V14P-C3-01": 3.0, "V14P-C2-02": 2.0, "V14P-C1-03": 1.0}


def test_weight_lookup_by_constraint_id() -> None:
    assert weight_for_prompt("... [V14P-C2-02] 删除前确认 ...", WEIGHTS) == 2.0
    assert weight_for_prompt("[V14P-C3-01] 路径", WEIGHTS) == 3.0


def test_weight_falls_back_to_one() -> None:
    assert weight_for_prompt("no constraint id here", WEIGHTS) == 1.0
    assert weight_for_prompt("", WEIGHTS) == 1.0


def test_empty_constraint_id_is_ignored() -> None:
    assert weight_for_prompt("anything", {"": 9.0, "V14P-C1-03": 1.0}) == 1.0


def test_normalised_weights_keep_mean_one() -> None:
    scaled = normalised_weights([3.0, 2.0, 1.0])
    assert sum(scaled) / len(scaled) == 1.0
    assert scaled[0] > scaled[1] > scaled[2]


def test_normalised_weights_handle_empty_and_zero() -> None:
    assert normalised_weights([]) == []
    assert normalised_weights([0.0, 0.0]) == [1.0, 1.0]
