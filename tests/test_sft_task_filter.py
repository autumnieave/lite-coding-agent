"""Task filtering for teacher probes; no network."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from training.build_sft_dataset import filter_specs_by_task


@dataclass
class Spec:
    task_id: str


def test_filter_keeps_requested_order_and_rejects_unknown() -> None:
    specs = [Spec("a"), Spec("b"), Spec("c")]
    assert [s.task_id for s in filter_specs_by_task(specs, "c,a")] == ["a", "c"]
    assert filter_specs_by_task(specs, "") == specs
    with pytest.raises(ValueError, match="unknown task ids"):
        filter_specs_by_task(specs, "missing")
