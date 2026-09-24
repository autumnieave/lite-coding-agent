"""Tests for the v8 paired-supervision dataset builder."""

from __future__ import annotations

from collections import Counter

from training.cross_template import build_paired_train_v8 as builder


def test_paired_rows_have_expected_shape() -> None:
    rows = builder.build_paired_rows()
    assert len(rows) == 72
    kinds = Counter(row["id"].split(":")[2] for row in rows)
    assert kinds == {"edit_tests": 24, "delete_without_confirm": 24, "path_escape": 24}


def test_each_row_injects_target_plus_four_constraints() -> None:
    rows = builder.build_paired_rows()
    for row in rows:
        layer = next(
            message["content"]
            for message in row["messages"]
            if message["content"].startswith(builder.LAYER_HEADING)
        )
        lines = [line for line in layer.splitlines() if line.startswith("- [")]
        assert len(lines) == 5
        target = row["metadata"]["target_constraint_id"]
        assert target in layer
        assert row["completion"].startswith("拒绝：")
        assert target in row["completion"]


def test_constraint_codes_and_requests_are_unique() -> None:
    rows = builder.build_paired_rows()
    codes = [row["metadata"]["target_constraint_id"] for row in rows]
    assert len(set(codes)) == 18
    requests = [next(m["content"] for m in row["messages"] if m["role"] == "user") for row in rows]
    assert len(set(requests)) == 72
    for row in rows:
        assert row["id"].startswith("v8:paired:")


def test_target_position_is_not_fixed() -> None:
    rows = builder.build_paired_rows()
    positions = []
    for row in rows:
        layer = next(
            message["content"]
            for message in row["messages"]
            if message["content"].startswith(builder.LAYER_HEADING)
        )
        ids = [
            line[3 : 3 + builder.CODE_LENGTH]
            for line in layer.splitlines()
            if line.startswith("- [")
        ]
        positions.append(ids.index(row["metadata"]["target_constraint_id"]))
    assert len(set(positions)) >= 4


def test_contamination_reports_verbatim_and_shingles() -> None:
    rows = builder.build_paired_rows()
    heldout = ["完全不同的约束文本。", "完全不同的请求文本。"]
    result = builder.contamination(rows, heldout)
    assert result["verbatim_hits"] == []
    assert result["request_shingle16_hits"] == []


def test_contamination_detects_a_verbatim_probe() -> None:
    rows = builder.build_paired_rows()
    request = next(m["content"] for m in rows[0]["messages"] if m["role"] == "user")
    result = builder.contamination(rows[:1], ["约束文本。", request])
    assert rows[0]["id"] in result["request_verbatim_hits"]


def test_validate_rows_reports_no_failures() -> None:
    report = builder.validate_rows(builder.build_paired_rows())
    assert report["failures"] == []
