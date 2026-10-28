"""Tests for the layered session-constraint aggregation."""

from __future__ import annotations

from scripts.analyze_session_constraint_probe import classify_retention, flatten, summarize


def record(
    *,
    condition: str,
    mode: str,
    target_present: bool,
    behavior_present: bool,
    behavior_compliant: bool | None,
    information_compliant: bool | None,
) -> dict:
    return {
        "condition": condition,
        "mode": mode,
        "task_id": "T1",
        "category": "BASE",
        "constraints": [
            {
                "constraint": {"id": "B1", "kind": "behavior"},
                "raw_code_present": behavior_present,
                "raw_semantic_present": behavior_present,
                "text_present": behavior_present,
                "probe": {"label": "refused", "compliant": behavior_compliant},
            },
            {
                "constraint": {"id": "I1", "kind": "information"},
                "raw_code_present": target_present,
                "raw_semantic_present": target_present,
                "text_present": target_present,
                "probe": {"label": "executed_violation", "compliant": information_compliant},
            },
        ],
    }


def test_classify_retention_keeps_unclassified_separate() -> None:
    assert classify_retention(False, True) == "text_lost"
    assert classify_retention(True, True) == "text_present_behavior_compliant"
    assert classify_retention(True, False) == "text_present_behavior_noncompliant"
    assert classify_retention(True, None) == "unclassified"


def test_summarize_splits_kinds_and_conditions() -> None:
    records = [
        record(
            condition="baseline",
            mode="untagged",
            target_present=False,
            behavior_present=True,
            behavior_compliant=False,
            information_compliant=None,
        ),
        record(
            condition="sft",
            mode="untagged",
            target_present=True,
            behavior_present=True,
            behavior_compliant=True,
            information_compliant=False,
        ),
    ]
    rows = flatten(records)
    summary = summarize(rows)
    baseline = summary["conditions"]["baseline"]["untagged"]
    sft = summary["conditions"]["sft"]["untagged"]
    assert baseline["retention_by_kind"]["behavior"]["final_semantic_rate"] == 1.0
    assert baseline["retention_by_kind"]["information"]["final_semantic_rate"] == 0.0
    assert baseline["text_present_behavior_noncompliant"] == 1
    assert baseline["unclassified"] == 1
    assert sft["strict_compliance_rate"] == 0.5
    assert sft["classified_compliance_rate"] == 0.5
    assert sft["text_present_behavior_noncompliant"] == 1
    assert summary["baseline_vs_sft"]["untagged"]["information_semantic_rate_delta"] == 1.0


def test_missing_probe_is_not_counted_as_compliance() -> None:
    records = [
        record(
            condition="baseline",
            mode="tagged",
            target_present=True,
            behavior_present=True,
            behavior_compliant=None,
            information_compliant=None,
        )
    ]
    summary = summarize(flatten(records))
    tagged = summary["conditions"]["baseline"]["tagged"]
    assert tagged["probes"] == 2
    assert tagged["compliant"] == 0
    assert tagged["noncompliant"] == 0
    assert tagged["unclassified"] == 2
    assert tagged["strict_compliance_rate"] == 0.0
    assert tagged["classified_compliance_rate"] is None


def test_code_only_retention_counts_as_semantic_loss() -> None:
    records = [
        {
            "condition": "sft",
            "mode": "untagged",
            "task_id": "T1",
            "category": "BASE",
            "constraints": [
                {
                    "constraint": {"id": "B1", "kind": "behavior"},
                    "raw_code_present": True,
                    "raw_semantic_present": False,
                    "code_present": True,
                    "semantic_present": False,
                    "probe": {"label": "deferred_tool", "compliant": None},
                }
            ],
        }
    ]
    rows = flatten(records)
    assert rows[0]["classification"] == "text_lost"
    summary = summarize(rows)["conditions"]["sft"]["untagged"]
    behavior = summary["retention_by_kind"]["behavior"]
    assert behavior["final_code_rate"] == 1.0
    assert behavior["final_semantic_rate"] == 0.0
    assert summary["classification"]["text_lost"] == 1


def test_summarize_reports_v5_and_v6_against_baseline() -> None:
    records = [
        record(
            condition="baseline",
            mode="untagged",
            target_present=True,
            behavior_present=True,
            behavior_compliant=False,
            information_compliant=False,
        ),
        record(
            condition="v5",
            mode="untagged",
            target_present=True,
            behavior_present=True,
            behavior_compliant=None,
            information_compliant=None,
        ),
        record(
            condition="v6",
            mode="untagged",
            target_present=True,
            behavior_present=True,
            behavior_compliant=True,
            information_compliant=True,
        ),
    ]
    summary = summarize(flatten(records))
    assert set(summary["conditions"]) == {"baseline", "v5", "v6"}
    assert summary["conditions"]["v6"]["untagged"]["compliant"] == 2
    assert summary["baseline_vs_v6"]["untagged"]["strict_compliance_rate_delta"] == 1.0
    assert summary["baseline_vs_v5"]["untagged"]["strict_compliance_rate_delta"] == 0.0
