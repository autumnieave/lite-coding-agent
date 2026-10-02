"""Tests for task-level aggregation of repeated constraint probes."""

from __future__ import annotations

from scripts.analyze_grpo_constraint_signal import flatten, render_markdown, summarize


def record(task_id: str, kind: str, compliant: bool | None, rollout: int) -> dict:
    return {
        "condition": "v6",
        "mode": "untagged",
        "task_id": task_id,
        "category": "BASE",
        "rollout": rollout,
        "constraints": [
            {
                "constraint": {"id": f"{kind}-{task_id}", "kind": kind},
                "semantic_present": True,
                "probe": {
                    "compliant": compliant,
                    "label": "compliant_tool" if compliant else "ambiguous",
                },
            }
        ],
    }


def test_group_distribution_separates_endpoints_and_intermediate() -> None:
    records = []
    records.extend(record("task-a", "behavior", True, rollout) for rollout in range(8))
    records.extend(record("task-b", "behavior", rollout < 4, rollout) for rollout in range(8))
    records.extend(record("task-c", "information", None, rollout) for rollout in range(8))
    records.extend(record("task-d", "information", False, rollout) for rollout in range(8))

    result = summarize(flatten(records))

    assert result["overall"]["groups"] == 4
    assert result["overall"]["probes"] == 32
    assert result["overall"]["patterns"]["all_compliant"] == 1
    assert result["overall"]["patterns"]["all_noncompliant"] == 1
    assert result["overall"]["patterns"]["all_unclassified"] == 1
    assert result["overall"]["patterns"]["intermediate_compliant"] == 1
    assert result["verdict"]["group_pattern"] == "intermediate_present"
    assert result["by_kind"]["behavior"]["compliant"] == 12
    assert result["by_kind"]["information"]["unclassified"] == 8


def test_markdown_reports_sampling_and_boundary() -> None:
    rows = flatten([record("task-a", "behavior", True, rollout) for rollout in range(8)])
    result = summarize(rows)
    markdown = render_markdown(
        result,
        {
            "adapter_sha256": "adapter-sha",
            "config": {
                "temperature": 0.7,
                "top_p": 0.9,
                "rollouts": 8,
                "decoding": "sampled",
            },
        },
    )
    assert "temperature=0.7" in markdown
    assert "采样口径" in markdown
    assert "不是 held-out" in markdown
    assert "| task-a | behavior | 8 | 0 | 0 | all_compliant |" in markdown
