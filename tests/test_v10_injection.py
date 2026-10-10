"""v10 severity injection and layered eval framework tests (CPU only)."""

from __future__ import annotations

from collections import Counter

from scripts.v10.build_conflict_tasks import _build_delete_tasks
from training.v10_eval_set import (
    build_eval_probes,
    find_shingle_overlaps,
    layer_counts,
    unique_task_count,
)
from training.v10_injection import (
    SAMPLES_PER_SEVERITY,
    SEVERITY_RULES,
    build_injection_samples,
    consistency_report,
)


def _task_ids(count: int = 60) -> list[str]:
    return [f"V10-TEST-TASK-{index:03d}" for index in range(count)]


def test_injection_builds_three_equal_tiers() -> None:
    samples = build_injection_samples(_task_ids())
    assert len(samples) == SAMPLES_PER_SEVERITY * 3
    counts = Counter(sample.severity for sample in samples)
    assert counts == {3: 12, 2: 12, 1: 12}
    assert len({sample.task_id for sample in samples}) == 36


def test_injection_labels_match_the_severity_ladder() -> None:
    samples = build_injection_samples(_task_ids())
    for sample in samples:
        rule = SEVERITY_RULES[sample.severity]
        assert sample.violation == rule.violation
        assert sample.style == rule.style
        assert sample.expected_label == rule.expected_label
        assert sample.verifier_label == rule.expected_label
        assert sample.match is True


def test_injection_consistency_passes_threshold() -> None:
    report = consistency_report(build_injection_samples(_task_ids()))
    assert report["total"] == 36
    assert report["matched"] == 36
    assert report["rate"] == 1.0
    assert report["passes_threshold"] is True
    assert report["verifier_labels"] == {
        "ambiguous": 12,
        "deferred_tool": 12,
        "executed_violation": 12,
    }


def test_safe_completions_are_compliant_without_tool_calls() -> None:
    samples = build_injection_samples(_task_ids())
    for sample in samples:
        assert sample.safe_compliant is True
        assert sample.safe_label in {"refused", "confirmed"}
        assert sample.safe_completion != sample.content


def test_training_rows_use_safe_completion_not_defect() -> None:
    samples = build_injection_samples(_task_ids())
    for sample in samples:
        row = sample.to_training_row()
        assert row["completion"] == sample.safe_completion
        assert sample.content not in row["completion"]
        assert row["metadata"]["source"] == "v10_defect_injection"
        assert row["metadata"]["severity"] == sample.severity
        user_messages = [
            message["content"] for message in row["messages"] if message["role"] == "user"
        ]
        assert user_messages == [sample.probe]
        layers = [
            message["content"]
            for message in row["messages"]
            if "[SYSTEM CONTEXT" in message["content"]
        ]
        assert sample.constraint_id in layers[0]


def test_defect_records_keep_rejected_side() -> None:
    samples = build_injection_samples(_task_ids())
    record = samples[0].to_defect_record()
    assert record["defect_content"] == samples[0].content
    assert record["defect_tool_calls"] == [dict(call) for call in samples[0].tool_calls]
    assert record["expected_label"] == samples[0].expected_label


def _pool_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(8):
        rows.append(
            {
                "task_id": f"V10-PATH-{index:02d}",
                "constraint_type": "path",
                "family": "path_params",
            }
        )
    for index in range(10):
        rows.append(
            {
                "task_id": f"V10-EDIT-{index:02d}",
                "constraint_type": "edit",
                "family": "03",
            }
        )
    families = [
        "public_grep",
        "json_constraints",
        "policy_extract",
        "selective_aggregate",
        "search_collect",
        "audit_reconcile",
        "edit_verify",
        "01",
        "02",
    ]
    for index in range(13):
        rows.append(
            {
                "task_id": f"V10-NORMAL-{index:02d}",
                "constraint_type": "information",
                "family": families[index % len(families)],
            }
        )
    return rows


def test_eval_framework_has_four_layers_and_minimum_counts() -> None:
    probes = build_eval_probes(_pool_rows(), _build_delete_tasks(), set())
    assert len(probes) == 72
    assert layer_counts(probes) == {"delete": 18, "edit": 14, "normal": 12, "path": 28}
    assert unique_task_count(probes) == {"delete": 6, "edit": 9, "normal": 12, "path": 8}
    assert all(len(probe.fixture_files) > 0 for probe in probes)
    assert all(
        probe.conflict_trigger == "pending_gpu" for probe in probes if probe.layer != "normal"
    )
    targeted = [probe for probe in probes if probe.backfill_feature]
    assert len(targeted) == 10
    assert [probe.probe_id for probe in targeted] == [
        f"V10-EVAL-PATH-{number:03d}" for number in range(19, 29)
    ]
    assert {probe.backfill_feature for probe in targeted} == {
        "plausible_shared_path",
        "explicit_outside_write",
        "task_goal_outside_write",
    }


def test_eval_task_ids_do_not_overlap_training_ids() -> None:
    training_ids = {"V10-PATH-00", "V10-EDIT-00", "V10-NORMAL-00"}
    probes = build_eval_probes(_pool_rows(), _build_delete_tasks(), training_ids)
    assert not ({probe.task_id for probe in probes} & training_ids)


def test_shingle_detector_finds_verbatim_and_rejects_unrelated() -> None:
    probes = build_eval_probes(_pool_rows(), _build_delete_tasks(), set())
    duplicate = [(f"dup:{probe.probe_id}", probe.probe) for probe in probes[:2]]
    assert find_shingle_overlaps(probes[:2], duplicate, 16)
    assert find_shingle_overlaps(probes[:2], [("unrelated", "完全无关的另一段文本。")], 16) == []
