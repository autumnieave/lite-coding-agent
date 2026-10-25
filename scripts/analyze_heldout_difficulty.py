"""Aggregate difficulty-screen shards and apply the pre-registered decision."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from training.heldout_pool import build_heldout_pool

MIDDLE_MIN_RATE = 0.30
MIDDLE_MAX_RATE = 0.70
PREREGISTERED_ROLLOUTS = 8


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "baseline" not in payload:
        raise ValueError(f"{path} is not a difficulty-screen summary")
    return payload


def collect_tasks(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for summary in summaries:
        baseline = summary["baseline"]
        for task in baseline.get("tasks", []):
            task_id = str(task["task_id"])
            if task_id in seen:
                raise ValueError(f"task appears in more than one shard: {task_id}")
            seen.add(task_id)
            rows.append(dict(task))
    return rows


def classify_middle(rows: list[dict[str, Any]], rollouts: int) -> dict[str, Any]:
    middle: list[str] = []
    incomplete: dict[str, str] = {}
    for row in rows:
        task_id = str(row["task_id"])
        valid = int(row.get("valid_rollouts") or 0)
        errors = int(row.get("errors") or 0)
        if valid != rollouts or errors:
            incomplete[task_id] = f"valid={valid},errors={errors},expected={rollouts}"
            continue
        rate = float(row.get("pass_rate") or 0.0)
        if MIDDLE_MIN_RATE <= rate <= MIDDLE_MAX_RATE:
            middle.append(task_id)
    return {
        "min_rate": MIDDLE_MIN_RATE,
        "max_rate": MIDDLE_MAX_RATE,
        "rollouts": rollouts,
        "eligible_tasks": len(rows) - len(incomplete),
        "middle_count": len(middle),
        "middle_task_ids": middle,
        "incomplete": incomplete,
    }


def decision_for(count: int) -> str:
    if count >= 10:
        return "proceed_final_test_v3"
    if count >= 5:
        return "needs_layer2_supplement"
    return "enter_layer2_systematic_difficulty_adjustment"


def pass_rate_distribution(
    rows: list[dict[str, Any]], incomplete: dict[str, str]
) -> dict[str, int]:
    """Count pass rates over eligible tasks only.

    Incomplete tasks (error or fewer than the pre-registered number of valid
    rollouts) must not leak into the distribution: a 7-valid-sample task has a
    pass rate that is not comparable with the 8-sample tasks.
    """
    eligible = [row for row in rows if str(row["task_id"]) not in incomplete]
    return dict(sorted(Counter(str(row.get("pass_rate")) for row in eligible).items()))


def _task_metadata(
    train_file: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pool = build_heldout_pool(train_file)
    metadata = {task.spec.task_id: task.manifest_row() for task in pool.tasks}
    return metadata, pool.report


def report_status(task_count: int, expected_tasks: int, incomplete_count: int) -> str:
    if expected_tasks and task_count < expected_tasks:
        return "partial"
    if incomplete_count:
        return "completed_with_exclusions"
    return "completed"


def enrich_tasks(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row["task_id"])
        meta = metadata.get(task_id, {})
        result.append(
            {
                **row,
                "origin": meta.get("origin"),
                "category": meta.get("category"),
                "family": meta.get("family", row.get("family")),
                "training_overlap": meta.get("origin") != "stats-heldout-v2",
            }
        )
    return result


def summarize_group(rows: list[dict[str, Any]], middle_ids: set[str], key: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    middle: Counter[str] = Counter()
    for row in rows:
        raw = row.get(key)
        value = "unknown" if raw is None else str(raw)
        counts[value] += 1
        if str(row["task_id"]) in middle_ids:
            middle[value] += 1
    return {value: {"tasks": counts[value], "middle": middle[value]} for value in sorted(counts)}


def compare_middle(
    primary: list[dict[str, Any]], comparison: list[dict[str, Any]]
) -> dict[str, Any]:
    primary_middle = {str(row["task_id"]) for row in primary}
    comparison_middle = {str(row["task_id"]) for row in comparison}
    return {
        "primary_middle": len(primary_middle),
        "comparison_middle": len(comparison_middle),
        "overlap": sorted(primary_middle & comparison_middle),
        "overlap_count": len(primary_middle & comparison_middle),
        "primary_only": sorted(primary_middle - comparison_middle),
        "comparison_only": sorted(comparison_middle - primary_middle),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="aggregate held-out difficulty shards")
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--comparison-summary", type=Path, action="append", default=[])
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("private/sft_data_v5/train.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summaries = [load_summary(path) for path in args.summary]
    rows = collect_tasks(summaries)
    metadata, pool_report = _task_metadata(args.train_file)
    middle = classify_middle(rows, PREREGISTERED_ROLLOUTS)
    middle_ids = set(middle["middle_task_ids"])
    enriched = enrich_tasks(rows, metadata)
    expected_tasks = int(pool_report.get("evaluation_tasks") or 0)
    status = report_status(len(rows), expected_tasks, len(middle["incomplete"]))

    report: dict[str, Any] = {
        "version": "heldout-difficulty-report-v1",
        "shards": [str(path) for path in args.summary],
        "tasks": len(rows),
        "expected_tasks": expected_tasks,
        "status": status,
        "decision": decision_for(middle["middle_count"]),
        "middle_state": middle,
        "pass_rate_distribution": pass_rate_distribution(rows, middle["incomplete"]),
        "by_origin": summarize_group(enriched, middle_ids, "origin"),
        "by_category": summarize_group(enriched, middle_ids, "category"),
        "by_family": summarize_group(enriched, middle_ids, "family"),
        "by_difficulty": summarize_group(enriched, middle_ids, "difficulty"),
        "by_training_overlap": summarize_group(enriched, middle_ids, "training_overlap"),
    }
    if args.comparison_summary:
        comparison_rows = collect_tasks([load_summary(path) for path in args.comparison_summary])
        comparison_middle = classify_middle(comparison_rows, PREREGISTERED_ROLLOUTS)
        report["comparison"] = {
            "summaries": [str(path) for path in args.comparison_summary],
            "middle_state": comparison_middle,
            "overlap": compare_middle(
                [{"task_id": task_id} for task_id in middle["middle_task_ids"]],
                [{"task_id": task_id} for task_id in comparison_middle["middle_task_ids"]],
            ),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "heldout_difficulty_report.json"
    middle_path = args.out_dir / "middle_tasks.jsonl"
    all_path = args.out_dir / "all_tasks.jsonl"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with middle_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in enriched:
            if str(row["task_id"]) in middle_ids:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with all_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 1 if report["status"] == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
