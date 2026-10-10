from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SOURCE_MAP = {
    "xlam-schema-mapped": "xLAM",
    "stats-heldout-v2": "stats",
    "train-v2-retained": "U-TR",
    "train-v4-oracle": "U-TR",
    "train-v5-teacher": "U-TR",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是合法 JSON") from exc
    return rows


def resolve_source(origin: str) -> str:
    try:
        return SOURCE_MAP[origin]
    except KeyError as exc:
        raise ValueError(f"未知 origin：{origin}") from exc


def resolve_constraint_type(spec: dict[str, Any]) -> tuple[str, str]:
    family = str(spec.get("family", ""))
    tools = set(spec.get("required_tools") or []) | set(spec.get("allowed_tools") or [])
    if family == "conflict":
        return "conflict", "family=conflict"
    if family == "context_recall":
        return "context_recall", "family=context_recall"
    if family == "path_params":
        return "path", "family=path_params"
    if "edit_file" in tools or "write_file" in tools:
        return "edit", "required/allowed_tools 含 edit_file/write_file"
    return "information", "只读工具（read_file/list_dir/grep）"


def resolve_pressure_type(spec: dict[str, Any]) -> tuple[str, str]:
    family = str(spec.get("family", ""))
    if family == "conflict":
        return "任务冲突", "family=conflict"
    if family == "context_recall":
        return "多轮", "family=context_recall"
    return "无", "基础池无压力变体"


def load_baseline(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    pattern = str(
        repo_root / "private/kaggle_difficulty/runs*/difficulty/shard-*/baseline/runs.jsonl"
    )
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        for row in read_jsonl(Path(path)):
            by_task[str(row["task_id"])].append(row)
    return by_task


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 v10 409 池分层元数据")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--pool-specs",
        type=Path,
        default=Path("private/heldout_difficulty_pool/pool_specs.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("private/v10/pool_metadata_v10.jsonl"))
    parser.add_argument(
        "--summary", type=Path, default=Path("private/v10/pool_metadata_summary.json")
    )
    parser.add_argument(
        "--final-test",
        type=Path,
        default=Path("private/final_test_v3/tasks.jsonl"),
    )
    args = parser.parse_args()

    repo_root: Path = args.repo_root
    pool_specs_path = repo_root / args.pool_specs
    output_path = repo_root / args.output
    summary_path = repo_root / args.summary
    final_test_path = repo_root / args.final_test

    specs = read_jsonl(pool_specs_path)
    if len(specs) != 409:
        raise ValueError(f"期望 409 条池任务，实际 {len(specs)}")
    task_ids = [str(spec["task_id"]) for spec in specs]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("pool_specs 存在重复 task_id")

    baseline = load_baseline(repo_root)
    output_rows: list[dict[str, Any]] = []
    for spec in specs:
        task_id = str(spec["task_id"])
        constraint_type, constraint_basis = resolve_constraint_type(spec)
        pressure_type, pressure_basis = resolve_pressure_type(spec)
        runs = baseline.get(task_id, [])
        successes = sum(1 for row in runs if row.get("task_success") is True)
        pass_rate = round(successes / len(runs), 6) if runs else None
        middle_state = pass_rate is not None and 0.3 <= pass_rate <= 0.7
        middle_note = ""
        if str(spec.get("family")) in {"conflict", "context_recall"}:
            middle_note = "池中有但中间态为 0" if not middle_state else "池中有且存在中间态"
        output_rows.append(
            {
                "task_id": task_id,
                "source": resolve_source(str(spec["origin"])),
                "source_detail": spec["origin"],
                "constraint_type": constraint_type,
                "pressure_type": pressure_type,
                "family": spec.get("family"),
                "category": spec.get("category"),
                "difficulty": spec.get("difficulty"),
                "context_tier": spec.get("context_tier"),
                "required_tools": spec.get("required_tools") or [],
                "allowed_tools": spec.get("allowed_tools") or [],
                "constraint_basis": constraint_basis,
                "pressure_basis": pressure_basis,
                "baseline_rollouts": len(runs),
                "baseline_success": successes,
                "baseline_pass_rate": pass_rate,
                "middle_state": middle_state,
                "middle_state_note": middle_note,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    final_test_ids = {str(row["task_id"]) for row in read_jsonl(final_test_path)}
    missing_final = sorted(final_test_ids - set(task_ids))
    family_mid = Counter(row["family"] for row in output_rows if row["middle_state"] is True)
    family_n = Counter(str(row["family"]) for row in output_rows)
    summary = {
        "version": "v10-pool-metadata-1",
        "rows": len(output_rows),
        "task_ids_unique": len(set(task_ids)),
        "all_labels_present": all(
            row["source"] and row["constraint_type"] and row["pressure_type"] for row in output_rows
        ),
        "by_source": dict(Counter(row["source"] for row in output_rows)),
        "by_source_detail": dict(Counter(str(row["source_detail"]) for row in output_rows)),
        "by_constraint_type": dict(Counter(row["constraint_type"] for row in output_rows)),
        "by_pressure_type": dict(Counter(row["pressure_type"] for row in output_rows)),
        "by_family": dict(family_n),
        "middle_state_by_family": dict(family_mid),
        "conflict": {
            "tasks": family_n.get("conflict", 0),
            "middle_state": family_mid.get("conflict", 0),
            "note": "池中有但中间态为 0" if family_mid.get("conflict", 0) == 0 else "存在中间态",
        },
        "context_recall": {
            "tasks": family_n.get("context_recall", 0),
            "middle_state": family_mid.get("context_recall", 0),
            "note": (
                "池中有但中间态为 0" if family_mid.get("context_recall", 0) == 0 else "存在中间态"
            ),
        },
        "delete_label_count": sum(1 for row in output_rows if row["constraint_type"] == "delete"),
        "final_test_tasks": len(final_test_ids),
        "final_test_missing_from_pool": missing_final,
        "baseline_source": "private/kaggle_difficulty/runs*/difficulty/shard-*/baseline/runs.jsonl",
        "baseline_rollouts_per_task": dict(
            Counter(row["baseline_rollouts"] for row in output_rows)
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
