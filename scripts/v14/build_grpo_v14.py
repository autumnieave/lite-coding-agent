"""Freeze the v14 GRPO prompt pool from the middle-state task evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""

    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _layer(task: Mapping[str, Any]) -> str:
    category = str(task.get("category") or "")
    if category == "path_escape":
        return "path"
    if category == "delete_without_confirm":
        return "delete"
    if category == "edit_tests":
        return "edit"
    return "policy"


def _middle_ids(analysis: Mapping[str, Any], *, key: str) -> set[str]:
    rows = analysis.get("task_rows") or analysis.get("per_task") or []
    return {str(row["task_id"]) for row in rows if bool(row.get(key))}


def build(
    *,
    pool_analysis: Path,
    granular_analysis: Path,
    pool_tasks: Path,
    granular_tasks: Path,
    eval_file: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Build the frozen 29-task prompt pool (policy+path+delete, no edit)."""

    pool = {str(row["task_id"]): dict(row) for row in load_jsonl(pool_tasks)}
    granular = {str(row["task_id"]): dict(row) for row in load_jsonl(granular_tasks)}
    tasks = {**pool, **granular}
    pool_middle = _middle_ids(
        json.loads(pool_analysis.read_text(encoding="utf-8")), key="middle_2_6"
    )
    granular_middle = _middle_ids(
        json.loads(granular_analysis.read_text(encoding="utf-8")), key="middle_state"
    )
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for task_id in sorted(pool_middle | granular_middle):
        task = tasks.get(task_id)
        if task is None:
            missing.append(task_id)
            continue
        layer = _layer(task)
        if layer == "edit":
            continue
        selected.append(
            {
                "task_id": task_id,
                "category": str(task.get("category") or ""),
                "layer": layer,
                "severity": int(task.get("severity") or 0),
                "source_set": "pool_middle" if task_id in pool_middle else "granular_middle",
                "task": task,
            }
        )
    if missing:
        raise ValueError(f"missing task definitions: {missing}")
    eval_ids = {str(row["task_id"]) for row in load_jsonl(eval_file)}
    overlap = sorted({row["task_id"] for row in selected} & eval_ids)
    if overlap:
        raise ValueError(f"GRPO/eval task overlap: {overlap}")
    counts = Counter(row["layer"] for row in selected)
    expected = {"policy": 14, "path": 9, "delete": 6}
    if dict(counts) != expected:
        raise ValueError(f"layer gate mismatch: {dict(counts)} != {expected}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / "prompts.jsonl"
    with prompt_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "version": "v14-grpo-prompts-1",
        "tasks": len(selected),
        "by_layer": dict(sorted(counts.items())),
        "by_severity": dict(sorted(Counter(row["severity"] for row in selected).items())),
        "by_source_set": dict(sorted(Counter(row["source_set"] for row in selected).items())),
        "edit_excluded": sorted(
            task_id
            for task_id in (pool_middle | granular_middle)
            if task_id in tasks and _layer(tasks[task_id]) == "edit"
        ),
        "task_overlap_with_eval": overlap,
        "prompts_sha256": sha256_file(prompt_path),
        "inputs": {
            "pool_analysis": str(pool_analysis),
            "pool_analysis_sha256": sha256_file(pool_analysis),
            "granular_analysis": str(granular_analysis),
            "granular_analysis_sha256": sha256_file(granular_analysis),
            "pool_tasks": str(pool_tasks),
            "pool_tasks_sha256": sha256_file(pool_tasks),
            "granular_tasks": str(granular_tasks),
            "granular_tasks_sha256": sha256_file(granular_tasks),
            "eval_file": str(eval_file),
            "eval_sha256": sha256_file(eval_file),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build v14 GRPO prompt pool")
    parser.add_argument(
        "--pool-analysis",
        type=Path,
        default=Path("private/v14/step0_precheck/analysis60_v2/middle_state_tasks.json"),
    )
    parser.add_argument(
        "--granular-analysis",
        type=Path,
        default=Path("private/v14/step0_5_granular/analysis.json"),
    )
    parser.add_argument(
        "--pool-tasks", type=Path, default=Path("private/v14/step0_precheck/tasks_60.jsonl")
    )
    parser.add_argument(
        "--granular-tasks", type=Path, default=Path("private/v14/step0_5_granular/tasks.jsonl")
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("private/v14/step2_benchmark/eval_set_frozen.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("private/v14/step5_grpo/prompts"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build(
        pool_analysis=args.pool_analysis,
        granular_analysis=args.granular_analysis,
        pool_tasks=args.pool_tasks,
        granular_tasks=args.granular_tasks,
        eval_file=args.eval_file,
        out_dir=args.out_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
