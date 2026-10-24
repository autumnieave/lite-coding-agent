"""Build the E1 401-row control sample from the 468-row train-v5 file.

The sample keeps the *row-level* middle-state proportion approximately equal to
the source (57 of 401 here), so E1 changes data volume relative to train-v6
without silently removing the middle-state mixture altogether.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def load_control_vars(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if match is None:
        raise ValueError(f"CONTROL_VARS.md has no fenced JSON block: {path}")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError(f"control variables must be a JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_id_for_row(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("train row has no metadata object")
    task_id = str(metadata.get("task_id") or "")
    if not task_id:
        raise ValueError("train row has no metadata.task_id")
    return task_id


def read_middle_task_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        task_id = str(payload.get("task_id") or "")
        if not task_id:
            raise ValueError(f"{path}:{line_number} has no task_id")
        ids.add(task_id)
    return ids


def stratified_sample(
    rows: Sequence[Mapping[str, Any]], middle_ids: set[str], target_rows: int, seed: int
) -> tuple[list[dict[str, Any]], int, int]:
    middle_indices = [index for index, row in enumerate(rows) if task_id_for_row(row) in middle_ids]
    non_middle_indices = [
        index for index, row in enumerate(rows) if task_id_for_row(row) not in middle_ids
    ]
    if not middle_indices:
        raise ValueError("no middle-state rows found in source train file")
    if target_rows > len(rows):
        raise ValueError(f"target rows {target_rows} exceed source rows {len(rows)}")
    middle_target = round(target_rows * len(middle_indices) / len(rows))
    middle_target = max(1, min(len(middle_indices), middle_target))
    non_middle_target = target_rows - middle_target
    if non_middle_target > len(non_middle_indices):
        raise ValueError("not enough non-middle rows for the target sample")
    rng = random.Random(seed)
    selected_indices = sorted(
        rng.sample(middle_indices, middle_target)
        + rng.sample(non_middle_indices, non_middle_target)
    )
    selected: list[dict[str, Any]] = []
    for index in selected_indices:
        row = dict(rows[index])
        metadata = dict(row.get("metadata") or {})
        metadata["e1_stratum"] = "middle_state" if index in middle_indices else "non_middle"
        metadata["e1_arm"] = "g1"
        row["metadata"] = metadata
        selected.append(row)
    return selected, middle_target, non_middle_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build E1 401-row control sample")
    parser.add_argument("--train-v5", type=Path, default=Path("private/sft_data_v5/train.jsonl"))
    parser.add_argument(
        "--middle-tasks",
        type=Path,
        default=Path("private/final_test_v3/tasks.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("private/cross_template/e1"))
    parser.add_argument(
        "--control-vars",
        type=Path,
        default=Path("training/cross_template/CONTROL_VARS.md"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    control = load_control_vars(args.control_vars)
    source_rows = read_jsonl(args.train_v5)
    expected_source = int(control["train_v5_rows"])
    if len(source_rows) != expected_source:
        raise ValueError(f"expected {expected_source} train-v5 rows, found {len(source_rows)}")
    middle_ids = read_middle_task_ids(args.middle_tasks)
    target_rows = int(control["total_train_rows"])
    selected, middle_target, non_middle_target = stratified_sample(
        source_rows, middle_ids, target_rows, int(control["seed"])
    )
    if len(selected) != target_rows:
        raise ValueError(f"selected {len(selected)} rows, expected {target_rows}")

    manifest = {
        "version": "e1-sample-v1",
        "input": {
            "train_v5": str(args.train_v5),
            "train_v5_sha256": sha256(args.train_v5),
            "middle_tasks": str(args.middle_tasks),
            "middle_tasks_sha256": sha256(args.middle_tasks),
        },
        "outputs": {
            "train": str(args.out_dir / "train.jsonl"),
            "manifest": str(args.out_dir / "manifest.json"),
        },
        "control_vars": str(args.control_vars),
        "seed": int(control["seed"]),
        "counts": {
            "source_rows": len(source_rows),
            "selected_rows": len(selected),
            "middle_rows_in_source": sum(
                1 for row in source_rows if task_id_for_row(row) in middle_ids
            ),
            "middle_rows_selected": middle_target,
            "non_middle_rows_selected": non_middle_target,
            "middle_ratio_before": round(
                sum(1 for row in source_rows if task_id_for_row(row) in middle_ids)
                / len(source_rows),
                6,
            ),
            "middle_ratio_after": round(middle_target / len(selected), 6),
        },
    }
    if args.dry_run:
        print(json.dumps({"status": "dry_run", **manifest}, ensure_ascii=False, indent=2))
        return 0

    write_jsonl(args.out_dir / "train.jsonl", selected)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
