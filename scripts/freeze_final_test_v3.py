"""Freeze the difficulty-screen middle-state tasks as ``final-test-v3``.

The frozen file is a *difficulty-calibrated diagnostic set*.  Most pool tasks
also appear in train-v5, so training-overlap is recorded per task and reported
separately; only ``stats-heldout-v2`` tasks support a clean held-out claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.heldout_pool import HeldoutPool, build_heldout_pool


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def freeze(
    pool: HeldoutPool,
    middle_rows: Sequence[Mapping[str, Any]],
    *,
    created_at: str,
    sources: Sequence[str],
) -> tuple[bytes, dict[str, Any], list[str]]:
    by_id = {task.spec.task_id: task for task in pool.tasks}
    selected: list[tuple[Any, Mapping[str, Any]]] = []
    missing: list[str] = []
    for row in sorted(middle_rows, key=lambda item: str(item["task_id"])):
        task_id = str(row["task_id"])
        task = by_id.get(task_id)
        if task is None:
            missing.append(task_id)
            continue
        selected.append((task, row))
    if not selected:
        raise ValueError("no middle-state task exists in the pool")

    lines: list[str] = []
    clean = 0
    families: Counter[str] = Counter()
    origins: Counter[str] = Counter()
    for task, row in selected:
        overlap = task.origin != "stats-heldout-v2"
        if not overlap:
            clean += 1
        families[task.spec.family] += 1
        origins[task.origin] += 1
        payload = {
            "task_id": task.spec.task_id,
            "manifest": task.manifest_row(),
            "pass_rate": float(row.get("pass_rate") or 0.0),
            "official_pass_rate": float(row.get("official_pass_rate", row.get("pass_rate") or 0.0)),
            "task_success": int(row.get("task_success") or 0),
            "valid_rollouts": int(row.get("valid_rollouts") or 0),
            "errors": int(row.get("errors") or 0),
            "training_overlap": overlap,
            "spec": task.spec.to_dict(),
        }
        lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    body = ("\n".join(lines) + "\n").encode("utf-8")
    manifest: dict[str, Any] = {
        "version": "final-test-v3",
        "created_at": created_at,
        "sources": list(sources),
        "middle_state": {
            "min_rate": 0.30,
            "max_rate": 0.70,
            "rollouts_per_task": 8,
        },
        "task_count": len(selected),
        "clean_heldout_tasks": clean,
        "training_overlap_tasks": len(selected) - clean,
        "by_family": dict(sorted(families.items())),
        "by_origin": dict(sorted(origins.items())),
        "tasks_sha256": _sha256_bytes(body),
        "missing_task_ids": missing,
        "claim_boundary": (
            "training_overlap_tasks were part of train-v5 and cannot support a "
            "generalization claim; only clean_heldout_tasks do, and only if the "
            "evaluation model was not trained on them."
        ),
    }
    return body, manifest, missing


def rows_from_rejudge(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in report["per_task"]:
        if not row["complete"]:
            continue
        lenient = float(row["lenient_rate"])
        if not 0.30 <= lenient <= 0.70:
            continue
        rows.append(
            {
                "task_id": row["task_id"],
                "pass_rate": lenient,
                "official_pass_rate": row["official_rate"],
                "task_success": int(row["official_success"]) + int(row["rescued"]),
                "valid_rollouts": row["valid_rollouts"],
                "errors": row["errors"],
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="freeze final-test-v3")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--middle", type=Path)
    source.add_argument("--rejudge", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("private/sft_data_v5/train.jsonl"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pool = build_heldout_pool(args.train_file)
    sources: list[str] = [str(args.train_file)]
    selection = "official_middle"
    if args.rejudge is not None:
        report = json.loads(args.rejudge.read_text(encoding="utf-8"))
        rows = rows_from_rejudge(report)
        sources.append(str(args.rejudge))
        selection = "lenient_rejudge"
    else:
        rows = [
            json.loads(line)
            for line in args.middle.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        sources.append(str(args.middle))
    created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    body, manifest, missing = freeze(
        pool,
        rows,
        created_at=created_at,
        sources=sources,
    )
    manifest["selection"] = selection
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = args.out_dir / "tasks.jsonl"
    tasks_path.write_bytes(body)
    manifest["tasks_file"] = str(tasks_path)
    manifest_path = args.out_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
