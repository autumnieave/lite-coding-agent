"""Summarise rollout-level success/compliance by layer for v14 runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.v10_conflict import load_tasks  # noqa: E402


def wilson(k: int, n: int) -> dict[str, float] | None:
    if n == 0:
        return None
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"low": round(max(0.0, centre - half), 4), "high": round(min(1.0, centre + half), 4)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for path in args.runs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(str(row["task_id"]), int(row.get("rollout_index", -1)))] = row
    rows = [rows[key] for key in sorted(rows)]

    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = tasks[str(row["task_id"])]
        layer = (
            task.category
            if task.category in {"path_escape", "delete_without_confirm", "edit_tests"}
            else "policy"
        )
        by_layer[layer].append(row)

    def block(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        classes = Counter(str((row.get("v10") or {}).get("classification")) for row in items)
        explicit = classes.get("explicit_compliance", 0)
        violation = classes.get("execution_violation", 0)
        safe = sum(1 for row in items if (row.get("checks") or {}).get("execution_compliant"))
        return {
            "rollouts": n,
            "explicit_compliance": explicit,
            "explicit_compliance_rate": round(explicit / n, 4) if n else None,
            "explicit_compliance_ci95": wilson(explicit, n),
            "execution_violation": violation,
            "execution_violation_rate": round(violation / n, 4) if n else None,
            "execution_compliant": safe,
            "execution_compliant_rate": round(safe / n, 4) if n else None,
            "classifications": dict(classes),
        }

    summary = {
        "tasks": len(tasks),
        "rollouts": len(rows),
        "overall": block(rows),
        "by_layer": {layer: block(items) for layer, items in sorted(by_layer.items())},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
