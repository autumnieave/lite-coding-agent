"""Select a deterministic stratified subset for the session-constraint probe.

The 45-task GRPO middle-state pool holds tasks whose measured pass rate is
between 20% and 80%. This selector takes 8 tasks (BASE 3 / HARD 2 / TOOL 3),
ordered by a stable SHA-256 key, so the layered probe is reproducible without
hand-picking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_POOL = Path("private/grpo_middle_pool/prompts.jsonl")
DEFAULT_OUT = Path("private/session_constraint/tasks.json")
ALLOCATION = {"BASE": 3, "HARD": 2, "TOOL": 3}
ALLOCATION_64 = {"BASE": 6, "HARD": 4, "TOOL": 6}
SELECTION_VERSION = "session-constraint-selection-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(payload)
    return rows


def stable_key(task_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_VERSION}:{task_id}".encode()).hexdigest()


def parse_allocation(value: str) -> dict[str, int]:
    """Parse ``BASE=6,HARD=4,TOOL=6`` into a category allocation."""
    allocation: dict[str, int] = {}
    for item in value.split(","):
        name, separator, raw_count = item.partition("=")
        category = name.strip().upper()
        if not separator or not category or not raw_count.strip():
            raise ValueError(f"invalid allocation item: {item!r}")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(f"invalid allocation count: {item!r}") from exc
        if count <= 0:
            raise ValueError(f"allocation count must be positive: {item!r}")
        if category in allocation:
            raise ValueError(f"duplicate allocation category: {category}")
        allocation[category] = count
    if not allocation:
        raise ValueError("allocation must not be empty")
    return allocation


def select_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    allocation: Mapping[str, int] = ALLOCATION,
) -> list[dict[str, Any]]:
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        category = str(row.get("category") or "").strip().upper()
        if not task_id or category not in allocation:
            continue
        by_category[category].append(row)
    selected: list[dict[str, Any]] = []
    for category, wanted in allocation.items():
        candidates = sorted(
            by_category.get(category, []), key=lambda row: stable_key(str(row["task_id"]))
        )
        if len(candidates) < wanted:
            raise ValueError(f"not enough {category} tasks: wanted {wanted}, got {len(candidates)}")
        selected.extend(dict(row) for row in candidates[:wanted])
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="select session-constraint probe tasks")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--allocation",
        default=None,
        help="optional category allocation, e.g. BASE=6,HARD=4,TOOL=6",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_jsonl(args.pool)
    allocation = parse_allocation(args.allocation) if args.allocation else ALLOCATION
    selected = select_rows(rows, allocation=allocation)
    payload = {
        "version": SELECTION_VERSION,
        "source_pool": {"path": str(args.pool), "sha256": sha256(args.pool), "rows": len(rows)},
        "allocation": dict(allocation),
        "rule": "stratified by category, stable SHA-256(version:task_id) order",
        "tasks": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "sha256": sha256(args.out),
                "task_ids": [row["task_id"] for row in selected],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
