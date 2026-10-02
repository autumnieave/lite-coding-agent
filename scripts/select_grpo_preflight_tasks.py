"""Select a deterministic stratified subset of the frozen GRPO middle-state pool.

The full pool has 45 tasks (BASE 13, HARD 9, TOOL 23). The preflight uses 20
tasks, allocated proportionally by category and ordered by a stable SHA-256 key
so the selection is reproducible without hand-picking easy or hard cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CATEGORY_ORDER = ("BASE", "HARD", "TOOL")
DEFAULT_PROMPTS = Path("private/grpo_middle_pool/prompts.jsonl")
DEFAULT_OUT = Path("private/grpo_middle_pool/preflight_20.jsonl")
DEFAULT_MANIFEST = Path("private/grpo_middle_pool/preflight_20_manifest.json")


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


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _task_id(row: Mapping[str, Any]) -> str:
    task_id = str(row.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("prompt row has no task_id")
    return task_id


def _category(row: Mapping[str, Any]) -> str:
    category = str(row.get("category") or "UNKNOWN").strip()
    if not category:
        raise ValueError(f"{_task_id(row)} has no category")
    return category


def proportional_allocation(counts: Mapping[str, int], total: int) -> dict[str, int]:
    population = sum(counts.values())
    if total < 1 or total > population:
        raise ValueError(f"count must be between 1 and {population}, got {total}")
    categories = sorted(
        (category for category, count in counts.items() if count > 0),
        key=lambda category: (
            CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else len(CATEGORY_ORDER),
            category,
        ),
    )
    exact = {category: total * counts[category] / population for category in categories}
    allocation = {category: math.floor(exact[category]) for category in categories}
    remainder = total - sum(allocation.values())
    for category in sorted(
        categories,
        key=lambda item: (-(exact[item] - allocation[item]), categories.index(item)),
    ):
        if remainder <= 0:
            break
        allocation[category] += 1
        remainder -= 1
    if remainder != 0:
        raise ValueError(f"allocation remainder left: {remainder}")
    return allocation


def select_preflight_tasks(
    prompts: Sequence[Mapping[str, Any]],
    *,
    count: int = 20,
    seed: int = 20261019,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in prompts:
        task_id = _task_id(row)
        if task_id in seen:
            continue
        seen.add(task_id)
        by_category[_category(row)].append(dict(row))
    counts = {category: len(rows) for category, rows in by_category.items()}
    allocation = proportional_allocation(counts, count)
    selected: list[dict[str, Any]] = []
    for category in sorted(
        by_category,
        key=lambda item: (
            CATEGORY_ORDER.index(item) if item in CATEGORY_ORDER else len(CATEGORY_ORDER),
            item,
        ),
    ):
        rows = sorted(
            by_category[category],
            key=lambda row: hashlib.sha256(f"{seed}:{_task_id(row)}".encode()).hexdigest(),
        )
        selected.extend(rows[: allocation[category]])
    if len(selected) != count:
        raise ValueError(f"selected {len(selected)} tasks, expected {count}")
    return selected, allocation


def build_manifest(
    *,
    prompts_path: Path,
    out_path: Path,
    selected: Sequence[Mapping[str, Any]],
    allocation: Mapping[str, int],
    count: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "version": "grpo-preflight-selection-v1",
        "source_prompts": {
            "path": str(prompts_path),
            "sha256": sha256(prompts_path),
        },
        "selection": {
            "count": count,
            "seed": seed,
            "allocation": dict(allocation),
            "rule": "proportional by category, stable SHA-256(seed:task_id) order",
        },
        "category_counts": dict(
            sorted(Counter(str(row.get("category")) for row in selected).items())
        ),
        "pass_rate_counts": dict(
            sorted(Counter(str(row.get("pass_rate")) for row in selected).items())
        ),
        "task_ids": [_task_id(row) for row in selected],
        "output": {
            "path": str(out_path),
            "sha256": sha256(out_path),
            "rows": len(selected),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="select the GRPO preflight task subset")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261019)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected, allocation = select_preflight_tasks(
        load_jsonl(args.prompts),
        count=args.count,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, selected)
    manifest = build_manifest(
        prompts_path=args.prompts,
        out_path=args.out,
        selected=selected,
        allocation=allocation,
        count=args.count,
        seed=args.seed,
    )
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
