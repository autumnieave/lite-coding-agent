"""Freeze the middle-state task pool used by the GRPO reward-variance preflight.

The v1 pool sampled train-v5 directly, so it could not answer whether tasks with
intermediate baseline pass rates have group variance. This builder selects the
frozen 30-70% middle-state tasks, joins their prompts from the task-spec file,
and writes a manifest with source and output hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

POOL_VERSION = "grpo-middle-pool-v2"
DEFAULT_MIDDLE_TASKS = Path("private/kaggle_difficulty/final/middle_tasks.jsonl")
DEFAULT_TASK_SPECS = Path("private/final_test_v3/tasks.jsonl")
DEFAULT_OUT_DIR = Path("private/grpo_middle_pool")


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
        spec = row.get("spec")
        if isinstance(spec, Mapping):
            task_id = str(spec.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("row has no task_id")
    return task_id


def select_middle_tasks(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_pass_rate: float = 0.3,
    max_pass_rate: float = 0.7,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    if not 0.0 <= min_pass_rate <= max_pass_rate <= 1.0:
        raise ValueError("pass-rate bounds must be within [0, 1] and ordered")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        task_id = _task_id(row)
        if task_id in seen:
            continue
        pass_rate = float(row.get("pass_rate", -1.0))
        expected = int(row.get("expected_rollouts", row.get("rollouts", 0)) or 0)
        valid = int(row.get("valid_rollouts", 0) or 0)
        errors = int(row.get("errors", 0) or 0)
        if require_complete and (expected < 1 or valid != expected or errors != 0):
            continue
        if not (min_pass_rate <= pass_rate <= max_pass_rate):
            continue
        seen.add(task_id)
        selected.append(dict(row))
    return selected


def index_task_specs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        spec = row.get("spec")
        payload = spec if isinstance(spec, Mapping) else row
        task_id = str(row.get("task_id") or payload.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("task-spec row has no task_id")
        by_id[task_id] = payload
    return by_id


def build_prompt_pool(
    middle_rows: Sequence[Mapping[str, Any]],
    spec_rows: Sequence[Mapping[str, Any]],
    *,
    min_pass_rate: float = 0.3,
    max_pass_rate: float = 0.7,
) -> list[dict[str, Any]]:
    selected = select_middle_tasks(
        middle_rows,
        min_pass_rate=min_pass_rate,
        max_pass_rate=max_pass_rate,
    )
    specs = index_task_specs(spec_rows)
    missing = sorted({_task_id(row) for row in selected} - set(specs))
    if missing:
        raise ValueError(f"middle-state task ids missing from task specs: {missing}")
    prompts: list[dict[str, Any]] = []
    for row in selected:
        task_id = _task_id(row)
        spec = specs[task_id]
        user_turns = spec.get("user_turns")
        if not isinstance(user_turns, list) or not user_turns:
            raise ValueError(f"{task_id} has no non-empty user_turns")
        prompts.append(
            {
                "task_id": task_id,
                "category": row.get("category") or spec.get("category"),
                "family": row.get("family") or spec.get("family"),
                "difficulty": row.get("difficulty") or spec.get("difficulty"),
                "pass_rate": row.get("pass_rate"),
                "variance": row.get("variance"),
                "valid_rollouts": row.get("valid_rollouts"),
                "training_overlap": row.get("training_overlap"),
                "user_turns": [str(item) for item in user_turns],
                "constraints": spec.get("constraints", []),
                "allowed_tools": spec.get("allowed_tools", []),
                "required_tools": spec.get("required_tools", []),
                "steps_limit": spec.get("steps_limit"),
            }
        )
    return prompts


def build_manifest(
    *,
    middle_path: Path,
    specs_path: Path,
    out_dir: Path,
    prompts: Sequence[Mapping[str, Any]],
    min_pass_rate: float,
    max_pass_rate: float,
) -> dict[str, Any]:
    task_ids_path = out_dir / "task_ids.jsonl"
    prompts_path = out_dir / "prompts.jsonl"
    return {
        "version": POOL_VERSION,
        "source_middle_tasks": {
            "path": str(middle_path),
            "sha256": sha256(middle_path),
        },
        "source_task_specs": {
            "path": str(specs_path),
            "sha256": sha256(specs_path),
        },
        "selection": {
            "min_pass_rate": min_pass_rate,
            "max_pass_rate": max_pass_rate,
            "require_complete_rollouts": True,
            "rule": "pass_rate in [min, max], valid_rollouts == expected_rollouts, errors == 0",
        },
        "tasks": len(prompts),
        "category_counts": dict(
            sorted(Counter(str(row.get("category") or "UNKNOWN") for row in prompts).items())
        ),
        "family_counts": dict(
            sorted(Counter(str(row.get("family") or "UNKNOWN") for row in prompts).items())
        ),
        "pass_rate_counts": dict(
            sorted(Counter(str(row.get("pass_rate")) for row in prompts).items())
        ),
        "training_overlap_counts": dict(
            sorted(Counter(str(row.get("training_overlap")) for row in prompts).items())
        ),
        "task_ids": [str(row["task_id"]) for row in prompts],
        "outputs": {
            "task_ids.jsonl": {
                "sha256": sha256(task_ids_path),
                "rows": len(prompts),
            },
            "prompts.jsonl": {
                "sha256": sha256(prompts_path),
                "rows": len(prompts),
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="freeze the GRPO middle-state prompt pool")
    parser.add_argument("--middle-tasks", type=Path, default=DEFAULT_MIDDLE_TASKS)
    parser.add_argument("--task-specs", type=Path, default=DEFAULT_TASK_SPECS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-pass-rate", type=float, default=0.3)
    parser.add_argument("--max-pass-rate", type=float, default=0.7)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    middle_rows = load_jsonl(args.middle_tasks)
    spec_rows = load_jsonl(args.task_specs)
    prompts = build_prompt_pool(
        middle_rows,
        spec_rows,
        min_pass_rate=args.min_pass_rate,
        max_pass_rate=args.max_pass_rate,
    )
    if not prompts:
        raise SystemExit("no middle-state tasks matched the pass-rate selection")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = [
        {
            "task_id": row["task_id"],
            "category": row.get("category"),
            "family": row.get("family"),
            "pass_rate": row.get("pass_rate"),
            "variance": row.get("variance"),
            "valid_rollouts": row.get("valid_rollouts"),
        }
        for row in prompts
    ]
    write_jsonl(args.out_dir / "task_ids.jsonl", task_ids)
    write_jsonl(args.out_dir / "prompts.jsonl", prompts)
    manifest = build_manifest(
        middle_path=args.middle_tasks,
        specs_path=args.task_specs,
        out_dir=args.out_dir,
        prompts=prompts,
        min_pass_rate=args.min_pass_rate,
        max_pass_rate=args.max_pass_rate,
    )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
