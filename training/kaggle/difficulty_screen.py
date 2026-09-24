"""Screen the frozen task pool for baseline middle-state difficulty.

Pre-registration for this runner:

* every unique task is sampled exactly eight times;
* sampling uses temperature=0.7, top_p=0.9, seed=base_seed+rollout_index;
* a task is middle-state when its valid sample pass rate is in [0.30, 0.70];
* tasks with any rollout error or fewer than eight valid samples are reported
  but excluded from the middle-state count.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from training.heldout_pool import HeldoutPool, PoolTask, build_heldout_pool
from training.kaggle.stats_pilot import (
    gpu_memory_snapshot,
    load_provider,
    run_condition,
)

MIDDLE_MIN_RATE = 0.30
MIDDLE_MAX_RATE = 0.70
PREREGISTERED_ROLLOUTS = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="held-out difficulty screen")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("private/sft_data_v5/train.jsonl"),
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=PREREGISTERED_ROLLOUTS)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--task-ids-file", type=Path, default=None)
    parser.add_argument("--origins", default="")
    parser.add_argument("--categories", default="")
    parser.add_argument("--staging", action="store_true")
    parser.add_argument("--staging-tasks", type=int, default=5)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--load-timeout", type=float, default=240.0)
    parser.add_argument("--rollout-timeout", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--device-map", default=None)
    return parser


def _comma_set(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _read_task_ids(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def staging_tasks(pool: HeldoutPool, limit: int) -> list[PoolTask]:
    """Pick the shortest task from each origin before filling remaining slots."""

    groups: dict[str, list[PoolTask]] = {}
    for task in pool.tasks:
        groups.setdefault(task.origin, []).append(task)
    for group in groups.values():
        group.sort(key=lambda task: (task.spec.steps_limit, task.spec.task_id))
    selected: list[PoolTask] = []
    while len(selected) < limit:
        added = False
        for origin in sorted(groups):
            if len(selected) >= limit:
                break
            group = groups[origin]
            if group:
                selected.append(group.pop(0))
                added = True
        if not added:
            break
    return selected


def select_tasks(
    pool: HeldoutPool,
    *,
    shard_index: int,
    shard_count: int,
    task_limit: int = 0,
    task_ids: set[str] | None = None,
    origins: set[str] | None = None,
    categories: set[str] | None = None,
    staging: bool = False,
    staging_count: int = 5,
) -> list[PoolTask]:
    if shard_count < 1:
        raise ValueError("shard-count must be at least one")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if staging:
        return staging_tasks(pool, staging_count)

    selected: list[PoolTask] = []
    for index, task in enumerate(pool.tasks):
        if index % shard_count != shard_index:
            continue
        if task_ids is not None and task.spec.task_id not in task_ids:
            continue
        if origins and task.origin not in origins:
            continue
        if categories and task.category not in categories:
            continue
        selected.append(task)
    if task_limit > 0:
        selected = selected[:task_limit]
    return selected


def task_list_sha256(task_ids: Sequence[str]) -> str:
    payload = "\n".join(task_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def add_middle_state_summary(
    summary: dict[str, Any],
    *,
    expected_rollouts: int,
) -> dict[str, Any]:
    middle: list[str] = []
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    for task in summary.get("tasks", []):
        task_id = str(task["task_id"])
        valid = int(task.get("valid_rollouts") or 0)
        errors = int(task.get("errors") or 0)
        rate = float(task.get("pass_rate") or 0.0)
        if errors or valid != expected_rollouts:
            excluded[task_id] = f"valid={valid},errors={errors},expected={expected_rollouts}"
            continue
        eligible.append(task_id)
        if MIDDLE_MIN_RATE <= rate <= MIDDLE_MAX_RATE:
            middle.append(task_id)
    summary["middle_state"] = {
        "min_rate": MIDDLE_MIN_RATE,
        "max_rate": MIDDLE_MAX_RATE,
        "expected_rollouts": expected_rollouts,
        "eligible_tasks": len(eligible),
        "count": len(middle),
        "tasks": middle,
        "excluded_incomplete": excluded,
    }
    summary["pass_rate_distribution"] = dict(
        sorted(Counter(str(task.get("pass_rate")) for task in summary.get("tasks", [])).items())
    )
    return summary


def exit_code_for_status(status: str) -> int:
    """A degraded run still carries usable rollouts; only load errors are fatal."""

    return 1 if status == "error" else 0


def _plan_payload(
    args: argparse.Namespace,
    pool: HeldoutPool,
    selected: Sequence[PoolTask],
    rollouts: int,
) -> dict[str, Any]:
    task_ids = [task.spec.task_id for task in selected]
    return {
        "status": "planned",
        "mode": "staging" if args.staging else "screen",
        "pool": pool.report,
        "rollouts": rollouts,
        "preregistered_rollouts": PREREGISTERED_ROLLOUTS,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "base_seed": args.seed,
            "seed_policy": "base_seed_plus_rollout_index",
        },
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "tasks": task_ids,
        "task_count": len(task_ids),
        "task_list_sha256": task_list_sha256(task_ids),
        "by_origin": dict(sorted(Counter(task.origin for task in selected).items())),
        "by_category": dict(sorted(Counter(task.category for task in selected).items())),
    }


async def main_async(args: argparse.Namespace) -> int:
    pool = build_heldout_pool(args.train_file)
    task_ids = _read_task_ids(args.task_ids_file) if args.task_ids_file else None
    selected = select_tasks(
        pool,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        task_limit=args.task_limit,
        task_ids=task_ids,
        origins=_comma_set(args.origins) or None,
        categories=_comma_set(args.categories) or None,
        staging=args.staging,
        staging_count=args.staging_tasks,
    )
    if not selected:
        raise ValueError("task selection is empty")
    if args.staging:
        rollouts = 1
    else:
        rollouts = args.rollouts
        if rollouts != PREREGISTERED_ROLLOUTS:
            raise ValueError(
                f"full screen rollouts are pre-registered at {PREREGISTERED_ROLLOUTS}; "
                f"got {rollouts}"
            )
    args.rollouts = rollouts
    # stats_pilot.run_condition uses this legacy switch to cap rollout jobs.
    # The difficulty runner shards by task instead, so it must stay disabled.
    args.limit = 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    plan = _plan_payload(args, pool, selected, rollouts)
    plan_path = args.out_dir / "difficulty_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.list_only:
        return 0

    if args.preflight_only:
        started = time.perf_counter()
        try:
            provider = load_provider(args, None)
            await provider.aclose()
            result = {
                "status": "ok",
                "load_seconds": round(time.perf_counter() - started, 1),
                "gpu": gpu_memory_snapshot(),
            }
        except Exception as exc:
            result = {
                "status": "error",
                "error_kind": type(exc).__name__,
                "error": str(exc)[:2000],
            }
        (args.out_dir / "preflight.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if result["status"] == "ok" else 2

    specs = [task.spec for task in selected]
    summary = await run_condition("baseline", specs=specs, args=args)
    summary = add_middle_state_summary(summary, expected_rollouts=rollouts)
    summary["mode"] = "staging" if args.staging else "screen"
    summary["pool"] = pool.report
    summary["task_list_sha256"] = plan["task_list_sha256"]
    output = {"plan": plan, "baseline": summary}
    summary_path = args.out_dir / "difficulty_summary.json"
    summary_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)

    if args.staging:
        ok = (
            summary.get("status") == "completed"
            and int(summary.get("errors") or 0) == 0
            and int(summary.get("task_success") or 0) >= 1
        )
        print(f"STAGING_OK={ok}", flush=True)
        return 0 if ok else 2
    return exit_code_for_status(str(summary.get("status")))


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
