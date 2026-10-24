"""Baseline vs SFT pilot runner for the stage-1.5 statistical task set."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import signal
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from scripts.unified_eval import run_once
from training.kaggle.contracts import read_jsonl
from training.kaggle.transformers_provider import TransformersOfflineProvider
from training.stats_heldout import PILOT_TASK_IDS, STATS_TASK_SPECS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="stage-1.5 baseline/SFT pilot")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(PILOT_TASK_IDS))
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--load-timeout", type=float, default=240.0)
    parser.add_argument("--rollout-timeout", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


@contextmanager
def deadline(seconds: float, error_kind: str):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise(_signum: int, _frame: Any) -> None:
        raise TimeoutError(error_kind)

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def load_provider(args: argparse.Namespace, adapter_path: str | None) -> Any:
    with deadline(args.load_timeout, "model_load_timeout"):
        return TransformersOfflineProvider(
            args.model_path,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            max_tokens=args.max_tokens,
            adapter_path=adapter_path,
            quantization="nf4",
        )


def _completed(path: Path) -> set[tuple[str, int]]:
    return {
        (str(row["task_id"]), int(row["rollout_index"]))
        for row in read_jsonl(path)
        if row.get("task_id") and row.get("rollout_index") is not None and not row.get("error_kind")
    }


async def run_condition(
    condition: str,
    *,
    specs: Sequence[Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    adapter = args.adapter_path if condition == "sft" else None
    started = time.perf_counter()
    provider: Any = None
    try:
        provider = load_provider(args, adapter)
    except Exception as exc:
        return {
            "condition": condition,
            "status": "error",
            "error_kind": "model_load_timeout"
            if "model_load_timeout" in str(exc)
            else type(exc).__name__,
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "adapter_path": adapter,
            "records": 0,
        }
    assert provider is not None
    active_provider = provider

    out_dir = args.out_dir / condition
    trace_dir = args.trace_dir / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / "runs.jsonl"
    all_jobs = [(spec, rollout) for spec in specs for rollout in range(args.rollouts)]
    if args.limit:
        all_jobs = all_jobs[: args.limit]
    expected_by_task = Counter(spec.task_id for spec, _ in all_jobs)
    jobs = all_jobs
    if args.resume:
        done = _completed(run_path)
        jobs = [job for job in jobs if (job[0].task_id, job[1]) not in done]

    semaphore = asyncio.Semaphore(1)

    async def one(spec: Any, rollout: int) -> dict[str, Any]:
        async with semaphore:
            try:
                return await asyncio.wait_for(
                    run_once(
                        spec,
                        provider=active_provider,
                        rollout_index=rollout,
                        trace_dir=trace_dir,
                        sampling_config=active_provider.sampling_config,
                        system_variant="s0",
                    ),
                    timeout=args.rollout_timeout,
                )
            except TimeoutError:
                return {
                    "task_id": spec.task_id,
                    "split": spec.split,
                    "difficulty": spec.difficulty,
                    "context_tier": spec.context_tier,
                    "rollout_index": rollout,
                    "fixture_seed": spec.fixture_seed,
                    "error_kind": "inference_timeout",
                    "error": f"rollout exceeded {args.rollout_timeout}s",
                    "checks": {},
                    "task_success": False,
                    "within_steps": False,
                    "steps": 0,
                }

    rows: list[dict[str, Any]] = []
    with run_path.open("a", encoding="utf-8", newline="\n") as handle:
        for future in asyncio.as_completed([one(spec, rollout) for spec, rollout in jobs]):
            row = await future
            row["condition"] = condition
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
            print(
                f"[{condition} {len(rows)}/{len(jobs)}] {row['task_id']}"
                f"#{row['rollout_index']:02d} success={row.get('task_success')} "
                f"steps={row.get('steps')} error={row.get('error_kind', '')}",
                flush=True,
            )
    await provider.aclose()
    del provider
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    all_rows = read_jsonl(run_path)
    summary = summarize_condition(condition, all_rows, specs, expected_by_task)
    summary["duration_s"] = round(time.perf_counter() - started, 1)
    summary["adapter_path"] = adapter
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def summarize_condition(
    condition: str,
    rows: Sequence[Mapping[str, Any]],
    specs: Sequence[Any] = STATS_TASK_SPECS,
    expected_rollouts_by_task: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row.get("task_id") or ""), []).append(row)
    tasks: list[dict[str, Any]] = []
    for spec in specs:
        task_rows = by_task.get(spec.task_id, [])
        valid = [row for row in task_rows if not row.get("error_kind")]
        successes = sum(1 for row in valid if row.get("task_success"))
        rate = successes / len(valid) if valid else 0.0
        binary = [1.0 if row.get("task_success") else 0.0 for row in valid]
        sample_variance = (
            sum((value - rate) ** 2 for value in binary) / (len(binary) - 1)
            if len(binary) > 1
            else 0.0
        )
        expected = (
            int(expected_rollouts_by_task.get(spec.task_id, 0))
            if expected_rollouts_by_task is not None
            else len(task_rows)
        )
        tasks.append(
            {
                "task_id": spec.task_id,
                "family": spec.family,
                "difficulty": spec.difficulty,
                "expected_rollouts": expected,
                "rollouts": len(task_rows),
                "valid_rollouts": len(valid),
                "errors": len(task_rows) - len(valid),
                "task_success": successes,
                "pass_rate": round(rate, 4),
                "variance": round(sample_variance, 6),
                "bernoulli_variance": round(rate * (1.0 - rate), 6),
                "variance_kind": "sample_variance_of_binary_outcomes",
                "within_steps_failures": sum(
                    1 for row in valid if not row.get("within_steps", True)
                ),
                "mean_steps": round(sum(int(row.get("steps", 0)) for row in valid) / len(valid), 2)
                if valid
                else 0.0,
            }
        )
    rollouts = sum(row["rollouts"] for row in tasks)
    expected_rollouts = sum(row["expected_rollouts"] for row in tasks)
    successes = sum(row["task_success"] for row in tasks)
    errors = sum(row["errors"] for row in tasks)
    complete = rollouts == expected_rollouts and all(
        row["valid_rollouts"] == row["expected_rollouts"] for row in tasks
    )
    return {
        "condition": condition,
        "status": "completed" if expected_rollouts and complete and errors == 0 else "degraded",
        "expected_rollouts": expected_rollouts,
        "rollouts": rollouts,
        "errors": errors,
        "task_success": successes,
        "pass_rate": round(successes / rollouts, 4) if rollouts else 0.0,
        "tasks": tasks,
    }


async def main_async(args: argparse.Namespace) -> int:
    wanted = {item.strip() for item in args.tasks.split(",") if item.strip()}
    specs = [spec for spec in STATS_TASK_SPECS if spec.task_id in wanted]
    if not specs:
        raise ValueError("no matching stats tasks")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)

    if args.preflight_only:
        result: dict[str, Any] = {"tasks": [spec.task_id for spec in specs]}
        for condition in ("baseline", "sft"):
            adapter = args.adapter_path if condition == "sft" else None
            try:
                provider = load_provider(args, adapter)
                await provider.aclose()
                result[condition] = {"status": "ok", "adapter_path": adapter}
            except Exception as exc:
                result[condition] = {
                    "status": "error",
                    "error_kind": "model_load_timeout"
                    if "model_load_timeout" in str(exc)
                    else type(exc).__name__,
                    "error": str(exc)[:1000],
                }
            gc.collect()
        (args.out_dir / "preflight.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if all(result[name]["status"] == "ok" for name in ("baseline", "sft")) else 2

    summaries = {}
    for condition in ("baseline", "sft"):
        summaries[condition] = await run_condition(condition, specs=specs, args=args)
    output = {"tasks": [spec.task_id for spec in specs], "conditions": summaries}
    (args.out_dir / "pilot_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(item.get("status") == "completed" for item in summaries.values()) else 1


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
