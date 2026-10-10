"""Run three baseline rollouts for every v10 conflict task."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.unified_eval import Fixture, run_once  # noqa: E402
from training.kaggle.contracts import classify_exception  # noqa: E402
from training.kaggle.transformers_provider import TransformersOfflineProvider  # noqa: E402
from training.v10_conflict import (  # noqa: E402
    V10ConflictTask,
    classify_rollout,
    judge,
    load_tasks,
    materialize,
    rollout_seed,
    validate_tasks,
)

TRIGGER_STRICT = {"execution_violation", "task_priority"}
TRIGGER_INCLUSIVE = TRIGGER_STRICT | {"ambiguous_reference"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v10 conflict baseline")
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=Path("private/v10/conflict_tasks/tasks.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("private/v10/baseline"))
    parser.add_argument("--trace-dir", type=Path, default=Path("private/traces/v10_baseline"))
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _completed(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("task_id")
                and row.get("rollout_index") is not None
                and not row.get("error_kind")
            ):
                done.add((str(row["task_id"]), int(row["rollout_index"])))
    return done


def _fixture_builder(tasks: dict[str, V10ConflictTask]) -> Any:
    def build(spec: Any, root: Path) -> Fixture:
        task = tasks[spec.task_id]
        state = materialize(task, root)
        return Fixture(
            task_id=spec.task_id,
            kind="v10_conflict",
            expected_json={"task": task.to_dict(), "state": state},
        )

    return build


def _judge_fn(fixture: Fixture, replies: list[str], calls: list[Any]) -> dict[str, bool]:
    task = V10ConflictTask.from_dict(fixture.expected_json["task"])
    return judge(task, replies, calls)


async def _run_one(
    task: V10ConflictTask,
    *,
    provider: TransformersOfflineProvider,
    rollout_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    # The provider reseeds torch on every call; vary the seed per rollout or
    # temperature>0 still produces identical samples within a task.
    if args.temperature and args.temperature > 0:
        provider.seed = rollout_seed(args.seed, rollout_index)
    fixture_builder = _fixture_builder({task.task_id: task})
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            run_once(
                task.to_spec(),
                provider=provider,
                rollout_index=rollout_index,
                trace_dir=args.trace_dir,
                sampling_config=dict(provider.sampling_config),
                system_variant="s0",
                fixture_builder=fixture_builder,
                judge_fn=lambda spec, fixture, root, replies, calls: _judge_fn(
                    fixture, replies, calls
                ),
            ),
            timeout=300,
        )
    except TimeoutError:
        return {
            "task_id": task.task_id,
            "rollout_index": rollout_index,
            "error_kind": "inference_timeout",
            "error": "rollout exceeded 300s",
            "duration_s": round(time.perf_counter() - started, 1),
        }
    classification = classify_rollout(
        task,
        result.get("step_replies") or [],
        result.get("tool_calls") or [],
    )
    result["v10"] = classification
    result["triggered_strict"] = classification["classification"] in TRIGGER_STRICT
    result["triggered_inclusive"] = classification["classification"] in TRIGGER_INCLUSIVE
    result["duration_s"] = round(time.perf_counter() - started, 1)
    return result


def _summarise(
    tasks: dict[str, V10ConflictTask], rows: list[dict[str, Any]], out_dir: Path
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row.get("task_id") or ""), []).append(row)
    task_rows: list[dict[str, Any]] = []
    for task in tasks.values():
        rows_for_task = by_task.get(task.task_id, [])
        valid = [row for row in rows_for_task if not row.get("error_kind")]
        classes = Counter(
            str((row.get("v10") or {}).get("classification") or "error") for row in valid
        )
        strict = any(row.get("triggered_strict") for row in valid)
        inclusive = any(row.get("triggered_inclusive") for row in valid)
        task_rows.append(
            {
                "task_id": task.task_id,
                "source": task.source,
                "category": task.category,
                "severity": task.severity,
                "coverage_only": task.coverage_only,
                "expected_rollouts": task.baseline_required_rollouts,
                "rollouts": len(rows_for_task),
                "valid_rollouts": len(valid),
                "errors": len(rows_for_task) - len(valid),
                "classifications": dict(classes),
                "triggered_strict": strict,
                "triggered_inclusive": inclusive,
                "retained": task.coverage_only or strict or inclusive,
            }
        )
    summary = {
        "version": "v10-conflict-baseline-1",
        "tasks": len(tasks),
        "expected_rollouts": sum(task.baseline_required_rollouts for task in tasks.values()),
        "rollouts": len(rows),
        "valid_rollouts": sum(1 for row in rows if not row.get("error_kind")),
        "errors": sum(1 for row in rows if row.get("error_kind")),
        "triggered_tasks_strict": sum(1 for row in task_rows if row["triggered_strict"]),
        "triggered_tasks_inclusive": sum(1 for row in task_rows if row["triggered_inclusive"]),
        "retained_tasks": sum(1 for row in task_rows if row["retained"]),
        "by_source": dict(Counter(task.source for task in tasks.values())),
        "by_category": dict(Counter(task.category for task in tasks.values())),
        "by_classification": dict(
            Counter(
                str((row.get("v10") or {}).get("classification") or "error")
                for row in rows
                if not row.get("error_kind")
            )
        ),
        "task_rows": task_rows,
        "status": "completed"
        if len(rows) == sum(task.baseline_required_rollouts for task in tasks.values())
        and not any(row.get("error_kind") for row in rows)
        else "degraded",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


async def main_async(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks_file)
    validate_tasks(tasks)
    wanted = list(tasks.values())
    if args.limit:
        wanted = wanted[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "v10-conflict-baseline-1",
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "adapter_sha256": sha256(Path(args.adapter_path) / "adapter_model.safetensors")
        if args.adapter_path and (Path(args.adapter_path) / "adapter_model.safetensors").exists()
        else None,
        "tasks_file": str(args.tasks_file),
        "tasks_sha256": sha256(args.tasks_file),
        "rollouts_per_task": args.rollouts,
        "temperature": args.temperature,
        "seed": args.seed,
        "environment": {
            name: package_version(name)
            for name in ("torch", "transformers", "peft", "bitsandbytes", "pyarrow")
        },
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    provider: TransformersOfflineProvider | None = None
    try:
        provider = TransformersOfflineProvider(
            args.model_path,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            max_tokens=args.max_tokens,
            adapter_path=args.adapter_path,
            quantization="nf4",
            device_map=args.device_map,
        )
        if args.preflight_only:
            (args.out_dir / "preflight.json").write_text(
                json.dumps({"status": "ok", "manifest": manifest}, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            return 0
        run_path = args.out_dir / "runs.jsonl"
        done = _completed(run_path) if args.resume else set()
        jobs = [
            (task, rollout)
            for task in wanted
            for rollout in range(args.rollouts)
            if (task.task_id, rollout) not in done
        ]
        rows: list[dict[str, Any]] = []
        with run_path.open("a", encoding="utf-8", newline="\n") as handle:
            for index, (task, rollout) in enumerate(jobs, start=1):
                row = await _run_one(task, provider=provider, rollout_index=rollout, args=args)
                row["condition"] = "baseline"
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                rows.append(row)
                print(
                    f"[{index}/{len(jobs)}] {task.task_id}#{rollout:02d} "
                    f"class={(row.get('v10') or {}).get('classification')} "
                    f"error={row.get('error_kind', '')}",
                    flush=True,
                )
        existing = []
        if run_path.exists():
            with run_path.open("r", encoding="utf-8") as handle:
                existing = [json.loads(line) for line in handle if line.strip()]
        summary = _summarise({task.task_id: task for task in wanted}, existing, args.out_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0 if summary["status"] == "completed" else 1
    except Exception as exc:  # noqa: BLE001 - surface load failures as manifests
        error = {
            "status": "error",
            "error_kind": classify_exception(exc),
            "error": f"{type(exc).__name__}: {exc}"[:2000],
        }
        (args.out_dir / "preflight.json").write_text(
            json.dumps(error, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(error, ensure_ascii=False, indent=2), flush=True)
        return 2
    finally:
        if provider is not None:
            await provider.aclose()


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
