"""Full-loop held-out comparison for a trained LoRA adapter.

The probe-only evaluator cannot measure tool-call success. This module reruns the
same held-out val tasks with the real AgentLoop twice -- baseline and adapter --
and aggregates by BASE/HARD difficulty. It shells out so the evaluation process
uses exactly the same harness as the other experiments.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

STABLE_HITS = 6
STABLE_MIN_TOTAL = 10


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def load_val_tasks(val_file: Path) -> list[str]:
    rows = load_jsonl(val_file)
    tasks = {str((row.get("metadata") or {}).get("task_id") or "") for row in rows}
    tasks.discard("")
    if not tasks:
        raise ValueError(f"no task ids in val file: {val_file}")
    return sorted(tasks)


def build_eval_command(
    *,
    repo: Path,
    model: str,
    out_dir: Path,
    trace_dir: Path,
    tasks: Sequence[str],
    rollouts: int,
    train_version: str = "train-v2",
    adapter_path: str = "",
    max_model_len: int = 8192,
    concurrency: int = 1,
    provider: str = "offline",
    quantization: str = "nf4",
) -> list[str]:
    command = [
        sys.executable,
        str(repo / "scripts/difficulty_tuning.py"),
        "--spec-set",
        "train",
        "--train-version",
        train_version,
        "--tasks",
        ",".join(tasks),
        "--rollouts",
        str(rollouts),
        "--concurrency",
        str(concurrency),
        "--sampling-temperature",
        "0",
        "--sampling-seed",
        "20261019",
        "--provider",
        provider,
        "--offline-model",
        model,
        "--max-model-len",
        str(max_model_len),
        "--out-dir",
        str(out_dir),
        "--trace-dir",
        str(trace_dir),
    ]
    if adapter_path:
        command += ["--adapter-path", adapter_path]
    if provider == "transformers":
        command += ["--quantization", quantization]
    elif quantization not in {"", "nf4"}:
        command += ["--offline-quantization", quantization]
    return command


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _stability(hits: int, total: int) -> str:
    if total < STABLE_MIN_TOTAL:
        return "insufficient"
    if hits == 0:
        return "not_observed"
    return "stable" if hits >= STABLE_HITS else "intermittent"


def _trace_tool_stats(trace_dir: Path, task_id: str, rollout: int) -> tuple[int, int]:
    matches = sorted(trace_dir.rglob(f"{task_id}_rollout{rollout:02d}.jsonl"))
    if not matches:
        return 0, 0
    ok = total = 0
    for event in load_jsonl(matches[0]):
        for result in event.get("tool_results") or []:
            total += 1
            if (
                _bool(result.get("ok"))
                and not _bool(result.get("is_error"))
                and not _bool(result.get("protocol_error"))
                and not result.get("error_kind")
            ):
                ok += 1
    return ok, total


def summarize_condition(
    rows: Sequence[Mapping[str, Any]],
    specs: Sequence[Any],
    trace_dir: Path,
) -> dict[str, Any]:
    spec_by_task = {spec.task_id: spec for spec in specs}
    task_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        task_rows.setdefault(str(row.get("task_id") or ""), []).append(row)

    tasks: list[dict[str, Any]] = []
    stable_checks = 0
    valid_stable_checks = 0
    intermittent_checks = 0
    within_steps_failures = 0
    for task_id, spec in sorted(spec_by_task.items()):
        if task_id not in task_rows:
            continue
        task_rollouts = task_rows[task_id]
        valid = [row for row in task_rollouts if not row.get("error_kind") and "error" not in row]
        total = len(valid)
        valid_steps = [row for row in valid if _bool(row.get("within_steps", True))]
        step_failures = total - len(valid_steps)
        within_steps_failures += step_failures
        check_rows = []
        for check in spec.checks:
            hits = sum(1 for row in valid if not _bool((row.get("checks") or {}).get(check.name)))
            status = _stability(hits, total)
            valid_hits = sum(
                1 for row in valid_steps if not _bool((row.get("checks") or {}).get(check.name))
            )
            valid_status = _stability(valid_hits, len(valid_steps))
            if status == "stable":
                stable_checks += 1
            elif status == "intermittent":
                intermittent_checks += 1
            if valid_status == "stable":
                valid_stable_checks += 1
            check_rows.append(
                {
                    "check": check.name,
                    "hits": hits,
                    "total": total,
                    "stability": status,
                    "valid_hits": valid_hits,
                    "valid_total": len(valid_steps),
                    "valid_stability": valid_status,
                }
            )
        tool_ok = tool_total = 0
        for row in valid:
            ok, count = _trace_tool_stats(trace_dir, task_id, int(row.get("rollout_index") or 0))
            tool_ok += ok
            tool_total += count
        tasks.append(
            {
                "task_id": task_id,
                "difficulty": spec.difficulty,
                "kind": "hard" if spec.difficulty == "strong" else "base",
                "rollouts": total,
                "valid_rollouts": len(valid_steps),
                "task_success": sum(1 for row in valid if _bool(row.get("task_success"))),
                "within_steps_failures": step_failures,
                "tool_results_ok": tool_ok,
                "tool_results_total": tool_total,
                "stable_checks": sum(1 for row in check_rows if row["stability"] == "stable"),
                "valid_stable_checks": sum(
                    1 for row in check_rows if row["valid_stability"] == "stable"
                ),
                "checks": check_rows,
            }
        )

    successes = sum(row["task_success"] for row in tasks)
    rollouts = sum(row["rollouts"] for row in tasks)
    return {
        "records": len(rows),
        "errors": sum(1 for row in rows if row.get("error_kind") or "error" in row),
        "rollouts": rollouts,
        "task_success": {
            "hits": successes,
            "total": rollouts,
            "rate": round(successes / rollouts, 4) if rollouts else None,
        },
        "tool_results": {
            "ok": sum(row["tool_results_ok"] for row in tasks),
            "total": sum(row["tool_results_total"] for row in tasks),
            "rate": (
                round(
                    sum(row["tool_results_ok"] for row in tasks)
                    / max(1, sum(row["tool_results_total"] for row in tasks)),
                    4,
                )
                if sum(row["tool_results_total"] for row in tasks)
                else None
            ),
        },
        "stable_checks": stable_checks,
        "valid_stable_checks": valid_stable_checks,
        "intermittent_checks": intermittent_checks,
        "within_steps_failures": within_steps_failures,
        "tasks": tasks,
    }


def _by_kind(summary: Mapping[str, Any], kind: str) -> dict[str, Any]:
    tasks = [task for task in summary.get("tasks", []) if task.get("kind") == kind]
    rollouts = sum(int(task.get("rollouts") or 0) for task in tasks)
    valid_rollouts = sum(int(task.get("valid_rollouts") or 0) for task in tasks)
    successes = sum(int(task.get("task_success") or 0) for task in tasks)
    tool_ok = sum(int(task.get("tool_results_ok") or 0) for task in tasks)
    tool_total = sum(int(task.get("tool_results_total") or 0) for task in tasks)
    return {
        "tasks": len(tasks),
        "rollouts": rollouts,
        "valid_rollouts": valid_rollouts,
        "task_success": successes,
        "task_success_rate": round(successes / rollouts, 4) if rollouts else None,
        "tool_results_ok": tool_ok,
        "tool_results_total": tool_total,
        "tool_success_rate": round(tool_ok / tool_total, 4) if tool_total else None,
        "stable_checks": sum(int(task.get("stable_checks") or 0) for task in tasks),
        "valid_stable_checks": sum(int(task.get("valid_stable_checks") or 0) for task in tasks),
        "within_steps_failures": sum(int(task.get("within_steps_failures") or 0) for task in tasks),
    }


def run_compare(
    *,
    repo: Path,
    model: str,
    val_file: Path,
    adapter_path: Path,
    out_root: Path,
    trace_root: Path,
    rollouts: int,
    train_version: str = "train-v2",
    max_model_len: int = 8192,
    provider: str = "offline",
    quantization: str = "nf4",
    dry_run: bool = False,
    tasks: Sequence[str] | None = None,
    baseline_adapter_path: Path | None = None,
) -> dict[str, Any]:
    from training.train_split import build_train_specs

    selected_tasks = list(tasks) if tasks is not None else load_val_tasks(val_file)
    specs = [
        spec for spec in build_train_specs(train_version) if spec.task_id in set(selected_tasks)
    ]
    if len(specs) != len(selected_tasks):
        missing = sorted(set(selected_tasks) - {spec.task_id for spec in specs})
        raise ValueError(f"held-out task ids missing from {train_version}: {missing}")
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"adapter path does not exist: {adapter_path}")
    if baseline_adapter_path is not None and not baseline_adapter_path.is_dir():
        raise FileNotFoundError(f"baseline adapter path does not exist: {baseline_adapter_path}")

    summaries: dict[str, Any] = {}
    conditions = (
        ("baseline", str(baseline_adapter_path) if baseline_adapter_path else ""),
        ("sft", str(adapter_path)),
    )
    for condition, adapter in conditions:
        out_dir = out_root / condition
        trace_dir = trace_root / condition
        command = build_eval_command(
            repo=repo,
            model=model,
            out_dir=out_dir,
            trace_dir=trace_dir,
            tasks=selected_tasks,
            rollouts=rollouts,
            train_version=train_version,
            adapter_path=adapter,
            max_model_len=max_model_len,
            provider=provider,
            quantization=quantization,
        )
        if dry_run:
            summaries[condition] = {"command": command}
            continue
        subprocess.run(command, check=True)
        rows = load_jsonl(out_dir / "runs.jsonl")
        summaries[condition] = summarize_condition(rows, specs, trace_dir)

    result = finalize_result(
        summaries,
        tasks=selected_tasks,
        rollouts=rollouts,
        model=model,
        adapter_path=adapter_path,
        val_file=val_file,
        train_version=train_version,
        provider=provider,
        quantization=quantization,
        dry_run=dry_run,
        baseline_adapter_path=baseline_adapter_path,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def finalize_result(
    summaries: Mapping[str, Any],
    *,
    tasks: Sequence[str],
    rollouts: int,
    model: str,
    adapter_path: Path,
    val_file: Path,
    train_version: str,
    provider: str,
    quantization: str,
    dry_run: bool,
    baseline_adapter_path: Path | None = None,
) -> dict[str, Any]:
    result = {
        "status": "dry_run" if dry_run else "completed",
        "model": model,
        "adapter_path": str(adapter_path),
        "baseline_adapter_path": (
            str(baseline_adapter_path) if baseline_adapter_path is not None else None
        ),
        "val_file": str(val_file),
        "train_version": train_version,
        "held_out_tasks": tasks,
        "rollouts_per_task": rollouts,
        "sampling_config": {
            "temperature": 0.0,
            "top_p": None,
            "seed": 20261019,
            "provider": "vllm-offline" if provider == "offline" else "transformers-offline",
            "requested_provider": provider,
            "quantization": quantization,
            "is_default": False,
        },
        "conditions": summaries,
    }
    if not dry_run:
        result["by_difficulty"] = {
            condition: {
                "base": _by_kind(summary, "base"),
                "hard": _by_kind(summary, "hard"),
            }
            for condition, summary in summaries.items()
        }
        for condition in summaries:
            result["by_difficulty"][condition]["overall"] = {
                "tasks": len(summaries[condition]["tasks"]),
                "rollouts": summaries[condition]["rollouts"],
                "task_success": summaries[condition]["task_success"]["hits"],
                "task_success_rate": summaries[condition]["task_success"]["rate"],
                "tool_success_rate": summaries[condition]["tool_results"]["rate"],
                "stable_checks": summaries[condition]["stable_checks"],
                "valid_stable_checks": summaries[condition]["valid_stable_checks"],
                "within_steps_failures": summaries[condition]["within_steps_failures"],
            }
        baseline = result["by_difficulty"]["baseline"]
        sft = result["by_difficulty"]["sft"]
        result["delta"] = {
            kind: {
                "task_success_rate": _subtract(
                    sft[kind].get("task_success_rate"), baseline[kind].get("task_success_rate")
                ),
                "tool_success_rate": _subtract(
                    sft[kind].get("tool_success_rate"), baseline[kind].get("tool_success_rate")
                ),
                "stable_checks": int(sft[kind].get("stable_checks") or 0)
                - int(baseline[kind].get("stable_checks") or 0),
                "valid_stable_checks": int(sft[kind].get("valid_stable_checks") or 0)
                - int(baseline[kind].get("valid_stable_checks") or 0),
                "within_steps_failures": int(sft[kind].get("within_steps_failures") or 0)
                - int(baseline[kind].get("within_steps_failures") or 0),
            }
            for kind in ("base", "hard", "overall")
        }
    return result


def _subtract(after: Any, before: Any) -> float | None:
    if after is None or before is None:
        return None
    return round(float(after) - float(before), 4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT baseline vs adapter held-out comparison")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument(
        "--baseline-adapter",
        type=Path,
        default=None,
        help="adapter for the baseline condition; defaults to the unmodified base model",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="comma-separated held-out task ids; defaults to task ids in --val-file",
    )
    parser.add_argument("--out-root", type=Path, default=Path("/kaggle/working/heldout"))
    parser.add_argument("--trace-root", type=Path, default=Path("/kaggle/working/heldout_traces"))
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--train-version", default="train-v2")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--provider", choices=("offline", "transformers"), default="offline")
    parser.add_argument("--quantization", choices=("nf4", "none", "awq"), default="nf4")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_compare(
        repo=args.repo.resolve(),
        model=args.model,
        val_file=args.val_file,
        adapter_path=args.adapter_path,
        out_root=args.out_root,
        trace_root=args.trace_root,
        rollouts=args.rollouts,
        train_version=args.train_version,
        max_model_len=args.max_model_len,
        provider=args.provider,
        quantization=args.quantization,
        dry_run=args.dry_run,
        tasks=[item.strip() for item in args.tasks.split(",") if item.strip()] or None,
        baseline_adapter_path=args.baseline_adapter,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
