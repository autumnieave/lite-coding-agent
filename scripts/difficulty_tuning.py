"""Run the dev-only single-dimension difficulty ladder.

The runner never touches the frozen pilot test or the future train split. It
uses the same agent loop, trace recorder and judge as ``unified_eval.py`` so
results are comparable to the baseline distribution.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from unified_eval import (  # noqa: E402
    aggregate,
    run_once,
    run_self_test,
    write_csv,
)

from agent.core.config import find_env_file, load_env_file  # noqa: E402
from agent.core.llm import OpenAICompatProvider  # noqa: E402
from training.difficulty_tuning import (  # noqa: E402
    TUNING_TASK_SPECS,
    tuning_level,
    validate_tuning_specs,
)
from training.final_test import build_final_specs  # noqa: E402
from training.kaggle.contracts import read_jsonl  # noqa: E402
from training.stats_heldout import STATS_TASK_SPECS, validate_stats_specs  # noqa: E402
from training.system_variants import GUIDED_JSON_SCHEMA  # noqa: E402
from training.train_split import build_train_specs  # noqa: E402

DEFAULT_OUT_DIR = Path("private/difficulty_tuning")
DEFAULT_TRACE_DIR = Path("private/traces/difficulty_tuning")


def _dimension_map(specs: Sequence[Any] = TUNING_TASK_SPECS) -> dict[str, str]:
    return {spec.task_id: spec.tuning_dimension or spec.family for spec in specs}


def completed_job_keys(out_dir: Path) -> set[tuple[str, int]]:
    """Return successfully completed (task_id, rollout_index) pairs for ``--resume``.

    Timed-out rows carry ``error_kind`` and must be retried; otherwise a single
    timeout would permanently remove that rollout from the experiment.
    """

    return {
        (str(row["task_id"]), int(row["rollout_index"]))
        for row in read_jsonl(out_dir / "runs.jsonl")
        if "task_id" in row and "rollout_index" in row and not row.get("error_kind")
    }


def build_summary_rows(
    checks: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    specs: Sequence[Any] = TUNING_TASK_SPECS,
) -> list[dict[str, Any]]:
    dimensions = _dimension_map(specs)
    specs_by_id = {spec.task_id: spec for spec in specs}
    checks_by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in checks:
        checks_by_task.setdefault(str(row["task_id"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        spec = specs_by_id[task_id]
        level = spec.difficulty if not spec.tuning_dimension else tuning_level(task_id)
        task_checks = checks_by_task.get(task_id, [])
        max_hits = max((int(row["hits"]) for row in task_checks), default=0)
        max_total = max((int(row["total"]) for row in task_checks), default=0)
        stable_checks = sum(1 for row in task_checks if row["stability"] == "stable")
        rows.append(
            {
                "dimension": dimensions[task_id],
                "level": level,
                "task_id": task_id,
                "rollouts": task["rollouts"],
                "checks": len(task_checks),
                "stable_checks": stable_checks,
                "intermittent_checks": sum(
                    1 for row in task_checks if row["stability"] == "intermittent"
                ),
                "max_hits": max_hits,
                "max_total": max_total,
                "max_rate": round(max_hits / max_total, 4) if max_total else 0.0,
                "task_success": task["task_success"],
                "within_steps_failures": task["within_steps_failures"],
                "mean_steps": task["mean_steps"],
            }
        )
    order = {
        "constraint_count": 0,
        "context_length": 1,
        "tool_parameters": 2,
        "conflict": 3,
        "check_depth": 4,
    }
    level_order = {"base": 0, "none": 0, "weak": 1, "hard": 2, "strong": 2}
    rows.sort(
        key=lambda row: (
            order.get(str(row["dimension"]), 99),
            level_order.get(str(row["level"]), 99),
        )
    )
    return rows


def write_experiment(
    out_dir: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    rollouts: int,
    specs: Sequence[Any] = TUNING_TASK_SPECS,
    validate_tuning: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    if validate_tuning:
        validate_tuning_specs()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / "runs.jsonl"
    merged: dict[tuple[str, int], Mapping[str, Any]] = {}
    if resume:
        for row in read_jsonl(run_path):
            if "task_id" in row and "rollout_index" in row:
                merged[(str(row["task_id"]), int(row["rollout_index"]))] = row
    for row in records:
        merged[(str(row["task_id"]), int(row["rollout_index"]))] = row
    all_records = sorted(
        merged.values(),
        key=lambda row: (str(row["task_id"]), int(row["rollout_index"])),
    )
    run_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in all_records) + "\n",
        encoding="utf-8",
    )
    check_rows, task_rows = aggregate(all_records, specs)
    dimensions = _dimension_map(specs)
    enriched_checks = [{**row, "dimension": dimensions[str(row["task_id"])]} for row in check_rows]
    write_csv(
        out_dir / "checks.csv",
        enriched_checks,
        (
            "dimension",
            "task_id",
            "check",
            "kind",
            "hits",
            "total",
            "stability",
        ),
    )
    summary_rows = build_summary_rows(check_rows, task_rows, specs)
    write_csv(
        out_dir / "summary.csv",
        summary_rows,
        (
            "dimension",
            "level",
            "task_id",
            "rollouts",
            "checks",
            "stable_checks",
            "intermittent_checks",
            "max_hits",
            "max_total",
            "max_rate",
            "task_success",
            "within_steps_failures",
            "mean_steps",
        ),
    )
    stable = sum(1 for row in check_rows if row["stability"] == "stable")
    result = {
        "rollouts_requested_per_task": rollouts,
        "records": len(all_records),
        "new_records": len(records),
        "errors": sum(1 for row in all_records if "error" in row),
        "checks": len(check_rows),
        "stable_checks": stable,
        "summary": summary_rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


async def main_async(args: argparse.Namespace) -> int:
    if args.spec_set == "final-test":
        specs = build_final_specs()
        validate_tuning = False
    elif args.spec_set == "train":
        specs = build_train_specs(args.train_version)
        validate_tuning = False
    elif args.spec_set == "stats":
        validate_stats_specs()
        specs = STATS_TASK_SPECS
        validate_tuning = False
    else:
        specs = TUNING_TASK_SPECS
        validate_tuning_specs()
        validate_tuning = True
    if args.self_test:
        return run_self_test(specs)
    if args.rollouts < 1:
        print("--rollouts 必须 >= 1", file=sys.stderr)
        return 2
    if args.tasks:
        wanted = {item.strip() for item in args.tasks.split(",") if item.strip()}
        specs = tuple(spec for spec in specs if spec.task_id in wanted)
        if not specs:
            print("没有匹配的调参任务", file=sys.stderr)
            return 2

    jobs = [(spec, rollout) for spec in specs for rollout in range(args.rollouts)]
    if args.resume:
        done = completed_job_keys(args.out_dir)
        jobs = [job for job in jobs if (job[0].task_id, job[1]) not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        print("没有待运行任务；保留已有 runs.jsonl", flush=True)
        result = write_experiment(
            args.out_dir,
            (),
            rollouts=args.rollouts,
            specs=specs,
            validate_tuning=validate_tuning,
            resume=args.resume,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    env_path = find_env_file(Path.cwd())
    if env_path is not None:
        load_env_file(env_path)
    if args.provider == "offline":
        from training.kaggle.offline_provider import VLLMOfflineProvider

        offline_model = args.offline_model or os.environ.get("LLM_MODEL", "")
        if not offline_model:
            print("--provider offline 需要 --offline-model 或 LLM_MODEL", file=sys.stderr)
            return 2
        provider = VLLMOfflineProvider(
            offline_model,
            max_model_len=args.max_model_len,
            temperature=args.sampling_temperature,
            top_p=args.sampling_top_p,
            seed=args.sampling_seed,
            max_tokens=args.offline_max_tokens,
            guided_json=(GUIDED_JSON_SCHEMA if args.system_variant == "s2" else None),
            adapter_path=args.adapter_path or None,
            quantization=args.offline_quantization or None,
        )
        args.concurrency = 1
    elif args.provider == "transformers":
        from training.kaggle.transformers_provider import TransformersOfflineProvider

        offline_model = args.offline_model or os.environ.get("LLM_MODEL", "")
        if not offline_model:
            print("--provider transformers 需要 --offline-model 或 LLM_MODEL", file=sys.stderr)
            return 2
        provider = TransformersOfflineProvider(
            offline_model,
            max_model_len=args.max_model_len,
            temperature=args.sampling_temperature,
            top_p=args.sampling_top_p,
            seed=args.sampling_seed,
            max_tokens=args.offline_max_tokens,
            adapter_path=args.adapter_path or None,
            quantization=args.quantization,
        )
        args.concurrency = 1
    else:
        provider = OpenAICompatProvider.from_env(
            temperature=args.sampling_temperature,
            top_p=args.sampling_top_p,
            seed=args.sampling_seed,
        )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def worker(spec: Any, rollout: int) -> dict[str, Any]:
        async with semaphore:
            if args.rollout_timeout <= 0:
                return await run_once(
                    spec,
                    provider=provider,
                    rollout_index=rollout,
                    trace_dir=args.trace_dir,
                    sampling_config=provider.sampling_config,
                    system_variant=args.system_variant,
                )
            try:
                return await asyncio.wait_for(
                    run_once(
                        spec,
                        provider=provider,
                        rollout_index=rollout,
                        trace_dir=args.trace_dir,
                        sampling_config=provider.sampling_config,
                        system_variant=args.system_variant,
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
                    "error": f"rollout 超过 {args.rollout_timeout}s",
                    "checks": {},
                    "violated": 0,
                    "task_success": False,
                    "within_steps": False,
                    "steps": 0,
                    "tools_used": [],
                    "tool_selection_ok": False,
                    "params_ok": False,
                    "reply": "",
                    "stopped_reasons": [],
                    "duration_s": args.rollout_timeout,
                }

    records: list[dict[str, Any]] = []
    for coro in asyncio.as_completed([worker(*job) for job in jobs]):
        row = await coro
        records.append(row)
        if args.progress_every <= 1 or len(records) % args.progress_every == 0:
            print(
                f"[{len(records)}/{len(jobs)}] {row['task_id']}#{row['rollout_index']:02d} "
                f"success={row.get('task_success')} steps={row.get('steps')} "
                f"error={row.get('error', '')}",
                flush=True,
            )
    await provider.aclose()
    records.sort(key=lambda item: (item["task_id"], item["rollout_index"]))
    result = write_experiment(
        args.out_dir,
        records,
        rollouts=args.rollouts,
        specs=specs,
        validate_tuning=validate_tuning,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="dev-only difficulty ladder")
    parser.add_argument("--tasks", default="", help="逗号分隔的任务 ID；默认全部")
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条 rollout；0 表示全部")
    parser.add_argument(
        "--rollout-timeout",
        type=float,
        default=0,
        help="单条 rollout 的超时秒数；0 表示不启用",
    )
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="跳过 runs.jsonl 中已完成的任务")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--provider", choices=("server", "offline", "transformers"), default="server"
    )
    parser.add_argument("--offline-model", default="", help="vLLM 离线模式的模型路径或 Hub id")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--offline-max-tokens", type=int, default=512)
    parser.add_argument(
        "--offline-quantization",
        default="",
        help="vLLM 量化方式，例如 awq；留空则按模型 config 自动识别",
    )
    parser.add_argument(
        "--quantization",
        choices=("nf4", "none"),
        default="nf4",
        help="transformers provider 的量化方式；默认 NF4",
    )
    parser.add_argument("--sampling-temperature", type=float, default=0.0)
    parser.add_argument("--sampling-top-p", type=float, default=None)
    parser.add_argument("--sampling-seed", type=int, default=20261019)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument(
        "--spec-set",
        choices=("tuning", "final-test", "train", "stats"),
        default="tuning",
    )
    parser.add_argument(
        "--train-version",
        choices=("train-v1", "train-v2"),
        default="train-v2",
        help="--spec-set train 使用的冻结版本",
    )
    parser.add_argument("--system-variant", choices=("s0", "s1", "s1t", "s2"), default="s0")
    parser.add_argument(
        "--adapter-path",
        default="",
        help="可选 LoRA adapter；仅 offline provider 使用",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
