"""Validate stage-1.5 held-out tasks with the frozen DeepSeek teacher."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent.core.llm import OpenAICompatProvider  # noqa: E402
from scripts.unified_eval import run_once  # noqa: E402
from training.stats_heldout import (  # noqa: E402
    STATS_TASK_SPECS,
    policy_check_names,
    policy_compliance,
    validate_stats_specs,
)

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"
TEACHER_PROTOCOL = "openai_chat_completions_tool_calls_v1"
GATE_MIN_RATE = 0.40
POLICY_MIN_RATE = 0.60


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="stage-1.5 teacher gate validation")
    parser.add_argument("--out-dir", type=Path, default=Path("private/heldout_1_5_teacher"))
    parser.add_argument(
        "--trace-dir", type=Path, default=Path("private/traces/stats_heldout/teacher")
    )
    parser.add_argument("--report", type=Path, default=Path("docs/evidence/统计heldout_teacher.md"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--tasks", default="", help="comma-separated task ids")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _api_key() -> str:
    for name in ("LLM_API_KEY", "DASHSCOPE_API_KEY"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    raise RuntimeError("missing LLM_API_KEY or DASHSCOPE_API_KEY")


def _load_completed(path: Path) -> set[tuple[str, int]]:
    if not path.is_file():
        return set()
    completed: set[tuple[str, int]] = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("task_id")
            and row.get("rollout_index") is not None
            and not row.get("error_kind")
        ):
            completed.add((str(row["task_id"]), int(row["rollout_index"])))
    return completed


async def _preflight(provider: OpenAICompatProvider) -> dict[str, Any]:
    started = time.perf_counter()
    reply = await provider.chat(
        [
            {"role": "system", "content": "Reply with exactly OK."},
            {"role": "user", "content": "preflight"},
        ]
    )
    return {
        "ok": bool(reply.content),
        "content": reply.content[:80],
        "duration_s": round(time.perf_counter() - started, 2),
        "model": provider.model,
        "base_url": provider.base_url,
    }


async def _run_jobs(
    provider: OpenAICompatProvider,
    jobs: list[tuple[Any, int]],
    *,
    trace_dir: Path,
    timeout: float,
    concurrency: int,
    run_path: Path,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(spec: Any, rollout: int) -> dict[str, Any]:
        async with semaphore:
            try:
                return await asyncio.wait_for(
                    run_once(
                        spec,
                        provider=provider,
                        rollout_index=rollout,
                        trace_dir=trace_dir,
                        sampling_config=provider.sampling_config,
                        system_variant="s0",
                    ),
                    timeout=timeout,
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
                    "error": f"rollout exceeded {timeout}s",
                    "checks": {},
                    "task_success": False,
                    "within_steps": False,
                    "steps": 0,
                    "duration_s": timeout,
                }

    rows: list[dict[str, Any]] = []
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with run_path.open("a", encoding="utf-8", newline="\n") as handle:
        for future in asyncio.as_completed([one(spec, rollout) for spec, rollout in jobs]):
            row = await future
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
            print(
                f"[{len(rows)}/{len(jobs)}] {row['task_id']}#{row['rollout_index']:02d} "
                f"success={row.get('task_success')} steps={row.get('steps')} "
                f"error={row.get('error_kind', '')}",
                flush=True,
            )
    return rows


def _summary_row(spec: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error_kind")]
    gate = sum(1 for row in valid if row.get("task_success"))
    policy = sum(1 for row in valid if policy_compliance(spec, dict(row.get("checks") or {})))
    gate_rate = gate / len(valid) if valid else 0.0
    policy_rate = policy / len(valid) if valid else 0.0
    return {
        "task_id": spec.task_id,
        "family": spec.family,
        "difficulty": spec.difficulty,
        "context_tier": spec.context_tier,
        "rollouts": len(rows),
        "valid_rollouts": len(valid),
        "errors": len(rows) - len(valid),
        "gate_passes": gate,
        "gate_rate": round(gate_rate, 4),
        "policy_passes": policy,
        "policy_rate": round(policy_rate, 4),
        "gate_threshold_met": gate_rate >= GATE_MIN_RATE,
        "policy_threshold_met": policy_rate >= POLICY_MIN_RATE,
        "keep": gate_rate >= GATE_MIN_RATE and policy_rate >= POLICY_MIN_RATE,
        "policy_checks": list(policy_check_names(spec)),
        "failed_checks": sorted(
            {name for row in valid for name, ok in dict(row.get("checks") or {}).items() if not ok}
        ),
    }


def summarize(specs: list[Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), []).append(row)
    task_rows = [_summary_row(spec, by_task.get(spec.task_id, [])) for spec in specs]
    kept = [row["task_id"] for row in task_rows if row["keep"]]
    dropped = [row["task_id"] for row in task_rows if not row["keep"]]
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "teacher_protocol": TEACHER_PROTOCOL,
        "thresholds": {"gate_rate": GATE_MIN_RATE, "policy_rate": POLICY_MIN_RATE},
        "tasks": task_rows,
        "kept": kept,
        "dropped": dropped,
        "all_kept": len(kept) == len(task_rows),
    }


def write_report(path: Path, summary: dict[str, Any], preflight: dict[str, Any]) -> None:
    lines = [
        "# Stage 1.5 教师门禁验证",
        "",
        f"- 生成时间：`{summary['created_at']}`",
        f"- 教师模型：`{preflight['model']}`",
        f"- base_url：`{preflight['base_url']}`",
        f"- 协议：`{summary['teacher_protocol']}`",
        "- 采样：`temperature=0.0`，`top_p=None`，`seed=20261019`",
        f"- 保留门槛：门禁通过率 ≥ {GATE_MIN_RATE:.0%}，策略合规率 ≥ {POLICY_MIN_RATE:.0%}",
        "",
        "## 逐任务结果",
        "",
        "| 任务 | 族 | 难度 | rollout | 门禁通过 | 门禁率 | 策略合规 | 合规率 | 结论 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["tasks"]:
        lines.append(
            f"| `{row['task_id']}` | {row['family']} | {row['difficulty']} | "
            f"{row['valid_rollouts']}/{row['rollouts']} | {row['gate_passes']} | "
            f"{row['gate_rate']:.1%} | {row['policy_passes']} | "
            f"{row['policy_rate']:.1%} | {'保留' if row['keep'] else '调整或丢弃'} |"
        )
    lines += [
        "",
        "## 失败检查项",
        "",
    ]
    for row in summary["tasks"]:
        failed = ", ".join(f"`{item}`" for item in row["failed_checks"]) or "无"
        lines.append(f"- `{row['task_id']}`：{failed}")
    lines += [
        "",
        "## 判定",
        "",
        f"- 保留任务：{len(summary['kept'])}/15",
        f"- 需调整或丢弃：{len(summary['dropped'])}",
        "- 本报告只记录教师可解性，不代表 baseline/SFT 难度结论。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


async def main_async(args: argparse.Namespace) -> int:
    validate_stats_specs()
    specs = list(STATS_TASK_SPECS)
    if args.tasks:
        wanted = {item.strip() for item in args.tasks.split(",") if item.strip()}
        specs = [spec for spec in specs if spec.task_id in wanted]
    if not specs:
        raise ValueError("no matching stats tasks")
    provider = OpenAICompatProvider(
        api_key=_api_key(),
        model=args.model,
        base_url=args.base_url or None,
        timeout=120.0,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    try:
        preflight = await _preflight(provider)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "preflight.json").write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(preflight, ensure_ascii=False), flush=True)
        if args.preflight_only:
            return 0
        jobs = [(spec, rollout) for spec in specs for rollout in range(args.rollouts)]
        run_path = args.out_dir / "teacher_runs.jsonl"
        if args.resume:
            done = _load_completed(run_path)
            jobs = [job for job in jobs if (job[0].task_id, job[1]) not in done]
        if args.limit:
            jobs = jobs[: args.limit]
        if jobs:
            await _run_jobs(
                provider,
                jobs,
                trace_dir=args.trace_dir,
                timeout=args.timeout,
                concurrency=args.concurrency,
                run_path=run_path,
            )
    finally:
        await provider.aclose()

    all_rows: list[dict[str, Any]] = []
    if run_path.is_file():
        all_rows = [
            json.loads(line)
            for line in run_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    summary = summarize(specs, all_rows)
    summary.update(
        {
            "model": provider.model,
            "base_url": provider.base_url,
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
                "rollouts_per_task": args.rollouts,
            },
        }
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.report, summary, preflight)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["all_kept"] else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
