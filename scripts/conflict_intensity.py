"""受控冲突强度实验：弱/中/强约束对下的输出行为观察。"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trace_utils import TraceRecorder  # noqa: E402

from agent.core.config import find_env_file, load_env_file  # noqa: E402
from agent.core.llm import BaseProvider, OpenAICompatProvider  # noqa: E402

DEFAULT_TRACE_DIR = Path("private/traces/conflict_intensity")
DEFAULT_OUT = Path("private/traces/conflict_intensity_runs.jsonl")
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")
CONFLICT_MARKERS = ("unresolved_conflict", "unresolved_conflicts")


@dataclass(frozen=True, slots=True)
class ConflictGroup:
    name: str
    intensity: str
    constraints: tuple[tuple[str, str], ...]
    notes: str


GROUPS: tuple[ConflictGroup, ...] = (
    ConflictGroup(
        "weak_compatible",
        "weak",
        (
            ("W1", "log_prefix 字段必须存在且非空。"),
            ("W2", "log_prefix 字段必须使用 [WEAK-A]。"),
        ),
        "同槽位两条弱约束，可用 [WEAK-A] 同时满足。",
    ),
    ConflictGroup(
        "weak_independent",
        "weak",
        (
            ("W3", "timestamp 字段必须使用 UTC。"),
            ("W4", "log_prefix 字段必须使用 [WEAK-B]。"),
        ),
        "不同字段的弱约束，彼此没有取舍。",
    ),
    ConflictGroup(
        "medium_priority",
        "medium",
        (
            ("M1", "log_prefix 字段必须使用 [MID-A]。"),
            ("M2", "log_prefix 字段必须使用 [MID-B]。"),
            ("M3", "如果 M1 与 M2 冲突，必须优先满足 M1，并在 conflict_resolution 字段说明取舍。"),
        ),
        "同槽位冲突，但有明确优先级，要求模型做取舍。",
    ),
    ConflictGroup(
        "strong_mutual_exclusion",
        "strong",
        (
            ("S1", "log_prefix 字段必须使用 [STRONG-A]。"),
            ("S2", "log_prefix 字段必须使用 [STRONG-B]。"),
        ),
        "同槽位两个不同精确值，没有优先级或例外。",
    ),
)


def collect_keys(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        keys = [str(key) for key in payload]
        for value in payload.values():
            keys.extend(collect_keys(value))
        return keys
    if isinstance(payload, list):
        keys: list[str] = []
        for item in payload:
            keys.extend(collect_keys(item))
        return keys
    return []


def judge_output(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text.strip())
    except (TypeError, ValueError):
        payload = None
    keys = collect_keys(payload)
    snake_ok = bool(keys) and all(SNAKE_CASE.fullmatch(key) for key in keys)
    lowered = text.lower()
    unresolved = False
    conflict_resolution = False
    if isinstance(payload, dict):
        unresolved = any(bool(payload.get(key)) for key in CONFLICT_MARKERS if key in payload)
        conflict_resolution = bool(payload.get("conflict_resolution"))
    else:
        unresolved = any(marker in lowered for marker in CONFLICT_MARKERS)
    return {
        "json_valid": payload is not None,
        "snake_case_keys": snake_ok,
        "no_code_fence": "```" not in text,
        "unresolved_conflict": unresolved,
        "conflict_resolution": conflict_resolution,
        "format_failure": payload is None or not snake_ok,
        "parsed_keys": keys,
    }


def build_messages(group: ConflictGroup) -> list[dict[str, str]]:
    lines = "\n".join(f"- [{cid}] {content}" for cid, content in group.constraints)
    return [
        {
            "role": "system",
            "content": (
                "你是一个受约束的终端输出 Agent。必须遵守下面的约束；"
                "如果约束无法同时满足，请在 JSON 中用 unresolved_conflicts 说明。\n"
                f"{lines}"
            ),
        },
        {
            "role": "user",
            "content": (
                "输出一个 JSON 对象总结本次会话，字段至少包含 log_prefix、timestamp；"
                "键名必须使用 snake_case。只描述你已经完成的事情，不要编造工具结果。"
            ),
        },
    ]


def _trace_path(root: Path, group: ConflictGroup, rollout_index: int) -> Path:
    return root / group.name / f"rollout{rollout_index:02d}.jsonl"


async def run_rollout(
    *,
    provider: BaseProvider,
    group: ConflictGroup,
    rollout_index: int,
    trace_dir: Path,
    sampling_config: Mapping[str, Any],
    max_tokens_note: str,
) -> dict[str, Any]:
    messages = build_messages(group)
    trace_path = _trace_path(trace_dir, group, rollout_index)
    trace = TraceRecorder(
        trace_path,
        metadata={
            "experiment": "conflict_intensity",
            "sample_set": "controlled_conflict",
            "task_id": group.name,
            "group": group.name,
            "profile": "controlled",
            "seed": sampling_config.get("seed"),
            "rollout_index": rollout_index,
            "constraints_total": len(group.constraints),
            "step_labels": ["probe"],
            "intensity": group.intensity,
            "notes": group.notes,
            "max_tokens_note": max_tokens_note,
        },
        workspace=Path.cwd(),
        constraints=group.constraints,
        sampling_config=sampling_config,
    )
    trace.set_step(1, "probe")
    started = time.perf_counter()
    try:
        response = await provider.chat(messages)
        text = response.content or ""
        result = judge_output(text)
        merged_config = dict(sampling_config)
        merged_config.update(getattr(response, "metadata", {}) or {})
        trace.record_llm_turn(
            purpose="agent",
            messages=messages,
            response=response,
            sampling_config=merged_config,
        )
        trace.end_run(
            checks={
                "json_valid": bool(result["json_valid"]),
                "snake_case_keys": bool(result["snake_case_keys"]),
                "no_code_fence": bool(result["no_code_fence"]),
            },
            evidence=[],
            step_replies={"probe": text},
            duration_s=time.perf_counter() - started,
            extras={
                "intensity": group.intensity,
                "unresolved_conflict": bool(result["unresolved_conflict"]),
                "format_failure": bool(result["format_failure"]),
                "parsed_keys": result["parsed_keys"],
                "reply_chars": len(text),
            },
        )
        return {
            "group": group.name,
            "intensity": group.intensity,
            "rollout_index": rollout_index,
            "checks": {
                "json_valid": bool(result["json_valid"]),
                "snake_case_keys": bool(result["snake_case_keys"]),
                "no_code_fence": bool(result["no_code_fence"]),
            },
            "unresolved_conflict": bool(result["unresolved_conflict"]),
            "format_failure": bool(result["format_failure"]),
            "trace_path": trace_path.as_posix(),
            "sampling_config": merged_config,
            "reply_chars": len(text),
        }
    finally:
        trace.close()


async def main_async(args: argparse.Namespace) -> int:
    if args.rollouts < 1:
        print("--rollouts 必须大于等于 1", file=sys.stderr)
        return 2
    env_path = find_env_file(Path.cwd())
    if env_path is not None:
        load_env_file(env_path)
    provider = OpenAICompatProvider.from_env(
        temperature=args.sampling_temperature,
        top_p=args.sampling_top_p,
        seed=args.sampling_seed,
    )
    trace_dir = Path(args.trace_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampling_config = dict(provider.sampling_config)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def worker(group: ConflictGroup, rollout_index: int) -> dict[str, Any]:
        async with semaphore:
            return await run_rollout(
                provider=provider,
                group=group,
                rollout_index=rollout_index,
                trace_dir=trace_dir,
                sampling_config=sampling_config,
                max_tokens_note=args.max_tokens_note,
            )

    jobs = [(group, rollout) for group in GROUPS for rollout in range(args.rollouts)]
    records: list[dict[str, Any]] = []
    try:
        for coro in asyncio.as_completed([worker(*job) for job in jobs]):
            record = await coro
            records.append(record)
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        await provider.aclose()

    print("\n=== summary ===")
    for group in GROUPS:
        selected = [item for item in records if item["group"] == group.name]
        print(
            group.name,
            "rollouts=",
            len(selected),
            "unresolved=",
            sum(bool(item["unresolved_conflict"]) for item in selected),
            "format_failure=",
            sum(bool(item["format_failure"]) for item in selected),
        )
    print(f"records={out_path}")
    print(f"traces={trace_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="受控冲突强度实验")
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sampling-temperature", type=float, default=0.0)
    parser.add_argument("--sampling-top-p", type=float, default=None)
    parser.add_argument("--sampling-seed", type=int, default=20261019)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--max-tokens-note",
        default="single-turn controlled probe",
        help="写入 trace 的实验说明",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
