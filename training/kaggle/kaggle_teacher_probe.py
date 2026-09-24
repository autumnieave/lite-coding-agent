"""Run a small teacher-format probe with an offline Qwen model on Kaggle.

The probe does not change the agent loop: it reuses ``scripts.unified_eval.run_once`` and
records the same traces as the main experiments, then compares the teacher's tool-call
wire format and deterministic-gate pass rate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_TASKS = (
    "U-TR-A-C1-BASE",
    "U-TR-A-CHECK-BASE",
    "U-TR-A-TOOL-BASE",
)
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
SAMPLING = {"temperature": 0.0, "top_p": None, "seed": 20261019}


class ModelLoadTimeout(RuntimeError):
    """Raised when the offline model exceeds the pre-registered load budget."""


def _timeout_handler(signum: int, frame: Any) -> None:  # noqa: ARG001 - signal signature
    raise ModelLoadTimeout("model_load_timeout")


def _alarm(seconds: float) -> None:
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)


def _cancel_alarm() -> None:
    if hasattr(signal, "SIGALRM"):
        signal.setitimer(signal.ITIMER_REAL, 0)


def analyze_trace(path: Path) -> dict[str, Any]:
    """Extract tool-call wire formats and conversion counts from one trace."""

    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    turns = [event for event in events if event.get("event") == "llm_turn"]
    end = next((event for event in reversed(events) if event.get("event") == "run_end"), {})
    wire_counts: dict[str, int] = {}
    native_calls = 0
    assistant_turns = 0
    for turn in turns:
        config = turn.get("sampling_config") or {}
        wire = str(config.get("wire_format") or "unrecorded")
        wire_counts[wire] = wire_counts.get(wire, 0) + 1
        calls = turn.get("tool_calls") or []
        if calls:
            assistant_turns += 1
            native_calls += len(calls)
    checks = dict(end.get("checks") or {})
    return {
        "task_id": end.get("task_id") or (turns[-1].get("task_id") if turns else ""),
        "turns": len(turns),
        "wire_formats": wire_counts,
        "native_assistant_turns": assistant_turns,
        "native_tool_calls": native_calls,
        "checks": checks,
        "all_checks_true": bool(checks) and all(bool(value) for value in checks.values()),
    }


def _count_conversions(path: Path) -> tuple[int, bool]:
    from training.build_sft_dataset import (
        load_episode,
        normalize_teacher_messages,
        validate_teacher_messages,
    )

    episode = load_episode(path)
    if episode is None:
        return 0, False
    final_turn = next(
        (
            turn
            for turn in reversed(episode.turns)
            if (turn.get("response") or {}).get("content")
            and not (turn.get("response") or {}).get("tool_calls")
        ),
        None,
    )
    if final_turn is None:
        return 0, False
    messages, converted = normalize_teacher_messages(final_turn.get("messages") or [])
    try:
        validate_teacher_messages(messages)
    except ValueError:
        return converted, False
    return converted, True


def _load_specs(repo: Path, task_ids: Sequence[str]) -> list[Any]:
    sys.path.insert(0, str(repo))
    from training.train_split import build_train_specs

    wanted = set(task_ids)
    specs = [spec for spec in build_train_specs() if spec.task_id in wanted]
    missing = wanted - {spec.task_id for spec in specs}
    if missing:
        raise ValueError(f"unknown task ids: {sorted(missing)}")
    return specs


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from scripts.unified_eval import run_once
    from training.kaggle.offline_provider import VLLMOfflineProvider

    specs = _load_specs(repo, [item for item in args.tasks.split(",") if item.strip()])
    if args.limit:
        specs = specs[: args.limit]
    started = time.perf_counter()
    _alarm(args.load_timeout)
    try:
        provider = VLLMOfflineProvider(
            args.model,
            max_model_len=args.max_model_len,
            temperature=SAMPLING["temperature"],
            top_p=SAMPLING["top_p"],
            seed=SAMPLING["seed"],
            max_tokens=args.max_tokens,
            quantization=args.quantization or None,
        )
    finally:
        _cancel_alarm()
    load_s = round(time.perf_counter() - started, 1)
    rows: list[dict[str, Any]] = []
    infer_started = time.perf_counter()
    try:
        for spec in specs:
            record = await run_once(
                spec,
                provider=provider,
                rollout_index=0,
                trace_dir=Path(args.trace_dir),
                sampling_config=provider.sampling_config,
                system_variant="s0",
            )
            trace_path = Path(str(record.get("trace_path") or ""))
            if not trace_path.is_file():
                trace_path = Path(args.trace_dir) / spec.split / f"{spec.task_id}_rollout00.jsonl"
            analysis = analyze_trace(trace_path) if trace_path.is_file() else {}
            converted, valid = (
                _count_conversions(trace_path) if trace_path.is_file() else (0, False)
            )
            rows.append(
                {
                    "task_id": spec.task_id,
                    "task_success": record.get("task_success"),
                    "within_steps": record.get("within_steps"),
                    "steps": record.get("steps"),
                    "duration_s": record.get("duration_s"),
                    "checks": record.get("checks"),
                    "wire_formats": analysis.get("wire_formats"),
                    "native_tool_calls": analysis.get("native_tool_calls"),
                    "converted_tool_calls": converted,
                    "converter_valid": valid,
                    "trace_path": str(trace_path),
                }
            )
            print("TASK", spec.task_id, "success", record.get("task_success"), flush=True)
    finally:
        await provider.aclose()
    inference_s = round(time.perf_counter() - infer_started, 1)
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    summary = {
        "status": "completed",
        "model": args.model,
        "quantization": args.quantization or None,
        "gpu": gpu,
        "load_s": load_s,
        "inference_s": inference_s,
        "rollouts": len(rows),
        "gate_pass": sum(1 for row in rows if row.get("task_success")),
        "rows": rows,
        "sampling_config": dict(SAMPLING),
    }
    _write(Path(args.out_dir) / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen 7B teacher format probe")
    parser.add_argument("--repo", type=Path, default=Path("/kaggle/working/repo"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/teacher_probe"))
    parser.add_argument("--trace-dir", type=Path, default=Path("/kaggle/working/teacher_traces"))
    parser.add_argument("--load-timeout", type=float, default=180.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--quantization", default="")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = _load_specs(Path(args.repo).resolve(), [t for t in args.tasks.split(",") if t.strip()])
    preflight = {
        "stage": "preflight",
        "repo": str(args.repo),
        "model": args.model,
        "tasks": [spec.task_id for spec in specs[: args.limit]],
        "max_model_len": args.max_model_len,
        "load_timeout_s": args.load_timeout,
    }
    _write(Path(args.out_dir) / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    try:
        summary = asyncio.run(run_probe(args))
    except ModelLoadTimeout:
        _write(
            Path(args.out_dir) / "summary.json",
            {"status": "error", "error_kind": "model_load_timeout", "model": args.model},
        )
        return 2
    except Exception as exc:  # noqa: BLE001 - the kernel records the failure instead of hiding it
        _write(
            Path(args.out_dir) / "summary.json",
            {
                "status": "error",
                "error_kind": type(exc).__name__,
                "message": str(exc),
                "model": args.model,
            },
        )
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("gate_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
