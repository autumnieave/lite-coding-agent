"""Run the API arm of the API-vs-local 7B comparison.

The runner reuses the frozen 45-task manifest and ``unified_eval.run_once`` so
the API and local arms share fixtures, tools and the deterministic judge.
Raw outputs stay under ``private/``; only aggregate evidence is committed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_root in (REPO_ROOT, REPO_ROOT / "src", SCRIPT_DIR):
    value = str(import_root)
    if value not in sys.path:
        sys.path.insert(0, value)

from unified_eval import run_once  # noqa: E402

from agent.core.llm import BaseProvider, LLMError, LLMResponse, ToolCall  # noqa: E402
from training.heldout_pool import build_heldout_pool  # noqa: E402

DEFAULT_TASKS_FILE = Path("private/final_test_v3/tasks.jsonl")
DEFAULT_TRAIN_FILE = Path("private/sft_data_v5/train.jsonl")
DEFAULT_OUT_DIR = Path("private/api_ab")
DEFAULT_TRACE_DIR = Path("private/traces/api_ab")
DEFAULT_TASKS_SHA256 = "20e20b3beabaf0c48c00dae56d97c1aea356b1ac89c4cdb0fd9decc799c4fa5b"
DEFAULT_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"
API_KEY_ENVS = ("DASHSCOPE_API_KEY", "LLM_API_KEY")
STABLE_HITS = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_task_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            ids.append(text)
            continue
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        task_id = payload.get("task_id")
        if not task_id and isinstance(payload.get("metadata"), Mapping):
            task_id = payload["metadata"].get("task_id")
        if not task_id:
            raise ValueError(f"{path}:{line_number} has no task_id")
        ids.append(str(task_id))
    if len(ids) != len(set(ids)):
        duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
        raise ValueError(f"duplicate task ids: {duplicates}")
    return ids


def load_specs(train_file: Path, task_ids: Sequence[str]) -> dict[str, Any]:
    pool = build_heldout_pool(train_file)
    wanted = set(task_ids)
    specs = {task.spec.task_id: task.spec for task in pool.tasks if task.spec.task_id in wanted}
    missing = sorted(wanted - set(specs))
    if missing:
        raise ValueError(f"task ids missing from held-out pool: {missing}")
    return specs


def api_key_from_env(environ: Mapping[str, str] | None = None) -> tuple[str, str]:
    env = os.environ if environ is None else environ
    for name in API_KEY_ENVS:
        value = (env.get(name) or "").strip()
        if value:
            return value, name
    raise RuntimeError(f"missing API key; tried {', '.join(API_KEY_ENVS)}")


def _jsonable_usage(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}
    model_dump = getattr(raw, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    legacy_dump = getattr(raw, "dict", None)
    if callable(legacy_dump):
        dumped = legacy_dump(exclude_none=True)
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {}


def _usage_token(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key)
    return int(value) if isinstance(value, (int, float)) else 0


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


class ApiUsageProvider(BaseProvider):
    """OpenAI-compatible provider with per-rollout usage and latency accounting."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        base_url: str,
        temperature: float,
        top_p: float,
        seed: int | None = None,
        max_attempts: int = 3,
    ) -> None:
        self._client = client
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_attempts = max(1, max_attempts)
        self.requests: list[dict[str, Any]] = []

    @property
    def sampling_config(self) -> dict[str, Any]:
        return {
            "provider": "dashscope-openai-compatible",
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "seed_note": "not_sent; API-side sampling variance remains",
            "is_default": False,
        }

    def audit(self) -> dict[str, Any]:
        prompt_tokens = sum(
            _usage_token(item.get("usage") or {}, "prompt_tokens") for item in self.requests
        )
        completion_tokens = sum(
            _usage_token(item.get("usage") or {}, "completion_tokens") for item in self.requests
        )
        total_tokens = sum(
            _usage_token(item.get("usage") or {}, "total_tokens") for item in self.requests
        )
        reasoning_tokens = sum(
            _usage_token(
                (item.get("usage") or {}).get("completion_tokens_details") or {},
                "reasoning_tokens",
            )
            for item in self.requests
        )
        latencies = [float(item["latency_s"]) for item in self.requests]
        return {
            "requests": len(self.requests),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "api_tool_call_requests": sum(
                int(item.get("tool_call_count", 0) > 0) for item in self.requests
            ),
            "api_tool_calls": sum(int(item.get("tool_call_count", 0)) for item in self.requests),
            "fenced_json_turns": sum(bool(item.get("fenced_json")) for item in self.requests),
            "request_latencies_s": latencies,
            "model_versions": sorted(
                {str(item["model_version"]) for item in self.requests if item.get("model_version")}
            ),
            "system_fingerprints": sorted(
                {
                    str(item["system_fingerprint"])
                    for item in self.requests
                    if item.get("system_fingerprint")
                }
            ),
        }

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if tools:
            payload["tools"] = list(tools)
        if self.seed is not None:
            payload["seed"] = self.seed

        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
                response = await self._client.chat.completions.create(**payload)
                break
            except Exception as exc:  # noqa: BLE001 - provider boundary
                if attempt < self.max_attempts and _retryable(exc):
                    await asyncio.sleep(2 ** (attempt - 1))
                    continue
                raise LLMError(f"API call failed: {type(exc).__name__}: {exc}") from exc
        else:  # pragma: no cover - the loop always breaks or raises
            raise LLMError("API call failed without a response")

        latency = time.perf_counter() - started
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("API returned empty choices")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise LLMError("API response choices[0] has no message")

        calls: list[ToolCall] = []
        for index, raw_call in enumerate(getattr(message, "tool_calls", None) or []):
            function = getattr(raw_call, "function", None)
            name = getattr(function, "name", None) if function is not None else None
            if not name:
                continue
            calls.append(
                ToolCall(
                    id=getattr(raw_call, "id", None) or f"call_{len(self.requests)}_{index}",
                    name=str(name),
                    arguments=getattr(function, "arguments", None) or "{}",
                )
            )

        content = getattr(message, "content", None) or ""
        usage = _jsonable_usage(getattr(response, "usage", None))
        metadata = {
            "model_version": str(getattr(response, "model", None) or self.model),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "request_id": getattr(response, "_request_id", None)
            or getattr(response, "request_id", None),
            "response_id": getattr(response, "id", None),
            "created": getattr(response, "created", None),
            "usage": usage,
        }
        self.requests.append(
            {
                "latency_s": round(latency, 3),
                "attempts": attempt,
                "finish_reason": getattr(choices[0], "finish_reason", None),
                "usage": usage,
                "tool_call_count": len(calls),
                "content_chars": len(content),
                "fenced_json": "```" in content,
                "model_version": metadata["model_version"],
                "system_fingerprint": metadata["system_fingerprint"],
                "request_id": metadata["request_id"],
            }
        )
        return LLMResponse(content=content, tool_calls=tuple(calls), metadata=metadata)


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_runs(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_ids: Sequence[str],
    model: str,
    base_url: str,
    task_sha256: str,
    source_sha256: str | None,
) -> dict[str, Any]:
    wanted = set(task_ids)
    selected = [row for row in rows if str(row.get("task_id") or "") in wanted]
    errors = [row for row in selected if row.get("error_kind") or row.get("error")]
    valid = [row for row in selected if row not in errors]
    by_task: dict[str, list[Mapping[str, Any]]] = {task_id: [] for task_id in task_ids}
    for row in valid:
        by_task[str(row["task_id"])].append(row)

    per_task: list[dict[str, Any]] = []
    for task_id in task_ids:
        task_rows = by_task[task_id]
        success = sum(bool(row.get("task_success")) for row in task_rows)
        within = sum(bool(row.get("within_steps")) for row in task_rows)
        tool_ok = sum(bool(row.get("tool_selection_ok")) for row in task_rows)
        per_task.append(
            {
                "task_id": task_id,
                "valid": len(task_rows),
                "success": success,
                "success_rate": success / len(task_rows) if task_rows else 0.0,
                "within_steps": within,
                "tool_selection_ok": tool_ok,
                "stable": len(task_rows) >= 10 and success >= STABLE_HITS,
            }
        )

    request_latencies = [
        float(value)
        for row in valid
        for value in (row.get("api_audit") or {}).get("request_latencies_s") or []
    ]
    rollout_latencies = [
        float(row["duration_s"]) for row in valid if row.get("duration_s") is not None
    ]
    prompt_tokens = sum(int((row.get("api_audit") or {}).get("prompt_tokens", 0)) for row in valid)
    completion_tokens = sum(
        int((row.get("api_audit") or {}).get("completion_tokens", 0)) for row in valid
    )
    total_tokens = sum(int((row.get("api_audit") or {}).get("total_tokens", 0)) for row in valid)
    requests = sum(int((row.get("api_audit") or {}).get("requests", 0)) for row in valid)
    toolcall_requests = sum(
        int((row.get("api_audit") or {}).get("api_tool_call_requests", 0)) for row in valid
    )
    fenced_turns = sum(
        int((row.get("api_audit") or {}).get("fenced_json_turns", 0)) for row in valid
    )
    model_versions = sorted(
        {
            str(value)
            for row in valid
            for value in (row.get("api_audit") or {}).get("model_versions") or []
        }
    )
    task_rates = [row["success_rate"] for row in per_task]
    valid_counts = [row["valid"] for row in per_task]
    return {
        "version": "api-local-ab-run-v1",
        "run_date": time.strftime("%Y-%m-%d"),
        "model_requested": model,
        "base_url": base_url,
        "task_manifest_sha256": task_sha256,
        "source_runs_sha256": source_sha256,
        "tasks_requested": len(task_ids),
        "tasks_with_valid_rollouts": sum(1 for row in per_task if row["valid"]),
        "rollouts_requested": len(task_ids) * 10,
        "rollouts_valid": len(valid),
        "error_rows": len(errors),
        "errors_by_task": dict(sorted(Counter(str(row["task_id"]) for row in errors).items())),
        "equal_task_success_rate": statistics.fmean(task_rates) if task_rates else 0.0,
        "pooled_success_rate": (
            sum(row["success"] for row in per_task) / sum(valid_counts)
            if sum(valid_counts)
            else 0.0
        ),
        "stable_tasks": sum(bool(row["stable"]) for row in per_task),
        "stable_rate": (
            sum(bool(row["stable"]) for row in per_task) / len(per_task) if per_task else 0.0
        ),
        "within_steps_equal_task_rate": (
            statistics.fmean(row["within_steps"] / row["valid"] for row in per_task if row["valid"])
            if any(row["valid"] for row in per_task)
            else 0.0
        ),
        "tool_selection_equal_task_rate": (
            statistics.fmean(
                row["tool_selection_ok"] / row["valid"] for row in per_task if row["valid"]
            )
            if any(row["valid"] for row in per_task)
            else 0.0
        ),
        "api_requests": requests,
        "api_tool_call_requests": toolcall_requests,
        "api_fenced_json_turns": fenced_turns,
        "api_model_versions": model_versions,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "request_latency_median_s": _percentile(request_latencies, 0.5),
        "request_latency_p95_s": _percentile(request_latencies, 0.95),
        "rollout_latency_median_s": _percentile(rollout_latencies, 0.5),
        "rollout_latency_p95_s": _percentile(rollout_latencies, 0.95),
        "per_task": per_task,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(payload)
    return rows


def _append_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run the API arm of the API-vs-local comparison")
    parser.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS_FILE)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--tasks", default="", help="comma-separated task ids; default all 45")
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-task-sha", default=DEFAULT_TASKS_SHA256)
    return parser


async def main_async(args: argparse.Namespace) -> int:
    if args.rollouts < 1:
        raise ValueError("--rollouts must be >= 1")
    task_sha = sha256_file(args.tasks_file)
    if task_sha.lower() != args.expected_task_sha.lower():
        raise ValueError(
            f"task manifest SHA mismatch: expected {args.expected_task_sha}, got {task_sha}"
        )
    task_ids = read_task_ids(args.tasks_file)
    if args.tasks:
        wanted = [item.strip() for item in args.tasks.split(",") if item.strip()]
        unknown = sorted(set(wanted) - set(task_ids))
        if unknown:
            raise ValueError(f"unknown task ids: {unknown}")
        task_ids = wanted
    specs = load_specs(args.train_file, task_ids)

    api_key, key_env = api_key_from_env()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.out_dir / "runs.jsonl"
    existing = _load_jsonl(runs_path) if args.resume else []
    if not args.resume and runs_path.exists():
        runs_path.unlink()

    completed = {
        (str(row.get("task_id")), int(row.get("rollout_index", -1)))
        for row in existing
        if not row.get("error_kind") and not row.get("error")
    }
    jobs = [
        (task_id, rollout)
        for task_id in task_ids
        for rollout in range(args.rollouts)
        if (task_id, rollout) not in completed
    ]
    print(
        json.dumps(
            {
                "event": "start",
                "tasks": len(task_ids),
                "rollouts_per_task": args.rollouts,
                "jobs": len(jobs),
                "resume_completed": len(completed),
                "model": args.model,
                "base_url": args.base_url,
                "key_env": key_env,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def worker(task_id: str, rollout: int) -> dict[str, Any]:
        async with semaphore:
            provider = ApiUsageProvider(
                client=client,
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                top_p=args.top_p,
                max_attempts=args.max_attempts,
            )
            row = await run_once(
                specs[task_id],
                provider=provider,
                rollout_index=rollout,
                trace_dir=args.trace_dir,
                sampling_config=provider.sampling_config,
                system_variant="s0",
            )
            row["api_audit"] = provider.audit()
            row["condition"] = "api"
            async with write_lock:
                _append_row(runs_path, row)
            return row

    for finished, coro in enumerate(
        asyncio.as_completed([worker(task_id, rollout) for task_id, rollout in jobs]), start=1
    ):
        row = await coro
        print(
            f"[{finished}/{len(jobs)}] {row['task_id']}#{int(row['rollout_index']):02d} "
            f"success={row.get('task_success')} steps={row.get('steps')} "
            f"tokens={(row.get('api_audit') or {}).get('total_tokens', 0)} "
            f"error={row.get('error', '')}",
            flush=True,
        )

    close = getattr(client, "close", None)
    if close is not None:
        result = close()
        if hasattr(result, "__await__"):
            await result

    rows = _load_jsonl(runs_path)
    summary = summarize_runs(
        rows,
        task_ids=task_ids,
        model=args.model,
        base_url=args.base_url,
        task_sha256=task_sha,
        source_sha256=sha256_file(runs_path),
    )
    summary["api_key_env"] = key_env
    summary["temperature"] = args.temperature
    summary["top_p"] = args.top_p
    summary["seed"] = None
    summary["seed_note"] = "not_sent; API-side sampling variance remains"
    summary["concurrency"] = args.concurrency
    summary["system_variant"] = "s0"
    summary["tasks_file"] = str(args.tasks_file)
    summary["train_file"] = str(args.train_file)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "done", "summary": str(summary_path)}, ensure_ascii=False))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
