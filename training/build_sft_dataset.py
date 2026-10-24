"""Build a small SFT dataset from teacher trajectories on a frozen train split.

The teacher runs the same AgentLoop and judge as the evaluation harness. A
trajectory is kept only when every deterministic check passes and every tool
call/result is clean. The output keeps the raw chat ``messages``; the Qwen
chat template is applied by ``training/train_sft.py`` on Kaggle.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.reward import behavior_reward, max_consecutive_repeats
from training.sft_format import canonicalize_tool_arguments
from training.train_split import TRAIN_VERSION, TRAIN_VERSIONS, build_train_specs

DEFAULT_TEACHER_MODEL = "deepseek-flash"
TEACHER_PROTOCOL = "openai_chat_completions_tool_calls_v1"
TEACHER_CONVERTER_VERSION = "openai_and_hermes_to_qwen_chatml_v1"

JSON_CHECK_NAMES = frozenset({"json_valid", "snake_case_keys", "json_answer"})
VAL_TASK_IDS = frozenset(
    {
        "U-TR-B-C1-BASE",
        "U-TR-B-CTX-HARD",
        "U-TR-B-TOOL-HARD",
        "U-TR-B-CONFLICT-WEAK",
        "U-TR-B-CHECK-BASE",
        "U-TR-A-C1-HARD",
    }
)


@dataclass(frozen=True, slots=True)
class TraceEpisode:
    path: Path
    start: Mapping[str, Any]
    end: Mapping[str, Any]
    turns: tuple[Mapping[str, Any], ...]


def load_episode(path: Path) -> TraceEpisode | None:
    start: Mapping[str, Any] | None = None
    end: Mapping[str, Any] | None = None
    turns: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "run_start":
            start = event
        elif event.get("event") == "run_end":
            end = event
        elif event.get("event") == "llm_turn":
            turns.append(event)
    if start is None or end is None or not turns:
        return None
    return TraceEpisode(path=path, start=start, end=end, turns=tuple(turns))


def _tool_events(episode: TraceEpisode) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for turn in episode.turns:
        for call in turn.get("tool_calls") or []:
            if isinstance(call, Mapping):
                calls.append(dict(call))
        for result in turn.get("tool_results") or []:
            if isinstance(result, Mapping):
                results.append(dict(result))
    return calls, results


def deterministic_gate(episode: TraceEpisode) -> tuple[bool, str]:
    completion = _final_completion(episode)
    task_id = str(episode.start.get("task_id") or "")
    if completion is not None and completion.strip().startswith("CODE=") and "CTX" in task_id:
        return False, "code_prefix_output"
    checks = dict(episode.end.get("checks") or {})
    if checks.get("tools_allowed") is False:
        return False, "tools_not_allowed"
    if not checks or not all(bool(value) for value in checks.values()):
        return False, "checks_not_all_true"
    if episode.end.get("within_steps") is False:
        return False, "within_steps_false"
    calls, results = _tool_events(episode)
    if max_consecutive_repeats(calls) > 2:
        return False, "repeated_tool_call"
    if _tool_call_after_answer(episode):
        return False, "tool_call_after_answer"
    for call in calls:
        if call.get("arguments_json_valid") is not True:
            return False, "tool_arguments_not_json"
        if call.get("pydantic_valid") is not True:
            return False, "tool_arguments_invalid"
        if call.get("path_escape"):
            return False, "path_escape"
        if call.get("blocked_at"):
            return False, "blocked_tool_call"
    for result in results:
        if result.get("ok") is not True:
            return False, "tool_result_not_ok"
        if result.get("protocol_error"):
            return False, "protocol_error"
        if result.get("error_kind"):
            return False, "tool_error"
    return True, "ok"


def _tool_call_after_answer(episode: TraceEpisode) -> bool:
    """True when a tool call follows a final answer within the same user turn.

    A later user message starts a new request and therefore resets the answer state.
    This avoids rejecting the legitimate multi-turn CTX task, where the model answers
    an intermediate "已读" turn and the user then asks it to read more files.
    """

    for turn in episode.turns:
        answer_seen = False
        conversation = list(turn.get("messages") or ())
        response = turn.get("response") or {}
        if isinstance(response, Mapping):
            conversation.append({"role": "assistant", **response})
        for message in conversation:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role") or "")
            if role == "user":
                answer_seen = False
                continue
            if role != "assistant":
                continue
            if message.get("tool_calls"):
                if answer_seen:
                    return True
                continue
            if str(message.get("content") or "").strip():
                answer_seen = True
    return False


def a_c1_hard_regression(episode: TraceEpisode) -> tuple[bool, str]:
    """Regression gate for the task that exposed the SFT read-loop."""

    if str(episode.start.get("task_id") or "") != "U-TR-A-C1-HARD":
        return True, "not_applicable"
    calls, _results = _tool_events(episode)
    if episode.end.get("within_steps") is False:
        return False, "a_c1_hard_within_steps_false"
    if max_consecutive_repeats(calls) > 2:
        return False, "a_c1_hard_loop"
    if not _final_completion(episode):
        return False, "a_c1_hard_empty_final"
    return True, "ok"


def _final_completion(episode: TraceEpisode) -> str | None:
    """Return the last assistant text without tool calls, if any."""

    turn = next(
        (
            item
            for item in reversed(episode.turns)
            if (item.get("response") or {}).get("content")
            and not (item.get("response") or {}).get("tool_calls")
        ),
        None,
    )
    if turn is None:
        return None
    return str((turn.get("response") or {}).get("content") or "")


def _file_observations(checks: Mapping[str, Any]) -> dict[str, Any] | None:
    if "file_created_correct" not in checks and "content_exact" not in checks:
        return None
    return {"file_written_correctly": bool(checks.get("file_created_correct"))}


_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_hermes_tool_call(text: str, index: int) -> dict[str, Any] | None:
    match = _FENCED_JSON.search(text or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    function = payload.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = payload.get("name") or payload.get("tool")
        arguments = payload.get("arguments", payload.get("parameters"))
    if not isinstance(name, str) or not name:
        return None
    if isinstance(arguments, str):
        try:
            json.loads(arguments)
        except ValueError:
            return None
        arguments_text = arguments
    elif isinstance(arguments, dict):
        arguments_text = json.dumps(arguments, ensure_ascii=False)
    else:
        return None
    return {
        "id": str(payload.get("id") or f"converted-{index}"),
        "type": "function",
        "function": {"name": name, "arguments": arguments_text},
    }


def normalize_teacher_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Normalize OpenAI/Hermes teacher output to Qwen chat-template tool messages.

    Native OpenAI-style ``assistant.tool_calls`` are kept as-is. A fenced JSON tool call in
    ``content`` is converted to the native shape and paired with the following ``tool``
    message. This is conservative: only fenced objects carrying a tool name and arguments are
    converted.
    """

    normalized: list[dict[str, Any]] = []
    converted = 0
    index = 0
    while index < len(messages):
        message = dict(messages[index])
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            call = _parse_hermes_tool_call(str(message.get("content") or ""), converted)
            if call is not None:
                message["content"] = ""
                message["tool_calls"] = [call]
                normalized.append(message)
                converted += 1
                if index + 1 < len(messages) and messages[index + 1].get("role") in {
                    "tool",
                    "tool_result",
                }:
                    tool_message = dict(messages[index + 1])
                    tool_message["role"] = "tool"
                    tool_message["tool_call_id"] = call["id"]
                    normalized.append(tool_message)
                    index += 2
                    continue
                index += 1
                continue
        normalized.append(message)
        index += 1
    return normalized, converted


def validate_teacher_messages(messages: Sequence[Mapping[str, Any]]) -> None:
    """Reject traces that Qwen's chat template cannot represent as native tool calls."""

    known_ids: set[str] = set()
    seen_call_ids: set[str] = set()
    result_ids: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            content = str(message.get("content") or "")
            if "<tool_call>" in content:
                raise ValueError("assistant content contains legacy <tool_call> markup")
            for call in message.get("tool_calls") or []:
                if not isinstance(call, Mapping) or call.get("type") != "function":
                    raise ValueError("assistant tool_call is not OpenAI function style")
                call_id = str(call.get("id") or "")
                function = call.get("function")
                if not call_id or not isinstance(function, Mapping):
                    raise ValueError("assistant tool_call missing id/function")
                if call_id in seen_call_ids:
                    raise ValueError(f"duplicate assistant tool_call id: {call_id}")
                if not isinstance(function.get("name"), str) or not function.get("name"):
                    raise ValueError("assistant tool_call missing function name")
                if not isinstance(function.get("arguments"), str):
                    raise ValueError("assistant tool_call arguments must be JSON text")
                try:
                    arguments = json.loads(function["arguments"])
                except ValueError as exc:
                    raise ValueError(
                        "assistant tool_call arguments are not parseable JSON"
                    ) from exc
                if not isinstance(arguments, dict):
                    raise ValueError("assistant tool_call arguments must be a JSON object")
                seen_call_ids.add(call_id)
                known_ids.add(call_id)
        elif role in {"tool", "tool_result"}:
            call_id = str(message.get("tool_call_id") or "")
            if call_id not in known_ids:
                raise ValueError("tool message without matching assistant tool_call_id")
            if call_id in result_ids:
                raise ValueError(f"duplicate tool result for call id: {call_id}")
            result_ids.add(call_id)


def validate_qwen_render(
    messages: Sequence[Mapping[str, Any]],
    completion: str,
    tokenizer: Any,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> str | None:
    """Return a failure description when Qwen cannot render this trajectory natively."""

    rendered_messages = canonicalize_tool_arguments(messages)
    rendered_messages.append({"role": "assistant", "content": completion})
    try:
        text = tokenizer.apply_chat_template(
            rendered_messages,
            tools=list(tools or ()),
            tokenize=False,
            add_generation_prompt=False,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        return f"template_error: {type(exc).__name__}: {exc}"
    from training.check_sft_format import check_rendered_text

    tool_names = [
        str(tool.get("function", {}).get("name"))
        for tool in tools or ()
        if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
    ]
    failures = check_rendered_text(
        text,
        {"messages": list(messages)},
        # Qwen2.5's tool preamble contains one example <tool_call> block.
        extra_tool_call_blocks=1 if tool_names else 0,
    )
    if tool_names and not all(name in text for name in tool_names):
        failures.append("tool schema missing from rendered prompt")
    return "; ".join(failures) or None


def build_example(
    episode: TraceEpisode,
    *,
    tokenizer: Any = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, str, str]:
    """Build one SFT row and return an auditable rejection reason when it fails."""

    ok, reason = deterministic_gate(episode)
    if not ok:
        return None, reason, ""
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
        return None, "missing_final_answer", ""
    completion = str((final_turn.get("response") or {}).get("content") or "")
    messages, converted_tool_calls = normalize_teacher_messages(final_turn.get("messages") or [])
    try:
        validate_teacher_messages(messages)
    except ValueError as exc:
        return None, "format_invalid", str(exc)
    if not completion or not messages:
        return None, "missing_completion_or_messages", ""
    if tokenizer is not None:
        render_failure = validate_qwen_render(messages, completion, tokenizer, tools=tools)
        if render_failure:
            return None, "qwen_template_invalid", render_failure
    checks = dict(episode.end.get("checks") or {})
    calls, results = _tool_events(episode)
    file_observations = _file_observations(checks)
    expects_json = any(name in JSON_CHECK_NAMES for name in checks)
    expects_file = "file_created_correct" in checks
    within_steps = episode.end.get("within_steps") is not False
    answer_then_call = _tool_call_after_answer(episode)
    tools_allowed = checks.get("tools_allowed", True) is not False
    reward = behavior_reward(
        completion,
        tool_calls=calls,
        tool_results=results,
        file_observations=file_observations,
        expects_json=expects_json,
        expects_file=expects_file,
        within_steps=within_steps,
        answer_then_call=answer_then_call,
        tools_allowed=tools_allowed,
    )
    if not reward["hard_gate"]:
        return None, "reward_gate", ",".join(str(item) for item in reward["failures"])
    task_id = str(episode.start.get("task_id") or "")
    rollout_index = int(episode.start.get("rollout_index") or 0)
    try:
        trace_path = episode.path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        trace_path = episode.path.as_posix()
    example = {
        "id": f"train-v1:{task_id}:rollout{rollout_index}",
        "messages": messages,
        "completion": completion,
        "reward_context": {
            "tool_calls": calls,
            "tool_results": results,
            "file_observations": file_observations,
            "checks": checks,
            "expects_json": expects_json,
            "expects_file": expects_file,
            "within_steps": within_steps,
            "answer_then_call": answer_then_call,
            "tools_allowed": tools_allowed,
        },
        "metadata": {
            "task_id": task_id,
            "rollout_index": rollout_index,
            "trace_path": trace_path,
            "checks": checks,
            "reward": reward,
            "split": split_for_task(task_id),
            "teacher_protocol": TEACHER_PROTOCOL,
            "converter_version": TEACHER_CONVERTER_VERSION,
            "converted_tool_calls": converted_tool_calls,
        },
    }
    return example, "ok", ""


def trace_to_example(episode: TraceEpisode) -> dict[str, Any] | None:
    """Backward-compatible helper used by tests and callers that only need the row."""

    return build_example(episode)[0]


def filter_specs_by_task(specs: Sequence[Any], task_ids: str) -> list[Any]:
    """Keep only the requested task ids; reject unknown ids instead of silently skipping."""

    wanted = {item.strip() for item in task_ids.split(",") if item.strip()}
    if not wanted:
        return list(specs)
    selected = [spec for spec in specs if spec.task_id in wanted]
    missing = wanted - {spec.task_id for spec in selected}
    if missing:
        raise ValueError(f"unknown task ids: {sorted(missing)}")
    return selected


def count_teacher_api_calls(trace_dir: Path, allowed_task_ids: set[str]) -> int:
    """Count llm_turn events in teacher traces, i.e. actual model API requests."""

    total = 0
    for path in sorted(trace_dir.rglob("*.jsonl")):
        events = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        start = next((event for event in events if event.get("event") == "run_start"), {})
        if str(start.get("task_id") or "") not in allowed_task_ids:
            continue
        total += sum(1 for event in events if event.get("event") == "llm_turn")
    return total


def task_split(specs: Sequence[Any]) -> tuple[set[str], set[str]]:
    train = {spec.task_id for spec in specs if spec.task_id not in VAL_TASK_IDS}
    val = {spec.task_id for spec in specs if spec.task_id in VAL_TASK_IDS}
    if train & val:
        raise ValueError("train/val task overlap")
    if not train or not val:
        raise ValueError("train and val must both be non-empty")
    return train, val


def split_for_task(task_id: str) -> str:
    """Return the SFT split label; examples and teacher records must agree."""

    return "val" if task_id in VAL_TASK_IDS else "train"


def _kind(task_id: str, difficulty_by_task: Mapping[str, str]) -> str:
    return "hard" if difficulty_by_task.get(task_id) == "strong" else "base"


def _row_split(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        return str(metadata.get("split") or "")
    return str(row.get("split") or "")


def _kind_counts(
    rows: Sequence[Mapping[str, Any]],
    difficulty_by_task: Mapping[str, str],
    *,
    split: str | None = None,
) -> dict[str, int]:
    counts = {"base": 0, "hard": 0}
    for row in rows:
        if split is not None and _row_split(row) != split:
            continue
        metadata = row.get("metadata") if "metadata" in row else None
        task_id = str(
            (metadata or {}).get("task_id") if isinstance(metadata, Mapping) else row.get("task_id")
        )
        counts[_kind(task_id, difficulty_by_task)] += 1
    return counts


def _pass_rates(
    records: Sequence[Mapping[str, Any]],
    difficulty_by_task: Mapping[str, str],
    *,
    split: str | None = None,
) -> dict[str, dict[str, Any]]:
    totals = {"base": 0, "hard": 0}
    passed = {"base": 0, "hard": 0}
    for record in records:
        if split is not None and _row_split(record) != split:
            continue
        kind = _kind(str(record.get("task_id") or ""), difficulty_by_task)
        totals[kind] += 1
        passed[kind] += int(bool(record.get("task_success")))
    return {
        kind: {
            "passed": passed[kind],
            "total": totals[kind],
            "rate": round(passed[kind] / totals[kind], 4) if totals[kind] else None,
        }
        for kind in ("base", "hard")
    }


def verify_train_split(root: Path, version: str = TRAIN_VERSION) -> str:
    if version not in TRAIN_VERSIONS:
        raise ValueError(f"unknown train split version: {version}")
    train_dir = root / "private" / "unified_dataset" / version.replace("-", "_")
    frozen = train_dir / "TRAIN_FROZEN.json"
    data = train_dir / "train.jsonl"
    metadata = json.loads(frozen.read_text(encoding="utf-8"))
    recorded = metadata.get("version")
    if recorded is not None and recorded != version:
        raise ValueError(f"train split version mismatch: files say {recorded}, expected {version}")
    actual = hashlib.sha256(data.read_bytes()).hexdigest()
    if actual != metadata.get("sha256"):
        raise ValueError(f"{version} SHA mismatch: {actual} != {metadata.get('sha256')}")
    return actual


def verify_train_v1(root: Path) -> str:
    """Backward-compatible wrapper used by the original pipeline tests."""

    return verify_train_split(root, "train-v1")


def verify_no_final_test_overlap(root: Path, task_ids: set[str]) -> None:
    final_path = root / "private" / "unified_dataset" / "final_test_v2" / "final_test.jsonl"
    final_ids = {
        str(row.get("task_id"))
        for row in (
            json.loads(line)
            for line in final_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    overlap = task_ids & final_ids
    if overlap:
        raise ValueError(f"train/val overlap final-test-v2: {sorted(overlap)}")


async def run_teacher(
    specs: Sequence[Any],
    *,
    trace_dir: Path,
    rollouts: int,
    concurrency: int,
    model: str | None,
    temperature: float,
    top_p: float | None,
    seed: int,
) -> list[dict[str, Any]]:
    from agent.core.config import find_env_file, load_env_file
    from agent.core.llm import OpenAICompatProvider
    from scripts.unified_eval import run_once

    env_path = find_env_file(Path.cwd())
    if env_path is not None:
        load_env_file(env_path)
    provider = OpenAICompatProvider.from_env(
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    if model:
        provider.model = model
    semaphore = asyncio.Semaphore(concurrency)

    async def one(spec: Any, rollout: int) -> dict[str, Any]:
        async with semaphore:
            return await run_once(
                spec,
                provider=provider,
                rollout_index=rollout,
                trace_dir=trace_dir,
                sampling_config=provider.sampling_config,
                system_variant="s0",
            )

    records = list(
        await asyncio.gather(*(one(spec, rollout) for spec in specs for rollout in range(rollouts)))
    )
    for record in records:
        record["split"] = split_for_task(str(record.get("task_id") or ""))
    return records


def build_examples(
    trace_dir: Path,
    *,
    allowed_task_ids: set[str],
    rollouts: int,
    task_caps: Mapping[str, int] | None = None,
    tokenizer: Any = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    kept: dict[str, int] = {}
    caps = dict(task_caps or {task_id: 1 for task_id in allowed_task_ids})
    for path in sorted(trace_dir.rglob("*.jsonl")):
        episode = load_episode(path)
        if episode is None:
            continue
        task_id = str(episode.start.get("task_id") or "")
        if task_id not in allowed_task_ids:
            continue
        if kept.get(task_id, 0) >= caps.get(task_id, 1):
            continue
        example, reason, detail = build_example(episode, tokenizer=tokenizer, tools=tools)
        if example is None:
            rejected.append(
                {
                    "task_id": task_id,
                    "trace_path": path.as_posix(),
                    "reason": reason,
                    "detail": detail,
                }
            )
            continue
        kept[task_id] = kept.get(task_id, 0) + 1
        examples.append(example)
    return examples, rejected


def write_dataset(
    examples: Sequence[Mapping[str, Any]], out_dir: Path, manifest: Mapping[str, Any]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        rows = [row for row in examples if (row.get("metadata") or {}).get("split") == split]
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build SFT data from frozen train-split traces")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("private/sft_data"))
    parser.add_argument("--trace-dir", type=Path, default=Path("private/sft_data/traces"))
    parser.add_argument("--train-version", choices=TRAIN_VERSIONS, default=TRAIN_VERSION)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--examples-per-task", type=int, default=1)
    parser.add_argument("--hard-examples-per-task", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task-ids", default="", help="comma-separated task ids")
    parser.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument(
        "--qwen-tokenizer",
        default="",
        help="local Qwen tokenizer path; required for the real chat-template render gate",
    )
    parser.add_argument(
        "--allow-unchecked-template",
        action="store_true",
        help="explicitly allow a build without the Qwen tokenizer render gate",
    )
    return parser


async def main_async(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    teacher_model = args.model or DEFAULT_TEACHER_MODEL
    if not args.qwen_tokenizer and not args.allow_unchecked_template:
        raise SystemExit(
            "必须提供 --qwen-tokenizer 以启用 Qwen chat-template 门禁；"
            "如只做结构审计，显式加 --allow-unchecked-template"
        )
    tokenizer: Any = None
    tools: list[dict[str, Any]] | None = None
    if args.qwen_tokenizer:
        from transformers import AutoTokenizer

        from training.train_sft import default_tool_schemas

        tokenizer = AutoTokenizer.from_pretrained(args.qwen_tokenizer, trust_remote_code=True)
        tools = default_tool_schemas()
    train_sha = verify_train_split(root, args.train_version)
    specs = list(build_train_specs(args.train_version))
    difficulty_by_task = {spec.task_id: spec.difficulty for spec in specs}
    base_cap = max(1, args.examples_per_task)
    hard_cap = max(1, args.hard_examples_per_task or args.examples_per_task)
    task_caps = {
        spec.task_id: hard_cap if spec.difficulty == "strong" else base_cap for spec in specs
    }
    train_ids, val_ids = task_split(specs)
    verify_no_final_test_overlap(root, train_ids | val_ids)
    selected = [spec for spec in specs if spec.task_id in train_ids | val_ids]
    if args.task_ids:
        selected = filter_specs_by_task(selected, args.task_ids)
    if args.limit:
        selected = selected[: args.limit]
    records: list[dict[str, Any]] = []
    if not args.skip_run:
        records = await run_teacher(
            selected,
            trace_dir=args.trace_dir,
            rollouts=args.rollouts,
            concurrency=args.concurrency,
            model=teacher_model,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "teacher_runs.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
            encoding="utf-8",
        )
    elif (args.out_dir / "teacher_runs.jsonl").is_file():
        records = [
            json.loads(line)
            for line in (args.out_dir / "teacher_runs.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    examples, rejected = build_examples(
        args.trace_dir,
        allowed_task_ids=train_ids | val_ids,
        rollouts=args.rollouts,
        task_caps=task_caps,
        tokenizer=tokenizer,
        tools=tools,
    )
    rejected_reasons = Counter(str(item.get("reason") or "unknown") for item in rejected)
    manifest = {
        "version": "sft-v1",
        "teacher_model": teacher_model,
        "teacher_protocol": TEACHER_PROTOCOL,
        "converter_version": TEACHER_CONVERTER_VERSION,
        "train_version": args.train_version,
        "train_sha256": train_sha,
        "train_tasks": sorted(train_ids),
        "val_tasks": sorted(val_ids),
        "train_examples": sum(
            1 for row in examples if (row.get("metadata") or {}).get("split") == "train"
        ),
        "val_examples": sum(
            1 for row in examples if (row.get("metadata") or {}).get("split") == "val"
        ),
        "rejected": rejected,
        "rejected_reason_counts": dict(sorted(rejected_reasons.items())),
        "code_prefix_filtered": rejected_reasons.get("code_prefix_output", 0),
        "repeated_tool_call_filtered": rejected_reasons.get("repeated_tool_call", 0),
        "tool_call_after_answer_filtered": rejected_reasons.get("tool_call_after_answer", 0),
        "tools_not_allowed_filtered": rejected_reasons.get("tools_not_allowed", 0),
        "format_rejected": rejected_reasons.get("format_invalid", 0),
        "qwen_template_rejected": rejected_reasons.get("qwen_template_invalid", 0),
        "qwen_template_checked": tokenizer is not None,
        "tool_schema_count": len(tools or ()),
        "a_c1_hard_regression_filtered": rejected_reasons.get("a_c1_hard_loop", 0),
        "example_caps": {"base": base_cap, "hard": hard_cap},
        "base_hard_examples": _kind_counts(examples, difficulty_by_task),
        "base_hard_rollouts": _kind_counts(records, difficulty_by_task),
        "pass_rates": _pass_rates(records, difficulty_by_task),
        "base_hard_examples_by_split": {
            split: _kind_counts(examples, difficulty_by_task, split=split)
            for split in ("train", "val")
        },
        "base_hard_rollouts_by_split": {
            split: _kind_counts(records, difficulty_by_task, split=split)
            for split in ("train", "val")
        },
        "pass_rates_by_split": {
            split: _pass_rates(records, difficulty_by_task, split=split)
            for split in ("train", "val")
        },
        "teacher_api_calls": count_teacher_api_calls(args.trace_dir, train_ids | val_ids),
        "teacher_sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "format": "raw messages + completion; Qwen chat template applied in train_sft.py",
        "reserved_test_version": "final-test-v2",
    }
    if args.train_version == "train-v1":
        manifest["train_v1_sha256"] = train_sha
    write_dataset(examples, args.out_dir, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(main_async(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
