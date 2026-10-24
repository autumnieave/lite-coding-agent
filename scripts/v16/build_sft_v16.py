"""Build v16 SFT data that restores tool calls while keeping compliant controls."""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REFUSAL = re.compile(
    r"不能|不会|无法|拒绝|不允许|需要.{0,8}确认|请.{0,8}确认|建议.{0,4}(确认|保留)|未经.{0,8}确认"
)
COMPLETED = re.compile(r"已经完成|已完成|已成功|成功删除|成功写入|已将|已更新")
TOOL_CALL_BLOCK = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.DOTALL)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def infer_task_layer(task_id: str) -> str:
    if "PATH" in task_id:
        return "path"
    if "DEL" in task_id:
        return "delete"
    if "EDIT" in task_id:
        return "edit"
    return "policy"


def final_reply(trace_end: dict[str, Any]) -> str:
    replies = trace_end.get("step_replies") or {}
    if isinstance(replies, dict):
        if not replies:
            return ""
        return str(replies[sorted(replies)[-1]])
    if isinstance(replies, list):
        return str(replies[-1]) if replies else ""
    return str(replies)


def _assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    content = TOOL_CALL_BLOCK.sub("", str(response.get("content") or "")).strip()
    message: dict[str, Any] = {"role": "assistant", "content": content}
    calls = response.get("tool_calls") or []
    if calls:
        message["tool_calls"] = [
            {
                "id": str(call.get("id") or f"call_{index}"),
                "type": "function",
                "function": {
                    "name": str(call.get("name") or ""),
                    "arguments": str(call.get("arguments") or "{}"),
                },
            }
            for index, call in enumerate(calls)
        ]
    return message


def trace_to_messages(path: Path) -> list[dict[str, Any]]:
    events = load_jsonl(path)
    turns = [event for event in events if event.get("event") == "llm_turn"]
    if not turns:
        raise ValueError(f"trace has no llm_turn: {path}")
    messages = [dict(message) for message in turns[0]["messages"]]
    for turn in turns[:-1]:
        assistant = _assistant_message(turn.get("response") or {})
        messages.append(assistant)
        for result in turn.get("tool_results") or []:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(result.get("call_id") or ""),
                    "content": str(result.get("content") or ""),
                }
            )
    last = turns[-1]
    response = last.get("response") or {}
    if response.get("tool_calls"):
        messages.append(_assistant_message(response))
        for result in last.get("tool_results") or []:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(result.get("call_id") or ""),
                    "content": str(result.get("content") or ""),
                }
            )
    messages.append({"role": "assistant", "content": final_reply(events[-1])})
    for message in messages:
        if message.get("role") == "assistant":
            message["content"] = TOOL_CALL_BLOCK.sub("", str(message.get("content") or "")).strip()
    return messages


def normal_candidates(runs: list[dict[str, Any]], eval_tasks: set[str]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        if not row.get("task_success") or not row.get("tools_used"):
            continue
        if int(row.get("violated") or 0) != 0:
            continue
        if str(row.get("task_id")) in eval_tasks:
            continue
        if not Path(str(row.get("trace_path"))).exists():
            continue
        by_task[str(row["task_id"])].append(row)
    selected = []
    for task_id in sorted(by_task):
        rows = sorted(by_task[task_id], key=lambda row: int(row.get("rollout_index") or 0))
        selected.extend(rows[:5])
    return selected


def conflict_candidates(run_paths: list[Path], eval_tasks: set[str]) -> list[dict[str, Any]]:
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for path in run_paths:
        try:
            rows = load_jsonl(path)
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            v10 = row.get("v10") or {}
            if v10.get("classification") != "explicit_compliance":
                continue
            if v10.get("execution_violation"):
                continue
            if not row.get("tool_calls"):
                continue
            if not row.get("trace_path") or not Path(str(row["trace_path"])).exists():
                continue
            if str(row.get("task_id")) in eval_tasks:
                continue
            replies = row.get("step_replies") or []
            last = str(replies[-1]) if replies else ""
            if not REFUSAL.search(last) or COMPLETED.search(last):
                continue
            found[(str(row["task_id"]), int(row.get("rollout_index") or 0))] = row
    return [found[key] for key in sorted(found)]


def select_conflict(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_layer[infer_task_layer(str(row["task_id"]))].append(row)
    for layer_rows in by_layer.values():
        layer_rows.sort(key=lambda row: (-len(row.get("tool_calls") or []), str(row["task_id"])))
    targets = {"path": 52, "delete": 6, "edit": 0, "policy": 62}
    selected: list[dict[str, Any]] = []
    for layer, target in targets.items():
        selected.extend(by_layer.get(layer, [])[:target])
    if len(selected) < count:
        used = {(str(row["task_id"]), int(row.get("rollout_index") or 0)) for row in selected}
        for row in rows:
            key = (str(row["task_id"]), int(row.get("rollout_index") or 0))
            if key not in used:
                selected.append(row)
                used.add(key)
            if len(selected) >= count:
                break
    return selected[:count]


def select_controls(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    controls = []
    for row in rows:
        messages = row.get("messages") or []
        if any(message.get("tool_calls") for message in messages):
            continue
        completion = str(row.get("completion") or "")
        if not REFUSAL.search(completion):
            continue
        controls.append(row)
    return controls[:count]


def build(
    *,
    normal_runs: Path,
    conflict_glob: str,
    controls_path: Path,
    eval_files: list[Path],
    normal_count: int,
    conflict_count: int,
    control_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del seed  # selection is deterministic; reserved for future sampling variants
    eval_tasks = {str(row["task_id"]) for path in eval_files for row in load_jsonl(path)}
    normal = normal_candidates(load_jsonl(normal_runs), eval_tasks)[:normal_count]
    conflict = select_conflict(
        conflict_candidates(
            [Path(path) for path in sorted(glob.glob(conflict_glob, recursive=True))], eval_tasks
        ),
        conflict_count,
    )
    controls = select_controls(load_jsonl(controls_path), control_count)

    records: list[dict[str, Any]] = []
    for row in normal:
        messages = trace_to_messages(Path(str(row["trace_path"])))
        completion = str(messages[-1].get("content") or "")
        records.append(
            {
                "id": f"v16:normal:{row['task_id']}:{row.get('rollout_index')}",
                "messages": messages[:-1],
                "completion": completion,
                "metadata": {
                    "v16_role": "normal_tool_use",
                    "task_id": row["task_id"],
                    "rollout_index": row.get("rollout_index"),
                    "tools_used": row.get("tools_used") or [],
                    "steps": row.get("steps"),
                    "trace_path": row.get("trace_path"),
                },
            }
        )
    for row in conflict:
        messages = trace_to_messages(Path(str(row["trace_path"])))
        completion = str(messages[-1].get("content") or "")
        records.append(
            {
                "id": f"v16:conflict:{row['task_id']}:{row.get('rollout_index')}",
                "messages": messages[:-1],
                "completion": completion,
                "metadata": {
                    "v16_role": "conflict_tool_then_compliance",
                    "task_id": row["task_id"],
                    "rollout_index": row.get("rollout_index"),
                    "layer": infer_task_layer(str(row["task_id"])),
                    "tools_used": [call.get("name") for call in row.get("tool_calls") or []],
                    "steps": row.get("steps"),
                    "trace_path": row.get("trace_path"),
                },
            }
        )
    for row in controls:
        records.append(
            {
                "id": f"v16:control:{row['id']}",
                "messages": [
                    *row["messages"],
                ],
                "completion": str(row.get("completion") or ""),
                "metadata": {
                    "v16_role": "refusal_without_tool",
                    "source_id": row["id"],
                },
            }
        )

    roles = Counter(record["metadata"]["v16_role"] for record in records)
    tools = sum(
        1 for record in records if any(message.get("tool_calls") for message in record["messages"])
    )
    manifest = {
        "version": "v16-sft-1",
        "records": len(records),
        "roles": dict(roles),
        "records_with_tool_calls": tools,
        "tool_call_ratio": round(tools / len(records), 4) if records else 0.0,
        "normal_tasks": len(
            {
                record["metadata"].get("task_id")
                for record in records
                if record["metadata"]["v16_role"] == "normal_tool_use"
            }
        ),
        "conflict_tasks": len(
            {
                record["metadata"].get("task_id")
                for record in records
                if record["metadata"]["v16_role"] == "conflict_tool_then_compliance"
            }
        ),
        "conflict_layers": dict(
            Counter(
                record["metadata"]["layer"]
                for record in records
                if record["metadata"]["v16_role"] == "conflict_tool_then_compliance"
            )
        ),
        "eval_task_overlap": sorted(
            {
                str(record["metadata"].get("task_id"))
                for record in records
                if record["metadata"].get("task_id") in eval_tasks
            }
        ),
    }
    return records, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="build v16 SFT data")
    parser.add_argument(
        "--normal-runs",
        type=Path,
        default=Path("private/cross_template/p0_task_cost/v8/sft/runs.jsonl"),
    )
    parser.add_argument("--conflict-glob", default="private/**/runs.jsonl")
    parser.add_argument(
        "--controls", type=Path, default=Path("private/sft_data_v8_paired/train.jsonl")
    )
    parser.add_argument(
        "--v14-eval",
        type=Path,
        default=Path("private/v14/step2_benchmark/eval_set_frozen.jsonl"),
    )
    parser.add_argument("--v15-eval", type=Path, default=Path("private/v15/eval_v15/eval.jsonl"))
    parser.add_argument("--normal-count", type=int, default=180)
    parser.add_argument("--conflict-count", type=int, default=120)
    parser.add_argument("--control-count", type=int, default=31)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--out", type=Path, default=Path("private/v16/sft_v16/train.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("private/v16/sft_v16/manifest.json"))
    args = parser.parse_args()
    records, manifest = build(
        normal_runs=args.normal_runs,
        conflict_glob=args.conflict_glob,
        controls_path=args.controls,
        eval_files=[args.v14_eval, args.v15_eval],
        normal_count=args.normal_count,
        conflict_count=args.conflict_count,
        control_count=args.control_count,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
