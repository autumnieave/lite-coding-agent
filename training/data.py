"""从已有教师 trace 构造 SFT/GRPO 小数据集，并保持 train/val 按 run 隔离。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.reward import behavior_reward

PROBE_LABELS = ("probe_json", "probe")


@dataclass(frozen=True, slots=True)
class Episode:
    path: Path
    start: Mapping[str, Any]
    end: Mapping[str, Any]
    turns: tuple[Mapping[str, Any], ...]


def load_episode(path: Path) -> Episode | None:
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
    if start is None or end is None:
        return None
    return Episode(path=path, start=start, end=end, turns=tuple(turns))


def final_turn(episode: Episode) -> Mapping[str, Any] | None:
    for label in PROBE_LABELS:
        matches = [turn for turn in episode.turns if turn.get("step_label") == label]
        if matches:
            return matches[-1]
    return None


def _constraints(start: Mapping[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in start.get("constraints") or []:
        if isinstance(item, Mapping):
            cid = str(item.get("id") or item.get("constraint_id") or "")
            content = str(item.get("content") or "")
            if cid and content:
                result.append({"id": cid, "content": content})
    return result


def _base_system(turn: Mapping[str, Any]) -> str:
    for message in turn.get("messages") or []:
        if message.get("role") == "system":
            content = str(message.get("content") or "")
            if content and not content.startswith("[历史对话摘要]"):
                return content
    return "你是一个受约束的终端 coding agent。"


def _user_prompt(turn: Mapping[str, Any]) -> str:
    for message in reversed(turn.get("messages") or []):
        if message.get("role") == "user" and message.get("content"):
            return str(message["content"])
    return "完成当前任务，并严格遵守约束。"


def _compact_tools(episode: Episode) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for turn in episode.turns:
        for call in turn.get("tool_calls") or []:
            if isinstance(call, Mapping):
                calls.append(
                    {
                        "tool_name": call.get("tool_name"),
                        "arguments_json_valid": call.get("arguments_json_valid"),
                        "pydantic_valid": call.get("pydantic_valid"),
                        "path_escape": call.get("path_escape"),
                        "blocked_at": call.get("blocked_at"),
                    }
                )
        for result in turn.get("tool_results") or []:
            if isinstance(result, Mapping):
                results.append(
                    {
                        "tool_name": result.get("tool_name"),
                        "ok": result.get("ok"),
                        "error_kind": result.get("error_kind"),
                        "protocol_error": result.get("protocol_error"),
                        "validation_stage": result.get("validation_stage"),
                    }
                )
    return calls, results


def _file_observations(checks: Mapping[str, Any]) -> dict[str, Any] | None:
    if "lowercase_name" not in checks and "code_in_new_file" not in checks:
        return None
    name_ok = bool(checks.get("lowercase_name"))
    content_ok = bool(checks.get("code_in_new_file"))
    return {
        "file_name_lower": name_ok,
        "content_has_required_marker": content_ok,
        "file_written_correctly": name_ok and content_ok,
    }


def build_example(episode: Episode) -> dict[str, Any] | None:
    turn = final_turn(episode)
    if turn is None:
        return None
    response = turn.get("response") or {}
    completion = str(response.get("content") or "")
    if not completion:
        return None
    start = episode.start
    end = episode.end
    task_id = str(start.get("task_id") or "")
    run_id = int(start.get("run_id") or 0)
    rollout_index = int(start.get("rollout_index") or 0)
    constraints = _constraints(start)
    constraint_lines = "\n".join(f"- [{item['id']}] {item['content']}" for item in constraints)
    system = _base_system(turn) + "\n\n关键约束：\n" + constraint_lines
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": _user_prompt(turn)},
    ]
    checks = dict(end.get("checks") or {})
    calls, results = _compact_tools(episode)
    file_observations = _file_observations(checks)
    expects_json = "json_valid" in checks or "snake_case_keys" in checks
    expects_file = file_observations is not None
    reward = behavior_reward(
        completion,
        tool_calls=calls,
        tool_results=results,
        file_observations=file_observations,
        expects_json=expects_json,
        expects_file=expects_file,
    )
    return {
        "id": f"{start.get('experiment')}:{task_id}:run{run_id}:rollout{rollout_index}",
        "messages": messages,
        "completion": completion,
        "expects_json": expects_json,
        "expects_file": expects_file,
        "reward_context": {
            "tool_calls": calls,
            "tool_results": results,
            "file_observations": file_observations,
            "checks": checks,
            "expects_json": expects_json,
            "expects_file": expects_file,
        },
        "metadata": {
            "task_id": task_id,
            "run_id": run_id,
            "rollout_index": rollout_index,
            "group": start.get("group"),
            "experiment": start.get("experiment"),
            "trace_path": episode.path.as_posix(),
            "constraint_ids": [item["id"] for item in constraints],
            "reward": reward,
        },
    }


def split_for(task_id: str, run_id: int) -> str:
    if task_id == "stress":
        return "val" if run_id == 10 else "train"
    if task_id == "C4":
        return "val" if run_id in {4, 5} else "train"
    return "val" if run_id % 2 == 0 else "train"


def collect_examples(
    trace_root: Path,
    *,
    stress_runs: set[int],
    c4_runs: set[int],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int, int]] = set()
    for path in sorted(trace_root.rglob("*.jsonl")):
        episode = load_episode(path)
        if episode is None:
            continue
        start = episode.start
        key = (
            str(start.get("experiment") or ""),
            str(start.get("task_id") or ""),
            str(start.get("group") or ""),
            int(start.get("run_id") or 0),
            int(start.get("rollout_index") or 0),
        )
        if key in seen:
            continue
        seen.add(key)
        task_id = str(start.get("task_id") or "")
        run_id = int(start.get("run_id") or 0)
        group = str(start.get("group") or "")
        if (
            task_id == "stress"
            and run_id in stress_runs
            and group == "on"
            or task_id == "C4"
            and run_id in c4_runs
            and group == "off"
        ):
            example = build_example(episode)
        else:
            continue
        if example is not None:
            examples.append(example)
    return examples


def write_dataset(examples: Sequence[Mapping[str, Any]], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = [
        item
        for item in examples
        if split_for(str(item["metadata"]["task_id"]), int(item["metadata"]["run_id"])) == "train"
    ]
    val = [
        item
        for item in examples
        if split_for(str(item["metadata"]["task_id"]), int(item["metadata"]["run_id"])) == "val"
    ]
    train_keys = {(item["metadata"]["task_id"], item["metadata"]["run_id"]) for item in train}
    val_keys = {(item["metadata"]["task_id"], item["metadata"]["run_id"]) for item in val}
    if train_keys & val_keys:
        raise ValueError("train/val run keys overlap")
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with (out_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "total": len(examples),
        "train": len(train),
        "val": len(val),
        "train_run_keys": sorted([list(item) for item in train_keys]),
        "val_run_keys": sorted([list(item) for item in val_keys]),
        "split_rule": "stress run10=val; C4 run4/5=val; no run key crosses split",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构造训练/验证 JSONL")
    parser.add_argument("--trace-root", default="private/traces")
    parser.add_argument("--out-dir", default="private/training_data")
    parser.add_argument("--stress-runs", default="2,10")
    parser.add_argument("--c4-runs", default="1,2,3,4,5")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stress_runs = {int(item) for item in args.stress_runs.split(",") if item}
    c4_runs = {int(item) for item in args.c4_runs.split(",") if item}
    examples = collect_examples(Path(args.trace_root), stress_runs=stress_runs, c4_runs=c4_runs)
    manifest = write_dataset(examples, Path(args.out_dir))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
