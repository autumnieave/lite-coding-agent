"""Offline calibration of the v14 conflict reward on the SDG trajectory set.

Answers two questions before spending GPU time on GRPO:

1. Does the reward separate compliant from violating trajectories?
2. How many tasks have reward variance inside a rollout group (GRPO signal)?
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.reward_v14 import score_reply  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def task_dict(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": record["task_id"],
        "source": record["source"],
        "source_ref": record["source_ref"],
        "category": record["category"],
        "severity": record["severity"],
        "pressure_type": record["pressure_type"],
        "constraint_id": record["constraint_id"],
        "constraint_text": record["constraint_text"],
        "user_turns": record["user_turns"],
        "required_tools": [],
        "allowed_tools": [],
        "steps_limit": 6,
        "fixture_seed": 1,
        "fixture_files": [],
        "forbidden_paths": [],
        "forbidden_commands": [],
        "forbidden_tools": [],
        "safe_terms": [],
        "shortcut_terms": [],
        "intent_keywords": [],
        "safe_action": record.get("safe_action") or "",
        "shortcut": record.get("shortcut") or "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-sdg", type=Path, default=Path("private/v14/step1_sdg/train_sdg.jsonl")
    )
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    records = read_jsonl(args.train_sdg)
    per_label: dict[str, list[float]] = defaultdict(list)
    per_task: dict[str, list[float]] = defaultdict(list)
    per_task_text: dict[str, list[float]] = defaultdict(list)
    buckets = Counter()
    for record in records:
        task = task_dict(record)
        reply = record["trajectory"].get("reply") or ""
        calls = record["trajectory"].get("tool_calls") or []
        result = score_reply(task, reply, calls)
        text_result = score_reply(task, reply, [], from_text=True)
        per_label[record["verifier_label"]].append(result["reward"])
        per_task[record["task_id"]].append(result["reward"])
        per_task_text[record["task_id"]].append(text_result["reward"])
        buckets[result["bucket"]] += 1

    def spread(values: list[float]) -> dict[str, float]:
        return {
            "mean": round(sum(values) / len(values), 3) if values else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }

    groups = {task_id: spread(values) for task_id, values in sorted(per_task.items())}
    groups_text = {task_id: spread(values) for task_id, values in sorted(per_task_text.items())}
    with_variance = [tid for tid, stat in groups.items() if stat["min"] < stat["max"]]
    with_variance_text = [tid for tid, stat in groups_text.items() if stat["min"] < stat["max"]]
    report = {
        "records": len(records),
        "buckets": dict(buckets),
        "reward_by_verifier_label": {
            label: spread(values) for label, values in sorted(per_label.items())
        },
        "tasks": len(per_task),
        "tasks_with_reward_variance": len(with_variance),
        "tasks_with_reward_variance_text_only": len(with_variance_text),
        "mixed_tasks": sorted(with_variance),
        "per_task": groups,
        "per_task_text_only": groups_text,
    }
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"per_task", "per_task_text_only"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
