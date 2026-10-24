"""Annotate whether conflict probes require tool use and measure no-call collapse."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ACTION_VERBS = re.compile(
    r"读取|读一下|查看|写入|写回|保存|导出|复制|移动|追加|同步|执行|运行|跑一下|"
    r"删|移除|清空|清理|写到|修改|编辑|改|替换|重写|标记为|"
    r"\bcat\b|\brm\b|\bcp\b|\bmv\b|\bsed\b|\bpytest\b|\bpython\b",
    re.IGNORECASE,
)

CONFLICT_LAYERS = ("path", "delete", "edit")


def requires_tool(probe: str) -> bool:
    """True when the probe asks the agent to perform an action, not merely explain."""
    return bool(ACTION_VERBS.search(probe))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def call_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in load_jsonl(path):
        counts[str(row["probe_id"])] = len(row.get("tool_calls") or [])
    return counts


def annotate(
    dataset: str,
    eval_rows: list[dict[str, Any]],
    counts: dict[str, int],
) -> list[dict[str, Any]]:
    annotated = []
    for row in eval_rows:
        layer = str(row.get("layer") or "")
        if layer not in CONFLICT_LAYERS:
            continue
        probe_id = str(row["probe_id"])
        required = requires_tool(str(row.get("probe") or ""))
        calls = int(counts.get(probe_id, 0))
        annotated.append(
            {
                "dataset": dataset,
                "probe_id": probe_id,
                "task_id": row.get("task_id"),
                "layer": layer,
                "probe": row.get("probe"),
                "requires_tool": required,
                "tool_call_count": calls,
                "required_but_no_call": required and calls == 0,
            }
        )
    return annotated


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def block(items: list[dict[str, Any]]) -> dict[str, Any]:
        required = sum(1 for row in items if row["requires_tool"])
        no_call = sum(1 for row in items if row["required_but_no_call"])
        return {
            "probes": len(items),
            "requires_tool": required,
            "required_but_no_call": no_call,
            "no_call_ratio": round(no_call / required, 4) if required else None,
            "probes_with_any_call": sum(1 for row in items if row["tool_call_count"] > 0),
        }

    by_dataset = {
        dataset: block([row for row in rows if row["dataset"] == dataset])
        for dataset in sorted({row["dataset"] for row in rows})
    }
    by_dataset_layer = {
        f"{dataset}:{layer}": block(
            [row for row in rows if row["dataset"] == dataset and row["layer"] == layer]
        )
        for dataset in sorted({row["dataset"] for row in rows})
        for layer in CONFLICT_LAYERS
    }
    overall = block(rows)
    return {
        "version": "v16-collapse-diagnosis-1",
        "overall": overall,
        "by_dataset": by_dataset,
        "by_dataset_layer": by_dataset_layer,
        "requires_tool_labels": dict(Counter(str(row["requires_tool"]) for row in rows)),
        "conclusion": (
            "model_collapse" if (overall["no_call_ratio"] or 0.0) >= 0.5 else "task_design_issue"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="diagnose v16 tool-call collapse")
    parser.add_argument(
        "--v14-eval",
        type=Path,
        default=Path("private/v14/step2_benchmark/eval_set_frozen.jsonl"),
    )
    parser.add_argument("--v15-eval", type=Path, default=Path("private/v15/eval_v15/eval.jsonl"))
    parser.add_argument(
        "--v14-runs", type=Path, default=Path("private/v14/grpo_base_v14_runs.jsonl")
    )
    parser.add_argument(
        "--v15-runs", type=Path, default=Path("private/v15/grpo_base_v15_runs.jsonl")
    )
    parser.add_argument("--out", type=Path, default=Path("private/v16/collapse_annotation.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("private/v16/collapse_summary.json"))
    args = parser.parse_args()

    rows = annotate("v14", load_jsonl(args.v14_eval), call_counts(args.v14_runs)) + annotate(
        "v15", load_jsonl(args.v15_eval), call_counts(args.v15_runs)
    )
    summary = summarize(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
