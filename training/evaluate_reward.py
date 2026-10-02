"""Validate `behavior_reward` against the original trace checks.

The reward is recomputed from the stored completion and compact tool evidence, then compared
with `run_end.checks` in the source trace. This catches a reward that only looks plausible in
unit tests but disagrees with the deterministic evaluation layer.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.reward import behavior_reward


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def trace_checks(trace_path: Path) -> dict[str, Any]:
    end: dict[str, Any] | None = None
    for line in trace_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "run_end":
            end = event
    return dict((end or {}).get("checks") or {})


def compare_to_checks(result: Mapping[str, Any], checks: Mapping[str, Any]) -> dict[str, bool]:
    observed = result.get("checks") or {}
    comparisons: dict[str, bool] = {}
    if "json_valid" in checks:
        comparisons["json_parseable"] = bool(observed.get("json_parseable")) == bool(
            checks["json_valid"]
        )
    if "snake_case_keys" in checks:
        comparisons["snake_case_keys"] = bool(observed.get("snake_case_keys")) == bool(
            checks["snake_case_keys"]
        )
    if "file_written_correctly" in checks and checks["file_written_correctly"] is not None:
        comparisons["file_written_correctly"] = bool(
            observed.get("file_written_correctly")
        ) == bool(checks["file_written_correctly"])
    return comparisons


def evaluate_examples(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    comparison_totals: dict[str, int] = {}
    comparison_matches: dict[str, int] = {}
    for example in examples:
        context = example.get("reward_context") or {}
        result = behavior_reward(
            str(example.get("completion") or ""),
            tool_calls=context.get("tool_calls") or (),
            tool_results=context.get("tool_results") or (),
            file_observations=context.get("file_observations"),
            expects_json=bool(context.get("expects_json", True)),
            expects_file=bool(context.get("expects_file", False)),
        )
        trace_path = Path(str((example.get("metadata") or {}).get("trace_path") or ""))
        checks = trace_checks(trace_path) if trace_path.is_file() else {}
        comparisons = compare_to_checks(result, checks)
        for name, matched in comparisons.items():
            comparison_totals[name] = comparison_totals.get(name, 0) + 1
            comparison_matches[name] = comparison_matches.get(name, 0) + int(matched)
        rows.append(
            {
                "id": example.get("id"),
                "hard_gate": result["hard_gate"],
                "failures": result["failures"],
                "score": result["score"],
                "comparisons": comparisons,
                "mismatches": [name for name, matched in comparisons.items() if not matched],
            }
        )
    comparison_summary = {
        name: {
            "matches": comparison_matches.get(name, 0),
            "total": total,
            "rate": round(comparison_matches.get(name, 0) / total, 4) if total else None,
        }
        for name, total in sorted(comparison_totals.items())
    }
    return {
        "samples": len(rows),
        "hard_gate_pass": sum(1 for row in rows if row["hard_gate"]),
        "hard_gate_fail": sum(1 for row in rows if not row["hard_gate"]),
        "comparisons": comparison_summary,
        "mismatch_rows": [row for row in rows if row["mismatches"]],
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="验证 reward 与原始 trace 检查是否一致")
    parser.add_argument("--data-dir", default="private/training_data")
    parser.add_argument("--out", default="private/training_data/reward_validation.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    examples: list[dict[str, Any]] = []
    for name in ("train.jsonl", "val.jsonl"):
        path = data_dir / name
        if path.is_file():
            examples.extend(load_jsonl(path))
    summary = evaluate_examples(examples)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = {key: value for key, value in summary.items() if key != "rows"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0 if not summary["mismatch_rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
