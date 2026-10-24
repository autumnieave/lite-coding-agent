"""G0 多 seed 分块统计：同一批 runs.jsonl 按种子块拆开。

G0 三块互不重叠（见 docs/experiments/g0_range_g2a_prereg.md §2）：
A = 20261101–20261110 + 20261201–20261210（复现块）
B = 20261301–20261310 + 20261401–20261410
C = 20261501–20261510 + 20261601–20261610
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

BLOCKS: dict[str, tuple[range, range]] = {
    "A": (range(20261101, 20261111), range(20261201, 20261211)),
    "B": (range(20261301, 20261311), range(20261401, 20261411)),
    "C": (range(20261501, 20261511), range(20261601, 20261611)),
}


def block_of(seed: int) -> str | None:
    for name, (first, second) in BLOCKS.items():
        if seed in first or seed in second:
            return name
    return None


def summarize_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sessions = len(rows)
    failed_sessions = 0
    probes = {"path_escape": 0, "edit_tests": 0, "delete_without_confirm": 0}
    labels: dict[str, int] = {}
    format_fail = 0
    semantic = 0
    total_constraints = 0
    replenished = 0
    for row in rows:
        replenished += int(row.get("replenished", 0))
        checks = row.get("format", {}).get("checks", {})
        if any(value is False for value in checks.values()):
            format_fail += 1
        hit = False
        for name, item in row.get("violations", {}).items():
            label = str(item.get("label", ""))
            labels[label] = labels.get(label, 0) + 1
            if label == "executed_violation":
                probes[name] = probes.get(name, 0) + 1
                hit = True
        if hit:
            failed_sessions += 1
        for item in row.get("constraints", []):
            total_constraints += 1
            semantic += int(bool(item.get("semantic_present")))
    return {
        "sessions": sessions,
        "violation_sessions": failed_sessions,
        "probe_failures": probes,
        "probes_total": sessions * 3,
        "labels": dict(sorted(labels.items())),
        "format_failed_sessions": format_fail,
        "semantic_present": semantic,
        "constraints_total": total_constraints,
        "replenished": replenished,
    }


def summarize_by_block(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {name: [] for name in (*BLOCKS, "unassigned")}
    for row in rows:
        grouped[block_of(int(row["seed"])) or "unassigned"].append(row)
    blocks = {name: summarize_block(subset) for name, subset in grouped.items() if subset}
    totals = [item["probe_failures"] for item in blocks.values()]
    per_block = []
    for item in totals:
        per_block.append(sum(item.values()))
    result: dict[str, Any] = {"blocks": blocks}
    if per_block:
        result["probe_violations_by_block"] = per_block
        result["min"] = min(per_block)
        result["max"] = max(per_block)
        result["mean"] = round(sum(per_block) / len(per_block), 2)
    return result


def render(summary: Mapping[str, Any]) -> str:
    lines = [
        "# G0 多 seed 越界范围",
        "",
        "| 块 | 会话 | 会话级越界 | 探针级越界 | path_escape | edit_tests | "
        "delete_without_confirm | 格式失败 | 语义保留 | 补录 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in summary["blocks"].items():
        probes = block["probe_failures"]
        lines.append(
            f"| {name} | {block['sessions']} | {block['violation_sessions']} | "
            f"{sum(probes.values())}/{block['probes_total']} | "
            f"{probes['path_escape']} | {probes['edit_tests']} | "
            f"{probes['delete_without_confirm']} | {block['format_failed_sessions']} | "
            f"{block['semantic_present']}/{block['constraints_total']} | "
            f"{block['replenished']} |"
        )
    if "mean" in summary:
        denominator = next(iter(summary["blocks"].values()))["probes_total"]
        lines += [
            "",
            f"三块探针级越界：min {summary['min']} / max {summary['max']} / "
            f"mean {summary['mean']}（分母 {denominator}）。",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Split G0 rows by seed block")
    parser.add_argument("runs", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.runs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize_by_block(rows)
    if args.json_out:
        args.json_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    text = render(summary)
    if args.md_out:
        args.md_out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
