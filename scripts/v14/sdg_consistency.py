"""Stratified manual-review sampling and agreement scoring for v14 SDG labels."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LABELS = (
    "explicit_compliance",
    "ambiguous_reference",
    "deferred_tool",
    "execution_violation",
    "task_priority",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def stratified(records: list[dict[str, Any]], sample: int, seed: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        buckets[rec["verifier_label"]].append(rec)
    rng = random.Random(seed)
    for items in buckets.values():
        rng.shuffle(items)
    picked: list[dict[str, Any]] = []
    keys = [key for key in LABELS if buckets.get(key)]
    index = 0
    while len(picked) < sample and keys:
        key = keys[index % len(keys)]
        if buckets[key]:
            picked.append(buckets[key].pop())
        index += 1
        if index > sample * 10:
            break
    return picked


def render(records: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# v14 步骤 1：SDG 标注人工核对表",
        "",
        "每条记录给出任务约束、用户轮次、工具调用与最终回复；请在 "
        "`review_template.jsonl` 里填 `manual_label`。",
        "",
        "可选标签：explicit_compliance / ambiguous_reference / deferred_tool / "
        "execution_violation / task_priority。",
        "",
    ]
    for rec in records:
        lines += [
            f"## {rec['sample_id']}",
            "",
            f"- verifier: `{rec['verifier_label']}`"
            f"（severity={rec['severity']}, layer={rec['category']}）",
            f"- 约束 {rec['constraint_id']}：{rec['constraint_text']}",
            f"- 安全动作：{rec['safe_action']}",
            f"- 违规捷径：{rec['shortcut']}",
            "",
            "**用户轮次**",
            "",
        ]
        for turn in rec["user_turns"]:
            lines.append(f"> {turn}")
        lines += ["", "**工具调用**", ""]
        calls = rec["trajectory"]["tool_calls"]
        if not calls:
            lines.append("（无工具调用）")
        for call in calls:
            lines.append(f"- `{call.get('name')}` {call.get('arguments')}")
        lines += ["", "**回复**", ""]
        replies = rec["trajectory"]["step_replies"] or [rec["trajectory"]["reply"]]
        for reply in replies:
            lines.append(f"> {str(reply)[:600]}")
            lines.append("")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def score(template: list[dict[str, Any]]) -> dict[str, Any]:
    filled = [row for row in template if row.get("manual_label")]
    agree = sum(1 for row in filled if row["manual_label"] == row["verifier_label"])
    confusion = Counter((row["verifier_label"], row["manual_label"]) for row in filled)
    disagreements = [
        {
            "sample_id": row["sample_id"],
            "verifier": row["verifier_label"],
            "manual": row["manual_label"],
            "note": row.get("note", ""),
        }
        for row in filled
        if row["manual_label"] != row["verifier_label"]
    ]
    return {
        "reviewed": len(filled),
        "agreement": agree,
        "agreement_rate": round(agree / len(filled), 3) if filled else None,
        "confusion": {f"{v}->{m}": n for (v, m), n in sorted(confusion.items())},
        "disagreements": disagreements,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--score", type=Path, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.score:
        result = score(load_jsonl(args.score))
        (args.out_dir / "verifier_consistency.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    records = load_jsonl(args.records)
    picked = stratified(records, args.sample, args.seed)
    render(picked, args.out_dir / "review_sheet.md")
    template = [
        {
            "sample_id": rec["sample_id"],
            "verifier_label": rec["verifier_label"],
            "manual_label": "",
            "note": "",
        }
        for rec in picked
    ]
    (args.out_dir / "review_template.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in template) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(dict(Counter(row["verifier_label"] for row in template)), ensure_ascii=False))
    print(f"sampled {len(template)} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
