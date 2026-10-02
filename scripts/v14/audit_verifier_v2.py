"""Audit a labeller against the frozen 40-sample manual review sheet."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_v2 import classify  # noqa: E402

from training.v10_conflict import V10ConflictTask, load_tasks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    tasks: dict[str, V10ConflictTask] = load_tasks(args.tasks)
    review = {}
    for line in args.review.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            review[row["sample_id"]] = row

    rows: list[dict[str, Any]] = []
    for line in args.runs.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        sample_id = f"{run['task_id']}#{run.get('rollout_index')}"
        task = tasks.get(str(run["task_id"]))
        if task is None:
            continue
        label = classify(
            task,
            run.get("step_replies") or [run.get("reply") or ""],
            run.get("tool_calls") or [],
        )
        rows.append(
            {
                "sample_id": sample_id,
                "v1": (run.get("v10") or {}).get("classification"),
                "v2": label["label"],
                "v2_detail": label["detail"],
            }
        )
    by_id = {row["sample_id"]: row for row in rows}

    confusion: Counter[tuple[str, str]] = Counter()
    per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    v1_agree = v2_agree = 0
    residual: list[dict[str, str]] = []
    for sample_id, ref in review.items():
        gold = ref["manual_label"]
        row = by_id.get(sample_id)
        if row is None:
            continue
        v1_agree += int(row["v1"] == gold)
        v2_agree += int(row["v2"] == gold)
        confusion[(row["v2"], gold)] += 1
        per_class[gold]["n"] += 1
        per_class[gold]["correct"] += int(row["v2"] == gold)
        if row["v2"] != gold:
            residual.append(
                {
                    "sample_id": sample_id,
                    "manual": gold,
                    "v1": str(row["v1"]),
                    "v2": row["v2"],
                    "v2_detail": str(row["v2_detail"]),
                    "note": ref.get("note", ""),
                }
            )

    n = len(review)
    result = {
        "reviewed": n,
        "v1_agree": v1_agree,
        "v1_rate": round(v1_agree / n, 4) if n else 0.0,
        "v2_agree": v2_agree,
        "v2_rate": round(v2_agree / n, 4) if n else 0.0,
        "per_class": {
            key: {"correct": val["correct"], "n": val["n"]}
            for key, val in sorted(per_class.items())
        },
        "confusion": {f"{pred}->{gold}": cnt for (pred, gold), cnt in sorted(confusion.items())},
        "residual": residual,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    lines = [
        "# v14 步骤 1：verifier 标注一致率（40 条分层抽检）",
        "",
        f"- 抽检条数：{n}（步骤 0 的 480 条 rollout 上分层抽取）",
        "- 原始 verifier（runs.jsonl 里的 `v10.classification`）："
        f"{v1_agree}/{n} = {v1_agree / n:.1%}",
        f"- 规则更强的 v2 判定器（`scripts/v14/label_v2.py`）：{v2_agree}/{n} = {v2_agree / n:.1%}",
        "",
        "## 逐类召回（v2 vs 人工）",
        "",
        "| 人工标签 | v2 正确 / 总数 |",
        "| --- | --- |",
    ]
    for key, val in sorted(per_class.items()):
        lines.append(f"| {key} | {val['correct']}/{val['n']} |")
    lines += [
        "",
        "## 混淆矩阵（v2 → 人工）",
        "",
        "| v2 判定 | 人工标签 | 条数 |",
        "| --- | --- | --- |",
    ]
    for (pred, gold), cnt in sorted(confusion.items()):
        lines.append(f"| {pred} | {gold} | {cnt} |")
    lines += [
        "",
        "## 残余分歧逐条",
        "",
        "| 样本 | 人工 | v1 | v2 | 说明 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in residual:
        lines.append(
            f"| {item['sample_id']} | {item['manual']} | {item['v1']} | {item['v2']} "
            f"({item['v2_detail']}) | {item['note']} |"
        )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "residual"}, ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
