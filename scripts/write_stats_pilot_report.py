"""Render the stage-1.5b pilot report from a Kaggle pilot summary."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _task_map(condition: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["task_id"]: row for row in condition.get("tasks", [])}


def _pct(value: float) -> str:
    return f"{value:.0%}"


def decide(summary: dict[str, Any]) -> tuple[str, list[str]]:
    conditions = summary.get("conditions", {})
    if set(conditions) != {"baseline", "sft"}:
        return "未判定", ["缺少 baseline 或 SFT 条件，不能判定。"]
    if any(condition.get("status") != "completed" for condition in conditions.values()):
        return "未判定", ["存在 harness error 或 rollout 不完整；先修复，再判饱和。"]
    baseline = _task_map(conditions["baseline"])
    sft = _task_map(conditions["sft"])
    task_ids = summary.get("tasks", [])
    sft_rates = [float(sft[task_id]["pass_rate"]) for task_id in task_ids]
    baseline_rates = [float(baseline[task_id]["pass_rate"]) for task_id in task_ids]
    zero_below_90 = sum(rate < 0.9 for rate in sft_rates)
    all_perfect = all(rate == 1.0 for rate in sft_rates)
    both_nine_of_ten = sum(
        base >= 0.9 and trained >= 0.9
        for base, trained in zip(baseline_rates, sft_rates, strict=True)
    )
    notes = [
        f"SFT 10/10 的任务数：{sum(rate == 1.0 for rate in sft_rates)}/{len(sft_rates)}。",
        f"SFT 通过率 <90% 的任务数：{zero_below_90}/{len(sft_rates)}。",
        f"baseline 与 SFT 都 ≥90% 的任务数：{both_nine_of_ten}/{len(sft_rates)}。",
    ]
    if all(rate == 0.0 for rate in baseline_rates):
        notes.append(
            "baseline 在全部 pilot 任务上为 0/10，存在地板效应；"
            "pilot 可以判定“未饱和”，但不能估计总体 baseline 方差或支持广泛训练增益。"
        )
    positive_sft = [task_id for task_id in task_ids if float(sft[task_id]["pass_rate"]) > 0]
    if len(positive_sft) == 1:
        notes.append(f"SFT 唯一有通过样本的任务是 `{positive_sft[0]}`；效应不能外推到其他族。")
    if all_perfect:
        notes.append("用户本轮门禁：5/5 任务 SFT 均为 10/10，判定饱和；只允许加难一轮。")
        return "加难（回 1.5a）", notes
    if zero_below_90 >= 3:
        notes.append("用户本轮门禁：至少 3 个任务 SFT <90%，判定不饱和；进入 1.5c。")
        return "进入 1.5c 功效分析", notes
    notes.append("本轮结果落在两条门禁之间的灰区；不自行解释为显著，报告后等确认。")
    return "灰区（报告等确认）", notes


def _failed_checks(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    failures: Counter[str] = Counter()
    for row in rows:
        for name, passed in (row.get("checks") or {}).items():
            if not passed:
                failures[name] += 1
    return failures


def build_document(
    summary: dict[str, Any],
    *,
    kernel: str,
    adapter_sha: str,
    runs_by_condition: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    staging: dict[str, Any] | None = None,
) -> str:
    conditions = summary.get("conditions", {})
    baseline = _task_map(conditions.get("baseline", {}))
    sft = _task_map(conditions.get("sft", {}))
    decision, notes = decide(summary)
    lines = [
        "# 阶段 1.5b Pilot 评测",
        "",
        f"- Kaggle kernel：`{kernel}`",
        f"- train-v5 adapter SHA-256：`{adapter_sha}`",
        "- 模型：`Qwen/Qwen2.5-7B-Instruct`，NF4，transformers 离线 provider（不起 vLLM server）",
        "- 采样：`temperature=0.0`、`top_p=None`、`seed=20261019`",
        "- 规模：5 个任务 × 10 rollout × baseline/SFT = 100 条",
        "- 方差口径：每个任务 10 个二元结果的样本方差；另列 Bernoulli `p(1-p)`。",
    ]
    if staging:
        staging_conditions = list(staging.get("conditions", {}).values())
        staging_errors = sum(int(row.get("errors") or 0) for row in staging_conditions)
        staging_success = sum(int(row.get("task_success") or 0) for row in staging_conditions)
        lines.append(
            f"- staging：5 任务 × 1 rollout × 2 组，0 error={staging_errors == 0}，"
            f"合计 success={staging_success}；通过后跑 full。"
        )
    lines.extend(
        [
            "",
            "## 逐任务结果",
            "",
            "| 任务 | baseline 通过 | baseline 率 | baseline 方差 | SFT 通过 | SFT 率 | "
            "SFT 方差 | 变化 | stable(≥6/10) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for task_id in summary.get("tasks", []):
        base = baseline.get(task_id, {})
        trained = sft.get(task_id, {})
        base_rate = float(base.get("pass_rate", 0.0))
        sft_rate = float(trained.get("pass_rate", 0.0))
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{task_id}`",
                    f"{base.get('task_success', 0)}/{base.get('valid_rollouts', 0)}",
                    _pct(base_rate),
                    f"{base.get('variance', 0.0):.6f}",
                    f"{trained.get('task_success', 0)}/{trained.get('valid_rollouts', 0)}",
                    _pct(sft_rate),
                    f"{trained.get('variance', 0.0):.6f}",
                    _pct(sft_rate - base_rate),
                    f"baseline={base.get('task_success', 0) >= 6}, "
                    f"SFT={trained.get('task_success', 0) >= 6}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 条件汇总",
            "",
            "| 条件 | 状态 | 有效 rollout | error | task_success | 通过率 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ("baseline", "sft"):
        row = conditions.get(name, {})
        lines.append(
            f"| {name} | {row.get('status', 'missing')} | {row.get('rollouts', 0)} | "
            f"{row.get('errors', 0)} | {row.get('task_success', 0)} | "
            f"{_pct(float(row.get('pass_rate', 0.0)))} |"
        )
    if runs_by_condition:
        lines.extend(
            [
                "",
                "## 失败检查项计数",
                "",
                "| 任务 | baseline 失败检查（10 次中） | SFT 失败检查（10 次中） |",
                "| --- | --- | --- |",
            ]
        )
        for task_id in summary.get("tasks", []):
            cells: list[str] = []
            for condition_name in ("baseline", "sft"):
                rows = [
                    row
                    for row in runs_by_condition[condition_name]
                    if row.get("task_id") == task_id
                ]
                failures = _failed_checks(rows)
                cells.append(
                    ", ".join(f"{name}={count}" for name, count in failures.most_common())
                    if failures
                    else "无"
                )
            lines.append(f"| `{task_id}` | {cells[0]} | {cells[1]} |")
    lines.extend(["", "## 判定", "", f"**{decision}**", ""])
    lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 本报告只写实际 Kaggle rollout；没有跑成功的条件不写通过。",
            "- stable 只在每个任务 10 次 rollout 内计算，不替代 1.5c 的样本量/功效分析。",
            "- 5 个 pilot 任务覆盖 5 个工作流族；结果不外推到全部 15 个任务。",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    conditions = summary.get("conditions", {})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "condition",
                "task_id",
                "family",
                "difficulty",
                "expected_rollouts",
                "valid_rollouts",
                "errors",
                "task_success",
                "pass_rate",
                "variance",
                "bernoulli_variance",
                "within_steps_failures",
                "stable_pass_6_of_10",
            ),
        )
        writer.writeheader()
        for condition_name in ("baseline", "sft"):
            for row in conditions.get(condition_name, {}).get("tasks", []):
                writer.writerow(
                    {
                        "condition": condition_name,
                        "task_id": row.get("task_id", ""),
                        "family": row.get("family", ""),
                        "difficulty": row.get("difficulty", ""),
                        "expected_rollouts": row.get("expected_rollouts", 0),
                        "valid_rollouts": row.get("valid_rollouts", 0),
                        "errors": row.get("errors", 0),
                        "task_success": row.get("task_success", 0),
                        "pass_rate": row.get("pass_rate", 0.0),
                        "variance": row.get("variance", 0.0),
                        "bernoulli_variance": row.get("bernoulli_variance", 0.0),
                        "within_steps_failures": row.get("within_steps_failures", 0),
                        "stable_pass_6_of_10": int(row.get("task_success", 0)) >= 6,
                    }
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="render stage-1.5b pilot markdown")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--staging-summary", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/evidence/统计heldout_pilot.md"))
    parser.add_argument("--csv-out", type=Path, default=Path("docs/evidence/统计heldout_pilot.csv"))
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="optional full-run root containing baseline/runs.jsonl and sft/runs.jsonl",
    )
    parser.add_argument("--kernel", default="autumnsq/lite-agent-stage-1-5-stats-pilot")
    parser.add_argument(
        "--adapter-sha",
        default="f5b685d4da795a783fbcb21b15880daebfa08c64b1661792f7e2c6caab29fea7",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    staging = (
        json.loads(args.staging_summary.read_text(encoding="utf-8"))
        if args.staging_summary
        else None
    )
    runs_by_condition: dict[str, list[dict[str, Any]]] | None = None
    if args.runs_root:
        runs_by_condition = {}
        for condition_name in ("baseline", "sft"):
            path = args.runs_root / condition_name / "runs.jsonl"
            runs_by_condition[condition_name] = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    document = build_document(
        summary,
        kernel=args.kernel,
        adapter_sha=args.adapter_sha,
        runs_by_condition=runs_by_condition,
        staging=staging,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document, encoding="utf-8", newline="\n")
    write_csv(args.csv_out, summary)
    print(f"wrote {args.out} ({len(document)} chars)")
    print(f"wrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
