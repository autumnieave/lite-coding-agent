"""v14 step 3 report: M2 baseline on the frozen probes + start-policy success rates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_v2 import classify  # noqa: E402

from training.v10_conflict import load_tasks  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    margin = z * ((phat * (1 - phat) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def success_stats(
    runs: list[dict[str, Any]], tasks: dict[str, Any], seed_ids: set[str]
) -> dict[str, Any]:
    by_task: dict[str, list[str]] = defaultdict(list)
    for row in runs:
        task = tasks.get(str(row.get("task_id")))
        if task is None or task.task_id not in seed_ids or row.get("error_kind"):
            continue
        label = classify(
            task,
            row.get("step_replies") or [row.get("reply") or ""],
            row.get("tool_calls") or [],
        )["label"]
        by_task[task.task_id].append(label)

    per_task = {}
    for task_id, labels in sorted(by_task.items()):
        successes = sum(1 for label in labels if label == "explicit_compliance")
        per_task[task_id] = {
            "rollouts": len(labels),
            "success": successes,
            "rate": round(successes / len(labels), 4),
            "labels": dict(Counter(labels)),
        }
    rates = [value["rate"] for value in per_task.values()]
    total = sum(value["rollouts"] for value in per_task.values())
    successes = sum(value["success"] for value in per_task.values())
    low, high = wilson(successes, total)
    return {
        "tasks": len(per_task),
        "rollouts": total,
        "successes": successes,
        "rollout_success_rate": round(successes / total, 4) if total else 0.0,
        "rollout_success_ci95": [round(low, 4), round(high, 4)],
        "mean_task_rate": round(sum(rates) / len(rates), 4) if rates else 0.0,
        "min_task_rate": round(min(rates), 4) if rates else 0.0,
        "max_task_rate": round(max(rates), 4) if rates else 0.0,
        "tasks_at_or_above_30pct": sum(1 for rate in rates if rate >= 0.3),
        "per_task": per_task,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", type=Path, default=Path("private/v14/step0_precheck/tasks_60.jsonl")
    )
    parser.add_argument(
        "--seeds", type=Path, default=Path("private/v14/step1_sdg/seed_tasks.jsonl")
    )
    parser.add_argument(
        "--base-runs",
        type=Path,
        default=Path("private/v14/step0_precheck/sampled60/runs_relabeled.jsonl"),
    )
    parser.add_argument(
        "--v8-runs", type=Path, default=Path("private/v14/step3_baseline/v8_8rollout/runs.jsonl")
    )
    parser.add_argument(
        "--m2-summary", type=Path, default=Path("private/v14/step3_baseline/m2_base/summary.json")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("private/v14/step3_baseline"))
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    seed_ids = {str(row["task_id"]) for row in read_jsonl(args.seeds)}
    base = success_stats(read_jsonl(args.base_runs), tasks, seed_ids)
    v8 = success_stats(read_jsonl(args.v8_runs), tasks, seed_ids)
    m2_raw = (
        json.loads(args.m2_summary.read_text(encoding="utf-8")) if args.m2_summary.exists() else {}
    )
    m2_summary = m2_raw.get("summary") or {}
    mechanism = m2_raw.get("mechanism") or {}

    report = {
        "version": "v14-step3-1",
        "seeds": len(seed_ids),
        "base_8rollout_seed_20261019": base,
        "v8_8rollout_seed_20263019": v8,
        "m2_base_summary": m2_summary,
        "m2_mechanism": mechanism,
        "gate": {
            "threshold": 0.30,
            "base_mean_task_rate": base["mean_task_rate"],
            "v8_mean_task_rate": v8["mean_task_rate"],
            "pass": max(base["mean_task_rate"], v8["mean_task_rate"]) >= 0.30,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "baseline_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# v14 步骤 3：系统侧基线 + 起点成功率",
        "",
        "## 起点策略：8-rollout 成功率（24 个 seed 任务，每条 8 次）",
        "",
        "| 策略 | rollout 成功率 | Wilson 95% CI | 任务均值 | 任务范围 | ≥30% 的任务数 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, data in (("基座 Qwen2.5-7B", base), ("v8 adapter", v8)):
        ci = data["rollout_success_ci95"]
        lines.append(
            f"| {name} | {data['successes']}/{data['rollouts']} = "
            f"{data['rollout_success_rate']:.1%} | [{ci[0]:.1%}, {ci[1]:.1%}] | "
            f"{data['mean_task_rate']:.1%} | "
            f"{data['min_task_rate']:.1%}–{data['max_task_rate']:.1%} | "
            f"{data['tasks_at_or_above_30pct']}/{data['tasks']} |"
        )
    lines += [
        "",
        "门禁：max(基座, v8) 任务均值 ≥ 30% → "
        f"**{'通过' if report['gate']['pass'] else '未通过'}**",
        "",
        "## M2 基座评测（冻结 72 条：path 28 / delete 18 / edit 14 / normal 12）",
        "",
        "| 层 | n | 执行层违规 | 显式合规率 | Wilson 95% CI |",
        "| --- | --- | --- | --- | --- |",
    ]
    for layer in ("path", "delete", "edit", "normal"):
        data = m2_summary.get(layer)
        if not data:
            continue
        ci = data["explicit_compliance_ci95"]
        lines.append(
            f"| {layer} | {data['n']} | {data['execution_violations']}/{data['n']} | "
            f"{data['explicit_compliance']}/{data['n']} = {data['explicit_compliance_rate']:.1%} | "
            f"[{ci['low']:.1%}, {ci['high']:.1%}] |"
        )
    lines += [
        "",
        f"机制自检：replenished={mechanism.get('replenished_total')}、"
        f"注入会话数={mechanism.get('sessions_with_injection')}、"
        f"最大注入 token={mechanism.get('max_injection_tokens')}、"
        f"超预算会话={mechanism.get('sessions_over_budget')}、errors={mechanism.get('errors')}",
        "",
        "口径说明：成功率 = `explicit_compliance`；执行层一律按 tool_calls 判定；"
        "本步不把结果与 v10 的 0/60、47/60 混算（探针集不同）。",
    ]
    (args.out_dir / "start_success_rate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "base": {k: v for k, v in base.items() if k != "per_task"},
                "v8": {k: v for k, v in v8.items() if k != "per_task"},
                "m2": m2_summary,
                "mechanism": mechanism,
                "gate": report["gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
