"""Retrospective statistical audit of the v14 DPO arms.

This is a worked example for the v15 analysis plan, not a replacement for the
v15 GRPO analysis. It uses probe-level exact McNemar tests, Wilson CIs,
task-cluster bootstrap sensitivity intervals, and BH-FDR across all arm/layer
comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.v15_stats import (  # noqa: E402
    benjamini_hochberg,
    cluster_bootstrap_rate_diff,
    mcnemar_exact,
    minimum_detectable_risk_difference,
    wilson_ci,
)

LAYERS = ("path", "delete", "edit")


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_probe(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["probe_id"]): row for row in summary["rows"]}


def _outcome(row: dict[str, Any]) -> bool:
    return bool(row.get("execution_violation"))


def compare(
    base: dict[str, dict[str, Any]],
    arm: dict[str, dict[str, Any]],
    *,
    layer: str | None,
) -> dict[str, Any]:
    keys = sorted(set(base) & set(arm))
    if layer is not None:
        keys = [key for key in keys if base[key]["layer"] == layer]
    else:
        keys = [key for key in keys if base[key]["layer"] != "normal"]
    base_values = [_outcome(base[key]) for key in keys]
    arm_values = [_outcome(arm[key]) for key in keys]
    base_only = sum(
        base_value and not arm_value
        for base_value, arm_value in zip(base_values, arm_values, strict=True)
    )
    arm_only = sum(
        not base_value and arm_value
        for base_value, arm_value in zip(base_values, arm_values, strict=True)
    )
    base_count = sum(base_values)
    arm_count = sum(arm_values)
    n = len(keys)
    test = mcnemar_exact(base_only, arm_only)
    clusters: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for key, base_value, arm_value in zip(keys, base_values, arm_values, strict=True):
        clusters[str(base[key]["task_id"])].append((base_value, arm_value))
    cluster_ci = cluster_bootstrap_rate_diff(list(clusters.values()))
    mdr = minimum_detectable_risk_difference(n, test.discordant) if n else None
    base_ci = wilson_ci(base_count, n) if n else None
    arm_ci = wilson_ci(arm_count, n) if n else None
    return {
        "layer": layer or "all_non_normal",
        "n": n,
        "base_violations": base_count,
        "arm_violations": arm_count,
        "base_rate": base_count / n if n else None,
        "arm_rate": arm_count / n if n else None,
        "risk_difference": (arm_count - base_count) / n if n else None,
        "base_wilson_ci95": None if base_ci is None else [base_ci.low, base_ci.high],
        "arm_wilson_ci95": None if arm_ci is None else [arm_ci.low, arm_ci.high],
        "cluster_bootstrap_diff_ci95": [cluster_ci.low, cluster_ci.high],
        "mcnemar_base_only": base_only,
        "mcnemar_arm_only": arm_only,
        "mcnemar_discordant": test.discordant,
        "mcnemar_exact_p": test.p_value,
        "minimum_detectable_risk_difference_80pct_power": mdr,
    }


def analyze(base_path: Path, arm_paths: list[Path]) -> dict[str, Any]:
    base = rows_by_probe(load_summary(base_path))
    tests: list[dict[str, Any]] = []
    for arm_path in arm_paths:
        arm = rows_by_probe(load_summary(arm_path))
        label = arm_path.parent.name
        for layer in (None, *LAYERS):
            comparison = compare(base, arm, layer=layer)
            comparison["arm"] = label
            comparison["summary_path"] = str(arm_path)
            tests.append(comparison)
    q_values = benjamini_hochberg([float(row["mcnemar_exact_p"]) for row in tests])
    for row, q_value in zip(tests, q_values, strict=True):
        row["bh_fdr_q"] = q_value
    return {
        "version": "v15-v14-dpo-stat-audit-1",
        "base_summary": str(base_path),
        "arms": len(arm_paths),
        "comparisons": len(tests),
        "method": {
            "unit": "probe; 60 conflict probes (path 28 / delete 18 / edit 14)",
            "paired_test": "exact two-sided McNemar on discordant pairs",
            "rate_ci": "Wilson 95%",
            "cluster_sensitivity": "task-cluster bootstrap 20,000 draws",
            "multiplicity": "Benjamini-Hochberg FDR across arm x layer tests",
            "power": "exact binomial McNemar power; MDR at 80% power and observed discordant count",
        },
        "tests": tests,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# v14 DPO 统计审计（v15 统计工具示例）",
        "",
        "本表是方法演示，不是 v15 GRPO 结果。单位为 probe，60 条冲突探针；",
        "p 值为双侧精确 McNemar，q 值为全部 arm x 层比较的 BH-FDR。",
        "",
        "| arm | 层 | 基座违规 | arm 违规 | 风险差 | McNemar b/c | p | q | "
        "任务簇 bootstrap 95% CI |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in payload["tests"]:
        ci = row["cluster_bootstrap_diff_ci95"]
        lines.append(
            f"| {row['arm']} | {row['layer']} | {row['base_violations']}/{row['n']} | "
            f"{row['arm_violations']}/{row['n']} | {row['risk_difference']:+.3f} | "
            f"{row['mcnemar_base_only']}/{row['mcnemar_arm_only']} | "
            f"{row['mcnemar_exact_p']:.4f} | {row['bh_fdr_q']:.4f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |"
        )
    lines += [
        "",
        "## 读法",
        "",
        "- risk difference = arm violation rate - base violation rate；负值表示 arm 更少违规。",
        "- Wilson CI 是单组比例区间；任务簇 bootstrap 是配对风险差的敏感性区间。",
        "- BH-FDR 只控制这组预先列出的 arm x 层比较，不用于挑选最好 arm。",
        "- 观察到的 discordant 对很少时，MDR 很大；不能把“不显著”写成“无差异”。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="statistical audit of v14 DPO arms")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("private/v14/step3_baseline/m2_base/summary.json"),
    )
    parser.add_argument(
        "--arms-glob",
        default="private/v14/step4b_dpo/remote/eval/*/summary.json",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("private/v15/statistics"))
    args = parser.parse_args()

    arm_paths = sorted(Path(".").glob(args.arms_glob))
    if not arm_paths:
        raise SystemExit(f"no arm summaries matched {args.arms_glob}")
    payload = analyze(args.base, arm_paths)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "v14_dpo_stat_audit.json"
    md_path = args.out_dir / "v14_dpo_stat_audit.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
