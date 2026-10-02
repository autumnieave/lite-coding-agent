"""Render docs/experiments/v14_step1_sdg.md from the step-1 artefacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", type=Path, default=Path("private/v14/step1_sdg/train_sdg.summary.json")
    )
    parser.add_argument(
        "--verifier",
        type=Path,
        default=Path("private/v14/step1_sdg/verifier_consistency_v2.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/v14_step1_sdg.md"))
    args = parser.parse_args()

    summary = load(args.summary)
    verifier = load(args.verifier)
    source_counts = summary.get("by_source_set") or {}
    granular = summary.get("granular_middle_state") or {}
    layer_role = summary["by_layer_role"]
    layers = ("path_escape", "delete_without_confirm", "edit_tests", "policy")
    layers_present = [
        layer
        for layer in ("path_escape", "delete_without_confirm", "edit_tests")
        if layer_role.get(f"{layer}:positive", 0) > 0 and layer_role.get(f"{layer}:negative", 0) > 0
    ]
    gate_ok = verifier["v2_rate"] >= 0.7 and len(layers_present) == 3

    lines = [
        "# v14 步骤 1：SDG 轨迹合成与标注质量",
        "",
        "日期：2026-11-01（初版）；2026-11-02 增量补造",
        f"状态：完成（{'通过门禁' if gate_ok else '未通过门禁'}）",
        "",
        "## 1. 设计",
        "",
        "- 种子任务：24 个（16 个步骤 0 中间态 + 8 个有方差），来自冲突池 60。",
        "- 采样：每任务 20 次 rollout，`temperature=0.8`、`top_p=0.9`、"
        "`seed=20262019`（避开步骤 0 的 20261019）。",
        "- 轨迹总数：两份 shard 各 12 任务 × 20 次 = 480。",
        "- 标注：`scripts/v14/label_v2.py`（执行层优先 → 伪工具调用按调用判 → "
        "完成声称 → 拒绝词 → 捷径词 → 提及 → 延迟）。",
        f"- 数据合入：stage0 {source_counts.get('stage0', 0)} 条 + "
        f"SDG {source_counts.get('sdg', 0)} 条 + "
        f"定向补造 {source_counts.get('granular_sdg', 0)} 条；"
        "与 72 条评测集任务 ID 重叠的行整行剔除。",
        "",
        "## 2. verifier 一致率（40 条分层抽检，人工冻结标签）",
        "",
        "| 判定器 | 一致数 | 一致率 |",
        "| --- | --- | --- |",
        f"| 原始 verifier（`v10.classification`，修正后规则） | "
        f"{verifier['v1_agree']}/40 | {verifier['v1_rate']:.1%} |",
        f"| v2 规则判定器（本步采用） | {verifier['v2_agree']}/40 | {verifier['v2_rate']:.1%} |",
        "",
        "逐类召回（v2 vs 人工）：",
        "",
        "| 人工标签 | 正确/总数 |",
        "| --- | --- |",
    ]
    for label, values in verifier["per_class"].items():
        lines.append(f"| {label} | {values['correct']}/{values['n']} |")
    lines += [
        "",
        f"残余分歧：{len(verifier['residual'])} 条；其中 4 条为「只读探查/讨要路径」"
        "被判成 task_priority，已在 `build_sdg.py` 用 `soft_text_only_no_write` "
        "守卫降级为 weak，不进负样本。",
        "",
        f"门禁：一致率 ≥70% 且三层都有 → **{'通过' if gate_ok else '未通过'}**"
        f"（一致率 {verifier['v2_rate']:.1%}；三层齐备：{', '.join(layers_present) or '无'}）。",
        "",
        "## 3. 训练集分布",
        "",
        f"- 轨迹数：{summary['records']}；任务数：{summary['tasks']}；"
        f"与评测集任务交集：{len(summary['task_overlap_with_eval'])}。",
        f"- 来源：stage0 {source_counts.get('stage0', 0)} 条；"
        f"SDG {source_counts.get('sdg', 0)} 条；"
        f"定向补造 {source_counts.get('granular_sdg', 0)} 条。",
        f"- 剔除：与评测集任务重叠 {summary['dropped_eval_overlap']} 条。",
        "",
        "| 层 | positive | negative | weak | 合计 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for layer in layers:
        positive = layer_role.get(f"{layer}:positive", 0)
        negative = layer_role.get(f"{layer}:negative", 0)
        weak = layer_role.get(f"{layer}:weak", 0)
        lines.append(
            f"| {layer} | {positive} | {negative} | {weak} | {positive + negative + weak} |"
        )
    lines += [
        "",
        "| severity | 条数 |",
        "| --- | --- |",
    ]
    for sev, count in summary["by_severity"].items():
        lines.append(f"| {sev} | {count} |")
    if granular:
        lines += [
            "",
            "### 定向补造中间态",
            "",
            f"- 8-rollout 中间态任务：{granular['middle_state_total']}/22；"
            "只对这些任务补 20-rollout SDG。",
            "",
            "| 层 | 任务数 | 中间态 |",
            "| --- | ---: | ---: |",
        ]
        for layer, data in granular["by_layer"].items():
            lines.append(f"| {layer} | {data['tasks']} | {data['middle_state']} |")
        gate = granular.get("layer_gate") or {}
        lines += [
            "",
            f"- policy + path 合计中间态：{gate.get('policy_path_middle_state_combined')}"
            f"（既有 {gate.get('policy_path_middle_state_pool')} + 新增 "
            f"{gate.get('policy_path_middle_state_new')}）。",
            f"- delete 新增中间态：{gate.get('delete_target_3_met') and '≥3' or '<3'}；"
            f"edit 新增中间态：{gate.get('edit_target_3_met') and '≥3' or '<3'}。",
            f"- edit 是否进入 GRPO："
            f"{'是' if gate.get('edit_target_3_met') else '否（SPA 兜底）'}；"
            f"delete 是否进入 GRPO："
            f"{'是' if gate.get('delete_target_3_met') else '否（SPA 兜底）'}。",
        ]
    lines += [
        "",
        "## 4. 边界",
        "",
        "- 全部轨迹来自基座 Qwen2.5-7B（无 adapter），不是 Teacher 模型；"
        "「SDG」在这里指「规则标注的自我 rollout」，不是蒸馏。",
        "- 伪工具调用（`<tool_call>`/`ronic`/`portunel`）既算格式失败，"
        "也算「想违规但没执行」；两者在文档里分开报。",
        "- 40 条抽检标签由单人标注，未做双人一致性；一致率是「规则 vs 单人」的上界估计。",
        f"- `soft_text_only_no_write` 守卫共降级 {summary['soft_guard_downgrades']} 条"
        "（文本像走捷径但无写操作 → weak，不进负样本）。",
        "- P0 只读误判的量化与修复见 `docs/experiments/v14_step1_p0_fixes.md`。",
        "",
        "## 5. 产物",
        "",
        "- `private/v14/step1_sdg/train_sdg.jsonl`（训练轨迹）",
        "- `private/v14/step1_sdg/verifier_consistency.md`（逐条分歧）",
        "- `private/v14/step1_sdg/severity_distribution.md`（分布）",
        "- `scripts/v14/build_train_sdg.py`、`scripts/v14/label_v2.py`、"
        "`scripts/v14/audit_verifier_v2.py`",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} (gate_ok={gate_ok})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
