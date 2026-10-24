"""Render the stage-1.5a task design document from frozen metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from training.stats_heldout import PILOT_TASK_IDS, STATS_TASK_SPECS  # noqa: E402

FAMILY_PURPOSE = {
    "policy_extract": "单文件策略抽取：从带注释/优先级规则的配置中提取最终生效值。",
    "selective_aggregate": "选择性聚合：先列目录，只读取匹配文件，排除诱饵后求和。",
    "search_collect": "检索收集：用 grep 定位命中文件，只读命中文件并汇总唯一 ID。",
    "edit_verify": "受控编辑：只改目标行，修改后重新读取验证，禁止 write_file。",
    "audit_reconcile": "审计对账：多文件按 REV 合并，只暴露冲突键并保留来源。",
}
FAMILY_WORKFLOW = {
    "policy_extract": "W1",
    "selective_aggregate": "W2",
    "search_collect": "W3",
    "edit_verify": "W4",
    "audit_reconcile": "W5",
}
WORKFLOW_COUNTS = {
    "W1": "78 次（单文件读取）",
    "W2": "20 次（list_dir -> read_file）",
    "W3": "15 次（read_file -> grep -> list_dir -> read_file）",
    "W4": "14 次（read_file -> edit_file）",
    "W5": "11 次（list_dir -> read_file -> grep -> read_file）",
}


def _canonical_manifest_sha() -> str:
    payload = "".join(
        json.dumps(spec.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for spec in STATS_TASK_SPECS
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_document(manifest: dict[str, Any], teacher: dict[str, Any]) -> str:
    expected_sha = _canonical_manifest_sha()
    if manifest["manifest_sha256"] != expected_sha:
        raise ValueError(
            f"frozen manifest SHA mismatch: {manifest['manifest_sha256']} != {expected_sha}"
        )
    teacher_by_id = {row["task_id"]: row for row in teacher["tasks"]}
    lines = [
        "# 阶段 1.5a 任务构造",
        "",
        f"- 冻结版本：`{manifest['version']}`",
        f"- 冻结时间：`{manifest['created_at']}`",
        f"- 任务数：{manifest['task_count']}（5 个族 × 3 个变体）",
        f"- task manifest SHA-256：`{manifest['manifest_sha256']}`",
        f"- fixture seed 范围：`{manifest['seed_range'][0]}–{manifest['seed_range'][1]}`",
        "- 任务 split：全部为 `test`，不进入 SFT train/val。",
        "",
        "## 种子来源",
        "",
        "任务不是纯人工另写，而是从 train-v5 的 assistant-only 轨迹中挖掘高频工作流后做变体化：",
        "",
        "| 工作流 | train-v5 轨迹出现次数 | 变体族 |",
        "| --- | ---: | --- |",
    ]
    for workflow, description in manifest["workflow_sources"].items():
        family = next(key for key, value in FAMILY_WORKFLOW.items() if value == workflow)
        lines.append(
            f"| `{workflow}` {_cell(description)} | {_cell(WORKFLOW_COUNTS[workflow])} | "
            f"`{family}` |"
        )
    lines.extend(
        [
            "",
            "## 任务族与设计目的",
            "",
            "| 族 | 任务数 | 设计目的 | pilot 任务 |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for family, purpose in FAMILY_PURPOSE.items():
        tasks = [spec.task_id for spec in STATS_TASK_SPECS if spec.family == family]
        pilot = [task_id for task_id in tasks if task_id in PILOT_TASK_IDS]
        lines.append(
            f"| `{family}` | {len(tasks)} | {_cell(purpose)} | "
            f"{', '.join(f'`{item}`' for item in pilot)} |"
        )
    lines.extend(
        [
            "",
            "## 任务总表",
            "",
            "| 任务 | 族 | 难度 | 上下文 | 策略检查 | 最少工具调用 | 步数上限 | fixture SHA-256 |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for spec in STATS_TASK_SPECS:
        entry = next(row for row in manifest["tasks"] if row["task_id"] == spec.task_id)
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{spec.task_id}`",
                    f"`{spec.family}`",
                    spec.difficulty,
                    spec.context_tier,
                    ", ".join(f"`{name}`" for name in entry["policy_checks"]),
                    str(spec.min_tool_calls),
                    str(spec.steps_limit),
                    f"`{entry['fixture_sha256']}`",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 逐任务设计",
            "",
        ]
    )
    for spec in STATS_TASK_SPECS:
        entry = next(row for row in manifest["tasks"] if row["task_id"] == spec.task_id)
        teacher_row = teacher_by_id[spec.task_id]
        lines.extend(
            [
                f"### `{spec.task_id}`",
                "",
                f"- 设计目的：{FAMILY_PURPOSE[spec.family]}",
                f"- 种子工作流：`{FAMILY_WORKFLOW[spec.family]}`，"
                f"{WORKFLOW_COUNTS[FAMILY_WORKFLOW[spec.family]]}",
                f"- 用户任务：{_cell(spec.user_turns[0])}",
                "- 约束组合："
                + "；".join(f"`{item.id}` {_cell(item.text)}" for item in spec.constraints),
                "- 策略约束检查：" + "、".join(f"`{name}`" for name in entry["policy_checks"]),
                f"- 最小必要工具调用：{spec.min_tool_calls}；步数上限：{spec.steps_limit}",
                "- 确定性检查项：" + "、".join(f"`{check.name}`" for check in spec.checks),
                f"- fixture：seed `{spec.fixture_seed}`，"
                f"目录 `{entry['fixture_dir']}`，SHA-256 `{entry['fixture_sha256']}`",
                f"- 教师门禁：{teacher_row['gate_passes']}/{teacher_row['rollouts']} "
                f"({teacher_row['gate_rate']:.0%})；策略合规 "
                f"{teacher_row['policy_passes']}/{teacher_row['rollouts']} "
                f"({teacher_row['policy_rate']:.0%})；结论："
                f"{'保留' if teacher_row['keep'] else '剔除'}",
                "",
            ]
        )
    lines.extend(
        [
            "## 冻结与不重叠校验",
            "",
            "- `validate_stats_specs()` 校验 15 个任务 ID、prompt、fixture seed 唯一，"
            "并与 train-v1..v5、dev/tuning、final-test-v2 的保留 ID/seed 集合求交；无交集。",
            "- 每个任务冻结前重新生成 fixture，并校验 prompt 引用的文件、行号、"
            "token 与 fixture 一致。",
            "- `fixture_sha256` 对目录内所有文件的相对路径和字节内容做 SHA-256；"
            "manifest SHA 对 15 条 canonical spec JSONL 做 SHA-256。",
            "- 任务只用于 stage-1.5 held-out 评测，不进入 SFT train-v5/v6，"
            "也不与 final-test-v2 混用。",
            "",
            "## 教师验证口径",
            "",
            f"- 教师：`{teacher.get('teacher_model', 'deepseek-v4.1-flash')}`；"
            f"协议：`{teacher['teacher_protocol']}`。",
            "- 每任务 5 次 rollout；保留门槛为门禁通过率 ≥40% 且策略合规率 ≥60%。",
            "- 保留 15/15；教师验证只证明任务可解，不代表 baseline/SFT 难度或训练增益。",
            "",
            "## Pilot 任务",
            "",
            "、".join(f"`{task_id}`" for task_id in PILOT_TASK_IDS),
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="render stage-1.5a task design markdown")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("private/heldout_1_5/STATS_TASKS_FROZEN.json"),
    )
    parser.add_argument(
        "--teacher-summary",
        type=Path,
        default=Path("private/heldout_1_5_teacher_v2/summary.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/evidence/阶段1.5a_任务构造.md"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    teacher = json.loads(args.teacher_summary.read_text(encoding="utf-8"))
    document = build_document(manifest, teacher)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(document)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
