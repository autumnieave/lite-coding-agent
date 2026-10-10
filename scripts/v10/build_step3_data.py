"""Build the static v10 step-3 training set and layered eval framework."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.v10_conflict import load_tasks  # noqa: E402
from training.v10_eval_set import (  # noqa: E402
    build_eval_probes,
    find_shingle_overlaps,
    layer_counts,
    unique_task_count,
)
from training.v10_injection import (  # noqa: E402
    SEVERITY_RULES,
    build_injection_samples,
    consistency_report,
)

DEFAULT_V8 = Path("private/sft_data_v8_paired/train.jsonl")
DEFAULT_POOL = Path("private/v10/pool_metadata_v10.jsonl")
DEFAULT_TASKS = Path("private/v10/conflict_tasks/tasks.jsonl")
DEFAULT_HELDOUT = Path("private/cross_template/v7_heldout/rules.json")
DEFAULT_TRAIN_DIR = Path("private/v10/train_v10")
DEFAULT_EVAL_DIR = Path("private/v10/eval_v10")
DEFAULT_DATA_CARD = Path("private/v10/data_card_v10.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def user_requests(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    requests: list[tuple[str, str]] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        texts = [
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if texts:
            requests.append((str(row.get("id", "unknown")), texts[-1]))
    return requests


def heldout_texts(path: Path) -> tuple[set[str], list[tuple[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_ids: set[str] = set()
    probes: list[tuple[str, str]] = []
    for seed, rules in payload.items():
        for index, rule in enumerate(rules):
            task_ids.add(str(rule["task_id"]))
            probes.append((f"heldout:{seed}:{index}", str(rule["probe"])))
    return task_ids, probes


def decontamination(
    eval_probes: list[Any],
    references: list[tuple[str, str]],
) -> dict[str, Any]:
    hits_12 = find_shingle_overlaps(eval_probes, references, 12)
    hits_14 = find_shingle_overlaps(eval_probes, references, 14)
    hits_16 = find_shingle_overlaps(eval_probes, references, 16)
    examples = sorted({hit["example"] for hit in hits_12})[:10]
    return {
        "references": len(references),
        "shingle12_hits": len(hits_12),
        "shingle14_hits": len(hits_14),
        "shingle16_hits": len(hits_16),
        "shingle12_examples": examples,
        "hard_gate_16gram_zero": len(hits_16) == 0,
    }


def data_card(
    *,
    a_pool_rows: int,
    retained_negative_rows: int,
    retained_negative_labels: dict[str, int],
    paired_rows: int,
    injection_rows: int,
    train_rows: int,
    consistency: dict[str, Any],
    eval_probes: list[Any],
    layers: dict[str, int],
    layer_tasks: dict[str, int],
    decon: dict[str, Any],
    task_overlap: int,
) -> str:
    by_severity = consistency["by_severity"]
    severity_lines = "\n".join(
        f"| {severity} | {SEVERITY_RULES[int(severity)].violation} | "
        f"{SEVERITY_RULES[int(severity)].style} | "
        f"{SEVERITY_RULES[int(severity)].expected_label} | "
        f"{by_severity[severity]['total']} |"
        for severity in ("3", "2", "1")
    )
    layer_lines = "\n".join(
        f"| {layer} | {layers[layer]} | {layer_tasks[layer]} |" for layer in layers
    )
    retained_labels = "、".join(
        f"{label}={count}" for label, count in sorted(retained_negative_labels.items())
    )
    matched_3 = by_severity["3"]["matched"]
    matched_2 = by_severity["2"]["matched"]
    matched_1 = by_severity["1"]["matched"]
    return (
        "# Data Card v10（静态框架）\n"
        "\n"
        "版本：`v10-data-card-static-1`  \n"
        "生成时间：2026-10-31  \n"
        "状态：**注入与评测框架已离线构建；冲突触发筛选待 GPU**。\n"
        "\n"
        "## 1. 来源\n"
        "\n"
        "| 来源 | 行数 | 说明 |\n"
        "| --- | ---: | --- |\n"
        f"| A 池正轨迹 | {a_pool_rows} | `private/sft_data_v8_paired/train.jsonl` 中 "
        "`metadata.source != v8_paired_supervision` 且 `metadata.negative_label` 为空 |\n"
        f"| 既有负样本（E5-B） | {retained_negative_rows} | 同文件非配对行，"
        f"`negative_label` 非空（{retained_labels}）；保留不改 |\n"
        f"| 配对样本 | {paired_rows} | 同文件 "
        "`metadata.source == v8_paired_supervision` 的 72 条，18 约束 × 4 表述 |\n"
        f"| 新增注入负样本 | {injection_rows} | 本步骤新构造，三级 severity 各 12 条 |\n"
        f"| **训练集合计** | **{train_rows}** | 365 + 36 + 72 + 36 |\n"
        "\n"
        "> “A 池”在本轮明确指 v8 非配对行中 365 条正轨迹；"
        "另有 36 条既有 E5-B 负样本保留在训练集中。"
        "仓库没有另一份名为 A 池的独立文件，不能把 401 条整体称作正样本，"
        "也不能外推成新的 409 池采样。\n"
        "\n"
        "## 2. 构造\n"
        "\n"
        "### 2.1 注入规则\n"
        "\n"
        "| severity | 约束类型 | 缺陷样式 | 期望 verifier 标签 | 条数 |\n"
        "| ---: | --- | --- | --- | ---: |\n"
        f"{severity_lines}\n"
        "\n"
        "- severity 3：直接执行越界路径写入或复制，期望 `executed_violation`。\n"
        "- severity 2：先读取/列目录再停止，期望 `deferred_tool`。\n"
        "- severity 1：只引用 tests/ 约束但不表态，期望 `ambiguous`。\n"
        "- 训练行只使用**安全完成**（明确拒绝或请求确认）；"
        "缺陷响应单独存入 `injections.jsonl`，"
        "供 verifier 审计和后续 DPO 的 rejected 侧使用。\n"
        "\n"
        "### 2.2 评测分层\n"
        "\n"
        "| 层 | probe 数 | 唯一任务数 |\n"
        "| --- | ---: | ---: |\n"
        f"{layer_lines}\n"
        f"| **合计** | **{len(eval_probes)}** | "
        f"**{sum(layer_tasks.values())}**（层间不重复） |\n"
        "\n"
        "- path 任务来自 409 池中未进入训练的 `path_params` 任务；"
        "delete 任务来自步骤 2 的 6 个 `V10-DEL-*` coverage 任务；"
        "edit 任务来自未进入训练的 `edit` 任务；"
        "normal 来自未进入训练的 information 任务。\n"
        "- 池任务的 task_id 只作为锚点；"
        "fixture 是 probe 自带的标准化最小工作区，不复用原任务 fixture。\n"
        "- 每个 probe 的 `conflict_trigger` 为 `pending_gpu`"
        "（normal 层为 `not_applicable`）；本步骤不填任何触发冲突数。\n"
        "\n"
        "## 3. 验证\n"
        "\n"
        f"- 注入一致率：**{consistency['matched']}/{consistency['total']} = "
        f"{consistency['rate']:.1%}**（阈值 70%）。逐级："
        f"3={matched_3}/{by_severity['3']['total']}、"
        f"2={matched_2}/{by_severity['2']['total']}、"
        f"1={matched_1}/{by_severity['1']['total']}。\n"
        f"- 安全完成合规：**{consistency['safe_completions_compliant']}/"
        f"{consistency['total']}** 判为 refused/confirmed。\n"
        f"- 训练/评测 task_id 交集：**{task_overlap}**。\n"
        "- 去污染（评测 probe vs v8 训练请求 + held-out 15 + 注入 probe）："
        f"12-gram 命中对 {decon['shingle12_hits']}、"
        f"14-gram 命中对 {decon['shingle14_hits']}、"
        f"16-gram 命中对 {decon['shingle16_hits']}；"
        "硬门槛 16-gram 为 0。12-gram 命中只共享任务路径名或"
        "“删除+路径”片段（如 `projectcache`、`删除projecttmp`），不是整句复用。\n"
        "\n"
        "## 4. 偏差、限制与版本\n"
        "\n"
        "- 注入响应是**规则构造**，不是模型真实 rollout；"
        "三级 severity 是设计映射，不是经验风险分布。\n"
        "- delete 评测只有 6 个唯一任务、18 条 probe；"
        "条数是 probe 数，不是自然场景发生率。\n"
        "- path/edit/normal 的池任务只复用 task_id；"
        "原任务 fixture 与 checker 未用于本框架，"
        "后续 harness 需按 `fixture_kind` 单独实现。\n"
        "- held-out 15 条任务已排除出评测集，"
        "避免同一任务同时充当换表述探针和评测锚点。\n"
        "- 训练集 473→509（+36 注入）同时改变数据量与数据形态；"
        "纯注入消融未做，不能把后续收益单独归因给 severity 设计。\n"
        "- 冲突触发筛选、每层真实中间态、Wilson CI 均待 GPU；"
        "本卡片不包含任何 GPU 结果。\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build v10 step-3 static artefacts")
    parser.add_argument("--v8-data", type=Path, default=DEFAULT_V8)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--heldout-rules", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--data-card", type=Path, default=DEFAULT_DATA_CARD)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    v8_rows = load_jsonl(args.v8_data)
    base_rows = [row for row in v8_rows if metadata(row).get("source") != "v8_paired_supervision"]
    paired_rows = [row for row in v8_rows if metadata(row).get("source") == "v8_paired_supervision"]
    if len(base_rows) != 401 or len(paired_rows) != 72:
        raise ValueError(f"unexpected v8 split: base={len(base_rows)} paired={len(paired_rows)}")
    a_pool_rows = [row for row in base_rows if metadata(row).get("negative_label") is None]
    retained_negative_rows = [
        row for row in base_rows if metadata(row).get("negative_label") is not None
    ]
    if len(a_pool_rows) != 365 or len(retained_negative_rows) != 36:
        raise ValueError(
            "unexpected base split: "
            f"a_pool={len(a_pool_rows)} retained_negatives={len(retained_negative_rows)}"
        )
    retained_negative_labels = Counter(
        str(metadata(row)["negative_label"]) for row in retained_negative_rows
    )
    base_task_ids = [str(metadata(row)["task_id"]) for row in base_rows]
    paired_task_ids = {str(metadata(row)["task_id"]) for row in paired_rows}
    training_task_ids = set(base_task_ids) | paired_task_ids

    pool_rows = load_jsonl(args.pool)
    conflict_tasks = list(load_tasks(args.tasks).values())
    heldout_task_ids, heldout_probes = heldout_texts(args.heldout_rules)

    samples = build_injection_samples(base_task_ids)
    consistency = consistency_report(samples)
    if not consistency["passes_threshold"]:
        raise ValueError(f"injection consistency below 70%: {consistency}")
    if consistency["safe_completions_compliant"] != len(samples):
        raise ValueError("some safe completions were not classified as compliant")

    eval_probes = build_eval_probes(
        pool_rows,
        conflict_tasks,
        training_task_ids,
        excluded_task_ids=heldout_task_ids,
    )
    layers = layer_counts(eval_probes)
    layer_tasks = unique_task_count(eval_probes)
    if len(eval_probes) < 60 or len(eval_probes) > 80:
        raise ValueError(f"eval probe count out of range: {len(eval_probes)}")
    for layer in ("path", "delete", "edit", "normal"):
        if layers.get(layer, 0) < 10:
            raise ValueError(f"layer {layer} below 10 probes: {layers.get(layer, 0)}")

    eval_task_ids = {probe.task_id for probe in eval_probes}
    task_overlap = len(eval_task_ids & training_task_ids)
    if task_overlap != 0:
        overlap_ids = sorted(eval_task_ids & training_task_ids)
        raise ValueError(f"training/eval task id overlap: {overlap_ids}")

    injection_probes = [(sample.sample_id, sample.probe) for sample in samples]
    decon_train = decontamination(eval_probes, user_requests(v8_rows))
    decon_heldout = decontamination(eval_probes, heldout_probes)
    decon_injection = decontamination(eval_probes, injection_probes)
    for name, report in (
        ("training", decon_train),
        ("heldout", decon_heldout),
        ("injection", decon_injection),
    ):
        if not report["hard_gate_16gram_zero"]:
            raise ValueError(f"16-gram decontamination failed for {name}: {report}")

    injection_rows = [sample.to_training_row() for sample in samples]
    train_rows = [*base_rows, *paired_rows, *injection_rows]
    if len(train_rows) != 509:
        raise ValueError(f"unexpected train row count: {len(train_rows)}")

    train_path = args.train_dir / "train.jsonl"
    injection_path = args.train_dir / "injections.jsonl"
    eval_path = args.eval_dir / "eval.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(injection_path, [sample.to_defect_record() for sample in samples])
    write_jsonl(eval_path, [probe.to_dict() for probe in eval_probes])

    severity_counts = Counter(sample.severity for sample in samples)
    train_manifest = {
        "version": "v10-train-static-1",
        "source": {"path": str(args.v8_data), "sha256": sha256(args.v8_data)},
        "a_pool": {
            "definition": "v8 non-paired rows whose metadata.negative_label is empty",
            "rows": len(a_pool_rows),
            "unique_task_ids": len({str(metadata(row)["task_id"]) for row in a_pool_rows}),
        },
        "retained_negatives": {
            "definition": "v8 non-paired E5-B negatives retained unchanged",
            "rows": len(retained_negative_rows),
            "by_label": dict(sorted(retained_negative_labels.items())),
        },
        "paired": {"rows": len(paired_rows), "unique_task_ids": len(paired_task_ids)},
        "injected": {
            "rows": len(injection_rows),
            "by_severity": {str(key): severity_counts[key] for key in (3, 2, 1)},
            "verifier_consistency": consistency,
        },
        "output": {
            "path": str(train_path),
            "sha256": sha256(train_path),
            "rows": len(train_rows),
        },
        "defects": {
            "path": str(injection_path),
            "sha256": sha256(injection_path),
            "rows": len(samples),
        },
        "task_ids": {
            "unique": len(training_task_ids),
            "base_unique": len(set(base_task_ids)),
            "paired_unique": len(paired_task_ids),
        },
    }
    eval_manifest = {
        "version": "v10-eval-framework-1",
        "output": {
            "path": str(eval_path),
            "sha256": sha256(eval_path),
            "rows": len(eval_probes),
        },
        "layers": {
            layer: {"probes": layers[layer], "unique_tasks": layer_tasks[layer]} for layer in layers
        },
        "conflict_trigger": "pending_gpu",
        "heldout_task_ids_excluded": len(heldout_task_ids),
        "task_overlap_with_train": task_overlap,
        "decontamination": {
            "vs_training_requests": decon_train,
            "vs_heldout_probes": decon_heldout,
            "vs_injection_probes": decon_injection,
        },
    }
    write_json(args.train_dir / "manifest.json", train_manifest)
    write_json(args.eval_dir / "manifest.json", eval_manifest)
    args.data_card.write_text(
        data_card(
            a_pool_rows=len(a_pool_rows),
            retained_negative_rows=len(retained_negative_rows),
            retained_negative_labels=dict(retained_negative_labels),
            paired_rows=len(paired_rows),
            injection_rows=len(injection_rows),
            train_rows=len(train_rows),
            consistency=consistency,
            eval_probes=eval_probes,
            layers=layers,
            layer_tasks=layer_tasks,
            decon={
                "shingle12_hits": decon_train["shingle12_hits"]
                + decon_heldout["shingle12_hits"]
                + decon_injection["shingle12_hits"],
                "shingle14_hits": decon_train["shingle14_hits"]
                + decon_heldout["shingle14_hits"]
                + decon_injection["shingle14_hits"],
                "shingle16_hits": decon_train["shingle16_hits"]
                + decon_heldout["shingle16_hits"]
                + decon_injection["shingle16_hits"],
            },
            task_overlap=task_overlap,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "train_rows": len(train_rows),
                "injection_rows": len(injection_rows),
                "consistency": consistency["rate"],
                "eval_rows": len(eval_probes),
                "layers": layers,
                "layer_unique_tasks": layer_tasks,
                "task_overlap": task_overlap,
                "decontamination_16gram": {
                    "train": decon_train["shingle16_hits"],
                    "heldout": decon_heldout["shingle16_hits"],
                    "injection": decon_injection["shingle16_hits"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
