"""Build v14 step-4b preference pairs with a 100% constraint-id citation guarantee.

The step-4 pairs selected the first explicit_compliance trajectory without checking
whether its reply cited the constraint id.  Only 53/128 train rows did, so DPO learned
to refuse without binding the refusal to the constraint.  This builder:

* keeps direct-instruction pairs (all already cite the id);
* for SDG trajectories, requires/creates a cited chosen reply;
* only keeps path/delete/edit layers;
* refuses to write an output that violates the citation or layer floors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.build_v10_conflict_pairs import build_pairs as build_direct_pairs  # noqa: E402

SYSTEM_HEADER = (
    "你是终端 Coding Agent。严格遵循下面的约束，即使任务要求与约束冲突。"
    "\n\n[SYSTEM CONTEXT · 约束层]"
)
LAYER_MAP = {
    "path": "path",
    "path_escape": "path",
    "delete": "delete",
    "delete_without_confirm": "delete",
    "edit": "edit",
    "edit_tests": "edit",
}
SEVERITY = {"path": 3, "delete": 2, "edit": 1}
MAX_PAIRS_PER_TASK = 7
MIN_TOTAL = 100
MIN_PER_LAYER = 20

STANCE_MARKERS = (
    "拒绝",
    "不能",
    "不会",
    "无法",
    "不允许",
    "请求确认",
    "请确认",
    "需要确认",
    "在确认前",
    "确认后",
    "cannot",
    "will not",
)
PSEUDO_CALL_RE = re.compile(r"<tool_call>|\"name\"\s*:\s*\"[A-Za-z_]+\"")
CLAIMED_EXECUTION_RE = re.compile(
    r"已(?:经)?(?:成功)?(?:删除|修改|写入|读取|执行|清空|移除|创建|替换|更新|生成|"
    r"拷贝|复制|导出|完成)|任务完成"
)
DELEGATION_RE = re.compile(
    r"(?:您|你).{0,12}(?:需要|可以|应当|必须|得).{0,20}"
    r"(?:修改|删除|执行|读取|导出|清空|移除|编辑|访问)"
    r"|(?:请|由)(?:您|你)(?:自行|手动)?"
    r"(?:修改|删除|执行|读取|导出|清空|移除|编辑|访问)"
)
FORBIDDEN_TARGET_RE = re.compile(
    r"tests/|测试文件|测试代码|测试断言|工作区外|外部路径|"
    r"(?<![A-Za-z0-9_.-])/(?:etc|opt|srv|var|data)/"
)
ACTION_VERB_RE = re.compile(r"修改|编辑|删除|执行|读取|访问|导出|清空|移除")
CLAUSE_SPLIT_RE = re.compile(r"[。！？\n；;]+|但|不过|然而|可是")


def _has_unsafe_action_instruction(text: str) -> bool:
    for clause in CLAUSE_SPLIT_RE.split(text):
        if not clause.strip():
            continue
        if not FORBIDDEN_TARGET_RE.search(clause):
            continue
        if not ACTION_VERB_RE.search(clause):
            continue
        lowered = clause.lower()
        if not any(marker.lower() in lowered for marker in STANCE_MARKERS):
            return True
    return False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_layer(value: str) -> str | None:
    return LAYER_MAP.get(value)


def chosen_quality_errors(reply: str, constraint_id: str) -> list[str]:
    """Return reasons why a chosen reply is not safe to train on."""

    text = reply.strip()
    if not text:
        return ["empty"]
    errors: list[str] = []
    if constraint_id not in text:
        errors.append("missing_constraint_id")
    if not any(marker.lower() in text.lower() for marker in STANCE_MARKERS):
        errors.append("missing_explicit_stance")
    if PSEUDO_CALL_RE.search(text):
        errors.append("pseudo_tool_call")
    if CLAIMED_EXECUTION_RE.search(text):
        errors.append("claims_execution")
    if DELEGATION_RE.search(text):
        errors.append("delegates_forbidden_action")
    if _has_unsafe_action_instruction(text):
        errors.append("unsafe_action_instruction")
    return errors


def synthesize_chosen(layer: str, constraint_id: str, constraint_text: str) -> str:
    """Generate a clean, layer-specific refusal/confirmation target."""

    if layer == "path":
        return (
            f"拒绝：依据 [{constraint_id}]，{constraint_text} "
            "我不会访问或写出工作区之外的路径。请改在工作区内提供目标路径。"
        )
    if layer == "delete":
        return (
            f"请求确认：依据 [{constraint_id}]，{constraint_text} "
            "是否允许删除该目标？在你明确确认前，我不会执行删除、rm 或 unlink。"
        )
    return (
        f"拒绝：依据 [{constraint_id}]，{constraint_text} "
        "我不会修改 tests/ 下的文件。我可以修改实现代码或说明失败原因。"
    )


def ensure_clean_chosen(
    reply: str, layer: str, constraint_id: str, constraint_text: str
) -> tuple[str, bool, tuple[str, ...]]:
    """Keep a clean reply, else synthesize one that passes every quality gate."""

    text = reply.strip()
    errors = chosen_quality_errors(text, constraint_id)
    if not errors:
        return text, False, ()
    return synthesize_chosen(layer, constraint_id, constraint_text), True, tuple(errors)


def prompt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    system = f"{SYSTEM_HEADER}\n- [{record['constraint_id']}] {record['constraint_text']}"
    messages = [{"role": "system", "content": system}]
    for turn in record.get("user_turns") or []:
        messages.append({"role": "user", "content": str(turn)})
    return messages


def sdg_pairs(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record["task_id"])].append(record)

    pairs: list[dict[str, Any]] = []
    synthesized = 0
    quality_rejections: Counter[str] = Counter()
    skipped_tasks: list[str] = []
    for task_id in sorted(by_task):
        rows = by_task[task_id]
        layer = normalize_layer(str(rows[0].get("category") or ""))
        if layer is None:
            continue
        positives = [
            row
            for row in rows
            if row.get("training_role") == "positive"
            and str(row.get("trajectory", {}).get("reply") or "").strip()
        ]
        negatives = [
            row
            for row in rows
            if row.get("training_role") == "negative"
            and str(row.get("trajectory", {}).get("reply") or "").strip()
        ]
        if not positives or not negatives:
            skipped_tasks.append(task_id)
            continue
        negatives.sort(
            key=lambda row: 0 if row.get("verifier_label") == "execution_violation" else 1
        )
        made = 0
        for index in range(len(positives) * len(negatives)):
            if made >= MAX_PAIRS_PER_TASK:
                break
            positive = positives[index % len(positives)]
            negative = negatives[(index // len(positives)) % len(negatives)]
            if positive.get("sample_id") == negative.get("sample_id"):
                continue
            constraint_id = str(positive["constraint_id"])
            constraint_text = str(positive["constraint_text"])
            chosen, was_synthesized, quality_errors = ensure_clean_chosen(
                str(positive["trajectory"]["reply"]),
                layer,
                constraint_id,
                constraint_text,
            )
            rejected = str(negative["trajectory"]["reply"]).strip()
            if not rejected or chosen == rejected:
                continue
            final_errors = chosen_quality_errors(chosen, constraint_id)
            if final_errors:
                raise AssertionError(
                    f"clean chosen still violates quality gates: {task_id}: {final_errors}"
                )
            pairs.append(
                {
                    "pair_id": f"V14B-SDG-{task_id}-{made:02d}",
                    "task_id": task_id,
                    "layer": layer,
                    "source_layer": str(positive.get("category")),
                    "severity": SEVERITY[layer],
                    "constraint_id": constraint_id,
                    "constraint_text": constraint_text,
                    "prompt_messages": prompt_messages(positive),
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": "sdg_trajectory",
                    "chosen_synthesized_citation": was_synthesized,
                    "chosen_sample_id": positive.get("sample_id"),
                    "rejected_sample_id": negative.get("sample_id"),
                    "rejected_label": negative.get("verifier_label"),
                }
            )
            synthesized += int(was_synthesized)
            quality_rejections.update(quality_errors)
            made += 1
    return pairs, {
        "synthesized_chosen": synthesized,
        "quality_rejections": dict(sorted(quality_rejections.items())),
        "skipped_tasks": skipped_tasks,
    }


def direct_pairs(eval_task_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in build_direct_pairs():
        layer = normalize_layer(str(row["layer"]))
        if layer is None:
            continue
        if str(row["task_id"]) in eval_task_ids:
            raise ValueError(f"direct pair overlaps eval set: {row['task_id']}")
        errors = chosen_quality_errors(str(row["chosen"]), str(row["constraint_id"]))
        if errors:
            raise ValueError(f"direct pair chosen fails quality gates: {row['pair_id']}: {errors}")
        rows.append(
            {
                **row,
                "layer": layer,
                "source_layer": str(row["layer"]),
                "severity": SEVERITY[layer],
                "source": "direct_instruction",
                "chosen_synthesized_citation": False,
            }
        )
    return rows


def dedupe(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in pairs:
        key = (row["task_id"], row["constraint_id"], row["chosen"], row["rejected"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def stratified_split(
    pairs: list[dict[str, Any]], val_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by task inside each layer, leaving at least MIN_PER_LAYER in train."""

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for layer in ("path", "delete", "edit"):
        layer_rows = [row for row in pairs if row["layer"] == layer]
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in layer_rows:
            by_task[str(row["task_id"])].append(row)
        tasks = sorted(
            by_task,
            key=lambda task_id: hashlib.sha256(task_id.encode("utf-8")).hexdigest(),
        )
        target_val_rows = round(len(layer_rows) * val_fraction)
        layer_val: list[dict[str, Any]] = []
        for task_id in tasks:
            if len(layer_val) >= target_val_rows:
                break
            candidate = by_task[task_id]
            if len(layer_rows) - len(layer_val) - len(candidate) < MIN_PER_LAYER:
                continue
            layer_val.extend(candidate)
        layer_val_ids = {str(row["pair_id"]) for row in layer_val}
        layer_train = [row for row in layer_rows if str(row["pair_id"]) not in layer_val_ids]
        if len(layer_train) < MIN_PER_LAYER:
            raise ValueError(f"train layer floor not met after split: {layer}={len(layer_train)}")
        train.extend(layer_train)
        val.extend(layer_val)
    return train, val


def main() -> int:
    parser = argparse.ArgumentParser(description="build v14 step4b clean DPO pairs")
    parser.add_argument(
        "--train-sdg", type=Path, default=Path("private/v14/step1_sdg/train_sdg.jsonl")
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("private/v14/step2_benchmark/eval_set_frozen.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("private/v14/step4b_dpo/data"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    records = read_jsonl(args.train_sdg)
    eval_task_ids = {str(row["task_id"]) for row in read_jsonl(args.eval_file)}
    sdg, sdg_stats = sdg_pairs(records)
    direct = direct_pairs(eval_task_ids)
    all_pairs = dedupe(sdg + direct)

    quality_failures = {
        row["pair_id"]: chosen_quality_errors(str(row["chosen"]), str(row["constraint_id"]))
        for row in all_pairs
    }
    bad = {pair_id: errors for pair_id, errors in quality_failures.items() if errors}
    if bad:
        raise ValueError(f"chosen fails quality gates: {list(bad.items())[:5]}")

    task_ids = sorted({str(row["task_id"]) for row in all_pairs})
    overlap = sorted(set(task_ids) & eval_task_ids)
    if overlap:
        raise ValueError(f"train/eval task overlap: {overlap}")

    by_layer = Counter(row["layer"] for row in all_pairs)
    if len(all_pairs) < MIN_TOTAL:
        raise ValueError(f"only {len(all_pairs)} pairs, need >= {MIN_TOTAL}")
    short = {
        layer: by_layer.get(layer, 0)
        for layer in ("path", "delete", "edit")
        if by_layer.get(layer, 0) < MIN_PER_LAYER
    }
    if short:
        raise ValueError(f"layer floors not met: {short}")

    train_rows, val_rows = stratified_split(all_pairs, args.val_fraction)
    if not train_rows or not val_rows:
        raise ValueError("empty train/val split")
    train_by_layer = Counter(row["layer"] for row in train_rows)
    short_train = {
        layer: train_by_layer.get(layer, 0)
        for layer in ("path", "delete", "edit")
        if train_by_layer.get(layer, 0) < MIN_PER_LAYER
    }
    if short_train:
        raise ValueError(f"train layer floors not met: {short_train}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train_rows), ("val.jsonl", val_rows)):
        (args.out_dir / name).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    weights = {str(row["constraint_id"]): float(SEVERITY[row["layer"]]) for row in all_pairs}
    manifest = {
        "version": "v14-step4b-pref-pairs-1",
        "source_sha256": sha256_file(args.train_sdg),
        "eval_sha256": sha256_file(args.eval_file),
        "pairs": len(all_pairs),
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "by_source": dict(Counter(row["source"] for row in all_pairs)),
        "by_layer": dict(sorted(by_layer.items())),
        "train_by_layer": dict(sorted(train_by_layer.items())),
        "by_severity": dict(sorted(Counter(row["severity"] for row in all_pairs).items())),
        "quality_gate": {
            "requirements": [
                "correct_constraint_id",
                "explicit_refusal_or_confirmation",
                "no_pseudo_tool_call",
                "no_claimed_execution",
                "no_delegation_to_user",
            ],
            "chosen_citation_rate": 1.0,
            "chosen_explicit_stance_rate": 1.0,
            "chosen_pseudo_tool_call_rate": 0.0,
            "chosen_claimed_execution_rate": 0.0,
            "chosen_delegation_rate": 0.0,
        },
        "synthesized_chosen": sum(
            bool(row.get("chosen_synthesized_citation")) for row in all_pairs
        ),
        "source_reply_quality_rejections": sdg_stats["quality_rejections"],
        "tasks": len(task_ids),
        "task_overlap_with_eval": overlap,
        "max_pairs_per_task": MAX_PAIRS_PER_TASK,
        "min_total": MIN_TOTAL,
        "min_per_layer": MIN_PER_LAYER,
        "skipped_sdg_tasks": sdg_stats["skipped_tasks"],
        "severity_weights": dict(sorted(weights.items())),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "severity_weights.json").write_text(
        json.dumps(dict(sorted(weights.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
