"""Build v14 preference pairs from labelled SDG trajectories + direct-conflict pairs.

Two sources:

* ``sdg``    - real rollouts from step 1: chosen = explicit_compliance trajectory,
               rejected = execution_violation / task_priority trajectory on the same
               task prompt. Capped per task so one task cannot dominate.
* ``direct`` - the v10 direct-instruction pairs (user orders the forbidden action,
               chosen refuses and cites the constraint id, rejected claims完成).
               Reused verbatim; v10 already proved they do not overlap the eval set.

The output is the TRL DPO row shape (``prompt_messages`` / ``chosen`` / ``rejected``)
plus metadata used by the severity-weighted trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
SEVERITY = {"path_escape": 3, "delete_without_confirm": 2, "edit_tests": 1, "policy": 1}
MAX_PAIRS_PER_TASK = 3


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    constraint_id = str(record["constraint_id"])
    constraint_text = str(record["constraint_text"])
    system = f"{SYSTEM_HEADER}\n- [{constraint_id}] {constraint_text}"
    messages = [{"role": "system", "content": system}]
    for turn in record.get("user_turns") or []:
        messages.append({"role": "user", "content": str(turn)})
    return messages


def sdg_pairs(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task_id"]].append(record)

    pairs: list[dict[str, Any]] = []
    skipped_empty = 0
    for task_id in sorted(by_task):
        rows = by_task[task_id]
        positives = [
            row
            for row in rows
            if row["training_role"] == "positive" and (row["trajectory"].get("reply") or "").strip()
        ]
        negatives = [
            row
            for row in rows
            if row["training_role"] == "negative" and (row["trajectory"].get("reply") or "").strip()
        ]
        if not positives or not negatives:
            continue
        made = 0
        for index in range(len(positives) * len(negatives)):
            if made >= MAX_PAIRS_PER_TASK:
                break
            positive = positives[index % len(positives)]
            negative = negatives[(index // len(positives)) % len(negatives)]
            if positive["sample_id"] == negative["sample_id"]:
                continue
            pairs.append(
                {
                    "pair_id": f"V14-SDG-{task_id}-{made:02d}",
                    "task_id": task_id,
                    "layer": positive["category"],
                    "severity": int(positive["severity"]),
                    "constraint_id": positive["constraint_id"],
                    "constraint_text": positive["constraint_text"],
                    "prompt_messages": prompt_messages(positive),
                    "chosen": positive["trajectory"]["reply"],
                    "rejected": negative["trajectory"]["reply"],
                    "source": "sdg_trajectory",
                    "chosen_sample_id": positive["sample_id"],
                    "rejected_sample_id": negative["sample_id"],
                }
            )
            made += 1
    return pairs, {"skipped_empty": skipped_empty}


def direct_pairs(eval_task_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in build_direct_pairs():
        if row["task_id"] in eval_task_ids:
            raise ValueError(f"direct pair overlaps eval set: {row['task_id']}")
        rows.append(
            {
                **row,
                "source": "direct_instruction",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="build v14 DPO preference pairs")
    parser.add_argument(
        "--train-sdg", type=Path, default=Path("private/v14/step1_sdg/train_sdg.jsonl")
    )
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("private/v14/step4_dpo/data"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    records = read_jsonl(args.train_sdg)
    eval_task_ids = {str(row["task_id"]) for row in read_jsonl(args.eval_file)}

    pairs, sdg_stats = sdg_pairs(records)
    direct = direct_pairs(eval_task_ids)
    all_pairs = pairs + direct

    task_ids = sorted({row["task_id"] for row in all_pairs})
    overlap = sorted(set(task_ids) & eval_task_ids)
    if overlap:
        raise ValueError(f"train/eval task overlap: {overlap}")

    # Deterministic val split by task id, so a task never appears on both sides.
    def is_val(task_id: str) -> bool:
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
        return (int(digest[:8], 16) % 1000) / 1000.0 < args.val_fraction

    train_rows = [row for row in all_pairs if not is_val(row["task_id"])]
    val_rows = [row for row in all_pairs if is_val(row["task_id"])]
    if not val_rows:
        val_rows = all_pairs[-max(1, len(all_pairs) // 10) :]
        train_rows = all_pairs[: -len(val_rows)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train_rows), ("val.jsonl", val_rows)):
        (args.out_dir / name).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    weights = {}
    for row in all_pairs:
        weights[str(row["constraint_id"])] = float(SEVERITY.get(str(row["layer"]), 1))
    manifest = {
        "version": "v14-step4-pref-pairs-1",
        "source_sha256": sha256_file(args.train_sdg),
        "eval_sha256": sha256_file(args.eval_file),
        "pairs": len(all_pairs),
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "by_source": dict(Counter(row["source"] for row in all_pairs)),
        "by_layer": dict(sorted(Counter(row["layer"] for row in all_pairs).items())),
        "by_severity": dict(sorted(Counter(row["severity"] for row in all_pairs).items())),
        "task_overlap_with_eval": overlap,
        "tasks": len(task_ids),
        "max_pairs_per_task": MAX_PAIRS_PER_TASK,
        "severity_weights": dict(sorted(weights.items())),
        **sdg_stats,
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
