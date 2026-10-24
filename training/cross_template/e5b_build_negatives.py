"""Build the E5-B positive/negative SFT mix from baseline probe failures.

The negative rows are *not* re-synthesised from model outputs. Each one is tied
to one ``executed_violation`` item in ``baseline.jsonl`` and is replaced with a
short refusal (behavior constraints) or read-only response (information
constraints). The 365 positives are a deterministic 401-task sample from
train-v6, leaving exactly 401 training rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def load_control_vars(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if match is None:
        raise ValueError(f"CONTROL_VARS.md has no fenced JSON block: {path}")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError(f"control variables must be a JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_violations(
    records: Iterable[Mapping[str, Any]],
) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    for record in records:
        for item in record.get("constraints") or []:
            probe = item.get("probe") or {}
            if probe.get("label") == "executed_violation":
                yield record, item


def negative_completion(kind: str, constraint_id: str, constraint_text: str) -> str:
    if kind == "behavior":
        return f"拒绝：{constraint_id} 约束为“{constraint_text}”，该请求会违反约束，不能执行。"
    return f"只读：{constraint_id} 已限定当前任务文件，不能处理其他文件；本次不执行越界写入。"


def build_negative_rows(
    records: Sequence[Mapping[str, Any]], expected_negatives: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record, item in iter_violations(records):
        constraint = item.get("constraint") or {}
        constraint_id = str(constraint.get("id") or "")
        kind = str(constraint.get("kind") or "")
        if not constraint_id or kind not in {"behavior", "information"}:
            raise ValueError("executed_violation item lacks constraint id/kind")
        task_id = str(record.get("task_id") or "")
        mode = str(record.get("mode") or "")
        probe = str(constraint.get("probe") or "")
        context = str(record.get("final_context") or "")
        if not task_id or not mode or not probe or not context:
            raise ValueError(f"incomplete negative source for {task_id}/{mode}/{constraint_id}")
        label = "behavior_refusal" if kind == "behavior" else "information_read_only"
        rows.append(
            {
                "id": f"e5b:negative:{task_id}:{mode}:{constraint_id}",
                "messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": probe},
                ],
                "completion": negative_completion(
                    kind, constraint_id, str(constraint.get("text") or "")
                ),
                "metadata": {
                    "task_id": task_id,
                    "mode": mode,
                    "constraint_id": constraint_id,
                    "constraint_kind": kind,
                    "negative_label": label,
                    "source": "baseline_executed_violation",
                    "training_version": "e5b-g1",
                    "split": "train",
                },
            }
        )
    if len(rows) != expected_negatives:
        raise ValueError(
            f"expected {expected_negatives} executed_violation negatives, found {len(rows)}"
        )
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("negative sample ids are not unique")
    return rows


def sample_positives(
    rows: Sequence[Mapping[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if count < 1 or count > len(rows):
        raise ValueError(f"positive sample count out of range: {count} of {len(rows)}")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), count))
    sampled: list[dict[str, Any]] = []
    for index in indices:
        row = dict(rows[index])
        metadata = dict(row.get("metadata") or {})
        metadata["e5b_arm"] = "g1"
        row["metadata"] = metadata
        sampled.append(row)
    return sampled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build E5-B positive/negative mix")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "private/kaggle_pull_session_constraint64_v1/session_constraint64/baseline.jsonl"
        ),
    )
    parser.add_argument("--positives", type=Path, default=Path("private/sft_data_v6/train.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("private/cross_template/e5b"))
    parser.add_argument(
        "--control-vars",
        type=Path,
        default=Path("training/cross_template/CONTROL_VARS.md"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    control = load_control_vars(args.control_vars)
    total_rows = int(control["total_train_rows"])
    negatives_expected = int(control["e5b_negatives"])
    if total_rows <= negatives_expected:
        raise ValueError("positive sample count must be positive")

    baseline_records = read_jsonl(args.baseline)
    positives_all = read_jsonl(args.positives)
    if len(positives_all) != total_rows:
        raise ValueError(f"expected {total_rows} train-v6 positives, found {len(positives_all)}")
    negatives = build_negative_rows(baseline_records, negatives_expected)
    positives = sample_positives(
        positives_all, total_rows - negatives_expected, int(control["seed"])
    )
    combined = positives + negatives
    if len(combined) != total_rows:
        raise ValueError(f"combined row count is {len(combined)}, expected {total_rows}")
    if len({str(row.get("id")) for row in combined}) != len(combined):
        raise ValueError("combined rows contain duplicate ids")

    manifest = {
        "version": "e5b-build-v1",
        "input": {
            "baseline": str(args.baseline),
            "baseline_sha256": sha256(args.baseline),
            "positives": str(args.positives),
            "positives_sha256": sha256(args.positives),
        },
        "outputs": {
            "train": str(args.out_dir / "train.jsonl"),
            "negatives": str(args.out_dir / "negatives.jsonl"),
            "manifest": str(args.out_dir / "manifest.json"),
        },
        "control_vars": str(args.control_vars),
        "seed": int(control["seed"]),
        "counts": {
            "positives": len(positives),
            "negatives": len(negatives),
            "total": len(combined),
        },
        "negatives": [
            {
                "id": row["id"],
                "task_id": row["metadata"]["task_id"],
                "mode": row["metadata"]["mode"],
                "kind": row["metadata"]["constraint_kind"],
                "constraint_id": row["metadata"]["constraint_id"],
                "negative_label": row["metadata"]["negative_label"],
            }
            for row in negatives
        ],
    }
    if args.dry_run:
        print(json.dumps({"status": "dry_run", **manifest}, ensure_ascii=False, indent=2))
        return 0

    write_jsonl(args.out_dir / "train.jsonl", combined)
    write_jsonl(args.out_dir / "negatives.jsonl", negatives)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
