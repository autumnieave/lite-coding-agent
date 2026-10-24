"""Build the v16 GRPO prompt pool from the frozen v14 pool plus v15 seeds.

The v14 29-task pool is preserved verbatim. The ten v15 targeted candidates are
filtered by their 8-rollout middle-state precheck: only candidates with 2-6/8
explicit_compliance are admitted. The two excluded candidates stay visible in the
manifest so the pool size is never overstated.
"""

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

DEFAULT_V14_PROMPTS = REPO_ROOT / "private/v14/step5_grpo/prompts/prompts.jsonl"
DEFAULT_V15_CANDIDATES = REPO_ROOT / "private/v15/tasks_candidates.jsonl"
DEFAULT_V15_PRECHECK = REPO_ROOT / "private/v15/seed_precheck_analysis.json"
DEFAULT_EVAL_SETS = (
    REPO_ROOT / "private/v14/step2_benchmark/eval_set_frozen.jsonl",
    REPO_ROOT / "private/v15/eval_v15/eval.jsonl",
)
DEFAULT_OUT_DIR = REPO_ROOT / "private/v16/grpo_v16/prompts"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of objects."""

    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of raw bytes."""

    return hashlib.sha256(payload).hexdigest()


def build_pool(
    v14_prompts: list[dict[str, Any]],
    v15_candidates: list[dict[str, Any]],
    precheck: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge pools and return ``(records, report)``.

    The helper is deliberately pure so tests can exercise the exclusion logic
    without touching the filesystem.
    """

    candidate_meta = {str(row["task_id"]): row for row in v15_candidates}
    precheck_rows = {str(row["task_id"]): row for row in precheck.get("candidates", [])}

    missing = sorted(set(candidate_meta) - set(precheck_rows))
    if missing:
        raise ValueError(f"v15 candidates missing precheck rows: {missing}")

    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for task_id, meta in candidate_meta.items():
        check = precheck_rows[task_id].get("precheck") or {}
        if check.get("retained") and check.get("middle_state_2_6"):
            retained.append(meta)
        else:
            excluded.append(
                {
                    "task_id": task_id,
                    "explicit_compliance": check.get("explicit_compliance"),
                    "rollouts": check.get("rollouts"),
                    "reason": "outside 2-6/8 middle-state band",
                }
            )

    records: list[dict[str, Any]] = []
    for row in v14_prompts:
        records.append(
            {
                "task_id": str(row["task_id"]),
                "category": str(row.get("category") or ""),
                "layer": str(row["layer"]),
                "severity": int(row.get("severity") or 0),
                "source_set": "v14_frozen_pool",
                "task": row["task"],
            }
        )
    for meta in sorted(retained, key=lambda row: str(row["task_id"])):
        check = precheck_rows[str(meta["task_id"])]
        layer = str(meta.get("layer") or check.get("layer") or "")
        if not layer:
            raise ValueError(f"v15 candidate has no layer: {meta['task_id']}")
        records.append(
            {
                "task_id": str(meta["task_id"]),
                "category": str(meta.get("category") or check.get("category") or ""),
                "layer": layer,
                "severity": int(meta.get("severity") or check.get("severity") or 0),
                "source_set": "v15g_middle_state",
                "task": meta.get("task") or meta,
            }
        )

    task_ids = [row["task_id"] for row in records]
    duplicated = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
    if duplicated:
        raise ValueError(f"duplicate task ids in pool: {duplicated}")

    by_layer = Counter(row["layer"] for row in records)
    report = {
        "version": "v16-grpo-prompts-1",
        "pool_size": len(records),
        "candidate_count": len(v15_candidates),
        "v14_records": len(v14_prompts),
        "v15_retained": len(retained),
        "v15_excluded": excluded,
        "by_layer": dict(sorted(by_layer.items())),
        "by_source_set": dict(sorted(Counter(row["source_set"] for row in records).items())),
        "notes": (
            "v14 pool preserved verbatim; v15 candidates admitted only when the 8-rollout "
            "precheck lands in 2-6/8 explicit_compliance. Pool size is 37, not 39."
        ),
    }
    return records, report


def load_eval_task_ids(paths: list[Path]) -> dict[str, list[str]]:
    """Return ``{path: task ids}`` for frozen evaluation sets."""

    out: dict[str, list[str]] = {}
    for path in paths:
        out[str(path)] = sorted({str(row["task_id"]) for row in load_jsonl(path)})
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Build the v16 GRPO prompt pool")
    parser.add_argument("--v14-prompts", type=Path, default=DEFAULT_V14_PROMPTS)
    parser.add_argument("--v15-candidates", type=Path, default=DEFAULT_V15_CANDIDATES)
    parser.add_argument("--v15-precheck", type=Path, default=DEFAULT_V15_PRECHECK)
    parser.add_argument("--eval", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    v14_prompts = load_jsonl(args.v14_prompts)
    v15_candidates = load_jsonl(args.v15_candidates)
    precheck = json.loads(args.v15_precheck.read_text(encoding="utf-8"))
    records, report = build_pool(v14_prompts, v15_candidates, precheck)

    eval_paths = args.eval if args.eval else list(DEFAULT_EVAL_SETS)
    overlaps: dict[str, list[str]] = {}
    for path, task_ids in load_eval_task_ids(eval_paths).items():
        pool_ids = {row["task_id"] for row in records}
        overlap = sorted(pool_ids & set(task_ids))
        if overlap:
            raise ValueError(f"prompt pool overlaps frozen eval set {path}: {overlap}")
        overlaps[path] = []

    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records
    ).encode("utf-8")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "prompts.jsonl").write_bytes(payload)

    report["prompts_sha256"] = sha256_bytes(payload)
    report["eval_overlap"] = overlaps
    (args.out_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
