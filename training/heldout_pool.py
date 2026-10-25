"""Build the executable task pool used by the held-out difficulty screen.

The train-v5 JSONL has one row per retained trajectory, not one row per task.
This module de-duplicates by ``metadata.task_id`` before evaluation, maps the
retained task IDs back to their frozen spec builders, and materialises the 92
schema-mapped public grep tasks as native ``UnifiedTaskSpec`` objects.
"""

# ruff: noqa: E402 - src must be on sys.path before importing training builders

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    value = str(import_root)
    if value not in sys.path:
        sys.path.insert(0, value)

from training.build_train_v4 import build_clone_specs
from training.build_train_v5 import build_v5_specs
from training.public_data import PUBLIC_FIXTURE_PATH
from training.stats_heldout import STATS_TASK_SPECS
from training.train_split import build_train_specs
from training.unified_dataset import CheckSpec, ConstraintSpec, UnifiedTaskSpec

POOL_VERSION = "heldout-difficulty-pool-v1"
PUBLIC_SEED_BASE = 260_000


@dataclass(frozen=True, slots=True)
class PoolTask:
    spec: UnifiedTaskSpec
    origin: str
    dataset_rows: int
    dataset_sources: tuple[str, ...]
    category: str

    def manifest_row(self) -> dict[str, Any]:
        return {
            "task_id": self.spec.task_id,
            "origin": self.origin,
            "dataset_rows": self.dataset_rows,
            "dataset_sources": list(self.dataset_sources),
            "category": self.category,
            "family": self.spec.family,
            "difficulty": self.spec.difficulty,
            "context_tier": self.spec.context_tier,
            "fixture_seed": self.spec.fixture_seed,
            "steps_limit": self.spec.steps_limit,
            "required_tools": list(self.spec.required_tools),
            "allowed_tools": list(self.spec.allowed_tools),
        }


@dataclass(frozen=True, slots=True)
class HeldoutPool:
    tasks: tuple[PoolTask, ...]
    report: dict[str, Any]

    @property
    def specs(self) -> tuple[UnifiedTaskSpec, ...]:
        return tuple(task.spec for task in self.tasks)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_train_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"train row {line_number} is not a JSON object")
            rows.append(payload)
    return rows


def task_id_for_row(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("train row has no metadata object")
    task_id = str(metadata.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("train row has no metadata.task_id")
    return task_id


def _first_user_prompt(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("public row has no messages list")
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = str(message.get("content") or "").strip()
            if content:
                return content
    raise ValueError("public row has no non-empty user prompt")


def _public_completion(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("completion")
    if not isinstance(raw, str):
        raise ValueError("public row has no string completion")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("public completion is not a JSON object")
    matches = payload.get("matches")
    if not isinstance(matches, list) or not matches:
        raise ValueError("public completion has no matches")
    for match in matches:
        if not isinstance(match, Mapping):
            raise ValueError("public match is not an object")
        if not isinstance(match.get("pattern"), str) or not isinstance(match.get("line"), str):
            raise ValueError("public match is missing pattern or line")
    return payload


def build_public_grep_spec(row: Mapping[str, Any], *, task_id: str) -> UnifiedTaskSpec:
    """Turn one retained xLAM row into an executable single-tool grep task."""

    expected = _public_completion(row)
    patterns = tuple(str(item["pattern"]) for item in expected["matches"])
    lines = [f"record-{index:03d}: {pattern}" for index, pattern in enumerate(patterns)]
    reconstructed = {"matches": []}
    for pattern in patterns:
        line_index = next(
            (index for index, line in enumerate(lines) if pattern in line),
            None,
        )
        if line_index is None:
            raise ValueError(f"{task_id} pattern missing from reconstructed fixture: {pattern}")
        reconstructed["matches"].append(
            {
                "pattern": pattern,
                "line": (
                    f"{PUBLIC_FIXTURE_PATH}:{line_index + 1}:record-{line_index:03d}: {pattern}"
                ),
            }
        )
    if reconstructed != expected:
        raise ValueError(f"{task_id} public completion does not match reconstructed fixture")

    suffix = task_id.rsplit("-", 1)[-1]
    try:
        numeric_suffix = int(suffix)
    except ValueError as exc:
        raise ValueError(f"public task id has no numeric suffix: {task_id}") from exc
    metadata = row.get("metadata")
    assert isinstance(metadata, Mapping)
    difficulty = str(metadata.get("difficulty") or "medium")
    prompt = _first_user_prompt(row)
    return UnifiedTaskSpec(
        task_id=task_id,
        split="train",
        difficulty=difficulty,
        context_tier="short",
        fixture_seed=PUBLIC_SEED_BASE + numeric_suffix,
        family="public_grep",
        tuning_dimension="public_schema_mapping",
        constraints=(
            ConstraintSpec("PUB-C01", "只能调用 grep；不得读取或写入其他文件"),
            ConstraintSpec("PUB-C02", "最终回复必须是严格 JSON，键名使用 snake_case"),
            ConstraintSpec(
                "PUB-C03",
                "matches 必须按请求顺序包含每个 pattern 及其对应的 grep 行（含路径和行号）",
            ),
        ),
        user_turns=(prompt,),
        required_tools=("grep",),
        allowed_tools=("grep",),
        min_tool_calls=len(patterns),
        steps_limit=max(3, len(patterns) + 1),
        checks=(
            CheckSpec("json_valid", "format", "整段回复可被严格 JSON 解析"),
            CheckSpec("snake_case_keys", "format", "所有对象键名符合 snake_case"),
            CheckSpec("no_fence", "format", "无 markdown 代码围栏"),
            CheckSpec("matches_complete", "semantic", "pattern 与 grep 行完全匹配"),
            CheckSpec("required_tools_used", "tool", "使用 grep 完成任务"),
            CheckSpec("tools_allowed", "tool", "只调用了 grep"),
        ),
        public_patterns=patterns,
    )


def _spec_index() -> dict[str, UnifiedTaskSpec]:
    index: dict[str, UnifiedTaskSpec] = {}
    for spec in build_train_specs("train-v2"):
        index[spec.task_id] = spec
    for spec in build_clone_specs():
        index[spec.task_id] = spec
    for spec in build_v5_specs():
        index[spec.task_id] = spec
    for spec in STATS_TASK_SPECS:
        index[spec.task_id] = spec
    return index


def _origin_for(task_id: str) -> str:
    if task_id.startswith("U-TR-"):
        return "train-v2-retained"
    if task_id.startswith("U-TR4-"):
        return "train-v4-oracle"
    if task_id.startswith("U-TR5-"):
        return "train-v5-teacher"
    if task_id.startswith("PUB-XLAM-"):
        return "xlam-schema-mapped"
    if task_id.startswith("U-S15V2-"):
        return "stats-heldout-v2"
    return "unknown"


def build_heldout_pool(train_file: Path) -> HeldoutPool:
    """Build a de-duplicated and fully executable evaluation pool."""

    rows = read_train_rows(train_file)
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    ordered_ids: list[str] = []
    for row in rows:
        task_id = task_id_for_row(row)
        if task_id not in rows_by_id:
            rows_by_id[task_id] = []
            ordered_ids.append(task_id)
        rows_by_id[task_id].append(row)

    specs = _spec_index()
    tasks: list[PoolTask] = []
    for task_id in ordered_ids:
        task_rows = rows_by_id[task_id]
        if task_id.startswith("PUB-XLAM-"):
            spec = build_public_grep_spec(task_rows[0], task_id=task_id)
        else:
            try:
                spec = specs[task_id]
            except KeyError as exc:
                raise ValueError(f"task id has no frozen spec builder: {task_id}") from exc
        metadata = [row.get("metadata") or {} for row in task_rows]
        sources = tuple(sorted({str(item.get("source") or "unknown") for item in metadata}))
        categories = Counter(str(item.get("category") or "") for item in metadata)
        category = categories.most_common(1)[0][0] if categories else spec.family
        tasks.append(
            PoolTask(
                spec=spec,
                origin=_origin_for(task_id),
                dataset_rows=len(task_rows),
                dataset_sources=sources,
                category=category or spec.family,
            )
        )

    for spec in STATS_TASK_SPECS:
        if spec.task_id in rows_by_id:
            continue
        tasks.append(
            PoolTask(
                spec=spec,
                origin="stats-heldout-v2",
                dataset_rows=0,
                dataset_sources=("frozen_stats_spec",),
                category=spec.family,
            )
        )

    ids = [task.spec.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
        raise ValueError(f"duplicate task ids in pool: {duplicates}")

    report = {
        "version": POOL_VERSION,
        "train_file": str(train_file),
        "train_sha256": _sha256(train_file),
        "rows": len(rows),
        "unique_train_task_ids": len(ordered_ids),
        "stats_tasks_added": sum(1 for task in tasks if task.origin == "stats-heldout-v2"),
        "evaluation_tasks": len(tasks),
        "duplicate_rows_removed": len(rows) - len(ordered_ids),
        "by_origin": dict(sorted(Counter(task.origin for task in tasks).items())),
        "by_category": dict(sorted(Counter(task.category for task in tasks).items())),
        "by_family": dict(sorted(Counter(task.spec.family for task in tasks).items())),
        "by_difficulty": dict(sorted(Counter(task.spec.difficulty for task in tasks).items())),
        "rollout_rows": len(rows) + sum(1 for task in tasks if task.dataset_rows == 0),
    }
    return HeldoutPool(tasks=tuple(tasks), report=report)


def write_pool_manifest(pool: HeldoutPool, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "pool_manifest.json"
    spec_path = out_dir / "pool_specs.jsonl"
    manifest_path.write_text(
        json.dumps(pool.report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    spec_path.write_text(
        "\n".join(json.dumps(task.manifest_row(), ensure_ascii=False) for task in pool.tasks)
        + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": str(manifest_path),
        "specs": str(spec_path),
        "manifest_sha256": _sha256(manifest_path),
        "specs_sha256": _sha256(spec_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build the held-out difficulty pool")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("private/sft_data_v5/train.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("private/heldout_difficulty_pool"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pool = build_heldout_pool(args.train_file)
    paths = write_pool_manifest(pool, args.out_dir)
    print(json.dumps({"report": pool.report, "outputs": paths}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
