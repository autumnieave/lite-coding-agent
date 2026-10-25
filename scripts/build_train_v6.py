"""Freeze train-v6 by removing the 45 clean-held-out tasks from train-v5.

The difficulty screen reported 45 middle-state tasks. They were part of
train-v5, so a clean held-out evaluation requires removing *every trajectory*
for those task ids before retraining. This script only filters: it does not
regenerate data, change prompts, or silently drop overlapping val rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VERSION = "train-v6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(payload)
    return rows


def read_task_ids(path: Path) -> list[str]:
    """Read task ids from JSONL (``task_id`` or ``metadata.task_id``) or text."""

    ids: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            ids.append(text)
            continue
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        metadata = payload.get("metadata")
        task_id = payload.get("task_id")
        if not task_id and isinstance(metadata, Mapping):
            task_id = metadata.get("task_id")
        if not task_id:
            raise ValueError(f"{path}:{line_number} has no task_id")
        ids.append(str(task_id))
    if len(ids) != len(set(ids)):
        duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
        raise ValueError(f"duplicate held-out task ids: {duplicates}")
    if not ids:
        raise ValueError(f"no task ids in {path}")
    return sorted(ids)


def task_id_for_row(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("train row has no metadata object")
    task_id = str(metadata.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("train row has no metadata.task_id")
    return task_id


def origin_for(task_id: str) -> str:
    if task_id.startswith("U-TR4-"):
        return "train-v4-oracle"
    if task_id.startswith("U-TR5-"):
        return "train-v5-teacher"
    if task_id.startswith("U-TR-"):
        return "train-v2-retained"
    if task_id.startswith("PUB-XLAM-"):
        return "xlam-schema-mapped"
    return "unknown"


def _distribution(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    values: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata")
        value = metadata.get(key) if isinstance(metadata, Mapping) else None
        values[str(value or "unknown")] += 1
    return dict(sorted(values.items()))


def _origin_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(origin_for(task_id_for_row(row)) for row in rows).items()))


def build_manifest(
    *,
    source_train: Path,
    source_val: Path,
    tasks_file: Path,
    created_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    task_ids = read_task_ids(tasks_file)
    wanted = set(task_ids)
    train_rows = read_jsonl(source_train)
    val_rows = read_jsonl(source_val)

    present = Counter(task_id_for_row(row) for row in train_rows)
    missing = [task_id for task_id in task_ids if task_id not in present]
    if missing:
        raise ValueError(f"held-out task ids missing from train file: {missing}")

    removed_train = [row for row in train_rows if task_id_for_row(row) in wanted]
    kept_train = [row for row in train_rows if task_id_for_row(row) not in wanted]
    removed_val = [row for row in val_rows if task_id_for_row(row) in wanted]
    kept_val = [row for row in val_rows if task_id_for_row(row) not in wanted]
    if removed_val:
        raise ValueError(
            "held-out tasks unexpectedly appear in val; refusing to silently drop them: "
            f"{sorted({task_id_for_row(row) for row in removed_val})}"
        )
    if not kept_train:
        raise ValueError("train-v6 is empty")

    kept_train_ids = {task_id_for_row(row) for row in kept_train}
    kept_val_ids = {task_id_for_row(row) for row in kept_val}
    leaked = sorted(kept_train_ids & wanted)
    if leaked:
        raise ValueError(f"held-out task ids still present in train-v6: {leaked}")
    split_overlap = sorted(kept_train_ids & kept_val_ids)
    if split_overlap:
        raise ValueError(f"task ids appear in both train and val: {split_overlap}")

    removed_by_task = Counter(task_id_for_row(row) for row in removed_train)
    manifest: dict[str, Any] = {
        "version": VERSION,
        "created_at": created_at,
        "purpose": "clean held-out retraining: remove the 45 middle-state tasks from train-v5",
        "source": {
            "train_file": str(source_train),
            "val_file": str(source_val),
            "train_sha256": sha256(source_train),
            "val_sha256": sha256(source_val),
            "tasks_file": str(tasks_file),
            "tasks_sha256": sha256(tasks_file),
        },
        "heldout": {
            "tasks": len(task_ids),
            "task_ids": task_ids,
            "removed_train_rows": len(removed_train),
            "removed_val_rows": len(removed_val),
            "removed_rows_by_task": dict(sorted(removed_by_task.items())),
        },
        "train_examples": len(kept_train),
        "val_examples": len(kept_val),
        "removed_train_distribution": {
            "by_origin": _origin_distribution(removed_train),
            "by_category": _distribution(removed_train, "category"),
            "by_difficulty": _distribution(removed_train, "difficulty"),
            "by_context_tier": _distribution(removed_train, "context_tier"),
        },
        "train_distribution": {
            "by_origin": _origin_distribution(kept_train),
            "by_category": _distribution(kept_train, "category"),
            "by_difficulty": _distribution(kept_train, "difficulty"),
            "by_context_tier": _distribution(kept_train, "context_tier"),
        },
        "val_distribution": {
            "by_origin": _origin_distribution(kept_val),
            "by_category": _distribution(kept_val, "category"),
            "by_difficulty": _distribution(kept_val, "difficulty"),
        },
        "checks": {
            "heldout_tasks_present_in_train": len(task_ids),
            "heldout_tasks_missing_from_train": missing,
            "heldout_tasks_remaining_in_train": leaked,
            "heldout_tasks_remaining_in_val": [],
            "train_val_task_overlap": split_overlap,
        },
    }
    return kept_train, kept_val, manifest


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def freeze(
    *,
    source_dir: Path,
    tasks_file: Path,
    out_dir: Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    timestamp = created_at or datetime.now(UTC).isoformat()
    train_rows, val_rows, manifest = build_manifest(
        source_train=source_dir / "train.jsonl",
        source_val=source_dir / "val.jsonl",
        tasks_file=tasks_file,
        created_at=timestamp,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    manifest["train_sha256"] = sha256(train_path)
    manifest["val_sha256"] = sha256(val_path)
    combined = hashlib.sha256()
    combined.update(train_path.read_bytes())
    combined.update(val_path.read_bytes())
    manifest["combined_sha256"] = combined.hexdigest()
    manifest["outputs"] = {"train": str(train_path), "val": str(val_path)}
    text = "\n".join(
        [
            VERSION,
            f"created_at={timestamp}",
            f"train_examples={manifest['train_examples']}",
            f"val_examples={manifest['val_examples']}",
            f"train_sha256={manifest['train_sha256']}",
            f"val_sha256={manifest['val_sha256']}",
            f"combined_sha256={manifest['combined_sha256']}",
            f"heldout_tasks={manifest['heldout']['tasks']}",
            f"removed_train_rows={manifest['heldout']['removed_train_rows']}",
            "",
        ]
    )
    (out_dir / "TRAIN_V6_FROZEN.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "TRAIN_V6_FROZEN.txt").write_text(text, encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="freeze train-v6 by removing clean held-out tasks")
    parser.add_argument("--source-dir", type=Path, default=Path("private/sft_data_v5"))
    parser.add_argument(
        "--tasks-file", type=Path, default=Path("private/final_test_v3/tasks.jsonl")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("private/sft_data_v6"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = freeze(
        source_dir=args.source_dir,
        tasks_file=args.tasks_file,
        out_dir=args.out_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
