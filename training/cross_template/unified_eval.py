"""Unified reporting for the 32-item constraint probe and 45-task evaluation.

The probe section filters ``mode == "untagged"`` and reports behavior and
information constraints separately. The task section groups task IDs by
``PUB-XLAM`` / ``U-TR`` and reports both pooled rollout rate and equal-weight
task mean. Inputs are labeled paths so the same script can compare G0/G1 or
baseline/train-v6 without hardcoded experiment names.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

KINDS = ("behavior", "information")
VIOLATION_LABELS = {"executed_violation", "claimed_execution"}


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


def load_task_ids(path: Path) -> set[str]:
    """Read a 45-task allowlist from JSONL or from a summary manifest."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"task id file is empty: {path}")
    if path.suffix == ".jsonl":
        ids: set[str] = set()
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            task_id = str(payload.get("task_id") or "") if isinstance(payload, dict) else ""
            if not task_id:
                raise ValueError(f"{path}:{line_number} has no task_id")
            ids.add(task_id)
        if not ids:
            raise ValueError(f"no task ids found in: {path}")
        return ids
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"task id manifest must be a JSON object: {path}")
    stack: list[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            candidate = node.get("task_ids")
            if (
                isinstance(candidate, list)
                and candidate
                and all(isinstance(value, str) for value in candidate)
            ):
                return set(candidate)
            stack.extend(value for value in node.values() if isinstance(value, dict))
        elif isinstance(node, list):
            stack.extend(value for value in node if isinstance(value, dict))
    raise ValueError(f"no heldout.task_ids or task_ids list in: {path}")


def expand_paths(raw: str) -> list[Path]:
    matches = sorted(glob.glob(raw))
    if not matches:
        matches = [raw]
    paths: list[Path] = []
    for match in matches:
        path = Path(match)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"input path does not exist: {match}")
    if not paths:
        raise FileNotFoundError(f"no JSONL files under: {raw}")
    return paths


def parse_labeled_path(raw: str) -> tuple[str, Path]:
    label, separator, path_text = raw.partition("=")
    if not separator or not label.strip() or not path_text.strip():
        raise ValueError(f"expected LABEL=PATH, got: {raw!r}")
    return label.strip(), Path(path_text)


def read_labeled_inputs(values: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in values:
        label, path = parse_labeled_path(raw)
        for resolved in expand_paths(str(path)):
            grouped[label].extend(read_jsonl(resolved))
    if not grouped:
        raise ValueError("no labeled inputs were provided")
    return dict(grouped)


def probe_items(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in records:
        if record.get("mode") != "untagged":
            continue
        for item in record.get("constraints") or []:
            constraint = item.get("constraint") or {}
            probe = item.get("probe") or {}
            items.append(
                {
                    "task_id": str(record.get("task_id") or ""),
                    "kind": str(constraint.get("kind") or "unknown"),
                    "constraint_id": str(constraint.get("id") or ""),
                    "label": str(probe.get("label") or "missing"),
                    "compliant": probe.get("compliant"),
                }
            )
    return items


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def summarize_probe(
    records: Sequence[Mapping[str, Any]], expected_items: int, expected_tasks: int
) -> dict[str, Any]:
    items = probe_items(records)
    if len(items) != expected_items:
        raise ValueError(f"expected {expected_items} untagged probe items, got {len(items)}")
    tasks = {item["task_id"] for item in items}
    if len(tasks) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} probe tasks, got {len(tasks)}")
    by_kind: dict[str, Any] = {}
    for kind in KINDS:
        group = [item for item in items if item["kind"] == kind]
        labels = Counter(item["label"] for item in group)
        compliant = sum(item["compliant"] is True for item in group)
        violation = sum(
            item["compliant"] is False or item["label"] in VIOLATION_LABELS for item in group
        )
        unclassified = len(group) - compliant - violation
        by_kind[kind] = {
            "items": len(group),
            "compliant": compliant,
            "violation": violation,
            "unclassified": unclassified,
            "compliance_rate": _rate(compliant, len(group)),
            "label_counts": dict(sorted(labels.items())),
        }
    compliant = sum(1 for item in items if item["compliant"] is True)
    violation = sum(
        1 for item in items if item["compliant"] is False or item["label"] in VIOLATION_LABELS
    )
    return {
        "records": len(records),
        "tasks": len(tasks),
        "items": len(items),
        "by_kind": by_kind,
        "overall": {
            "items": len(items),
            "compliant": compliant,
            "violation": violation,
            "unclassified": len(items) - compliant - violation,
            "compliance_rate": _rate(compliant, len(items)),
        },
    }


def task_source(task_id: str) -> str:
    return "PUB-XLAM" if task_id.startswith("PUB-XLAM-") else "U-TR"


def summarize_tasks(records: Sequence[Mapping[str, Any]], expected_tasks: int) -> dict[str, Any]:
    task_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    seen: set[tuple[str, str]] = set()
    for record in records:
        task_id = str(record.get("task_id") or "")
        if not task_id or "task_success" not in record:
            raise ValueError("task run row lacks task_id or task_success")
        rollout = record.get("rollout_index")
        key = (task_id, str(rollout) if rollout is not None else str(len(seen)))
        if key in seen:
            continue
        seen.add(key)
        task_counts[task_id][1] += 1
        task_counts[task_id][0] += int(record.get("task_success") is True)
    if len(task_counts) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} tasks, got {len(task_counts)}")
    sources = {
        source: {"tasks": 0, "rollouts": 0, "successes": 0} for source in ("PUB-XLAM", "U-TR")
    }
    task_means: list[float] = []
    for task_id, (successes, count) in task_counts.items():
        source = task_source(task_id)
        sources[source]["tasks"] += 1
        sources[source]["rollouts"] += count
        sources[source]["successes"] += successes
        task_means.append(successes / count)
    for source, values in sources.items():
        values["pass_rate"] = _rate(values["successes"], values["rollouts"])
        values["task_mean"] = (
            round(
                sum(
                    task_counts[task_id][0] / task_counts[task_id][1]
                    for task_id in task_counts
                    if task_source(task_id) == source
                )
                / values["tasks"],
                6,
            )
            if values["tasks"]
            else None
        )
    total_successes = sum(value[0] for value in task_counts.values())
    total_rollouts = sum(value[1] for value in task_counts.values())
    return {
        "tasks": len(task_counts),
        "rollouts": total_rollouts,
        "successes": total_successes,
        "pass_rate": _rate(total_successes, total_rollouts),
        "task_mean": round(sum(task_means) / len(task_means), 6),
        "by_source": sources,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="unified constraint/task evaluation")
    parser.add_argument(
        "--probe-run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="untagged probe JSONL(s); repeat with different labels to compare",
    )
    parser.add_argument(
        "--task-run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="45-task run JSONL(s); repeat with different labels or shards",
    )
    parser.add_argument(
        "--control-vars",
        type=Path,
        default=Path("training/cross_template/CONTROL_VARS.md"),
    )
    parser.add_argument(
        "--task-ids",
        type=Path,
        default=None,
        help="optional 45-task allowlist (JSONL or clean-heldout summary.json)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.probe_run and not args.task_run:
        raise ValueError("provide at least one --probe-run or --task-run")
    control = load_control_vars(args.control_vars)
    payload: dict[str, Any] = {
        "version": "unified-eval-v1",
        "control_vars": str(args.control_vars),
        "seed": int(control["seed"]),
        "probe": {},
        "tasks": {},
    }
    if args.probe_run:
        probe_runs = read_labeled_inputs(args.probe_run)
        payload["probe"] = {
            "expected_items": int(control["probe_items"]),
            "expected_tasks": int(control["probe_tasks"]),
            "runs": {
                label: summarize_probe(
                    records, int(control["probe_items"]), int(control["probe_tasks"])
                )
                for label, records in probe_runs.items()
            },
        }
    if args.task_run:
        task_runs = read_labeled_inputs(args.task_run)
        if args.task_ids is not None:
            allowlist = load_task_ids(args.task_ids)
            filtered: dict[str, list[dict[str, Any]]] = {}
            for label, records in task_runs.items():
                selected = [
                    record for record in records if str(record.get("task_id") or "") in allowlist
                ]
                seen = {str(record.get("task_id") or "") for record in selected}
                missing = allowlist - seen
                if missing:
                    raise ValueError(
                        f"{label}: {len(missing)} allowlisted tasks are missing from the run"
                    )
                filtered[label] = selected
            task_runs = filtered
        payload["tasks"] = {
            "expected_tasks": int(control["evaluation_tasks"]),
            "task_ids": str(args.task_ids) if args.task_ids is not None else None,
            "runs": {
                label: summarize_tasks(records, int(control["evaluation_tasks"]))
                for label, records in task_runs.items()
            },
        }
    if args.out is not None and not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
