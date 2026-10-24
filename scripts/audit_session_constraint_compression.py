"""Audit whether session constraints were compressed out of history.

The session-constraint probe does not persist ``compressed_out_of_history`` or
``replenished_ids``.  This script reconstructs the same visibility judgment used
by ``scripts.trace_utils.scan_constraints`` from the saved summary and final
context: a constraint is compressed out when its original declaration is gone
and its id/body are not verbatim in the summary message.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DECLARATION_MARKER = "以下是本次会话必须一直遵守的约束"
TAGGED_DECLARATION_MARKER = "[CONSTRAINT]"
REPLENISH_HEADING = "[约束补录]"
BODY_FRAGMENT_CHARS = 12


def audit_item(row: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    spec = item.get("constraint") if isinstance(item.get("constraint"), Mapping) else {}
    constraint_id = str(spec.get("id") or item.get("constraint_id") or "")
    body = str(spec.get("text") or "")
    raw_summary = str(row.get("raw_summary") or "")
    final_context = str(row.get("final_context") or "")

    declaration_present = (
        DECLARATION_MARKER in final_context or TAGGED_DECLARATION_MARKER in final_context
    )
    same_message_verbatim = bool(
        constraint_id and body and constraint_id in raw_summary and body in raw_summary
    )
    compressed_out_of_history = not declaration_present and not same_message_verbatim
    body_fragment_present = bool(body and body[:BODY_FRAGMENT_CHARS] in raw_summary)
    replenished = int(row.get("replenished") or 0)
    replenish_heading_present = REPLENISH_HEADING in final_context

    return {
        "condition": row.get("condition"),
        "mode": row.get("mode"),
        "task_id": row.get("task_id"),
        "category": row.get("category"),
        "constraint_kind": spec.get("kind") or item.get("kind"),
        "constraint_id": constraint_id,
        "constraint_text": body,
        "declaration_message_present": declaration_present,
        "same_message_verbatim": same_message_verbatim,
        "body_fragment_present": body_fragment_present,
        "semantic_present": bool(item.get("semantic_present")),
        "compressed_out_of_history": compressed_out_of_history,
        "replenished": replenished,
        "replenished_ids_empty": replenished == 0,
        "replenish_heading_present": replenish_heading_present,
    }


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename in ("baseline.jsonl", "sft.jsonl"):
        path = input_dir / filename
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for row in rows:
        for item in row.get("constraints") or []:
            items.append(audit_item(row, item))

    by_condition: dict[str, dict[str, int]] = {}
    for item in items:
        key = f"{item['condition']}:{item['mode']}"
        bucket = by_condition.setdefault(
            key,
            {
                "constraints": 0,
                "compressed_true": 0,
                "body_fragment_absent": 0,
                "semantic_absent": 0,
            },
        )
        bucket["constraints"] += 1
        bucket["compressed_true"] += int(item["compressed_out_of_history"])
        bucket["body_fragment_absent"] += int(not item["body_fragment_present"])
        bucket["semantic_absent"] += int(not item["semantic_present"])

    runs_with_compressed = sum(
        1
        for row in rows
        if any(
            audit_item(row, item)["compressed_out_of_history"]
            for item in (row.get("constraints") or [])
        )
    )
    return {
        "runs": len(rows),
        "constraints": len(items),
        "tier4_runs": sum(bool(row.get("tier4_triggered")) for row in rows),
        "runs_with_compressed_true": runs_with_compressed,
        "compressed_true": sum(int(item["compressed_out_of_history"]) for item in items),
        "replenished_nonzero_runs": sum(int(row.get("replenished") or 0) for row in rows),
        "runs_with_replenish_heading": sum(
            REPLENISH_HEADING in str(row.get("final_context") or "") for row in rows
        ),
        "by_condition": by_condition,
    }


def export(input_dir: Path, out_path: Path, *, summary_path: Path | None = None) -> dict[str, Any]:
    rows = load_rows(input_dir)
    summary = summarize(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(audit_item(rows[0], (rows[0].get("constraints") or [{}])[0]))
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for item in row.get("constraints") or []:
                writer.writerow(audit_item(row, item))
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="audit session-constraint compression")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("private/kaggle_pull_session_constraint_v2/session_constraint"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/evidence/会话约束compressed字段复核.csv")
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    summary = export(args.input_dir, args.out, summary_path=args.summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
