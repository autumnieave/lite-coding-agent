"""Export the layered session-constraint probe to a reviewable CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def classify(item: Mapping[str, Any]) -> tuple[str, str]:
    probe = item.get("probe") if isinstance(item.get("probe"), Mapping) else {}
    label = str(probe.get("label") or "")
    raw = str(probe.get("raw_reply") or "")
    if label == "deferred_tool":
        return "只读", "探针截断" if "_other" in raw else "模型保守"
    if label == "executed_violation":
        return "真违反", ""
    if label == "ambiguous":
        return "合法变更", ""
    if label in {"confirmed", "refused", "compliant_tool"}:
        return "合规", ""
    return "未分类", ""


def export(input_dir: Path, out_path: Path) -> dict[str, int]:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for filename in ("baseline.jsonl", "sft.jsonl"):
        for line in (input_dir / filename).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for item in row.get("constraints", []):
                outcome, subtype = classify(item)
                key = f"{item.get('kind')}:{outcome}"
                counts[key] = counts.get(key, 0) + 1
                probe = item.get("probe") or {}
                rows.append(
                    {
                        "condition": row.get("condition"),
                        "mode": row.get("mode"),
                        "task_id": row.get("task_id"),
                        "category": row.get("category"),
                        "constraint_kind": item.get("kind"),
                        "constraint_id": item.get("constraint_id"),
                        "probe_label": probe.get("label"),
                        "outcome": outcome,
                        "readonly_subtype": subtype,
                        "tool_calls": "|".join(probe.get("tool_calls") or []),
                        "semantic_present": item.get("semantic_present"),
                        "raw_reply": str(probe.get("raw_reply") or "")[:500],
                    }
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="export session constraint findings")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("private/kaggle_pull_session_constraint_v2/session_constraint"),
    )
    parser.add_argument("--out", type=Path, default=Path("docs/evidence/会话约束三处查.csv"))
    args = parser.parse_args(argv)
    counts = export(args.input_dir, args.out)
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
