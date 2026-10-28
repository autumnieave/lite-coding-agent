"""Compare tagged-mode replenishment before and after the ID+body fix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONDITIONS = ("v6_pre", "v6_post")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(payload)
    return rows


def _items(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    items: list[Mapping[str, Any]] = []
    for row in rows:
        items.extend(row.get("constraints") or [])
    return items


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = _items(rows)
    labels = Counter(str((item.get("probe") or {}).get("label") or "missing") for item in items)
    compliant = sum(item.get("probe", {}).get("compliant") is True for item in items)
    violating = sum(item.get("probe", {}).get("compliant") is False for item in items)
    unclassified = len(items) - compliant - violating
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in ("behavior", "information"):
        subset = [item for item in items if item.get("constraint", {}).get("kind") == kind]
        by_kind[kind] = {
            "total": len(subset),
            "semantic": sum(bool(item.get("semantic_present")) for item in subset),
            "code": sum(bool(item.get("code_present")) for item in subset),
            "compliant": sum(item.get("probe", {}).get("compliant") is True for item in subset),
            "violating": sum(item.get("probe", {}).get("compliant") is False for item in subset),
            "unclassified": sum(item.get("probe", {}).get("compliant") is None for item in subset),
        }
    return {
        "tasks": len(rows),
        "probes": len(items),
        "semantic": sum(bool(item.get("semantic_present")) for item in items),
        "code": sum(bool(item.get("code_present")) for item in items),
        "compliant": compliant,
        "violating": violating,
        "unclassified": unclassified,
        "replenished_sum": sum(int(row.get("replenished") or 0) for row in rows),
        "rows_with_replenished": sum(bool(row.get("replenished")) for row in rows),
        "labels": dict(sorted(labels.items())),
        "by_kind": by_kind,
    }


def _key(row: Mapping[str, Any], item: Mapping[str, Any]) -> tuple[str, str, str]:
    constraint_id = str(item.get("constraint", {}).get("id") or item.get("constraint_id") or "")
    return (str(row.get("task_id") or ""), str(row.get("mode") or ""), constraint_id)


def paired_transitions(
    pre_rows: Sequence[Mapping[str, Any]], post_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    pre = {
        (row["task_id"], row["mode"], item["constraint"]["id"]): item
        for row in pre_rows
        for item in row.get("constraints", [])
    }
    post = {
        (row["task_id"], row["mode"], item["constraint"]["id"]): item
        for row in post_rows
        for item in row.get("constraints", [])
    }
    if set(pre) != set(post):
        raise ValueError("pre/post probe keys do not match")
    semantic = Counter()
    compliance = Counter()
    for key in sorted(pre):
        before = pre[key]
        after = post[key]
        semantic[(bool(before.get("semantic_present")), bool(after.get("semantic_present")))] += 1
        compliance[
            (
                (before.get("probe") or {}).get("compliant"),
                (after.get("probe") or {}).get("compliant"),
            )
        ] += 1
    return {
        "semantic": {f"{before}->{after}": count for (before, after), count in semantic.items()},
        "compliance": {
            f"{before}->{after}": count for (before, after), count in compliance.items()
        },
    }


def analyze(input_dir: Path) -> dict[str, Any]:
    rows = {condition: load_jsonl(input_dir / f"{condition}.jsonl") for condition in CONDITIONS}
    return {
        "version": "replenish-tagged-analysis-v1",
        "conditions": {condition: summarize(rows[condition]) for condition in CONDITIONS},
        "paired": paired_transitions(rows["v6_pre"], rows["v6_post"]),
        "note": (
            "Tagged mode only; forced ID-only summary; directional evidence, no significance test."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="analyze tagged replenishment comparison")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze(args.input_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
