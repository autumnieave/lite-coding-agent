"""Aggregate the layered session-constraint probe into retention and behavior tables.

The experiment is directional: this script reports counts and rates, not
significance tests. It keeps unclassified probes separate instead of folding
them into either compliant or non-compliant.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

CONDITIONS = ("baseline", "v5", "v6", "sft")
MODES = ("untagged", "tagged")
KINDS = ("behavior", "information")
LABELS = (
    "text_lost",
    "text_present_behavior_noncompliant",
    "text_present_behavior_compliant",
    "unclassified",
)


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


def classify_retention(text_present: bool, compliant: bool | None) -> str:
    if not text_present:
        return "text_lost"
    if compliant is True:
        return "text_present_behavior_compliant"
    if compliant is False:
        return "text_present_behavior_noncompliant"
    return "unclassified"


def flatten(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for item in record.get("constraints") or []:
            spec = item.get("constraint") or {}
            probe = item.get("probe") or {}
            raw_code = bool(item.get("raw_code_present"))
            raw_semantic = bool(item.get("raw_semantic_present"))
            final_code = bool(item.get("code_present", item.get("text_present")))
            final_semantic = bool(item.get("semantic_present", item.get("text_present")))
            compliant = probe.get("compliant")
            rows.append(
                {
                    "condition": record.get("condition"),
                    "mode": record.get("mode"),
                    "task_id": record.get("task_id"),
                    "category": record.get("category"),
                    "kind": spec.get("kind"),
                    "constraint_id": spec.get("id"),
                    "raw_code_present": raw_code,
                    "raw_semantic_present": raw_semantic,
                    "raw_any_text_present": raw_code or raw_semantic,
                    "final_code_present": final_code,
                    "final_semantic_present": final_semantic,
                    "final_any_text_present": final_code or final_semantic,
                    "probe_label": probe.get("label"),
                    "compliant": compliant,
                    "classification": classify_retention(final_semantic, compliant),
                }
            )
    return rows


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["mode"]))].append(row)

    observed = {str(row["condition"]) for row in rows}
    ordered_conditions = [name for name in CONDITIONS if name in observed]
    ordered_conditions.extend(sorted(observed.difference(ordered_conditions)))
    conditions: dict[str, Any] = {}
    for condition in ordered_conditions:
        modes: dict[str, Any] = {}
        for mode in MODES:
            subset = grouped.get((condition, mode), [])
            if not subset:
                continue
            by_kind: dict[str, Any] = {}
            for kind in KINDS:
                kind_rows = [row for row in subset if row.get("kind") == kind]
                raw_code = sum(1 for row in kind_rows if row["raw_code_present"])
                raw_semantic = sum(1 for row in kind_rows if row["raw_semantic_present"])
                final_code = sum(1 for row in kind_rows if row["final_code_present"])
                final_semantic = sum(1 for row in kind_rows if row["final_semantic_present"])
                by_kind[kind] = {
                    "n": len(kind_rows),
                    "raw_code_present": raw_code,
                    "raw_code_rate": _rate(raw_code, len(kind_rows)),
                    "raw_semantic_present": raw_semantic,
                    "raw_semantic_rate": _rate(raw_semantic, len(kind_rows)),
                    "final_code_present": final_code,
                    "final_code_rate": _rate(final_code, len(kind_rows)),
                    "final_semantic_present": final_semantic,
                    "final_semantic_rate": _rate(final_semantic, len(kind_rows)),
                }
            probes = [row for row in subset if row.get("probe_label") is not None]
            compliant = sum(1 for row in probes if row.get("compliant") is True)
            noncompliant = sum(1 for row in probes if row.get("compliant") is False)
            unclassified = sum(1 for row in probes if row.get("compliant") is None)
            counted = compliant + noncompliant
            labels = Counter(str(row["probe_label"]) for row in probes)
            classifications = Counter(str(row["classification"]) for row in subset)
            modes[mode] = {
                "constraints": len(subset),
                "retention_by_kind": by_kind,
                "probes": len(probes),
                "compliant": compliant,
                "noncompliant": noncompliant,
                "unclassified": unclassified,
                "strict_compliance_rate": _rate(compliant, len(probes)),
                "classified_compliance_rate": _rate(compliant, counted),
                "probe_labels": dict(sorted(labels.items())),
                "classification": {label: classifications.get(label, 0) for label in LABELS},
                "text_present_behavior_noncompliant": classifications.get(
                    "text_present_behavior_noncompliant", 0
                ),
            }
        conditions[condition] = modes

    deltas: dict[str, Any] = {}
    for mode in MODES:
        baseline = conditions.get("baseline", {}).get(mode)
        if not baseline:
            continue
        for condition in ordered_conditions:
            if condition == "baseline":
                continue
            variant = conditions.get(condition, {}).get(mode)
            if not variant:
                continue
            deltas.setdefault(condition, {})[mode] = {
                "behavior_semantic_rate_delta": _delta(
                    baseline["retention_by_kind"]["behavior"]["final_semantic_rate"],
                    variant["retention_by_kind"]["behavior"]["final_semantic_rate"],
                ),
                "information_semantic_rate_delta": _delta(
                    baseline["retention_by_kind"]["information"]["final_semantic_rate"],
                    variant["retention_by_kind"]["information"]["final_semantic_rate"],
                ),
                "behavior_code_rate_delta": _delta(
                    baseline["retention_by_kind"]["behavior"]["final_code_rate"],
                    variant["retention_by_kind"]["behavior"]["final_code_rate"],
                ),
                "strict_compliance_rate_delta": _delta(
                    baseline["strict_compliance_rate"], variant["strict_compliance_rate"]
                ),
                "classified_compliance_rate_delta": _delta(
                    baseline["classified_compliance_rate"], variant["classified_compliance_rate"]
                ),
            }

    return {
        "version": "session-constraint-analysis-v1",
        "rows": len(rows),
        "conditions": conditions,
        "baseline_vs_sft": deltas.get("sft", {}),
        "baseline_vs_v5": deltas.get("v5", {}),
        "baseline_vs_v6": deltas.get("v6", {}),
        "baseline_vs_variants": deltas,
        "classification_basis": "final_semantic_present; code-only retention counts as text_lost",
        "note": "Directional evidence only; no McNemar or bootstrap is computed.",
    }


def _delta(baseline: float | None, sft: float | None) -> float | None:
    if baseline is None or sft is None:
        return None
    return round(sft - baseline, 4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="analyze session-constraint probe output")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records: list[dict[str, Any]] = []
    paths = sorted(args.input_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL probe outputs in {args.input_dir}")
    for path in paths:
        records.extend(load_jsonl(path))
    rows = flatten(records)
    summary = summarize(rows)
    out = args.out or (args.input_dir / "analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
