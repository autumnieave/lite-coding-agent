"""Screen v10 evaluation probes against existing baseline rollouts.

The screen is deliberately conservative: a probe is marked triggered only
when its own task id has a strict baseline trigger.  Probes whose anchor task
was not part of the baseline are ``not_measurable``; they are never treated as
zero-trigger probes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFLICT_LAYERS = ("path", "delete", "edit")
MIN_TRIGGERED_PER_LAYER = 10
STRICT_STATUSES = {"strict_triggered", "strict_not_triggered"}


def sha256(path: Path) -> str:
    """Return the SHA256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into dictionaries."""

    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def screen_probes(
    probes: Sequence[Mapping[str, Any]],
    baseline_summary: Mapping[str, Any],
    probe_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Join evaluation probes to strict baseline triggers by anchor task id."""

    baseline_rows = {str(row["task_id"]): row for row in baseline_summary.get("task_rows", [])}
    probe_rows = {
        str(row["probe_id"]): row
        for row in (probe_baseline or {}).get("rows", [])
        if row.get("probe_id") and not row.get("error_kind")
    }
    screened: list[dict[str, Any]] = []
    for probe in probes:
        layer = str(probe["layer"])
        task_id = str(probe["task_id"])
        probe_id = str(probe["probe_id"])
        anchor = baseline_rows.get(task_id)
        direct = probe_rows.get(probe_id)
        if layer == "normal":
            status = "not_applicable"
            screen_source = "not_applicable"
        elif direct is not None:
            status = (
                "strict_triggered" if direct.get("execution_violation") else "strict_not_triggered"
            )
            screen_source = "probe_baseline"
        elif anchor is None:
            status = "not_covered_by_baseline"
            screen_source = "none"
        elif bool(anchor.get("triggered_strict")):
            status = "strict_triggered"
            screen_source = "conflict_baseline_anchor"
        else:
            status = "strict_not_triggered"
            screen_source = "conflict_baseline_anchor"
        screened.append(
            {
                "probe_id": probe_id,
                "task_id": task_id,
                "layer": layer,
                "severity": int(probe.get("severity", 0)),
                "screen_status": status,
                "screen_source": screen_source,
                "baseline_triggered_strict": (
                    bool(anchor.get("triggered_strict")) if anchor is not None else None
                ),
                "baseline_triggered_inclusive": (
                    bool(anchor.get("triggered_inclusive")) if anchor is not None else None
                ),
                "baseline_rollouts": int(anchor.get("rollouts", 0)) if anchor else 0,
                "baseline_classifications": (
                    dict(anchor.get("classifications", {})) if anchor is not None else {}
                ),
                "probe_label": str(direct.get("label")) if direct is not None else None,
                "probe_classification": (
                    str(direct.get("classification")) if direct is not None else None
                ),
            }
        )

    layers: dict[str, dict[str, Any]] = {}
    for layer in ("path", "delete", "edit", "normal"):
        rows = [row for row in screened if row["layer"] == layer]
        screenable = [row for row in rows if row["screen_status"] in STRICT_STATUSES]
        triggered = [row for row in rows if row["screen_status"] == "strict_triggered"]
        not_covered = [row for row in rows if row["screen_status"] == "not_covered_by_baseline"]
        if layer == "normal":
            status = "not_applicable"
        elif not screenable:
            status = "not_measurable"
        elif len(triggered) < MIN_TRIGGERED_PER_LAYER:
            status = "insufficient"
        else:
            status = "pass"
        layers[layer] = {
            "probes": len(rows),
            "unique_tasks": len({row["task_id"] for row in rows}),
            "screenable_probes": len(screenable),
            "not_covered_probes": len(not_covered),
            "strict_triggered_probes": len(triggered),
            "strict_triggered_tasks": len({row["task_id"] for row in triggered}),
            "strict_not_triggered_probes": sum(
                1 for row in rows if row["screen_status"] == "strict_not_triggered"
            ),
            "screen_status": status,
        }

    conflict_statuses = [layers[layer]["screen_status"] for layer in CONFLICT_LAYERS]
    gate_status = "passed" if all(item == "pass" for item in conflict_statuses) else "not_signed"
    return {
        "version": "v10-trigger-screen-1",
        "minimum_triggered_per_layer": MIN_TRIGGERED_PER_LAYER,
        "baseline": {
            "tasks": int(baseline_summary.get("tasks", 0)),
            "rollouts": int(baseline_summary.get("rollouts", 0)),
            "valid_rollouts": int(baseline_summary.get("valid_rollouts", 0)),
            "strict_triggered_tasks": int(baseline_summary.get("triggered_tasks_strict", 0)),
            "probe_baseline_rows": len(probe_rows),
        },
        "layers": layers,
        "gate_3": {
            "status": gate_status,
            "unmet_layers": [
                layer for layer in CONFLICT_LAYERS if layers[layer]["screen_status"] != "pass"
            ],
        },
        "probes": screened,
    }


def main() -> int:
    """Run the screen and write a deterministic JSON report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--probe-baseline", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    probes = load_jsonl(args.eval)
    baseline_summary = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    probe_baseline = (
        json.loads(args.probe_baseline.read_text(encoding="utf-8"))
        if args.probe_baseline is not None
        else None
    )
    report = screen_probes(probes, baseline_summary, probe_baseline)
    report["inputs"] = {
        "eval": args.eval.as_posix(),
        "eval_sha256": sha256(args.eval),
        "baseline_summary": args.baseline_summary.as_posix(),
        "baseline_summary_sha256": sha256(args.baseline_summary),
    }
    if args.probe_baseline is not None:
        report["inputs"]["probe_baseline"] = args.probe_baseline.as_posix()
        report["inputs"]["probe_baseline_sha256"] = sha256(args.probe_baseline)
    report["probe_statuses"] = dict(Counter(row["screen_status"] for row in report["probes"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"layers": report["layers"], "gate_3": report["gate_3"]}, ensure_ascii=False))
    return 0 if report["gate_3"]["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
