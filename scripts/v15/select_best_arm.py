"""Select the v14 GRPO best arm using a deterministic, documented ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_reports(pattern: str) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted(Path(".").glob(pattern)):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["_report_path"] = str(path)
        report["_arm"] = path.parent.name
        reports.append(report)
    return reports


def rank_key(report: dict[str, Any]) -> tuple[Any, ...]:
    normal = report.get("normal") or {}
    citation = report.get("citation") or {}
    normal_rate = float(normal.get("rate") or 0.0)
    path_exec = int((report.get("layers") or {}).get("path", {}).get("execution_violations", 999))
    return (
        normal_rate < 0.9,
        int(report.get("total_execution_violations", 999)),
        path_exec,
        -float(citation.get("cited_own_rate") or 0.0),
        -normal_rate,
        str(report.get("_arm")),
    )


def select(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("no reports")
    ranked = sorted(reports, key=rank_key)
    best = ranked[0]
    return {
        "version": "v15-best-arm-1",
        "selection_criteria": [
            "normal completion >= 90% preferred; if none, lowest total violation wins "
            "with a warning",
            "lowest total execution violations on v14 72",
            "then lowest path execution violations",
            "then highest correct constraint-ID citation rate",
            "then highest normal completion rate",
            "then arm name for deterministic tie-break",
        ],
        "selection_bias": (
            "Selected on the v14 72-probe set; its v14 interval is optimistic. "
            "The v15 100-probe result is an extension check, not a second selection round."
        ),
        "arms": [
            {
                "arm": report["_arm"],
                "total_execution_violations": report.get("total_execution_violations"),
                "path_execution_violations": report.get("layers", {})
                .get("path", {})
                .get("execution_violations"),
                "citation_rate": (report.get("citation") or {}).get("cited_own_rate"),
                "normal_rate": (report.get("normal") or {}).get("rate"),
                "report_path": report.get("_report_path"),
            }
            for report in ranked
        ],
        "best_arm": best["_arm"],
        "best_report_path": best.get("_report_path"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="select best v14 GRPO arm")
    parser.add_argument("--reports-glob", default="private/v14/step5_grpo/arms/*/report.json")
    parser.add_argument("--out", type=Path, default=Path("private/v14/step5_grpo/best_arm.json"))
    args = parser.parse_args()
    payload = select(load_reports(args.reports_glob))
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
