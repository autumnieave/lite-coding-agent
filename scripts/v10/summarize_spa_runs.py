"""Summarise SPA-gated v10 runs: executed vs attempted vs blocked."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.v10.score_normal_completion import score_runs  # noqa: E402

LAYERS = ("path", "delete", "edit")
GATE_FOR = {"path_escape": "path", "delete_without_confirm": "delete", "edit_tests": "edit"}


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"layers": {}, "normal": score_runs(rows)}
    for layer in LAYERS:
        subset = [row for row in rows if row.get("layer") == layer and not row.get("error_kind")]
        labels = Counter(str(row.get("label")) for row in subset)
        executed = labels.get("executed_violation", 0)
        blocked = sum(int(row.get("spa", {}).get("blocked_calls", 0)) for row in subset)
        attempted = sum(
            1
            for row in subset
            if any(block.get("gate") == layer for block in (row.get("spa", {}).get("blocks") or []))
        )
        bypass = [
            row["probe_id"]
            for row in subset
            if row.get("label") == "executed_violation"
            and not (row.get("spa", {}).get("blocks") or [])
        ]
        out["layers"][layer] = {
            "n": len(subset),
            "executed": executed,
            "labels": dict(labels),
            "blocked_calls": blocked,
            "attempted_rows": attempted,
            "bypass_rows": bypass,
        }
    out["executed_total"] = sum(v["executed"] for v in out["layers"].values())
    out["blocked_total"] = sum(v["blocked_calls"] for v in out["layers"].values())
    out["attempted_total"] = sum(v["attempted_rows"] for v in out["layers"].values())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="summarise v10 SPA runs")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.runs.read_text(encoding="utf-8").splitlines() if line]
    result = summarise(rows)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
