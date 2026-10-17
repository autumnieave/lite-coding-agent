"""Replay the SPA gate over recorded v10 DPO traces (no model, no GPU)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent.core.action_gate import ActionGate  # noqa: E402
from training.v10_eval_harness import classify_eval_probe  # noqa: E402

LAYERS = ("path", "delete", "edit")


def replay(
    probes: dict[str, dict[str, Any]], rows: list[dict[str, Any]], root: Path
) -> dict[str, Any]:
    gate = ActionGate(root)
    out: dict[str, Any] = {"rows": [], "layers": {}}
    for row in rows:
        probe = probes[str(row["probe_id"])]
        decisions = []
        calls = []
        for call in row.get("tool_calls") or []:
            decision = gate.check(str(call["name"]), str(call.get("arguments") or ""))
            decisions.append(decision)
            calls.append(
                SimpleNamespace(
                    name=call["name"],
                    arguments=call.get("arguments") or "",
                    result=SimpleNamespace(
                        ok=bool(call.get("ok")) and decision.allowed,
                        error_kind=None if decision.allowed else "blocked_by_gate",
                    ),
                )
            )
        classified = classify_eval_probe(
            probe, str(row.get("reply") or ""), calls, scan_content_calls=False
        )
        out["rows"].append(
            {
                "probe_id": row["probe_id"],
                "layer": row["layer"],
                "before": row.get("label"),
                "after": classified["label"],
                "blocked": [d.gate for d in decisions if not d.allowed],
                "execution_violation_after": classified["execution_violation"],
            }
        )
    for layer in LAYERS:
        subset = [r for r in out["rows"] if r["layer"] == layer]
        out["layers"][layer] = {
            "n": len(subset),
            "before_executed": sum(1 for r in subset if r["before"] == "executed_violation"),
            "after_executed": sum(1 for r in subset if r["execution_violation_after"]),
            "after_labels": dict(Counter(str(r["after"]) for r in subset)),
            "blocked_calls": sum(len(r["blocked"]) for r in subset),
        }
    out["before_total"] = sum(v["before_executed"] for v in out["layers"].values())
    out["after_total"] = sum(v["after_executed"] for v in out["layers"].values())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="replay SPA gate over recorded traces")
    parser.add_argument("--eval-file", type=Path, default=Path("private/v10/eval_v10/eval.jsonl"))
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/tmp/workspace"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    probes = {
        str(row["probe_id"]): row
        for row in (
            json.loads(line)
            for line in args.eval_file.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    rows = [json.loads(line) for line in args.runs.read_text(encoding="utf-8").splitlines() if line]
    result = replay(probes, rows, args.root)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
