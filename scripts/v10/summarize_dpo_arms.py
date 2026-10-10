"""Compare the v10 step-5 DPO arms on the frozen layered eval set."""

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


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"layers": {}}
    for layer in LAYERS:
        subset = [row for row in rows if row.get("layer") == layer and not row.get("error_kind")]
        labels = Counter(str(row.get("label")) for row in subset)
        total = len(subset)
        executed = labels.get("executed_violation", 0)
        explicit = labels.get("refused", 0) + labels.get("confirmed", 0)
        out["layers"][layer] = {
            "n": total,
            "executed": executed,
            "explicit": explicit,
            "ambiguous": labels.get("ambiguous", 0),
            "deferred": labels.get("deferred_tool", 0),
            "other": sum(
                value
                for key, value in labels.items()
                if key
                not in {"executed_violation", "refused", "confirmed", "ambiguous", "deferred_tool"}
            ),
            "labels": dict(labels),
            "explicit_rate": explicit / total if total else 0.0,
        }
    out["conflict_total"] = sum(out["layers"][layer]["n"] for layer in LAYERS)
    out["executed_total"] = sum(out["layers"][layer]["executed"] for layer in LAYERS)
    out["normal"] = score_runs(rows)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="summarise v10 DPO arms")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result: dict[str, Any] = {}
    for arm in ("eval_init", "eval_equal", "eval_weighted"):
        path = args.root / arm / "runs.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        result[arm] = summarise(rows)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
