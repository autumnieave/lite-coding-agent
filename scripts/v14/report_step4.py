"""Aggregate v14 step-4 DPO arms: execution, citation, decoupling, normal-task proxy."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_id_citation import audit  # noqa: E402

from training.kaggle.session_constraint_probe import classify_probe  # noqa: E402
from training.v10_eval_harness import probe_constraint_spec  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def normal_completion(runs_path: Path) -> dict[str, Any]:
    """A normal probe counts as completed if at least one tool call succeeded."""

    rows = [row for row in read_jsonl(runs_path) if row.get("layer") == "normal"]
    completed = 0
    details = []
    for row in rows:
        calls = row.get("tool_calls") or []
        ok = any(bool(call.get("ok")) for call in calls)
        errored = bool(row.get("error_kind")) or row.get("stopped_reason") == "error"
        success = ok and not errored
        completed += int(success)
        details.append(
            {
                "probe_id": row.get("probe_id"),
                "completed": success,
                "calls": len(calls),
                "ok_calls": sum(1 for call in calls if call.get("ok")),
                "stopped_reason": row.get("stopped_reason"),
            }
        )
    total = len(rows)
    return {
        "n": total,
        "completed": completed,
        "rate": round(completed / total, 4) if total else 0.0,
        "definition": "至少一次 tool_calls.ok=True 且未报错",
        "details": details,
    }


def layer_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """Read the runner's raw summary (old verifier semantics, comparison only)."""

    layers = summary.get("summary") or {}
    out: dict[str, Any] = {}
    for layer in ("path", "delete", "edit", "normal"):
        item = layers.get(layer) or {}
        out[layer] = {
            "n": item.get("n", 0),
            "execution_violations": item.get("execution_violations", 0),
            "execution_violation_rate": item.get("execution_violation_rate", 0.0),
            "explicit_compliance": item.get("explicit_compliance", 0),
            "explicit_compliance_rate": item.get("explicit_compliance_rate", 0.0),
            "labels": item.get("labels", {}),
        }
    return out


def recompute_layers(runs_path: Path, eval_path: Path) -> dict[str, Any]:
    """Recompute per-layer labels from real tool_calls only (P0 rule)."""

    probes = {str(row["probe_id"]): row for row in read_jsonl(eval_path)}
    labels: dict[str, Counter] = {layer: Counter() for layer in ("path", "delete", "edit")}
    for row in read_jsonl(runs_path):
        probe = probes.get(str(row.get("probe_id")))
        if probe is None or str(probe.get("layer")) == "normal" or row.get("error_kind"):
            continue
        result = classify_probe(
            probe_constraint_spec(probe),
            str(row.get("reply") or ""),
            row.get("tool_calls") or [],
            scan_content_calls=False,
        )
        labels[str(probe["layer"])][str(result["label"])] += 1
    out: dict[str, Any] = {}
    for layer in ("path", "delete", "edit"):
        counts = labels[layer]
        total = sum(counts.values())
        executed = counts.get("executed_violation", 0)
        explicit = counts.get("refused", 0) + counts.get("confirmed", 0)
        out[layer] = {
            "n": total,
            "execution_violations": executed,
            "execution_violation_rate": round(executed / total, 4) if total else 0.0,
            "explicit_compliance": explicit,
            "explicit_compliance_rate": round(explicit / total, 4) if total else 0.0,
            "labels": dict(counts),
        }
    out["normal"] = {"n": 12, "labels": {"not_applicable": 12}}
    return out


def collapse_decoupling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_layer_total = Counter()
    per_layer_decoupled = Counter()
    for case in rows:
        per_layer_total[str(case.get("layer"))] += 1
        per_layer_decoupled[str(case.get("layer"))] += 1
    return {
        "total": len(rows),
        "path": per_layer_decoupled.get("path", 0),
        "delete": per_layer_decoupled.get("delete", 0),
        "edit": per_layer_decoupled.get("edit", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="report v14 DPO arms")
    parser.add_argument("--arms-dir", type=Path, default=Path("private/v14/step4_dpo/arms"))
    parser.add_argument("--eval-dir", type=Path, default=Path("private/v14/step4_dpo/eval"))
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("private/v14/step2_benchmark/eval_set_frozen.jsonl"),
    )
    parser.add_argument(
        "--baseline-runs",
        type=Path,
        default=Path("private/v14/step3_baseline/m2_base/runs.jsonl"),
    )
    parser.add_argument("--out", type=Path, default=Path("private/v14/step4_dpo/dpo_results.json"))
    parser.add_argument("--md", type=Path, default=Path("private/v14/step4_dpo/summary.md"))
    args = parser.parse_args()

    arms_summary_path = args.arms_dir / "arms_summary.json"
    arms_summary = (
        json.loads(arms_summary_path.read_text(encoding="utf-8"))
        if arms_summary_path.exists()
        else {}
    )

    baseline = (
        audit(args.baseline_runs, args.eval_file, "m2_base") if args.baseline_runs.exists() else {}
    )
    baseline_normal = normal_completion(args.baseline_runs)

    evaluated: dict[str, Any] = {}
    arm_dirs = (
        sorted(path for path in args.eval_dir.iterdir() if path.is_dir())
        if args.eval_dir.exists()
        else []
    )
    for arm_dir in arm_dirs:
        summary_path = arm_dir / "summary.json"
        runs_path = arm_dir / "runs.jsonl"
        if not summary_path.exists() or not runs_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        audit_path = arm_dir / "audit.json"
        arm_audit = (
            json.loads(audit_path.read_text(encoding="utf-8"))
            if audit_path.exists()
            else audit(runs_path, args.eval_file, arm_dir.name)
        )
        evaluated[arm_dir.name] = {
            "layers": recompute_layers(runs_path, args.eval_file),
            "raw_summary_layers": layer_metrics(summary),
            "mechanism": summary.get("mechanism", {}),
            "citation": {
                "probes": arm_audit.get("probes", 0),
                "cited_own_id": arm_audit.get("cited_own_id", 0),
                "cited_own_rate": arm_audit.get("cited_own_rate", 0.0),
                "cited_other_id": arm_audit.get("cited_other_id", 0),
                "cited_none": arm_audit.get("cited_none", 0),
            },
            "decoupling": {
                "count": arm_audit.get("decoupled_count", 0),
                "cases": collapse_decoupling(arm_audit.get("decoupled_cases") or []),
                "rate_of_executed": arm_audit.get("decoupled_rate_of_executed", 0.0),
            },
            "normal_completion": normal_completion(runs_path),
        }

    best = None
    for name, item in evaluated.items():
        total_exec = sum(
            item["layers"][layer]["execution_violations"] for layer in ("path", "delete", "edit")
        )
        key = (total_exec, -item["citation"]["cited_own_rate"])
        if best is None or key < best[0]:
            best = (key, name)

    result = {
        "version": "v14-step4-report-1",
        "arms_summary": arms_summary,
        "baseline_m2": {
            "citation": {
                "probes": baseline.get("probes", 0),
                "cited_own_id": baseline.get("cited_own_id", 0),
                "cited_own_rate": baseline.get("cited_own_rate", 0.0),
            },
            "decoupling": {
                "count": baseline.get("decoupled_count", 0),
                "cases": collapse_decoupling(baseline.get("decoupled_cases") or []),
                "rate_of_executed": baseline.get("decoupled_rate_of_executed", 0.0),
            },
            "normal_completion": baseline_normal,
        },
        "arms": evaluated,
        "best_arm_by_execution": best[1] if best else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# v14 步骤 4 DPO 汇总",
        "",
        "| arm | path 执行违规 | delete | edit | 合计 | 引用正确 ID | 脱钩 | normal 完成 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in sorted(evaluated.items()):
        layers = item["layers"]
        total_exec = sum(
            layers[layer]["execution_violations"] for layer in ("path", "delete", "edit")
        )
        citation = item["citation"]
        decoupled = item["decoupling"]
        normal = item["normal_completion"]
        lines.append(
            f"| {name} | {layers['path']['execution_violations']}/{layers['path']['n']} | "
            f"{layers['delete']['execution_violations']}/{layers['delete']['n']} | "
            f"{layers['edit']['execution_violations']}/{layers['edit']['n']} | {total_exec} | "
            f"{citation['cited_own_id']}/{citation['probes']} ({citation['cited_own_rate']:.1%}) | "
            f"{decoupled['count']} ({decoupled['cases']}) | "
            f"{normal['completed']}/{normal['n']} ({normal['rate']:.1%}) |"
        )
    base = result["baseline_m2"]
    lines.extend(
        [
            "",
            "## 基座 M2 对照",
            "",
            f"- 引用正确 ID：{base['citation']['cited_own_id']}/{base['citation']['probes']}"
            f" = {base['citation']['cited_own_rate']:.1%}",
            f"- 脱钩：{base['decoupling']['count']}，按层 {base['decoupling']['cases']}",
            f"- normal 完成代理：{base['normal_completion']['completed']}/"
            f"{base['normal_completion']['n']} = {base['normal_completion']['rate']:.1%}",
            "",
            f"- 执行违规最少的 arm：`{result['best_arm_by_execution']}`",
            "- 本表按真实 `tool_calls` 重算（`scan_content_calls=False`）。",
            "- runner 原始 summary 会把文本伪调用/只读越界算进执行层；仅作对照，不作为结论。",
            "- normal 完成代理定义：至少一次 `tool_calls.ok=True` 且未报错；不是完整任务成功判定。",
        ]
    )
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
