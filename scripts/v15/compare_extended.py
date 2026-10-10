"""Compare GRPO base vs best arm on the v14 primary and v15 extension sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.v15_stats import (  # noqa: E402
    benjamini_hochberg,
    mcnemar_exact,
    wilson_ci,
)

LAYERS = ("path", "delete", "edit")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    return {str(row["probe_id"]): row for row in payload.get("rows", [])}


def _paired(
    base_rows: dict[str, dict[str, Any]],
    best_rows: dict[str, dict[str, Any]],
    *,
    layer: str | None,
    probe_prefix: str | None = None,
) -> dict[str, Any]:
    keys = sorted(set(base_rows) & set(best_rows))
    if layer is not None:
        keys = [key for key in keys if base_rows[key].get("layer") == layer]
    else:
        keys = [key for key in keys if base_rows[key].get("layer") in LAYERS]
    if probe_prefix is not None:
        keys = [key for key in keys if key.startswith(probe_prefix)]
    base = [bool(base_rows[key].get("execution_violation")) for key in keys]
    best = [bool(best_rows[key].get("execution_violation")) for key in keys]
    base_only = sum(b and not c for b, c in zip(base, best, strict=True))
    best_only = sum(not b and c for b, c in zip(base, best, strict=True))
    test = mcnemar_exact(base_only, best_only)
    base_ci = wilson_ci(sum(base), len(base)) if base else None
    best_ci = wilson_ci(sum(best), len(best)) if best else None
    return {
        "n": len(keys),
        "base_violations": sum(base),
        "best_violations": sum(best),
        "risk_difference": (sum(best) - sum(base)) / len(base) if base else None,
        "base_wilson_ci95": None if base_ci is None else [base_ci.low, base_ci.high],
        "best_wilson_ci95": None if best_ci is None else [best_ci.low, best_ci.high],
        "base_only": base_only,
        "best_only": best_only,
        "mcnemar_p": test.p_value,
    }


def paired_tests(
    base_rows: dict[str, dict[str, Any]],
    best_rows: dict[str, dict[str, Any]],
    *,
    probe_prefix: str | None = None,
) -> list[dict[str, Any]]:
    tests = []
    for layer in (None, *LAYERS):
        row = _paired(base_rows, best_rows, layer=layer, probe_prefix=probe_prefix)
        row["layer"] = layer or "all_conflict"
        tests.append(row)
    q_values = benjamini_hochberg([float(row["mcnemar_p"]) for row in tests])
    for row, q_value in zip(tests, q_values, strict=True):
        row["bh_fdr_q"] = q_value
    return tests


def report_row(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": report.get("label"),
        "total_execution_violations": report.get("total_execution_violations"),
        "total_conflict_probes": report.get("total_conflict_probes"),
        "path": report.get("layers", {}).get("path"),
        "delete": report.get("layers", {}).get("delete"),
        "edit": report.get("layers", {}).get("edit"),
        "citation": report.get("citation"),
        "decoupling": report.get("decoupling"),
        "normal": report.get("normal"),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# v14 主判据 / v15 扩展验证",
        "",
        f"- base arm：`{payload['base_arm']}`",
        f"- best arm：`{payload['best_arm']}`",
        "- 两栏分报；v15 100 条不并入 v14 72 条，不做 172 条合并比例。",
        "",
        "## 指标",
        "",
        "| 数据集 | 模型 | 执行违规 | 引用正确 ID | 脱钩 | normal 完成 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (("v14", "v14 72"), ("v15", "v15 100")):
        for arm_key in ("base", "best"):
            row = payload[key][arm_key]
            citation = row["citation"] or {}
            normal = row["normal"] or {}
            lines.append(
                f"| {label} | {arm_key} | {row['total_execution_violations']}/"
                f"{row['total_conflict_probes']} | {citation.get('cited_own_id')}/"
                f"{citation.get('probes')} | {(row['decoupling'] or {}).get('count')} | "
                f"{normal.get('completed')}/{normal.get('n')} |"
            )
    lines += ["", "## 配对检验（base vs best，执行违规）", ""]
    for key, label in (("v14", "v14 72"), ("v15", "v15 100"), ("v15_new", "v15 新增 28")):
        lines += [
            f"### {label}",
            "",
            "| 层 | n | base | best | 风险差 | b/c | p | q |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
        for row in payload[key]["tests"]:
            lines.append(
                f"| {row['layer']} | {row['n']} | {row['base_violations']} | "
                f"{row['best_violations']} | {row['risk_difference']:+.3f} | "
                f"{row['base_only']}/{row['best_only']} | {row['mcnemar_p']:.4f} | "
                f"{row['bh_fdr_q']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="compare base vs best arm")
    parser.add_argument("--base-arm", required=True)
    parser.add_argument("--best-arm", required=True)
    parser.add_argument("--v14-base-report", type=Path, required=True)
    parser.add_argument("--v14-best-report", type=Path, required=True)
    parser.add_argument("--v15-base-report", type=Path, required=True)
    parser.add_argument("--v15-best-report", type=Path, required=True)
    parser.add_argument("--v14-base-summary", type=Path, required=True)
    parser.add_argument("--v14-best-summary", type=Path, required=True)
    parser.add_argument("--v15-base-summary", type=Path, required=True)
    parser.add_argument("--v15-best-summary", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    payload = {
        "version": "v15-extended-eval-1",
        "base_arm": args.base_arm,
        "best_arm": args.best_arm,
        "v14": {
            "base": report_row(load_json(args.v14_base_report)),
            "best": report_row(load_json(args.v14_best_report)),
            "tests": paired_tests(
                summary_rows(args.v14_base_summary), summary_rows(args.v14_best_summary)
            ),
        },
        "v15": {
            "base": report_row(load_json(args.v15_base_report)),
            "best": report_row(load_json(args.v15_best_report)),
            "tests": paired_tests(
                summary_rows(args.v15_base_summary), summary_rows(args.v15_best_summary)
            ),
        },
        "v15_new": {
            "tests": paired_tests(
                summary_rows(args.v15_base_summary),
                summary_rows(args.v15_best_summary),
                probe_prefix="V15-EVAL-",
            )
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json": str(args.out_json), "markdown": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
