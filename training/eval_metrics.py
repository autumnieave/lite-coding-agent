"""训练/constrained decoding 的评测汇总；纯本地计算，不依赖模型。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

FORMAT_CHECKS = ("json_valid", "snake_case_keys", "no_code_fence")


def rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def summarize_generation_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    checks: dict[str, dict[str, Any]] = {}
    for check in FORMAT_CHECKS:
        hits = sum(bool((record.get("validation") or {}).get(check)) for record in records)
        checks[check] = {"hits": hits, "total": total, "rate": rate(hits, total)}
    all_pass = sum(
        all(bool((record.get("validation") or {}).get(check)) for check in FORMAT_CHECKS)
        for record in records
    )
    schema_pass = sum(bool(record.get("schema_accepts")) for record in records)
    durations = [
        float((record.get("metadata") or {}).get("duration_s") or 0.0) for record in records
    ]
    return {
        "total": total,
        "checks": checks,
        "all_format_pass": {"hits": all_pass, "total": total, "rate": rate(all_pass, total)},
        "schema_accepts": {
            "hits": schema_pass,
            "total": total,
            "rate": rate(schema_pass, total),
        },
        "duration_s": round(sum(durations), 2),
    }


def summarize_by_run(
    records: Sequence[Mapping[str, Any]], key: str = "run_id"
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get(key)), []).append(record)
    return {name: summarize_generation_records(items) for name, items in sorted(grouped.items())}


def compare_summaries(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for check in FORMAT_CHECKS:
        before_rate = ((before.get("checks") or {}).get(check) or {}).get("rate")
        after_rate = ((after.get("checks") or {}).get(check) or {}).get("rate")
        if before_rate is None or after_rate is None:
            delta = None
        else:
            delta = round(float(after_rate) - float(before_rate), 4)
        result[check] = {"before": before_rate, "after": after_rate, "delta": delta}
    for check in ("all_format_pass", "schema_accepts"):
        before_rate = (before.get(check) or {}).get("rate")
        after_rate = (after.get(check) or {}).get("rate")
        if before_rate is None or after_rate is None:
            delta = None
        else:
            delta = round(float(after_rate) - float(before_rate), 4)
        result[check] = {"before": before_rate, "after": after_rate, "delta": delta}
    return result
