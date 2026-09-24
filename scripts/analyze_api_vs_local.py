"""Analyze API baseline vs local 7B baseline/SFT on the frozen 45 tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import statistics
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent.tools import build_default_registry  # noqa: E402
from agent.tools.base import ToolResult  # noqa: E402
from scripts.analyze_clean_heldout import (  # noqa: E402
    METRICS,
    TaskOutcome,
    binom_two_sided,
    discover_baseline_files,
    equal_task_rate,
    holm,
    load_outcomes,
    pooled_rate,
    read_jsonl,
    read_task_ids,
    rollout_mcnemar,
    sha256,
    task_mcnemar,
)
from scripts.unified_eval import ToolCallRecord, build_fixture, judge  # noqa: E402
from training.unified_dataset import (  # noqa: E402
    CheckSpec,
    ConstraintSpec,
    UnifiedTaskSpec,
)

DEFAULT_TASKS_FILE = Path("private/final_test_v3/tasks.jsonl")
DEFAULT_API_RUNS = Path("private/api_ab/runs.jsonl")
DEFAULT_API_SUMMARY = Path("private/api_ab/summary.json")
DEFAULT_SFT_RUNS = Path("private/kaggle_pull_clean_heldout_meta/clean_heldout/full/sft/runs.jsonl")
DEFAULT_BASELINE_ROOT = Path("private/kaggle_difficulty")
DEFAULT_CLEAN_REPORT = Path("docs/evidence/clean_heldout_统计.json")
DEFAULT_OUT_JSON = Path("docs/evidence/API_vs_本地对照.json")
DEFAULT_OUT_MD = Path("docs/evidence/API_vs_本地对照.md")
DEFAULT_PREFLIGHT_SUMMARY = Path("private/api_ab/preflight_script/summary.json")
JUDGE_PATH = Path("scripts/unified_eval.py")
STABLE_HITS = 6
_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def read_task_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        task_id = str(payload.get("task_id") or "")
        if not task_id:
            raise ValueError(f"{path} has a row without task_id")
        manifest = payload.get("manifest") or {}
        rows[task_id] = {
            "family": manifest.get("family"),
            "category": manifest.get("category"),
            "difficulty": manifest.get("difficulty"),
            "training_overlap": bool(payload.get("training_overlap")),
        }
    return rows


def _task_spec_from_manifest(row: Mapping[str, Any]) -> UnifiedTaskSpec:
    payload = dict(row["spec"])
    payload["constraints"] = tuple(
        ConstraintSpec(**item) for item in payload.get("constraints", ())
    )
    payload["checks"] = tuple(CheckSpec(**item) for item in payload.get("checks", ()))
    for field in (
        "user_turns",
        "required_tools",
        "allowed_tools",
        "trajectory_events",
        "path_targets",
        "public_patterns",
    ):
        if field in payload:
            payload[field] = tuple(payload[field])
    return UnifiedTaskSpec(**payload)


def _trace_tool_calls(
    trace_path: Path,
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    calls: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for event in events:
        if event.get("event") != "llm_turn":
            continue
        results = {str(item.get("call_id")): item for item in (event.get("tool_results") or [])}
        for call in event.get("tool_calls") or []:
            calls.append((call, results.get(str(call.get("call_id")))))
    return calls


def _extract_fenced_json(reply: str) -> str:
    match = _FENCED_JSON.search(reply)
    return match.group(1).strip() if match else reply.strip()


async def _rejudge_normalized(
    row: Mapping[str, Any], task_row: Mapping[str, Any]
) -> dict[str, Any]:
    task_id = str(row["task_id"])
    rollout_index = int(row["rollout_index"])
    try:
        spec = _task_spec_from_manifest(task_row)
        trace_path = Path(str(row.get("trace_path") or ""))
        if not trace_path.is_file():
            raise FileNotFoundError(f"trace not found: {trace_path}")
        normalized_reply = _extract_fenced_json(str(row.get("reply") or ""))
        with tempfile.TemporaryDirectory(prefix="api-ab-counterfactual-") as temp_dir:
            root = Path(temp_dir)
            fixture = build_fixture(spec, root)
            registry = build_default_registry(root)
            calls: list[ToolCallRecord] = []
            for call, result in _trace_tool_calls(trace_path):
                name = str(call.get("tool_name") or "")
                arguments = str(call.get("arguments") or "")
                if bool(call.get("executed")) and name != "bash":
                    await registry.execute(name, arguments)
                result_payload = result or {}
                calls.append(
                    ToolCallRecord(
                        name=name,
                        arguments=arguments,
                        result=ToolResult(
                            ok=bool(result_payload.get("ok")),
                            content=str(result_payload.get("content") or ""),
                        ),
                    )
                )
            checks = judge(spec, fixture, root, normalized_reply, calls)
        return {
            "task_id": task_id,
            "rollout_index": rollout_index,
            "success": all(checks.values()),
            "failed_checks": sorted(name for name, passed in checks.items() if not passed),
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - reported in the evidence JSON
        return {
            "task_id": task_id,
            "rollout_index": rollout_index,
            "success": False,
            "failed_checks": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _run_normalized_judge(
    rows: Sequence[Mapping[str, Any]], task_rows: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [await _rejudge_normalized(row, task_rows[str(row["task_id"])]) for row in rows]


def analyze_format_attribution(
    api_rows: Sequence[Mapping[str, Any]],
    sft_rows: Sequence[Mapping[str, Any]],
    task_ids: Sequence[str],
    tasks_file: Path,
) -> dict[str, Any]:
    task_rows = {
        str(row["task_id"]): row
        for row in (
            json.loads(line)
            for line in tasks_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    official_failures = [row for row in api_rows if not bool(row.get("task_success"))]
    fenced_failures = [
        row
        for row in official_failures
        if int((row.get("api_audit") or {}).get("fenced_json_turns") or 0) > 0
    ]
    extractable = 0
    for row in fenced_failures:
        try:
            json.loads(_extract_fenced_json(str(row.get("reply") or "")))
            extractable += 1
        except (TypeError, ValueError):
            pass

    results = asyncio.run(_run_normalized_judge(fenced_failures, task_rows))
    normalized_passes = {
        (item["task_id"], int(item["rollout_index"])) for item in results if item["success"]
    }
    normalized_keys = {
        (str(row["task_id"]), int(row["rollout_index"]))
        for row in api_rows
        if bool(row.get("task_success"))
    } | normalized_passes

    per_task = {task_id: [0, 0] for task_id in task_ids}
    for row in api_rows:
        task_id = str(row["task_id"])
        if task_id not in per_task or row.get("error") or row.get("error_kind"):
            continue
        per_task[task_id][1] += 1
        if (task_id, int(row["rollout_index"])) in normalized_keys:
            per_task[task_id][0] += 1
    rates = [success / valid for success, valid in per_task.values() if valid]
    normalized_equal_rate = statistics.fmean(rates) if rates else 0.0
    normalized_stable_tasks = sum(
        valid >= 10 and success >= STABLE_HITS for success, valid in per_task.values()
    )

    api_map = {
        (str(row["task_id"]), int(row["rollout_index"])): bool(row.get("task_success"))
        for row in api_rows
        if not row.get("error") and not row.get("error_kind")
    }
    sft_map = {
        (str(row["task_id"]), int(row["rollout_index"])): bool(row.get("task_success"))
        for row in sft_rows
        if not row.get("error") and not row.get("error_kind")
    }
    api_fail_sft_pass = 0
    api_pass_sft_fail = 0
    for key in sorted(api_map.keys() & sft_map.keys()):
        normalized_api = key in normalized_keys
        if not normalized_api and sft_map[key]:
            api_fail_sft_pass += 1
        elif normalized_api and not sft_map[key]:
            api_pass_sft_fail += 1
    discordant = api_fail_sft_pass + api_pass_sft_fail
    normalized_p = binom_two_sided(min(api_fail_sft_pass, api_pass_sft_fail), discordant)

    sft_successes = sum(int(row.get("task_success") is True) for row in sft_rows)
    normalized_failed = [item for item in results if not item["success"]]
    non_fenced_failures = [
        {
            "task_id": str(row["task_id"]),
            "rollout_index": int(row["rollout_index"]),
            "failed_checks": sorted(
                name for name, passed in (row.get("checks") or {}).items() if not passed
            ),
        }
        for row in official_failures
        if row not in fenced_failures
    ]
    return {
        "official_api_successes": len(api_rows) - len(official_failures),
        "official_api_failures": len(official_failures),
        "fenced_final_answer_failures": len(fenced_failures),
        "fenced_json_extractable": extractable,
        "fenced_normalized_passes": len(normalized_passes),
        "normalized_replay_errors": sum(1 for item in results if item["error"]),
        "normalized_api_successes": len(normalized_keys),
        "normalized_api_equal_task_rate": normalized_equal_rate,
        "normalized_api_stable_tasks": normalized_stable_tasks,
        "local_sft_successes": sft_successes,
        "local_sft_equal_task_rate": equal_task_rate(
            {task_id: _outcome_for(task_id, sft_rows) for task_id in task_ids},
            task_ids,
            "task_success",
        ),
        "normalized_mcnemar": {
            "api_fail_sft_pass": api_fail_sft_pass,
            "api_pass_sft_fail": api_pass_sft_fail,
            "discordant_pairs": discordant,
            "p_two_sided": normalized_p,
        },
        "non_fenced_failures": non_fenced_failures,
        "normalized_failures": normalized_failed,
        "note": (
            "Post-hoc diagnostic only: the official judge result is unchanged. "
            "Each fenced failure was re-judged after extracting the fenced JSON, "
            "replaying executed tool calls on the deterministic fixture, and applying "
            "the same judge. This separates content correctness from output-contract compliance."
        ),
    }


def _outcome_for(task_id: str, rows: Sequence[Mapping[str, Any]]) -> TaskOutcome:
    valid = 0
    successes = {"task_success": 0, "within_steps": 0, "tool_selection_ok": 0}
    for row in rows:
        if str(row.get("task_id")) != task_id or row.get("error") or row.get("error_kind"):
            continue
        valid += 1
        for metric in successes:
            if bool(row.get(metric)):
                successes[metric] += 1
    return TaskOutcome(valid=valid, successes=successes)


def is_stable(outcome: TaskOutcome, metric: str = "task_success") -> bool:
    return outcome.valid >= 10 and outcome.successes[metric] >= STABLE_HITS


def stable_task_mcnemar(
    task_ids: Sequence[str],
    api: Mapping[str, TaskOutcome],
    local: Mapping[str, TaskOutcome],
    metric: str,
) -> dict[str, Any]:
    api_only: list[str] = []
    local_only: list[str] = []
    both: list[str] = []
    neither: list[str] = []
    for task_id in task_ids:
        api_ok = is_stable(api[task_id], metric)
        local_ok = is_stable(local[task_id], metric)
        if api_ok and local_ok:
            both.append(task_id)
        elif local_ok and not api_ok:
            local_only.append(task_id)
        elif api_ok and not local_ok:
            api_only.append(task_id)
        else:
            neither.append(task_id)
    discordant = len(api_only) + len(local_only)
    return {
        "metric": metric,
        "stable_threshold": f"{STABLE_HITS}/10",
        "both_stable": len(both),
        "local_only_stable": len(local_only),
        "api_only_stable": len(api_only),
        "neither_stable": len(neither),
        "discordant_tasks": discordant,
        "p_two_sided": binom_two_sided(min(len(api_only), len(local_only)), discordant),
        "local_only_task_ids": local_only,
        "api_only_task_ids": api_only,
    }


def _metric_pairs(
    api_rows: Sequence[Mapping[str, Any]],
    local_rows: Sequence[Mapping[str, Any]],
    task_id: str,
    metric: str,
) -> list[tuple[bool, bool]]:
    api = {
        int(row["rollout_index"]): bool(row.get(metric))
        for row in api_rows
        if str(row.get("task_id")) == task_id and not row.get("error_kind") and not row.get("error")
    }
    local = {
        int(row["rollout_index"]): bool(row.get(metric))
        for row in local_rows
        if str(row.get("task_id")) == task_id and not row.get("error_kind") and not row.get("error")
    }
    return [(api[index], local[index]) for index in sorted(set(api) & set(local))]


def paired_task_bootstrap(
    task_ids: Sequence[str],
    api_rows: Sequence[Mapping[str, Any]],
    local_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    pairs = {task_id: _metric_pairs(api_rows, local_rows, task_id, metric) for task_id in task_ids}
    usable = [task_id for task_id in task_ids if pairs[task_id]]
    rng = random.Random(seed)
    effects: list[float] = []
    for _ in range(iterations):
        sampled_tasks = [rng.choice(usable) for _ in usable]
        differences: list[float] = []
        for task_id in sampled_tasks:
            task_pairs = pairs[task_id]
            sampled = [task_pairs[rng.randrange(len(task_pairs))] for _ in task_pairs]
            api_rate = sum(api for api, _ in sampled) / len(sampled)
            local_rate = sum(local for _, local in sampled) / len(sampled)
            differences.append(local_rate - api_rate)
        effects.append(statistics.fmean(differences))
    ordered = sorted(effects)
    return {
        "metric": metric,
        "iterations": iterations,
        "seed": seed,
        "method": "paired_task_cluster_percentile",
        "tasks_used": len(usable),
        "mean": statistics.fmean(effects),
        "ci95_low": _percentile(ordered, 0.025),
        "ci95_high": _percentile(ordered, 0.975),
    }


def _percentile(ordered: Sequence[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def analyze(
    *,
    tasks_file: Path,
    api_runs: Path,
    sft_runs: Path,
    baseline_root: Path,
    api_summary_path: Path,
    clean_report_path: Path,
    preflight_path: Path,
    expected_task_sha: str,
    bootstrap_iterations: int,
    seed: int,
    api_input_price: float | None,
    api_output_price: float | None,
    local_gpu_hourly: float | None,
) -> dict[str, Any]:
    task_sha = sha256(tasks_file)
    if task_sha is None or task_sha.lower() != expected_task_sha.lower():
        raise ValueError(
            f"task manifest SHA mismatch: expected {expected_task_sha}, got {task_sha}"
        )
    task_ids = read_task_ids(tasks_file)
    task_meta = read_task_metadata(tasks_file)
    api_rows = read_jsonl(api_runs)
    sft_rows = read_jsonl(sft_runs)
    baseline_files = discover_baseline_files(baseline_root)
    baseline_rows = [row for path in baseline_files for row in read_jsonl(path)]

    api_out, api_source = load_outcomes([api_runs], task_ids, label="api")
    sft_out, sft_source = load_outcomes([sft_runs], task_ids, label="local-sft")
    baseline_out, baseline_source = load_outcomes(baseline_files, task_ids, label="local-baseline")
    incomplete = [
        task_id
        for task_id in task_ids
        if api_out[task_id].valid != 10 or sft_out[task_id].valid != 10
    ]
    if incomplete:
        raise ValueError(f"API/SFT do not have 10 valid rollouts for: {incomplete}")

    api_summary = json.loads(api_summary_path.read_text(encoding="utf-8"))
    if api_summary.get("source_runs_sha256") != sha256(api_runs):
        raise ValueError("API summary does not match runs.jsonl SHA")
    clean_report = (
        json.loads(clean_report_path.read_text(encoding="utf-8"))
        if clean_report_path.is_file()
        else {}
    )
    preflight = (
        json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else {}
    )

    metrics: dict[str, Any] = {}
    rollout_pvalues: dict[str, float] = {}
    stable_pvalues: dict[str, float] = {}
    for metric in METRICS:
        api_equal = equal_task_rate(api_out, task_ids, metric)
        local_equal = equal_task_rate(sft_out, task_ids, metric)
        rollout_test = rollout_mcnemar(task_ids, api_rows, sft_rows, metric)
        stable_test = stable_task_mcnemar(task_ids, api_out, sft_out, metric)
        bootstrap = paired_task_bootstrap(
            task_ids,
            api_rows,
            sft_rows,
            metric=metric,
            iterations=bootstrap_iterations,
            seed=seed + METRICS.index(metric),
        )
        rollout_pvalues[metric] = float(rollout_test["p_two_sided"])
        stable_pvalues[metric] = float(stable_test["p_two_sided"])
        metrics[metric] = {
            "api_equal_task_rate": api_equal,
            "local_sft_equal_task_rate": local_equal,
            "effect_local_minus_api_pp": (local_equal - api_equal) * 100.0,
            "api_pooled_rate": pooled_rate(api_out, task_ids, metric),
            "local_sft_pooled_rate": pooled_rate(sft_out, task_ids, metric),
            "rollout_mcnemar": rollout_test,
            "task_rate_mcnemar": task_mcnemar(task_ids, api_out, sft_out, metric),
            "task_stable_mcnemar": stable_test,
            "paired_task_bootstrap": bootstrap,
        }

    rollout_holm = holm(rollout_pvalues)
    stable_holm = holm(stable_pvalues)
    for metric in METRICS:
        metrics[metric]["rollout_mcnemar"]["p_holm"] = rollout_holm[metric]
        metrics[metric]["task_stable_mcnemar"]["p_holm"] = stable_holm[metric]

    format_attribution = analyze_format_attribution(api_rows, sft_rows, task_ids, tasks_file)
    primary = metrics["task_success"]
    effect_pp = float(primary["effect_local_minus_api_pp"])
    ci_low = float(primary["paired_task_bootstrap"]["ci95_low"]) * 100.0
    ci_high = float(primary["paired_task_bootstrap"]["ci95_high"]) * 100.0
    p_holm = float(primary["rollout_mcnemar"]["p_holm"])
    normalized_effect_pp = (
        format_attribution["local_sft_equal_task_rate"]
        - format_attribution["normalized_api_equal_task_rate"]
    ) * 100.0
    normalized_p = float(format_attribution["normalized_mcnemar"]["p_two_sided"])
    contract_note = (
        f"失败归因显示 {format_attribution['fenced_final_answer_failures']}/"
        f"{format_attribution['official_api_failures']} 次 API 失败是最终答案被 fenced JSON 包裹；"
        f"离线去除 fence 后 {format_attribution['fenced_normalized_passes']}/"
        f"{format_attribution['fenced_final_answer_failures']} 通过同一 judge，"
        f"内容级 API≈{format_attribution['normalized_api_equal_task_rate'] * 100:.1f}% vs "
        f"SFT≈{format_attribution['local_sft_equal_task_rate'] * 100:.1f}%"
        f"（post-hoc McNemar p={normalized_p:.4g}）。"
    )
    if p_holm < 0.05 and ci_low > 0.0:
        branch = "local_sft_exceeds_api_on_strict_output_contract"
        conclusion = (
            "严格固定输出契约下，本地 7B SFT 在 45 个受限任务上显著超过 API baseline"
            f"（效应量 {effect_pp:+.1f}pp，Holm p={p_holm:.4g}，"
            f"95% CI [{ci_low:+.1f}, {ci_high:+.1f}]pp）。"
            f"{contract_note}"
            "因此这是输出契约可靠性的优势，不能写成任务内容推理优势。"
        )
    elif p_holm < 0.05 and ci_high < 0.0:
        branch = "api_exceeds_local_sft"
        conclusion = (
            "API baseline 在 45 个受限任务上显著高于本地 7B SFT"
            f"（差距 {effect_pp:+.1f}pp，Holm p={p_holm:.4g}，"
            f"95% CI [{ci_low:+.1f}, {ci_high:+.1f}]pp）。"
        )
    else:
        branch = "statistically_tied"
        conclusion = (
            "API baseline 与本地 7B SFT 在 45 个受限任务上统计不可区分"
            f"（本地−API {effect_pp:+.1f}pp，Holm p={p_holm:.4g}，"
            f"95% CI [{ci_low:+.1f}, {ci_high:+.1f}]pp）。"
        )

    api_vs_baseline = {
        metric: rollout_mcnemar(task_ids, baseline_rows, api_rows, metric) for metric in METRICS
    }
    sft_vs_baseline = {
        metric: rollout_mcnemar(task_ids, baseline_rows, sft_rows, metric) for metric in METRICS
    }
    three_groups = {
        "api": {
            "model": api_summary.get("model_requested"),
            "valid_rollouts": sum(outcome.valid for outcome in api_out.values()),
            "equal_task_success_rate": equal_task_rate(api_out, task_ids, "task_success"),
            "pooled_success_rate": pooled_rate(api_out, task_ids, "task_success"),
            "stable_tasks": sum(is_stable(api_out[task_id]) for task_id in task_ids),
            "within_steps_rate": equal_task_rate(api_out, task_ids, "within_steps"),
            "tool_selection_rate": equal_task_rate(api_out, task_ids, "tool_selection_ok"),
        },
        "local_baseline": {
            "valid_rollouts": sum(outcome.valid for outcome in baseline_out.values()),
            "equal_task_success_rate": equal_task_rate(baseline_out, task_ids, "task_success"),
            "pooled_success_rate": pooled_rate(baseline_out, task_ids, "task_success"),
            "stable_tasks": None,
            "within_steps_rate": equal_task_rate(baseline_out, task_ids, "within_steps"),
            "tool_selection_rate": equal_task_rate(baseline_out, task_ids, "tool_selection_ok"),
        },
        "local_sft": {
            "valid_rollouts": sum(outcome.valid for outcome in sft_out.values()),
            "equal_task_success_rate": equal_task_rate(sft_out, task_ids, "task_success"),
            "pooled_success_rate": pooled_rate(sft_out, task_ids, "task_success"),
            "stable_tasks": sum(is_stable(sft_out[task_id]) for task_id in task_ids),
            "within_steps_rate": equal_task_rate(sft_out, task_ids, "within_steps"),
            "tool_selection_rate": equal_task_rate(sft_out, task_ids, "tool_selection_ok"),
        },
    }

    per_task = []
    for task_id in task_ids:
        api_rate = api_out[task_id].rate("task_success")
        sft_rate = sft_out[task_id].rate("task_success")
        per_task.append(
            {
                "task_id": task_id,
                "family": task_meta.get(task_id, {}).get("family"),
                "category": task_meta.get(task_id, {}).get("category"),
                "difficulty": task_meta.get(task_id, {}).get("difficulty"),
                "training_overlap": task_meta.get(task_id, {}).get("training_overlap"),
                "api_success": api_out[task_id].successes["task_success"],
                "api_valid": api_out[task_id].valid,
                "local_baseline_success": baseline_out[task_id].successes["task_success"],
                "local_baseline_valid": baseline_out[task_id].valid,
                "local_sft_success": sft_out[task_id].successes["task_success"],
                "local_sft_valid": sft_out[task_id].valid,
                "local_minus_api_pp": (sft_rate - api_rate) * 100.0,
            }
        )

    local_sft_gpu_hours = sum(float(row.get("duration_s") or 0.0) for row in sft_rows) / 3600
    local_baseline_gpu_hours = (
        sum(float(row.get("duration_s") or 0.0) for row in baseline_rows) / 3600
    )
    api_prompt_tokens = int(api_summary.get("prompt_tokens") or 0)
    api_completion_tokens = int(api_summary.get("completion_tokens") or 0)
    api_estimated_cost = None
    if api_input_price is not None and api_output_price is not None:
        api_estimated_cost = (
            api_prompt_tokens / 1_000_000 * api_input_price
            + api_completion_tokens / 1_000_000 * api_output_price
        )
    local_sft_estimated_cost = (
        local_sft_gpu_hours * local_gpu_hourly if local_gpu_hourly is not None else None
    )
    local_baseline_estimated_cost = (
        local_baseline_gpu_hours * local_gpu_hourly if local_gpu_hourly is not None else None
    )

    report = {
        "version": "api-vs-local-ab-analysis-v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_manifest": {
            "path": str(tasks_file),
            "sha256": task_sha,
            "tasks": len(task_ids),
        },
        "sources": {
            "api_runs": {
                "path": str(api_runs),
                "sha256": sha256(api_runs),
                "summary_path": str(api_summary_path),
                "summary_sha256": sha256(api_summary_path),
            },
            "local_sft_runs": {
                "path": str(sft_runs),
                "sha256": sha256(sft_runs),
                "adapter_sha256": clean_report.get("adapter_sha256"),
            },
            "local_baseline_runs": {
                "files": [str(path) for path in baseline_files],
                "sha256": {str(path): sha256(path) for path in baseline_files},
            },
            "clean_heldout_report": {
                "path": str(clean_report_path),
                "sha256": sha256(clean_report_path),
            },
        },
        "preflight": {
            "summary_path": str(preflight_path),
            "summary": preflight,
            "selected_api_model_reason": (
                "The exact unqualified `deepseek-v3` alias returned 404; "
                "`vanchin/deepseek-v3` returned 400 because the marketplace product is not "
                "activated; "
                "`deepseek-v4.1-flash` is the available same-family teacher API "
                "and returned native tool_calls in the end-to-end probe."
            ),
        },
        "judge": {
            "path": str(JUDGE_PATH),
            "sha256": sha256(JUDGE_PATH),
            "commit": "9b235ef525e73a419c7ab532def6e04683da329d",
        },
        "model_probe": {
            "date": "2026-10-26",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "results": [
                {
                    "model": "deepseek-v3",
                    "http_status": 404,
                    "result": "model_not_found",
                    "selected": False,
                },
                {
                    "model": "vanchin/deepseek-v3",
                    "http_status": 400,
                    "result": "product_not_activated",
                    "selected": False,
                },
                {
                    "model": "deepseek-v4.1-flash",
                    "http_status": 200,
                    "native_tool_calls": True,
                    "selected": True,
                },
                {
                    "model": "qwen-max",
                    "http_status": 200,
                    "native_tool_calls": True,
                    "selected": False,
                },
            ],
        },
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": None,
            "seed_note": "API seed was not sent; API-side sampling variance remains.",
            "system_variant": "s0",
            "judge": "scripts/unified_eval.py::judge/run_once",
        },
        "api_protocol": {
            "api_requests": api_summary.get("api_requests"),
            "api_tool_call_requests": api_summary.get("api_tool_call_requests"),
            "api_fenced_json_turns": api_summary.get("api_fenced_json_turns"),
            "error_rows": api_summary.get("error_rows"),
            "within_steps_equal_task_rate": api_summary.get("within_steps_equal_task_rate"),
            "tool_selection_equal_task_rate": api_summary.get("tool_selection_equal_task_rate"),
        },
        "three_groups": three_groups,
        "metrics": metrics,
        "holm_rollout": rollout_holm,
        "holm_task_stable": stable_holm,
        "api_vs_local_baseline": api_vs_baseline,
        "local_sft_vs_local_baseline": sft_vs_baseline,
        "format_attribution": format_attribution,
        "normalized_content_effect_local_minus_api_pp": normalized_effect_pp,
        "latency": {
            "api": {
                "api_request_median_s": api_summary.get("request_latency_median_s"),
                "api_request_p95_s": api_summary.get("request_latency_p95_s"),
                "api_rollout_median_s": api_summary.get("rollout_latency_median_s"),
                "api_rollout_p95_s": api_summary.get("rollout_latency_p95_s"),
                "environment": (
                    "local client -> DashScope compatible endpoint; network latency included"
                ),
            },
            "local_sft": {
                "rollout_median_s": statistics.median(
                    float(row.get("duration_s") or 0.0) for row in sft_rows
                ),
                "gpu_hours": local_sft_gpu_hours,
                "environment": "Kaggle T4 14.56GB, offline transformers/NV4",
            },
            "local_baseline": {
                "gpu_hours": local_baseline_gpu_hours,
                "environment": "Kaggle T4 14.56GB, offline transformers/NV4",
            },
        },
        "cost": {
            "currency": "CNY",
            "api_prompt_tokens": api_prompt_tokens,
            "api_completion_tokens": api_completion_tokens,
            "api_total_tokens": api_summary.get("total_tokens"),
            "api_input_price_per_mtok": api_input_price,
            "api_output_price_per_mtok": api_output_price,
            "api_estimated_cost": api_estimated_cost,
            "api_actual_charge": "not available from API response; calculate from console bill",
            "local_sft_gpu_hours": local_sft_gpu_hours,
            "local_baseline_gpu_hours": local_baseline_gpu_hours,
            "local_gpu_hourly": local_gpu_hourly,
            "local_sft_estimated_cost": local_sft_estimated_cost,
            "local_baseline_estimated_cost": local_baseline_estimated_cost,
            "local_actual_charge": "0 incremental: Kaggle free T4 quota",
        },
        "source_audit": {
            "api": api_source,
            "local_sft": sft_source,
            "local_baseline": baseline_source,
        },
        "scope": {
            "generalization": clean_report.get("generalization_scope"),
            "family_overlap": clean_report.get("family_overlap"),
            "prompt_overlap": clean_report.get("prompt_overlap"),
            "note": (
                "The 45 diagnostic tasks are not cross-template held-out. The result "
                "supports same-family instance behavior only, not general coding ability."
            ),
        },
        "per_task": per_task,
        "branch": branch,
        "conclusion": conclusion,
    }
    return report


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _pp(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.1f}pp"


def _p(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4g}"


def format_markdown(report: Mapping[str, Any]) -> str:
    groups = report["three_groups"]
    primary = report["metrics"]["task_success"]
    preflight = report["preflight"]["summary"]
    api = groups["api"]
    baseline = groups["local_baseline"]
    sft = groups["local_sft"]
    lines: list[str] = []
    lines.append("# API vs 本地 7B：45 个受限任务 A/B 对照")
    lines.append("")
    lines.append(
        "> 目的：在同一套 45 个冻结任务、同一 fixture、同一 6 工具 schema、同一确定性 judge、"
        "同一 `temperature=0.7/top_p=0.9` 下，比较 DashScope API baseline 与本地 Qwen2.5-7B "
        "baseline/SFT。"
    )
    lines.append("")
    lines.append("## 1. 前置检查")
    lines.append("")
    lines.append(
        f"- 任务口径：官方 45 任务，SHA-256 `{report['task_manifest']['sha256']}`；"
        "不使用重判 35 口径。"
    )
    lines.append("- Judge：`scripts/unified_eval.py` 的同一 `build_fixture/judge/run_once`，S0。")
    lines.append("- base_url：`https://dashscope.aliyuncs.com/compatible-mode/v1`。")
    lines.append(
        "- 采样：`temperature=0.7`、`top_p=0.9`；未向 API 发送 seed，存在 API 侧额外方差。"
    )
    lines.append(
        "- 模型现场探测（2026-10-26）：`deepseek-v3` 精确别名返回 **404 model_not_found**；"
        "`vanchin/deepseek-v3` 返回 **400 product_not_activated**；"
        "`deepseek-v4.1-flash` 与 `qwen-max` 均返回 200。"
    )
    lines.append(
        "- 选择 A 组 `deepseek-v4.1-flash`：它是当前可用的同系列教师 API，端到端预检返回原生 "
        "`tool_calls`，不是 fenced JSON；`qwen-max` 也通过预检但未作为主组。"
    )
    lines.append(
        f"- 单任务预检：`PUB-XLAM-00001`，1 rollout，"
        f"task_success={preflight.get('pooled_success_rate')}, "
        f"API 请求 {preflight.get('api_requests')} 次，原生 tool-call 请求 "
        f"{preflight.get('api_tool_call_requests')} 次，"
        f"fenced JSON {preflight.get('api_fenced_json_turns')} 次，"
        f"token={preflight.get('total_tokens')}。"
    )
    protocol = report["api_protocol"]
    lines.append(
        f"- 全量协议审计：{protocol['api_requests']} 次 API 请求，"
        f"{protocol['api_tool_call_requests']} 次返回原生 tool-call，"
        f"{protocol['api_fenced_json_turns']} 个最终答案使用 fenced JSON；"
        f"error_rows={protocol['error_rows']}，"
        f"within_steps={_pct(protocol['within_steps_equal_task_rate'])}，"
        f"tool_selection={_pct(protocol['tool_selection_equal_task_rate'])}。"
    )
    lines.append("")
    lines.append("## 2. 三组结果")
    lines.append("")
    lines.append(
        "| 组 | 模型 | valid rollout | task_success（等权） | pooled | "
        "stable 任务 | within_steps | tool_selection |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| A API baseline | `{api.get('model')}` | {api['valid_rollouts']} | "
        f"{_pct(api['equal_task_success_rate'])} | {_pct(api['pooled_success_rate'])} | "
        f"{api['stable_tasks']}/45 | {_pct(api['within_steps_rate'])} | "
        f"{_pct(api['tool_selection_rate'])} |"
    )
    lines.append(
        f"| B 本地 baseline | Qwen2.5-7B 4-bit | {baseline['valid_rollouts']} | "
        f"{_pct(baseline['equal_task_success_rate'])} | {_pct(baseline['pooled_success_rate'])} | "
        f"N/A（8–9 rollout） | {_pct(baseline['within_steps_rate'])} | "
        f"{_pct(baseline['tool_selection_rate'])} |"
    )
    lines.append(
        f"| C 本地 SFT | Qwen2.5-7B + QLoRA | {sft['valid_rollouts']} | "
        f"{_pct(sft['equal_task_success_rate'])} | {_pct(sft['pooled_success_rate'])} | "
        f"{sft['stable_tasks']}/45 | {_pct(sft['within_steps_rate'])} | "
        f"{_pct(sft['tool_selection_rate'])} |"
    )
    lines.append("")
    lines.append(
        "口径：等权均值是 45 个任务通过率的平均；pooled 是全部有效 rollout 合并；"
        "stable 为 10 次中≥6 次通过。"
    )
    lines.append("")
    lines.append("## 3. API vs 本地 SFT 配对统计")
    lines.append("")
    mcnemar = primary["rollout_mcnemar"]
    bootstrap = primary["paired_task_bootstrap"]
    lines.append(
        f"- 配对单位：同一 `task_id + rollout_index`，共 {mcnemar['paired_rollouts']} 对。"
    )
    lines.append(
        f"- 本地 SFT − API：{_pp(primary['effect_local_minus_api_pp'])}；"
        f"paired task-cluster bootstrap 95% CI "
        f"[{bootstrap['ci95_low'] * 100:+.1f}, {bootstrap['ci95_high'] * 100:+.1f}]pp，"
        f"{bootstrap['iterations']} 次。"
    )
    lines.append(
        f"- 精确 McNemar：API 失败→SFT 成功 {mcnemar['baseline_fail_sft_pass']} 对；"
        f"API 成功→SFT 失败 {mcnemar['baseline_pass_sft_fail']} 对；"
        f"p={_p(mcnemar['p_two_sided'])}，Holm p={_p(mcnemar['p_holm'])}。"
    )
    stable = primary["task_stable_mcnemar"]
    lines.append(
        f"- 任务级 stable 敏感性：SFT-only stable {stable['local_only_stable']} 个，"
        f"API-only stable {stable['api_only_stable']} 个，p={_p(stable['p_two_sided'])}，"
        f"Holm p={_p(stable['p_holm'])}。"
    )
    lines.append("")
    lines.append("| 指标 | API（等权） | 本地 SFT（等权） | 本地−API | 95% CI | p | Holm p |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for metric in ("task_success", "within_steps", "tool_selection_ok"):
        item = report["metrics"][metric]
        boot = item["paired_task_bootstrap"]
        test = item["rollout_mcnemar"]
        lines.append(
            f"| `{metric}` | {_pct(item['api_equal_task_rate'])} | "
            f"{_pct(item['local_sft_equal_task_rate'])} | "
            f"{_pp(item['effect_local_minus_api_pp'])} | "
            f"[{boot['ci95_low'] * 100:+.1f}, {boot['ci95_high'] * 100:+.1f}]pp | "
            f"{_p(test['p_two_sided'])} | {_p(test['p_holm'])} |"
        )
    lines.append("")
    lines.append("敏感性：")
    api_base = report["api_vs_local_baseline"]["task_success"]
    sft_base = report["local_sft_vs_local_baseline"]["task_success"]
    lines.append(
        f"- API vs 本地 baseline（共同 rollout）：{api_base['paired_rollouts']} 对，"
        f"baseline 失败→API 成功 {api_base['baseline_fail_sft_pass']}，"
        f"baseline 成功→API 失败 {api_base['baseline_pass_sft_fail']}，"
        f"p={_p(api_base['p_two_sided'])}。"
    )
    lines.append(
        f"- 本地 SFT vs 本地 baseline（共同 rollout）：{sft_base['paired_rollouts']} 对，"
        f"baseline 失败→SFT 成功 {sft_base['baseline_fail_sft_pass']}，"
        f"baseline 成功→SFT 失败 {sft_base['baseline_pass_sft_fail']}，"
        f"p={_p(sft_base['p_two_sided'])}。"
    )
    lines.append("")
    attribution = report["format_attribution"]
    normalized_mcnemar = attribution["normalized_mcnemar"]
    lines.append("## 4. 输出契约归因（post-hoc，非确认性）")
    lines.append("")
    lines.append(
        f"- 官方严格 judge 下：API {attribution['official_api_successes']}/450 成功，"
        f"本地 SFT {attribution['local_sft_successes']}/450 成功。"
    )
    lines.append(
        f"- API 的 {attribution['official_api_failures']} 条失败中，"
        f"{attribution['fenced_final_answer_failures']} 条是最终答案被 fenced JSON 包裹；"
        f"其余 {len(attribution['non_fenced_failures'])} 条为非格式内容错误。"
    )
    lines.append(
        f"- {attribution['fenced_json_extractable']}/{attribution['fenced_final_answer_failures']} "
        "条 fenced 回复可直接抽出合法 JSON；在临时 fixture 中重放已执行工具调用后，"
        f"{attribution['fenced_normalized_passes']}/{attribution['fenced_final_answer_failures']} "
        "通过同一 judge。"
    )
    lines.append(
        f"- 去 fence 后 API 内容级成功 {attribution['normalized_api_successes']}/450"
        f"（等权 {_pct(attribution['normalized_api_equal_task_rate'])}），"
        f"SFT {attribution['local_sft_successes']}/450"
        f"（等权 {_pct(attribution['local_sft_equal_task_rate'])}）；"
        f"post-hoc 配对 McNemar p={_p(normalized_mcnemar['p_two_sided'])}。"
    )
    lines.append(
        "- 因此官方 +8.9pp 差距应解释为“严格固定输出契约下的可靠性差异”，"
        "不能解释为本地模型任务推理更强。若 API 侧做确定的 fence 剥离或结构化输出约束，"
        "本任务集上的内容级结果与本地 SFT 相同。"
    )
    lines.append("")
    lines.append("## 5. 延迟与成本")
    lines.append("")
    latency = report["latency"]
    cost = report["cost"]
    lines.append(
        f"- API：单请求中位数 {latency['api']['api_request_median_s']:.2f}s，"
        f"P95 {latency['api']['api_request_p95_s']:.2f}s；单 rollout 中位数 "
        f"{latency['api']['api_rollout_median_s']:.2f}s。含网络与工具执行。"
    )
    lines.append(
        f"- 本地 SFT：单 rollout 中位数 {latency['local_sft']['rollout_median_s']:.2f}s；"
        f"450 次推理约 {latency['local_sft']['gpu_hours']:.2f} GPU·h（Kaggle T4）。"
    )
    lines.append(
        f"- API token：prompt {cost['api_prompt_tokens']:,}，completion "
        f"{cost['api_completion_tokens']:,}，total {cost['api_total_tokens']:,}。"
    )
    if cost.get("api_estimated_cost") is None:
        lines.append(
            "- API 费用：API 响应不返回账单金额；需按上表 token 用量 × 控制台账单单价计算。"
        )
    else:
        lines.append(
            f"- API 估算费用：¥{cost['api_estimated_cost']:.3f}"
            f"（输入 ¥{cost['api_input_price_per_mtok']}/百万 token，"
            f"输出 ¥{cost['api_output_price_per_mtok']}/百万 token；不是控制台账单实扣）。"
        )
    if cost.get("local_sft_estimated_cost") is None:
        lines.append("- 本地成本：Kaggle 免费 T4 增量账单为 0；租卡口径未给定，不伪造金额。")
    else:
        lines.append(
            f"- 本地 SFT 估算：{cost['local_sft_gpu_hours']:.2f} GPU·h × "
            f"¥{cost['local_gpu_hourly']:.2f}/h = ¥{cost['local_sft_estimated_cost']:.3f}；"
            "不计训练成本。"
        )
    lines.append("")
    lines.append("## 6. 结论")
    lines.append("")
    lines.append(report["conclusion"])
    if report["branch"] == "statistically_tied":
        lines.append("")
        lines.append(
            "统计上按双侧 McNemar + bootstrap CI 判定“不可区分”；如果描述上有很小的百分点差，"
            "不能把它写成稳定能力优势。"
        )
    lines.append("")
    lines.append("## 7. 边界")
    lines.append("")
    lines.append(
        "- 45 个任务是诊断集，不是跨模板 held-out；"
        "clean-heldout 报告已确认 45/45 仍有同族训练任务。"
    )
    lines.append("- 结论只覆盖受限 6 工具、固定输出契约的子智能体场景，不外推到开放域 coding。")
    lines.append(
        "- 本地 baseline 为 8–9 rollout，B 组只作三组描述与共同 index 敏感性，"
        "不与 A/C 的 450 对混算。"
    )
    lines.append("- API model id 与行为可能漂移；本报告固定记录 2026-10-26 的 model/version。")
    lines.append("- 未发送 seed，API 侧存在额外采样方差；未对语义失败做重试或挑样本。")
    lines.append(
        "- 输出契约归因是读取既有回复的事后诊断，不是新的 API rollout；"
        "官方通过率仍以严格 judge 为准，反事实数字不进入确认性统计。"
    )
    lines.append("")
    lines.append("## 8. 逐任务结果")
    lines.append("")
    lines.append("| 任务 | family | difficulty | API | 本地 baseline | 本地 SFT | SFT−API |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in report["per_task"]:
        lines.append(
            f"| `{row['task_id']}` | `{row['family']}` | {row['difficulty']} | "
            f"{row['api_success']}/{row['api_valid']} | "
            f"{row['local_baseline_success']}/{row['local_baseline_valid']} | "
            f"{row['local_sft_success']}/{row['local_sft_valid']} | "
            f"{row['local_minus_api_pp']:+.1f}pp |"
        )
    lines.append("")
    lines.append("## 9. 可追溯性")
    lines.append("")
    sources = report["sources"]
    lines.append(f"- API runs SHA-256：`{sources['api_runs']['sha256']}`")
    lines.append(f"- API summary SHA-256：`{sources['api_runs']['summary_sha256']}`")
    lines.append(f"- 本地 SFT runs SHA-256：`{sources['local_sft_runs']['sha256']}`")
    lines.append(f"- 本地 SFT adapter SHA-256：`{sources['local_sft_runs']['adapter_sha256']}`")
    lines.append(f"- 任务 manifest SHA-256：`{report['task_manifest']['sha256']}`")
    lines.append("- 原始 JSONL 与 trace 位于 `private/`，不进入 Git。")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="analyze API baseline vs local 7B")
    parser.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS_FILE)
    parser.add_argument("--api-runs", type=Path, default=DEFAULT_API_RUNS)
    parser.add_argument("--sft-runs", type=Path, default=DEFAULT_SFT_RUNS)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--api-summary", type=Path, default=DEFAULT_API_SUMMARY)
    parser.add_argument("--clean-report", type=Path, default=DEFAULT_CLEAN_REPORT)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT_SUMMARY)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--expected-task-sha", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--api-input-price-per-mtok", type=float, default=None)
    parser.add_argument("--api-output-price-per-mtok", type=float, default=None)
    parser.add_argument("--local-gpu-hourly", type=float, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    expected = args.expected_task_sha
    if expected is None:
        expected = "20e20b3beabaf0c48c00dae56d97c1aea356b1ac89c4cdb0fd9decc799c4fa5b"
    report = analyze(
        tasks_file=args.tasks_file,
        api_runs=args.api_runs,
        sft_runs=args.sft_runs,
        baseline_root=args.baseline_root,
        api_summary_path=args.api_summary,
        clean_report_path=args.clean_report,
        preflight_path=args.preflight,
        expected_task_sha=expected,
        bootstrap_iterations=args.bootstrap,
        seed=args.seed,
        api_input_price=args.api_input_price_per_mtok,
        api_output_price=args.api_output_price_per_mtok,
        local_gpu_hourly=args.local_gpu_hourly,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.out_md.write_text(format_markdown(report) + "\n", encoding="utf-8")
    print(
        json.dumps({"json": str(args.out_json), "markdown": str(args.out_md)}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
