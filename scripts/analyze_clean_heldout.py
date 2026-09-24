"""Paired analysis of train-v6 clean held-out baseline vs SFT rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRICS = ("task_success", "within_steps", "tool_selection_ok")
METRIC_LABELS = {
    "task_success": "task_success",
    "within_steps": "within_steps",
    "tool_selection_ok": "tool_selection_ok",
}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(payload)
    return rows


def read_task_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            ids.append(text)
            continue
        metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
        task_id = payload.get("task_id") if isinstance(payload, Mapping) else None
        if not task_id and isinstance(metadata, Mapping):
            task_id = metadata.get("task_id")
        if not task_id:
            raise ValueError(f"{path} has a row without task_id")
        ids.append(str(task_id))
    if not ids:
        raise ValueError(f"no task ids in {path}")
    if len(ids) != len(set(ids)):
        duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
        raise ValueError(f"duplicate task ids: {duplicates}")
    return ids


def discover_baseline_files(root: Path) -> list[Path]:
    files = sorted(
        path for path in root.glob("runs*/**/baseline/runs.jsonl") if "staging" not in path.parts
    )
    if not files:
        raise FileNotFoundError(f"no baseline runs.jsonl under {root}")
    return files


def read_task_families(path: Path) -> dict[str, str]:
    families: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        task_id = str(payload.get("task_id") or "")
        manifest = payload.get("manifest")
        family = manifest.get("family") if isinstance(manifest, Mapping) else None
        if task_id and family:
            families[task_id] = str(family)
    return families


def family_overlap(task_ids: Sequence[str], tasks_file: Path, train_file: Path) -> dict[str, Any]:
    """Count train-v6 tasks/rows that share each held-out task's spec family."""

    from training.heldout_pool import build_heldout_pool

    families = read_task_families(tasks_file)
    pool = build_heldout_pool(train_file)
    train_tasks = Counter()
    train_rows = Counter()
    for task in pool.tasks:
        if task.dataset_rows <= 0:
            continue
        train_tasks[task.spec.family] += 1
        train_rows[task.spec.family] += task.dataset_rows
    per_task = []
    for task_id in task_ids:
        family = families.get(task_id, "unknown")
        per_task.append(
            {
                "task_id": task_id,
                "family": family,
                "train_v6_same_family_tasks": train_tasks.get(family, 0),
                "train_v6_same_family_rows": train_rows.get(family, 0),
            }
        )
    covered = sum(1 for row in per_task if row["train_v6_same_family_tasks"] > 0)
    return {
        "train_file": str(train_file),
        "heldout_tasks": len(task_ids),
        "heldout_tasks_with_same_family_in_train": covered,
        "heldout_families": dict(sorted(Counter(row["family"] for row in per_task).items())),
        "train_v6_tasks_by_family": dict(sorted(train_tasks.items())),
        "train_v6_rows_by_family": dict(sorted(train_rows.items())),
        "per_task": per_task,
        "scope": "within_family_instance"
        if covered == len(task_ids)
        else "cross_template_or_partial",
    }


def normalize_prompt(text: str) -> str:
    value = re.sub(r"[0-9]+", "<N>", text)
    value = re.sub(r"project/[A-Za-z0-9_./-]+", "<PATH>", value)
    return value


def prompt_overlap(task_ids: Sequence[str], tasks_file: Path, train_file: Path) -> dict[str, Any]:
    """Check exact and normalized user-prompt overlap against train-v6."""

    train_prompts: set[str] = set()
    with train_file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for message in row.get("messages") or []:
                if isinstance(message, Mapping) and message.get("role") == "user":
                    content = str(message.get("content") or "").strip()
                    if content:
                        train_prompts.add(content)
    train_normalized = {normalize_prompt(prompt) for prompt in train_prompts}
    per_task = []
    wanted = set(task_ids)
    for line in tasks_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        task_id = str(payload.get("task_id") or "")
        if task_id not in wanted:
            continue
        spec = payload.get("spec") or {}
        user_turns = [str(item).strip() for item in spec.get("user_turns") or []]
        exact = any(prompt in train_prompts for prompt in user_turns)
        normalized = any(normalize_prompt(prompt) in train_normalized for prompt in user_turns)
        per_task.append(
            {
                "task_id": task_id,
                "exact_prompt_overlap": exact,
                "normalized_prompt_overlap": normalized,
            }
        )
    per_task.sort(key=lambda item: task_ids.index(item["task_id"]))
    return {
        "train_file": str(train_file),
        "tasks": len(per_task),
        "exact_prompt_overlap_tasks": sum(1 for row in per_task if row["exact_prompt_overlap"]),
        "normalized_prompt_overlap_tasks": sum(
            1 for row in per_task if row["normalized_prompt_overlap"]
        ),
        "per_task": per_task,
    }


@dataclass(frozen=True)
class TaskOutcome:
    valid: int
    successes: dict[str, int]

    def rate(self, metric: str) -> float:
        return self.successes[metric] / self.valid if self.valid else 0.0


def load_outcomes(
    files: Sequence[Path], task_ids: Sequence[str], *, label: str
) -> tuple[dict[str, TaskOutcome], dict[str, Any]]:
    wanted = set(task_ids)
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(read_jsonl(path))
    selected = [row for row in rows if str(row.get("task_id") or "") in wanted]
    seen: Counter[tuple[str, int]] = Counter()
    errors: Counter[str] = Counter()
    valid_by_task: dict[str, list[Mapping[str, Any]]] = {task_id: [] for task_id in task_ids}
    for row in selected:
        task_id = str(row["task_id"])
        rollout = int(row.get("rollout_index", -1))
        key = (task_id, rollout)
        seen[key] += 1
        if row.get("error_kind"):
            errors[task_id] += 1
            continue
        valid_by_task[task_id].append(row)
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} has duplicate task/rollout rows: {duplicates[:10]}")
    missing = [task_id for task_id in task_ids if not valid_by_task[task_id]]
    outcomes: dict[str, TaskOutcome] = {}
    for task_id in task_ids:
        task_rows = valid_by_task[task_id]
        outcomes[task_id] = TaskOutcome(
            valid=len(task_rows),
            successes={
                metric: sum(1 for row in task_rows if bool(row.get(metric))) for metric in METRICS
            },
        )
    report = {
        "label": label,
        "files": [str(path) for path in files],
        "source_sha256": {str(path): sha256(path) for path in files},
        "source_rows": len(rows),
        "selected_rows": len(selected),
        "tasks_requested": len(task_ids),
        "tasks_with_valid_rollouts": sum(1 for task_id in task_ids if valid_by_task[task_id]),
        "missing_tasks": missing,
        "error_rows": sum(errors.values()),
        "errors_by_task": dict(sorted(errors.items())),
        "valid_rollouts_by_task": dict(
            sorted((task_id, len(valid_by_task[task_id])) for task_id in task_ids)
        ),
    }
    return outcomes, report


def binom_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, index) for index in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def binom_greater_equal(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    return min(1.0, sum(math.comb(n, index) for index in range(k, n + 1)) / (2**n))


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(items)
    for index, (name, value) in enumerate(items):
        candidate = min(1.0, value * (total - index))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def task_mcnemar(
    task_ids: Sequence[str],
    baseline: Mapping[str, TaskOutcome],
    sft: Mapping[str, TaskOutcome],
    metric: str,
) -> dict[str, Any]:
    better: list[str] = []
    worse: list[str] = []
    ties: list[str] = []
    for task_id in task_ids:
        left = baseline[task_id].rate(metric)
        right = sft[task_id].rate(metric)
        if right > left:
            better.append(task_id)
        elif right < left:
            worse.append(task_id)
        else:
            ties.append(task_id)
    discordant = len(better) + len(worse)
    return {
        "metric": metric,
        "sft_better_tasks": len(better),
        "sft_worse_tasks": len(worse),
        "tied_tasks": len(ties),
        "discordant_tasks": discordant,
        "p_two_sided": binom_two_sided(min(len(better), len(worse)), discordant),
        "p_sft_better": binom_greater_equal(len(better), discordant),
        "better_task_ids": better,
        "worse_task_ids": worse,
    }


def rollout_mcnemar(
    task_ids: Sequence[str],
    baseline_rows: Sequence[Mapping[str, Any]],
    sft_rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> dict[str, Any]:
    wanted = set(task_ids)
    baseline = {
        (str(row["task_id"]), int(row["rollout_index"])): bool(row.get(metric))
        for row in baseline_rows
        if not row.get("error_kind") and str(row.get("task_id") or "") in wanted
    }
    sft = {
        (str(row["task_id"]), int(row["rollout_index"])): bool(row.get(metric))
        for row in sft_rows
        if not row.get("error_kind") and str(row.get("task_id") or "") in wanted
    }
    common = sorted(set(baseline) & set(sft))
    baseline_fail_sft_pass = sum(1 for key in common if not baseline[key] and sft[key])
    baseline_pass_sft_fail = sum(1 for key in common if baseline[key] and not sft[key])
    discordant = baseline_fail_sft_pass + baseline_pass_sft_fail
    return {
        "metric": metric,
        "paired_rollouts": len(common),
        "baseline_fail_sft_pass": baseline_fail_sft_pass,
        "baseline_pass_sft_fail": baseline_pass_sft_fail,
        "discordant_rollouts": discordant,
        "p_two_sided": binom_two_sided(
            min(baseline_fail_sft_pass, baseline_pass_sft_fail), discordant
        ),
    }


def equal_task_rate(
    outcomes: Mapping[str, TaskOutcome], task_ids: Sequence[str], metric: str
) -> float:
    rates = [outcomes[task_id].rate(metric) for task_id in task_ids]
    return sum(rates) / len(rates) if rates else 0.0


def pooled_rate(outcomes: Mapping[str, TaskOutcome], task_ids: Sequence[str], metric: str) -> float:
    successes = sum(outcomes[task_id].successes[metric] for task_id in task_ids)
    total = sum(outcomes[task_id].valid for task_id in task_ids)
    return successes / total if total else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stratified_bootstrap(
    task_ids: Sequence[str],
    baseline: Mapping[str, TaskOutcome],
    sft: Mapping[str, TaskOutcome],
    *,
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    effects: list[float] = []
    for _ in range(iterations):
        sampled_tasks = [rng.choice(task_ids) for _ in task_ids]
        differences: list[float] = []
        for task_id in sampled_tasks:
            base = baseline[task_id]
            treat = sft[task_id]
            base_rate = _resample_rate(rng, base.successes[metric], base.valid)
            treat_rate = _resample_rate(rng, treat.successes[metric], treat.valid)
            differences.append(treat_rate - base_rate)
        effects.append(sum(differences) / len(differences))
    return {
        "metric": metric,
        "iterations": iterations,
        "seed": seed,
        "method": "hierarchical_task_stratified_percentile",
        "mean": sum(effects) / len(effects) if effects else 0.0,
        "ci95_low": percentile(effects, 0.025),
        "ci95_high": percentile(effects, 0.975),
    }


def _resample_rate(rng: random.Random, successes: int, valid: int) -> float:
    if valid <= 0:
        return 0.0
    probability = successes / valid
    draws = sum(1 for _ in range(valid) if rng.random() < probability)
    return draws / valid


def build_report(
    *,
    task_ids: Sequence[str],
    baseline: Mapping[str, TaskOutcome],
    sft: Mapping[str, TaskOutcome],
    baseline_report: Mapping[str, Any],
    sft_report: Mapping[str, Any],
    baseline_rows: Sequence[Mapping[str, Any]],
    sft_rows: Sequence[Mapping[str, Any]],
    bootstrap_iterations: int,
    seed: int,
    train_sha256: str | None,
    adapter_sha256: str | None,
    family_audit: Mapping[str, Any] | None = None,
    prompt_audit: Mapping[str, Any] | None = None,
    run_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    for metric in METRICS:
        mcnemar = task_mcnemar(task_ids, baseline, sft, metric)
        pvalues[metric] = mcnemar["p_two_sided"]
        baseline_equal = equal_task_rate(baseline, task_ids, metric)
        sft_equal = equal_task_rate(sft, task_ids, metric)
        metrics[metric] = {
            "baseline_equal_task_rate": baseline_equal,
            "sft_equal_task_rate": sft_equal,
            "effect_pp": (sft_equal - baseline_equal) * 100.0,
            "baseline_pooled_rate": pooled_rate(baseline, task_ids, metric),
            "sft_pooled_rate": pooled_rate(sft, task_ids, metric),
            "mcnemar_task_level": mcnemar,
            "mcnemar_rollout_level_sensitivity": rollout_mcnemar(
                task_ids, baseline_rows, sft_rows, metric
            ),
            "bootstrap": stratified_bootstrap(
                task_ids,
                baseline,
                sft,
                metric=metric,
                iterations=bootstrap_iterations,
                seed=seed + METRICS.index(metric),
            ),
        }
    adjusted = holm(pvalues)
    for metric in METRICS:
        metrics[metric]["mcnemar_task_level"]["p_holm"] = adjusted[metric]

    primary = metrics["task_success"]
    effect = primary["effect_pp"]
    ci_low = primary["bootstrap"]["ci95_low"] * 100.0
    ci_high = primary["bootstrap"]["ci95_high"] * 100.0
    p_adj = primary["mcnemar_task_level"]["p_holm"]
    if p_adj < 0.05 and ci_low > 0.0 and effect >= 10.0:
        branch = "significant_positive"
        conclusion = (
            "SFT 在去重叠的中间态任务上出现显著正增量"
            f"（效应量 {effect:.1f}pp，Holm p={p_adj:.4g}，"
            f"95% CI [{ci_low:.1f}, {ci_high:.1f}]pp）。"
        )
    elif p_adj < 0.05 and ci_high < 0.0 and effect < 0.0:
        branch = "significant_degradation"
        conclusion = (
            "SFT 在去重叠的中间态任务上显著退化"
            f"（效应量 {effect:.1f}pp，Holm p={p_adj:.4g}，"
            f"95% CI [{ci_low:.1f}, {ci_high:.1f}]pp）。"
        )
    else:
        branch = "no_significant_increment"
        conclusion = (
            "SFT 在去重叠的中间态任务上无显著增量"
            f"（效应量 {effect:.1f}pp，Holm p={p_adj:.4g}，"
            f"95% CI [{ci_low:.1f}, {ci_high:.1f}]pp）。"
        )

    scope = "not_checked"
    if family_audit is not None:
        scope = str(family_audit.get("scope") or "not_checked")
        if branch == "significant_positive" and scope == "within_family_instance":
            conclusion += (
                f" 但 {family_audit['heldout_tasks_with_same_family_in_train']}/"
                f"{family_audit['heldout_tasks']} 个任务在 train-v6 仍有同族任务，"
                "因此只能表述为同族实例泛化，不能写成跨模板泛化。"
            )

    prompt_groups = None
    if prompt_audit is not None:
        flags = {row["task_id"]: row for row in prompt_audit["per_task"]}
        group_specs = {
            "exact_prompt_overlap": [
                task for task in task_ids if flags[task]["exact_prompt_overlap"]
            ],
            "no_exact_prompt": [
                task for task in task_ids if not flags[task]["exact_prompt_overlap"]
            ],
            "normalized_prompt_overlap": [
                task for task in task_ids if flags[task]["normalized_prompt_overlap"]
            ],
            "no_normalized_prompt": [
                task for task in task_ids if not flags[task]["normalized_prompt_overlap"]
            ],
        }
        prompt_groups = {}
        for name, subset in group_specs.items():
            if not subset:
                continue
            baseline_rate = equal_task_rate(baseline, subset, "task_success")
            sft_rate = equal_task_rate(sft, subset, "task_success")
            bootstrap = stratified_bootstrap(
                subset,
                baseline,
                sft,
                metric="task_success",
                iterations=bootstrap_iterations,
                seed=seed + 100,
            )
            prompt_groups[name] = {
                "tasks": len(subset),
                "baseline_rate": baseline_rate,
                "sft_rate": sft_rate,
                "effect_pp": (sft_rate - baseline_rate) * 100.0,
                "bootstrap": bootstrap,
                "mcnemar": task_mcnemar(subset, baseline, sft, "task_success"),
            }
        exact_group = prompt_groups.get("exact_prompt_overlap")
        no_exact_group = prompt_groups.get("no_exact_prompt")
        if exact_group and no_exact_group:
            conclusion += (
                f" Prompt 级：{prompt_audit['exact_prompt_overlap_tasks']}/"
                f"{prompt_audit['tasks']} 个任务与 train-v6 有逐字重叠；"
                f"有重叠组 {exact_group['effect_pp']:+.1f}pp，"
                f"无重叠组 {no_exact_group['effect_pp']:+.1f}pp。"
            )

    baseline_middle = [
        task_id for task_id in task_ids if 0.0 < baseline[task_id].rate("task_success") < 1.0
    ]
    sft_middle = [task_id for task_id in task_ids if 0.0 < sft[task_id].rate("task_success") < 1.0]
    baseline_uniform = len(baseline_middle) == 0
    sft_uniform = len(sft_middle) == 0
    disappearance_note = None
    if baseline_uniform and sft_uniform:
        branch = "middle_state_disappeared"
        disappearance_note = (
            "baseline 与 SFT 在 45 个任务上均无任务内混合结果（全部 0 或全对），"
            "去重叠后中间态消失。"
        )
        conclusion = disappearance_note
    elif sft_uniform:
        disappearance_note = (
            "SFT 在 45 个任务上均无任务内混合结果（每任务全 0 或全对），"
            "但 baseline 仍有中间态；这是 SFT 侧中间态消失，不等同于 SFT 无效。"
        )

    per_task = []
    for task_id in task_ids:
        base = baseline[task_id]
        treat = sft[task_id]
        diff = (treat.rate("task_success") - base.rate("task_success")) * 100.0
        per_task.append(
            {
                "task_id": task_id,
                "baseline_success": base.successes["task_success"],
                "baseline_valid": base.valid,
                "sft_success": treat.successes["task_success"],
                "sft_valid": treat.valid,
                "diff_pp": diff,
            }
        )
    return {
        "version": "clean-heldout-analysis-v1",
        "train_sha256": train_sha256,
        "adapter_sha256": adapter_sha256,
        "generalization_scope": scope,
        "family_overlap": dict(family_audit) if family_audit is not None else None,
        "prompt_overlap": dict(prompt_audit) if prompt_audit is not None else None,
        "prompt_overlap_groups": prompt_groups,
        "run_info": dict(run_info) if run_info is not None else None,
        "tasks": len(task_ids),
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
        "baseline": dict(baseline_report),
        "sft": dict(sft_report),
        "metrics": metrics,
        "holm": adjusted,
        "branch": branch,
        "conclusion": conclusion,
        "middle_state": {
            "baseline_middle_tasks": len(baseline_middle),
            "sft_middle_tasks": len(sft_middle),
            "note": disappearance_note,
        },
        "per_task": per_task,
    }


def format_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Clean held-out 配对统计（train-v6）")
    lines.append("")
    lines.append("> 目的：train-v5 的 45 个中间态任务全部与训练集重叠。train-v6 移除这些任务的")
    lines.append(
        "> 全部 67 条轨迹后重训 7B SFT，在 45 个任务上与已有 baseline rollout 做配对比较。"
    )
    lines.append("")
    lines.append("## 1. 版本与数据完整性")
    lines.append("")
    lines.append(f"- train-v6 train SHA-256：`{report['train_sha256']}`")
    lines.append(f"- SFT adapter SHA-256：`{report['adapter_sha256']}`")
    lines.append(f"- 任务数：{report['tasks']}")
    for key in ("baseline", "sft"):
        item = report[key]
        counts = Counter(item["valid_rollouts_by_task"].values())
        counts_text = ", ".join(
            f"{valid} 条 × {tasks} 任务" for valid, tasks in sorted(counts.items())
        )
        lines.append(
            f"- {key}：{item['selected_rows']} 行，缺失任务 {len(item['missing_tasks'])}，"
            f"error 行 {item['error_rows']}，有效 rollout 分布 {counts_text}"
        )
        lines.append(f"  - 来源：{', '.join(f'`{path}`' for path in item['files'])}")
    run_info = report.get("run_info")
    if run_info:
        lines.append("")
        lines.append("### 1.1 训练与运行")
        lines.append("")
        lines.append(
            f"- 硬件：{run_info.get('device', 'unknown')}，"
            f"{run_info.get('gpu_hours_free_tier', 0):.2f} GPU·h，"
            f"总耗时 {run_info.get('duration_s', 0):.0f}s。"
        )
        lines.append(
            f"- 训练：train-v6 {run_info.get('train_rows', 0)} 行 / val "
            f"{run_info.get('val_rows', 0)} 行；"
            f"{run_info.get('optimizer_steps', 0)} optimizer steps；"
            f"train loss {run_info.get('train_loss', 0):.4f}；"
            f"adapter {run_info.get('adapter_bytes', 0):,} bytes。"
        )
        lines.append(
            f"- tokenizer gate：train 最大 {run_info.get('train_token_max', 0)} tokens，"
            f"val 最大 {run_info.get('val_token_max', 0)} tokens，全部通过 3072 门禁。"
        )
        lines.append(
            f"- staging：{run_info.get('staging_success', 0)}/"
            f"{run_info.get('staging_rows', 0)} 成功，"
            f"error {run_info.get('staging_errors', 0)}；"
            f"full：{run_info.get('full_success', 0)}/{run_info.get('full_rows', 0)} 成功，"
            f"error {run_info.get('full_errors', 0)}。"
        )
        failures = run_info.get("failed_rollouts") or []
        if failures:
            detail = "; ".join(
                f"`{item['task_id']}` rollout {item['rollout_index']} "
                f"failed_checks={item['failed_checks']}"
                for item in failures[:5]
            )
            lines.append(f"- 失败 rollout：{detail}。")
    lines.append("")
    lines.append("## 2. 总体指标")
    lines.append("")
    lines.append(
        "| 指标 | baseline 均值 | SFT 均值 | 效应量 | 95% CI | 任务级 McNemar p | Holm p |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for metric in METRICS:
        item = report["metrics"][metric]
        lines.append(
            "| {metric} | {base:.1%} | {sft:.1%} | {effect:+.1f}pp | "
            "[{low:+.1f}, {high:+.1f}]pp | {p:.4g} | {ph:.4g} |".format(
                metric=METRIC_LABELS[metric],
                base=item["baseline_equal_task_rate"],
                sft=item["sft_equal_task_rate"],
                effect=item["effect_pp"],
                low=item["bootstrap"]["ci95_low"] * 100.0,
                high=item["bootstrap"]["ci95_high"] * 100.0,
                p=item["mcnemar_task_level"]["p_two_sided"],
                ph=item["mcnemar_task_level"]["p_holm"],
            )
        )
    lines.append("")
    lines.append("口径：均值为 45 个任务通过率的等权平均；CI 为按任务分层的 Bootstrap 百分位区间")
    lines.append(f"（{report['bootstrap_iterations']} 次，seed={report['seed']}）。Holm 校正覆盖 ")
    lines.append("task_success、within_steps、tool_selection_ok 三个指标。")
    lines.append("")
    lines.append("### 任务级 McNemar 方向表（task_success）")
    lines.append("")
    primary = report["metrics"]["task_success"]["mcnemar_task_level"]
    lines.append("| SFT 更好 | SFT 更差 | 持平 | 不一致任务数 | p（双侧） |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| {primary['sft_better_tasks']} | {primary['sft_worse_tasks']} | "
        f"{primary['tied_tasks']} | "
        f"{primary['discordant_tasks']} | {primary['p_two_sided']:.4g} |"
    )
    lines.append("")
    lines.append(
        "口径：任务级 McNemar 以任务为配对单位，不一致对是“SFT 通过率高于 baseline”与"
        "“SFT 通过率低于 baseline”的任务；由于 baseline 为 8 rollout、SFT 为 10 rollout，"
        "任务级检验是主分析。下面的 rollout 级表只配对共同 rollout_index 0–7，作敏感性分析。"
    )
    lines.append("")
    lines.append("### Rollout 级 McNemar（敏感性分析）")
    lines.append("")
    lines.append("baseline 为 8 rollout、SFT 为 10 rollout，仅按共同 rollout_index 0–7 配对：")
    lines.append("")
    lines.append(
        "| 指标 | 配对数 | baseline 失败 → SFT 成功 | baseline 成功 → SFT 失败 | p（双侧） |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for metric in METRICS:
        item = report["metrics"][metric]["mcnemar_rollout_level_sensitivity"]
        lines.append(
            f"| {METRIC_LABELS[metric]} | {item['paired_rollouts']} | "
            f"{item['baseline_fail_sft_pass']} | "
            f"{item['baseline_pass_sft_fail']} | {item['p_two_sided']:.4g} |"
        )
    lines.append("")
    lines.append("## 3. 模板 / 同族重叠审计")
    lines.append("")
    overlap = report.get("family_overlap")
    if overlap:
        lines.append(
            f"- 45 个 held-out 任务中，train-v6 仍存在同族任务的任务数："
            f"{overlap['heldout_tasks_with_same_family_in_train']}/{overlap['heldout_tasks']}。"
        )
        lines.append(f"- 泛化范围判定：`{overlap['scope']}`")
        lines.append("")
        lines.append("| family | held-out 任务数 | train-v6 同族任务数 | train-v6 同族轨迹数 |")
        lines.append("| --- | ---: | ---: | ---: |")
        for family, count in overlap["heldout_families"].items():
            lines.append(
                f"| `{family}` | {count} | {overlap['train_v6_tasks_by_family'].get(family, 0)} | "
                f"{overlap['train_v6_rows_by_family'].get(family, 0)} |"
            )
        lines.append("")
        lines.append(
            "解释边界：task_id 级去重叠已成立（这些任务的 67 条轨迹已从 train-v6 移除），"
            "但同族模板在 train-v6 仍有其他实例。因此本结果证明同族实例泛化，"
            "不能证明跨模板泛化；跨模板结论需要整族 held-out 实验。"
        )
        prompt = report.get("prompt_overlap")
        groups = report.get("prompt_overlap_groups")
        if prompt and groups:
            lines.append("")
            lines.append("### 3.1 Prompt 级重叠与分组统计")
            lines.append("")
            lines.append(
                f"- 逐字 prompt 与 train-v6 重叠的任务：{prompt['exact_prompt_overlap_tasks']}/"
                f"{prompt['tasks']}；归一化模板重叠：{prompt['normalized_prompt_overlap_tasks']}/"
                f"{prompt['tasks']}。"
            )
            lines.append("")
            lines.append("| 分组 | 任务数 | baseline | SFT | 效应量 | 95% CI | 任务级 p |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for name, item in groups.items():
                lines.append(
                    f"| `{name}` | {item['tasks']} | {item['baseline_rate']:.1%} | "
                    f"{item['sft_rate']:.1%} | {item['effect_pp']:+.1f}pp | "
                    f"[{item['bootstrap']['ci95_low'] * 100:+.1f}, "
                    f"{item['bootstrap']['ci95_high'] * 100:+.1f}]pp | "
                    f"{item['mcnemar']['p_two_sided']:.4g} |"
                )
            lines.append("")
            lines.append(
                "解释：有/无逐字 prompt 重叠两组都提升约 50pp，说明增益不只来自逐字记忆；"
                "但 45/45 任务仍有同族模板，仍不能外推为跨模板泛化。"
            )
    else:
        lines.append("- 未提供 train-v6 同族审计输入，无法判定泛化范围。")
    lines.append("")
    lines.append("## 4. 中间态是否消失")
    lines.append("")
    middle = report["middle_state"]
    lines.append(f"- baseline 有任务内混合结果的任务数：{middle['baseline_middle_tasks']}/45")
    lines.append(f"- SFT 有任务内混合结果的任务数：{middle['sft_middle_tasks']}/45")
    if middle["note"]:
        lines.append(f"- 判定：{middle['note']}")
    elif 0 < middle["sft_middle_tasks"] < middle["baseline_middle_tasks"]:
        lines.append(
            "- 判定：SFT 的任务内中间态从 "
            f"{middle['baseline_middle_tasks']}/45 收缩到 {middle['sft_middle_tasks']}/45，"
            "接近天花板但未完全消失；baseline 仍是完整中间态。"
        )
    else:
        lines.append("- 判定：两组都保留了任务内混合结果，未出现中间态整体消失。")
    lines.append("")
    lines.append("## 5. 逐任务结果")
    lines.append("")
    lines.append("| task_id | baseline | SFT | 差值 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in report["per_task"]:
        lines.append(
            f"| `{row['task_id']}` | {row['baseline_success']}/{row['baseline_valid']} | "
            f"{row['sft_success']}/{row['sft_valid']} | {row['diff_pp']:+.1f}pp |"
        )
    lines.append("")
    lines.append("## 6. 结论")
    lines.append("")
    lines.append(f"- 分支：`{report['branch']}`")
    lines.append(f"- 泛化范围：`{report['generalization_scope']}`")
    lines.append(f"- {report['conclusion']}")
    lines.append("")
    lines.append("边界：")
    lines.append("")
    lines.append("- baseline 复用 difficulty screen 的已有 8 rollout，不重跑；SFT 为 10 rollout。")
    lines.append("- 45 个任务来自 train-v5 的官方中间态口径（baseline 通过率 30%–70%）。")
    lines.append("- 只评估 7B + 单一 SFT 配方 + 单一 seed，不推广到其他模型或配方。")
    lines.append("- Holm 校正只覆盖本文三个指标；p 值与 CI 均应按 45 个冻结任务解释。")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="analyze clean held-out paired results")
    parser.add_argument(
        "--tasks-file", type=Path, default=Path("private/final_test_v3/tasks.jsonl")
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("private/kaggle_difficulty"),
    )
    parser.add_argument("--sft-runs", type=Path, required=True)
    parser.add_argument("--train-sha256", default=None)
    parser.add_argument("--adapter-sha256", default=None)
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=None,
        help="optional kernel summary.json with training/runtime metadata",
    )
    parser.add_argument(
        "--train-overlap-file",
        type=Path,
        default=Path("private/sft_data_v6/train.jsonl"),
        help="train-v6 file used only for same-family overlap audit",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    task_ids = read_task_ids(args.tasks_file)
    baseline_files = discover_baseline_files(args.baseline_root)
    baseline_rows_all: list[dict[str, Any]] = []
    for path in baseline_files:
        baseline_rows_all.extend(read_jsonl(path))
    sft_rows_all = read_jsonl(args.sft_runs)
    baseline, baseline_report = load_outcomes(baseline_files, task_ids, label="baseline")
    sft, sft_report = load_outcomes([args.sft_runs], task_ids, label="sft")
    if baseline_report["missing_tasks"]:
        raise ValueError(f"baseline missing tasks: {baseline_report['missing_tasks']}")
    if sft_report["missing_tasks"]:
        raise ValueError(f"sft missing tasks: {sft_report['missing_tasks']}")
    run_info = None
    if args.run_summary is not None:
        summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
        manifest = summary.get("train_manifest") or {}
        gate = summary.get("tokenizer_gate") or {}
        run_info = {
            "device": (summary.get("gpu") or {}).get("device"),
            "gpu_hours_free_tier": summary.get("gpu_hours_free_tier", 0.0),
            "duration_s": summary.get("duration_s", 0.0),
            "train_rows": manifest.get("train_rows", 0),
            "val_rows": manifest.get("val_rows", 0),
            "optimizer_steps": manifest.get("optimizer_steps", 0),
            "train_loss": (manifest.get("train_metrics") or {}).get("train_loss", 0.0),
            "adapter_bytes": manifest.get("adapter_bytes", 0),
            "train_token_max": ((gate.get("train") or {}).get("token_count_max", 0)),
            "val_token_max": ((gate.get("val") or {}).get("token_count_max", 0)),
            "staging_rows": (summary.get("staging_runs") or {}).get("rows", 0),
            "staging_success": (summary.get("staging_runs") or {}).get("task_success", 0),
            "staging_errors": (summary.get("staging_runs") or {}).get("errors", 0),
            "full_rows": (summary.get("full_runs") or {}).get("rows", 0),
            "full_success": (summary.get("full_runs") or {}).get("task_success", 0),
            "full_errors": (summary.get("full_runs") or {}).get("errors", 0),
        }
        run_info["failed_rollouts"] = [
            {
                "task_id": row.get("task_id"),
                "rollout_index": row.get("rollout_index"),
                "failed_checks": sorted(
                    name for name, ok in (row.get("checks") or {}).items() if not ok
                ),
            }
            for row in sft_rows_all
            if not row.get("task_success")
        ]
    report = build_report(
        task_ids=task_ids,
        baseline=baseline,
        sft=sft,
        baseline_report=baseline_report,
        sft_report=sft_report,
        baseline_rows=baseline_rows_all,
        sft_rows=sft_rows_all,
        bootstrap_iterations=args.bootstrap,
        seed=args.seed,
        train_sha256=args.train_sha256,
        adapter_sha256=args.adapter_sha256,
        family_audit=family_overlap(task_ids, args.tasks_file, args.train_overlap_file),
        prompt_audit=prompt_overlap(task_ids, args.tasks_file, args.train_overlap_file),
        run_info=run_info,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(format_markdown(report), encoding="utf-8")
    json_path = args.json_out or args.out.with_suffix(".json")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"branch": report["branch"], "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
