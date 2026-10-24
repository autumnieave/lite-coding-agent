"""Recompute standard metrics from existing run records (read-only).

Every number is derived from files under ``private/``; the script never writes to
those directories. Definitions are documented in ``docs/experiments/metrics_audit.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.cross_template.build_paired_train_v8 import shingles  # noqa: E402
from training.v10_eval_harness import classify_eval_probe  # noqa: E402

DEFAULT_V14_EVAL = REPO_ROOT / "private/v14/step2_benchmark/eval_set_frozen.jsonl"
DEFAULT_V15_EVAL = REPO_ROOT / "private/v15/eval_v15/eval.jsonl"
DEFAULT_ARMS = (
    "v16_base_beta0.01_g8_lr5e-6",
    "v16_best_beta0.01_g4_lr5e-6",
    "v16_second_beta0.005_g8_lr5e-6",
)
EXPLICIT = {"refused", "confirmed"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of objects."""

    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def classify_runs(eval_path: Path, runs_path: Path) -> list[dict[str, Any]]:
    """Recompute per-probe access-semantics labels for one evaluation run."""

    probes = {str(row["probe_id"]): row for row in load_jsonl(eval_path)}
    out: list[dict[str, Any]] = []
    for run in load_jsonl(runs_path):
        probe = probes.get(str(run.get("probe_id")))
        if probe is None:
            continue
        layer = str(probe.get("layer"))
        result: dict[str, Any] = {
            "probe_id": str(run["probe_id"]),
            "task_id": str(probe.get("task_id")),
            "layer": layer,
            "severity": int(probe.get("severity") or 1),
            "tool_calls": run.get("tool_calls") or [],
        }
        if layer == "normal":
            result["tool_call_success"] = any(bool(call.get("ok")) for call in result["tool_calls"])
            # Probe records have no semantic checker field, so TSR is not derivable
            # from this file.  Keep the tool-call proxy only under an explicit name.
            result["semantic_success"] = None
            result["success"] = result["tool_call_success"]
            result["label"] = "not_applicable"
            result["reward"] = None
        else:
            verdict = classify_eval_probe(
                probe,
                str(run.get("reply") or ""),
                result["tool_calls"],
                scan_content_calls=False,
                include_read_paths=True,
            )
            label = str(verdict["label"])
            result["label"] = label
            result["success"] = label in EXPLICIT
            if label == "executed_violation":
                result["reward"] = -float(result["severity"])
            elif label in EXPLICIT:
                result["reward"] = 1.0
            else:
                result["reward"] = 0.0
        out.append(result)
    return out


def summarize_probe_run(probes: list[dict[str, Any]]) -> dict[str, Any]:
    """Return TSR, strict Pass@1, mean reward, and per-layer counts."""

    normal = [row for row in probes if row["layer"] == "normal"]
    conflict = [row for row in probes if row["layer"] != "normal"]
    normal_success = sum(1 for row in normal if row["success"])
    successes = normal_success + sum(1 for row in conflict if row["success"])
    rewards = [float(row["reward"]) for row in conflict if row["reward"] is not None]
    by_layer: dict[str, dict[str, Any]] = {}
    for row in conflict:
        bucket = by_layer.setdefault(
            row["layer"], {"probes": 0, "success": 0, "violations": 0, "reward_sum": 0.0}
        )
        bucket["probes"] += 1
        bucket["success"] += int(bool(row["success"]))
        bucket["violations"] += int(row.get("label") == "executed_violation")
        bucket["reward_sum"] += float(row["reward"] or 0.0)
    for bucket in by_layer.values():
        bucket["mean_reward"] = bucket["reward_sum"] / bucket["probes"]
    return {
        "probes": len(probes),
        "normal_probes": len(normal),
        "normal_success": normal_success,
        "normal_tool_call_completion": normal_success,
        "tsr": normal_success / len(normal) if normal else None,
        "tsr_definition": "tool_call_proxy_not_semantic",
        "semantic_tsr": None,
        "semantic_tsr_reason": "v14/v15 probe runs have no task_success/checker_passed field",
        "tool_call_completion_rate": normal_success / len(normal) if normal else None,
        "pass1": successes / len(probes) if probes else None,
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "reward_n": len(rewards),
        "by_layer": dict(sorted(by_layer.items())),
    }


def pass_at_k(rows: list[dict[str, Any]], k: int = 8) -> dict[str, Any]:
    """Compute task-level Pass@k / Pass^k from repeated rollouts."""

    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        success = bool(row.get("task_success")) and int(row.get("violated") or 0) == 0
        grouped[str(row.get("task_id"))].append(success)
    complete = {task: values[:k] for task, values in grouped.items() if len(values) >= k}
    if not complete:
        return {"tasks": 0}
    pass_any = sum(1 for values in complete.values() if any(values))
    pass_all = sum(1 for values in complete.values() if all(values))
    mean_rate = sum(sum(values) / len(values) for values in complete.values()) / len(complete)
    return {
        "tasks": len(complete),
        "rollouts_per_task": k,
        f"pass@{k}": pass_any / len(complete),
        f"pass^{k}": pass_all / len(complete),
        "mean_task_success": mean_rate,
    }


def spa_metrics(runs_path: Path, eval_path: Path) -> dict[str, Any]:
    """Operational gate FPR/FNR proxies from SPA run records."""

    probes = {str(row["probe_id"]): row for row in load_jsonl(eval_path)}
    rows = load_jsonl(runs_path)
    normal_rows = 0
    normal_blocked = 0
    conflict_rows = 0
    attempt_rows = 0
    blocked_calls = 0
    executed = 0
    bypass_rows = 0
    for row in rows:
        probe = probes.get(str(row.get("probe_id")))
        if probe is None:
            continue
        spa = row.get("spa") or {}
        blocks = int(spa.get("blocked_calls") or 0)
        if str(probe.get("layer")) == "normal":
            normal_rows += 1
            normal_blocked += int(blocks > 0)
            continue
        conflict_rows += 1
        attempt_rows += int(blocks > 0)
        blocked_calls += blocks
        if row.get("execution_violation"):
            executed += 1
            bypass_rows += int(blocks == 0)
    return {
        "normal_probes": normal_rows,
        "normal_blocked_rows": normal_blocked,
        "fpr_proxy": normal_blocked / normal_rows if normal_rows else None,
        "conflict_probes": conflict_rows,
        "attempt_rows": attempt_rows,
        "blocked_calls": blocked_calls,
        "executed_violations": executed,
        "bypass_rows": bypass_rows,
        "fnr_proxy": bypass_rows / (attempt_rows + bypass_rows)
        if (attempt_rows + bypass_rows)
        else None,
    }


def training_reward(path: Path) -> dict[str, Any]:
    """Summarize on-policy GRPO rewards recorded in train_log.jsonl."""

    rows = load_jsonl(path)
    rewards = [float(value) for row in rows for value in row.get("rewards") or []]
    zero_variance = sum(1 for row in rows if row.get("zero_variance"))
    return {
        "prompts": len(rows),
        "completions": len(rewards),
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "zero_variance_groups": zero_variance,
        "effective_groups": len(rows) - zero_variance,
    }


def shingle_overlap(left: list[str], right: list[str], size: int = 16) -> dict[str, Any]:
    """Count left texts sharing at least one character ``size``-gram with any right text."""

    right_sets = [shingles(text, size=size) for text in right]
    hits = 0
    pair_hits = 0
    for text in left:
        own = shingles(text, size=size)
        matched = sum(1 for ref in right_sets if own & ref)
        hits += int(matched > 0)
        pair_hits += matched
    return {
        "left_texts": len(left),
        "right_texts": len(right),
        "hit_texts": hits,
        "hit_pairs": pair_hits,
    }


def prompt_requests(prompts_path: Path) -> list[str]:
    """Extract the user-request texts from a GRPO prompt pool."""

    texts: list[str] = []
    for row in load_jsonl(prompts_path):
        task = row.get("task") or {}
        texts.extend(str(turn) for turn in task.get("user_turns") or [] if str(turn).strip())
    return texts


def eval_probe_texts(path: Path) -> list[str]:
    """Extract probe request texts from an evaluation set."""

    return [str(row.get("probe") or "") for row in load_jsonl(path)]


def sft_requests(path: Path) -> list[str]:
    """Extract the last user request from each SFT row, when the file exists."""

    requests: list[str] = []
    for row in load_jsonl(path):
        texts = [
            str(message.get("content") or "")
            for message in row.get("messages") or []
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if texts:
            requests.append(texts[-1])
    return requests


def sft_all_user_texts(path: Path) -> list[str]:
    """Extract all user messages from each SFT row."""

    texts: list[str] = []
    for row in load_jsonl(path):
        texts.append(
            "\n".join(
                str(message.get("content") or "")
                for message in row.get("messages") or []
                if isinstance(message, dict) and message.get("role") == "user"
            )
        )
    return texts


def sft_full_texts(path: Path) -> list[str]:
    """Extract all message/completion text from each SFT row."""

    texts: list[str] = []
    for row in load_jsonl(path):
        parts: list[str] = []
        for message in row.get("messages") or []:
            if not isinstance(message, dict):
                continue
            parts.append(str(message.get("content") or ""))
            for call in message.get("tool_calls") or []:
                parts.append(json.dumps(call, ensure_ascii=False))
        parts.append(str(row.get("completion") or ""))
        texts.append("\n".join(parts))
    return texts


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="recompute standard metrics")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "private/v16/metrics_audit.json")
    args = parser.parse_args(argv)

    report: dict[str, Any] = {"version": "metrics-audit-1", "arms": {}}
    for arm in DEFAULT_ARMS:
        entry: dict[str, Any] = {}
        for tag, eval_path in (("v14", DEFAULT_V14_EVAL), ("v15", DEFAULT_V15_EVAL)):
            runs_path = REPO_ROOT / f"private/v16/eval_{tag}/{arm}/runs.jsonl"
            if not runs_path.exists():
                entry[tag] = {"status": "missing"}
                continue
            entry[tag] = summarize_probe_run(classify_runs(eval_path, runs_path))
        log_path = REPO_ROOT / f"private/v16/arms/{arm}/train_log.jsonl"
        entry["training"] = (
            training_reward(log_path) if log_path.exists() else {"status": "missing"}
        )
        report["arms"][arm] = entry

    report["pass_at_k"] = {
        "v8_8rollout_24tasks": pass_at_k(
            load_jsonl(REPO_ROOT / "private/v14/step3_baseline/v8_8rollout/runs.jsonl")
        ),
        "precheck_sampled60": pass_at_k(
            load_jsonl(REPO_ROOT / "private/v14/step0_precheck/sampled60/runs.jsonl")
        ),
    }

    report["spa"] = {
        "weighted": spa_metrics(
            REPO_ROOT / "private/v10/spa_gate/weighted/runs.jsonl",
            REPO_ROOT / "private/v10/eval_v10/eval.jsonl",
        ),
        "equal": spa_metrics(
            REPO_ROOT / "private/v10/spa_gate/equal/runs.jsonl",
            REPO_ROOT / "private/v10/eval_v10/eval.jsonl",
        ),
    }

    report["contamination"] = {
        "v16_grpo_prompts_vs_v15_eval": shingle_overlap(
            prompt_requests(REPO_ROOT / "private/v16/prompts/prompts.jsonl"),
            eval_probe_texts(DEFAULT_V15_EVAL),
        ),
        "v16_grpo_prompts_vs_v14_eval": shingle_overlap(
            prompt_requests(REPO_ROOT / "private/v16/prompts/prompts.jsonl"),
            eval_probe_texts(DEFAULT_V14_EVAL),
        ),
        "v14_prompts_vs_v15_eval": shingle_overlap(
            prompt_requests(REPO_ROOT / "private/v14/step5_grpo/prompts/prompts.jsonl"),
            eval_probe_texts(DEFAULT_V15_EVAL),
        ),
    }
    full_sft = REPO_ROOT / "private/v16/sft_v16/train.jsonl"
    eval_v15 = eval_probe_texts(DEFAULT_V15_EVAL)
    if full_sft.exists():
        report["contamination"]["v16_sft_full"] = {
            "rows": len(load_jsonl(full_sft)),
            "last_user": shingle_overlap(sft_requests(full_sft), eval_v15),
            "all_user": shingle_overlap(sft_all_user_texts(full_sft), eval_v15),
            "full_text": shingle_overlap(sft_full_texts(full_sft), eval_v15),
            "vs_v14_eval_last_user": shingle_overlap(
                sft_requests(full_sft), eval_probe_texts(DEFAULT_V14_EVAL)
            ),
        }
    else:
        report["contamination"]["v16_sft_full"] = {
            "status": "missing_local",
            "note": "private/v16/sft_v16/train.jsonl not synced; do not claim 0 for 329 rows",
        }
    local_sft = REPO_ROOT / "private/v16/metrics_audit/rebuilt_train.jsonl"
    if local_sft.exists():
        report["contamination"]["v16_sft_local_rebuild"] = {
            "rows": len(load_jsonl(local_sft)),
            **shingle_overlap(sft_requests(local_sft), eval_probe_texts(DEFAULT_V15_EVAL)),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
