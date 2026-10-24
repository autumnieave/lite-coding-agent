"""GRPO prompt-pool variance preflight on frozen train-v5 trajectories.

Protocol: each row in train-v5 contributes its historical messages (system, user,
tool calls and tool results) and we resample only the final assistant completion
from the SFT adapter. This isolates output-level group variance without rerunning
the full agent loop, which is too expensive for a 100 x 8 preflight. The report
explicitly marks this scope; tool-selection variance is not covered here.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from training.reward import behavior_reward  # noqa: E402
from training.train_sft import default_tool_schemas  # noqa: E402

CATEGORY_ORDER = ("BASE", "HARD", "CTX", "TOOL", "CONFLICT")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _stable_order(rows: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> str:
        return hashlib.sha256(f"{seed}:{row.get('id')}".encode()).hexdigest()

    return sorted(rows, key=key)


def select_prompt_pool(
    rows: Sequence[Mapping[str, Any]],
    *,
    per_category: int = 20,
    seed: int = 20261019,
) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        category = str((row.get("metadata") or {}).get("category") or "UNKNOWN")
        prompt_hash = hashlib.sha256(
            json.dumps(row.get("messages") or (), ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        if prompt_hash in seen:
            continue
        seen.add(prompt_hash)
        by_category[category].append(dict(row))
    selected: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        selected.extend(_stable_order(by_category.get(category, ()), seed)[:per_category])
    for category in sorted(set(by_category) - set(CATEGORY_ORDER)):
        selected.extend(_stable_order(by_category[category], seed)[:per_category])
    return selected


def _normalized_json(text: str) -> Any | None:
    try:
        return json.loads(text.strip())
    except (TypeError, ValueError):
        return None


def completion_matches(
    completion: str,
    *,
    expected: str,
    expects_json: bool,
) -> bool:
    if expects_json:
        actual_payload = _normalized_json(completion)
        expected_payload = _normalized_json(expected)
        return actual_payload is not None and actual_payload == expected_payload
    return completion.strip() == expected.strip()


def score_completion(
    row: Mapping[str, Any],
    completion: str,
    *,
    tool_calls: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    context = row.get("reward_context") or {}
    expects_json = bool(context.get("expects_json", True))
    expected = str(row.get("completion") or "")
    success = completion_matches(completion, expected=expected, expects_json=expects_json)
    reward = behavior_reward(
        completion,
        tool_calls=context.get("tool_calls") or (),
        tool_results=context.get("tool_results") or (),
        file_observations=context.get("file_observations"),
        expects_json=expects_json,
        expects_file=bool(context.get("expects_file", False)),
        within_steps=bool(context.get("within_steps", True)),
        answer_then_call=False,
        tools_allowed=bool(context.get("tools_allowed", True)),
    )
    # The GRPO objective must not receive credit for a wrong answer that merely
    # has valid JSON. Keep the two signals separate in the report.
    pool_reward = float(reward["score"]) if success else 0.0
    return {
        "task_success": success,
        "behavior_reward": float(reward["score"]),
        "pool_reward": pool_reward,
        "reward_failures": list(reward["failures"]),
        "tool_calls_after_prompt": len(tool_calls),
    }


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


async def run_pool(
    *,
    train_file: Path,
    out_dir: Path,
    provider: Any,
    model_label: str,
    rollouts: int,
    per_category: int,
    temperature: float,
    seed: int,
    max_new_tokens: int,
    limit: int = 0,
) -> dict[str, Any]:
    rows = select_prompt_pool(_load_jsonl(train_file), per_category=per_category, seed=seed)
    if limit:
        rows = rows[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    runs_path.write_text("", encoding="utf-8")
    tools = default_tool_schemas()
    prompt_stats: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        task_id = str((row.get("metadata") or {}).get("task_id") or row.get("id"))
        category = str((row.get("metadata") or {}).get("category") or "UNKNOWN")
        samples: list[dict[str, Any]] = []
        for rollout in range(rollouts):
            if hasattr(provider, "seed"):
                provider.seed = seed + rollout
            response = await provider.chat(row["messages"], tools=tools)
            completion = str(response.content or "")
            scored = score_completion(row, completion, tool_calls=response.tool_calls)
            metadata = dict(response.metadata or {})
            sample = {
                "task_id": task_id,
                "source_id": row.get("id"),
                "category": category,
                "rollout_index": rollout,
                "completion": completion,
                "task_success": scored["task_success"],
                "behavior_reward": scored["behavior_reward"],
                "pool_reward": scored["pool_reward"],
                "reward_failures": scored["reward_failures"],
                "tool_calls_after_prompt": scored["tool_calls_after_prompt"],
                "provider": metadata.get("provider"),
                "model_version": metadata.get("model_version") or model_label,
                "system_fingerprint": metadata.get("system_fingerprint"),
                "request_id": metadata.get("request_id"),
                "prompt_tokens": metadata.get("prompt_tokens"),
                "completion_tokens": metadata.get("completion_tokens"),
                "raw_reply_head": metadata.get("raw_reply_head"),
                "sampling_config": {
                    "temperature": temperature,
                    "top_p": None,
                    "seed": seed + rollout,
                },
            }
            samples.append(sample)
            with runs_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        successes = sum(bool(sample["task_success"]) for sample in samples)
        pass_rate = successes / rollouts
        reward_values = [float(sample["pool_reward"]) for sample in samples]
        prompt_stats.append(
            {
                "task_id": task_id,
                "source_id": row.get("id"),
                "category": category,
                "rollouts": rollouts,
                "successes": successes,
                "pass_rate": round(pass_rate, 4),
                "reward_mean": round(sum(reward_values) / rollouts, 4),
                "reward_variance": round(_variance(reward_values), 6),
                "in_20_80": 0.2 <= pass_rate <= 0.8,
            }
        )
        print(
            f"[{index}/{len(rows)}] {category} {task_id} "
            f"pass={successes}/{rollouts} reward_var={prompt_stats[-1]['reward_variance']}",
            flush=True,
        )
    eligible = [row for row in prompt_stats if row["in_20_80"]]
    reward_nonzero = [row for row in prompt_stats if row["reward_variance"] > 0]
    summary = {
        "status": "completed",
        "protocol": "final-turn resampling on frozen successful train-v5 trajectories",
        "scope_note": (
            "Samples only the final assistant completion; does not rerun tool selection "
            "or file side effects. Use this as a reward/variance gate, not a full-agent score."
        ),
        "model_label": model_label,
        "train_file": str(train_file),
        "train_sha256": hashlib.sha256(train_file.read_bytes()).hexdigest(),
        "rollouts_per_prompt": rollouts,
        "temperature": temperature,
        "top_p": None,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "selected_prompts": len(prompt_stats),
        "category_counts": dict(Counter(row["category"] for row in prompt_stats)),
        "eligible_20_80": len(eligible),
        "reward_variance_positive": len(reward_nonzero),
        "pass_rate_distribution": {
            "min": min((row["pass_rate"] for row in prompt_stats), default=0.0),
            "p50": round(_percentile([row["pass_rate"] for row in prompt_stats], 0.5), 4),
            "max": max((row["pass_rate"] for row in prompt_stats), default=0.0),
        },
        "grpo_branch": "candidate" if len(eligible) >= 50 else "insufficient_pool",
        "grpo_note": (
            "prompt pool >=50 with 20-80% pass rate"
            if len(eligible) >= 50
            else "GRPO prompt 池不足，组内方差信号稀疏；GRPO 降级为可选"
        ),
        "prompts": prompt_stats,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    _write_csv(out_dir / "prompts.csv", prompt_stats)
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "task_id",
        "source_id",
        "category",
        "rollouts",
        "successes",
        "pass_rate",
        "reward_mean",
        "reward_variance",
        "in_20_80",
    )
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(json.dumps(row.get(field), ensure_ascii=False) for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRPO prompt-pool variance preflight")
    parser.add_argument("--train-file", type=Path, default=Path("private/sft_data_v5/train.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("private/grpo_prompt_pool"))
    parser.add_argument("--provider", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--staging", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.provider == "transformers":
        from training.kaggle.transformers_provider import TransformersOfflineProvider

        provider = TransformersOfflineProvider(
            args.model,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=None,
            seed=args.seed,
            max_tokens=args.max_new_tokens,
            adapter_path=args.adapter_path or None,
            quantization="nf4",
        )
    else:
        from training.kaggle.offline_provider import VLLMOfflineProvider

        provider = VLLMOfflineProvider(
            args.model,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=None,
            seed=args.seed,
            max_tokens=args.max_new_tokens,
            adapter_path=args.adapter_path or None,
            quantization="awq",
        )

    async def execute() -> dict[str, Any]:
        if args.staging:
            stage = await run_pool(
                train_file=args.train_file,
                out_dir=args.out_dir / "staging",
                provider=provider,
                model_label=args.model,
                rollouts=2,
                per_category=args.per_category,
                temperature=args.temperature,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                limit=2,
            )
            stage_successes = sum(int(row["successes"]) for row in stage["prompts"])
            if stage_successes == 0:
                raise RuntimeError("staging failed: 0 successful rollouts in 2 prompts x 2")
            print(
                json.dumps(
                    {
                        "staging": "passed",
                        "stage_successes": stage_successes,
                        "selected": stage["selected_prompts"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return await run_pool(
            train_file=args.train_file,
            out_dir=args.out_dir,
            provider=provider,
            model_label=args.model,
            rollouts=args.rollouts,
            per_category=args.per_category,
            temperature=args.temperature,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
        )

    summary = asyncio.run(execute())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(main())
