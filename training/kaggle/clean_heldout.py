"""Evaluate a trained adapter on the 45 clean held-out tasks.

The task specs are still resolved from train-v5: train-v6 removed those tasks
from training, not from the frozen evaluator. The runner reuses the same
full-loop harness and per-rollout seed policy as the difficulty screen, so the
only changed variable is the adapter.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.heldout_pool import HeldoutPool, build_heldout_pool


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_task_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            ids.append(text)
            continue
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        metadata = payload.get("metadata")
        task_id = payload.get("task_id")
        if not task_id and isinstance(metadata, Mapping):
            task_id = metadata.get("task_id")
        if not task_id:
            raise ValueError(f"{path}:{line_number} has no task_id")
        ids.append(str(task_id))
    if len(ids) != len(set(ids)):
        duplicates = sorted(item for item, count in _counts(ids).items() if count > 1)
        raise ValueError(f"duplicate held-out task ids: {duplicates}")
    if not ids:
        raise ValueError(f"no task ids in {path}")
    return ids


def _counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def select_specs(pool: HeldoutPool, task_ids: Sequence[str]) -> tuple[Any, ...]:
    by_id = {task.spec.task_id: task.spec for task in pool.tasks}
    missing = sorted(set(task_ids) - set(by_id))
    if missing:
        raise ValueError(f"held-out task ids missing from pool: {missing}")
    return tuple(by_id[task_id] for task_id in task_ids)


def build_namespace(
    *,
    model_path: Path,
    adapter_path: Path,
    out_dir: Path,
    trace_dir: Path,
    rollouts: int,
    limit: int,
    resume: bool,
    max_model_len: int,
    max_tokens: int,
    load_timeout: float,
    rollout_timeout: float,
    temperature: float,
    top_p: float,
    seed: int,
    device_map: str | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        model_path=str(model_path),
        adapter_path=str(adapter_path),
        out_dir=out_dir,
        trace_dir=trace_dir,
        rollouts=rollouts,
        limit=limit,
        resume=resume,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        load_timeout=load_timeout,
        rollout_timeout=rollout_timeout,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        device_map=device_map,
    )


async def run_clean_heldout(
    *,
    train_file: Path,
    task_ids_file: Path,
    model_path: Path,
    adapter_path: Path,
    out_dir: Path,
    trace_dir: Path,
    rollouts: int = 10,
    task_limit: int = 0,
    limit: int = 0,
    resume: bool = False,
    max_model_len: int = 8192,
    max_tokens: int = 512,
    load_timeout: float = 240.0,
    rollout_timeout: float = 300.0,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 20261019,
    device_map: str | None = None,
) -> dict[str, Any]:
    if not adapter_path.is_dir() or not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter missing adapter_config.json: {adapter_path}")
    pool = build_heldout_pool(train_file)
    task_ids = read_task_ids(task_ids_file)
    specs = select_specs(pool, task_ids)
    if task_limit:
        specs = specs[:task_limit]
    args = build_namespace(
        model_path=model_path,
        adapter_path=adapter_path,
        out_dir=out_dir,
        trace_dir=trace_dir,
        rollouts=rollouts,
        limit=limit,
        resume=resume,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        load_timeout=load_timeout,
        rollout_timeout=rollout_timeout,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        device_map=device_map,
    )
    from training.kaggle.stats_pilot import run_condition

    summary = await run_condition("sft", specs=specs, args=args)
    result = {
        "version": "clean-heldout-eval-v1",
        "train_file": str(train_file),
        "train_sha256": sha256(train_file),
        "task_ids_file": str(task_ids_file),
        "task_ids_sha256": sha256(task_ids_file),
        "tasks_requested": len(task_ids),
        "tasks_evaluated": len(specs),
        "task_ids": [spec.task_id for spec in specs],
        "adapter_path": str(adapter_path),
        "adapter_config_sha256": sha256(adapter_path / "adapter_config.json"),
        "adapter_model_sha256": sha256(adapter_path / "adapter_model.safetensors"),
        "model_path": str(model_path),
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "base_seed": seed,
            "seed_policy": "base_seed_plus_rollout_index",
            "rollouts": rollouts,
        },
        "condition": summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="clean held-out adapter evaluation")
    parser.add_argument("--train-file", type=Path, default=Path("private/sft_data_v5/train.jsonl"))
    parser.add_argument("--task-ids-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--task-limit", type=int, default=0, help="only first N tasks; 0 = all")
    parser.add_argument("--limit", type=int, default=0, help="only first N rollouts; 0 = all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--load-timeout", type=float, default=240.0)
    parser.add_argument("--rollout-timeout", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--device-map", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = asyncio.run(
        run_clean_heldout(
            train_file=args.train_file,
            task_ids_file=args.task_ids_file,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            out_dir=args.out_dir,
            trace_dir=args.trace_dir,
            rollouts=args.rollouts,
            task_limit=args.task_limit,
            limit=args.limit,
            resume=args.resume,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            load_timeout=args.load_timeout,
            rollout_timeout=args.rollout_timeout,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            device_map=args.device_map,
        )
    )
    return 0 if result["condition"].get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
