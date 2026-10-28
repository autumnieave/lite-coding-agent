"""Run the tagged-only replenishment comparison with a forced ID-only summary."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent.tools import build_default_registry
from training.kaggle.session_constraint_probe import (
    LocalModel,
    build_scenario,
    build_workspace,
    run_condition,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="tagged replenishment pre/post comparison")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--summary-max-tokens", type=int, default=512)
    parser.add_argument("--probe-max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261026)
    return parser


async def main_async(args: argparse.Namespace) -> int:
    payload = load_json(args.tasks)
    tasks: list[Mapping[str, Any]] = list(payload.get("tasks") or [])
    if len(tasks) != 32:
        raise ValueError(f"expected exactly 32 tasks for 64 tagged probes, got {len(tasks)}")
    scenarios = [build_scenario(row, index) for index, row in enumerate(tasks)]
    workspace = args.out_dir / "workspace"
    build_workspace(workspace, scenarios)
    tool_specs = build_default_registry(workspace).specs()
    started = time.perf_counter()

    model = await asyncio.to_thread(
        LocalModel,
        args.model_path,
        adapter_path=args.adapter_path,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    results = []
    for condition, legacy_verify in (("v6_pre", True), ("v6_post", False)):
        result = await run_condition(
            condition,
            model=model,
            tasks=tasks,
            tool_specs=tool_specs,
            out_path=args.out_dir / f"{condition}.jsonl",
            summary_max_tokens=args.summary_max_tokens,
            probe_max_tokens=args.probe_max_tokens,
            modes=("tagged",),
            legacy_verify=legacy_verify,
            summary_mode="id_only",
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = {
        "version": "replenish-tagged-compare-v1",
        "model_path": str(args.model_path),
        "adapter_path": str(args.adapter_path),
        "adapter_sha256": sha256(args.adapter_path / "adapter_model.safetensors"),
        "tasks_path": str(args.tasks),
        "tasks_sha256": sha256(args.tasks),
        "tasks": len(tasks),
        "modes": ["tagged"],
        "summary_mode": "id_only",
        "note": (
            "Forced ID-only summary; this tests the replenishment path, "
            "not natural summary quality."
        ),
        "conditions": results,
        "duration_s": round(time.perf_counter() - started, 1),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
