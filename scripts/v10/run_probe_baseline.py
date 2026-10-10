"""Run a one-shot base-model screen for the v10 path/edit probes."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.unified_eval import Fixture, run_once  # noqa: E402
from training.kaggle.contracts import classify_exception  # noqa: E402
from training.kaggle.transformers_provider import TransformersOfflineProvider  # noqa: E402
from training.v10_eval_harness import (  # noqa: E402
    classify_eval_probe,
    load_probes,
    materialize_probe,
    probe_to_spec,
    select_probes,
    summarize_layers,
    trigger_gate,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v10 path/edit probe baseline")
    parser.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--eval-file", type=Path, default=Path("private/v10/eval_v10/eval.jsonl"))
    parser.add_argument(
        "--out", type=Path, default=Path("private/v10/eval_v10/probe_baseline_results.json")
    )
    parser.add_argument("--layers", default="path,edit")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _fixture_builder(probe: dict[str, Any]) -> Any:
    def build(spec: Any, root: Path) -> Fixture:
        materialize_probe(probe, root)
        return Fixture(task_id=spec.task_id, kind="v10_eval_probe", expected_json={"probe": probe})

    return build


async def _run_probe(
    probe: dict[str, Any], *, provider: TransformersOfflineProvider, args: argparse.Namespace
) -> dict[str, Any]:
    started = time.perf_counter()
    spec = probe_to_spec(probe)

    def judge_probe(
        _spec: Any, _fixture: Any, _root: Path, replies: list[str], calls: list[Any]
    ) -> dict[str, bool]:
        observed = classify_eval_probe(probe, "\n".join(replies), calls)
        return {
            "execution_compliant": not observed["execution_violation"],
            "explicit_compliance": bool(observed["explicit_compliance"]),
            "no_ambiguous_reference": not observed["ambiguous_reference"],
        }

    try:
        result = await asyncio.wait_for(
            run_once(
                spec,
                provider=provider,
                rollout_index=0,
                trace_dir=args.trace_dir,
                sampling_config=dict(provider.sampling_config),
                system_variant="s0",
                fixture_builder=_fixture_builder(probe),
                judge_fn=judge_probe,
            ),
            timeout=300,
        )
    except TimeoutError:
        return {
            "probe_id": probe["probe_id"],
            "task_id": probe["task_id"],
            "layer": probe["layer"],
            "error_kind": "inference_timeout",
            "error": "probe exceeded 300s",
            "duration_s": round(time.perf_counter() - started, 1),
        }
    classification = classify_eval_probe(
        probe,
        str(result.get("reply") or ""),
        result.get("tool_calls") or [],
    )
    if result.get("error_kind"):
        return {
            "probe_id": probe["probe_id"],
            "task_id": probe["task_id"],
            "layer": probe["layer"],
            "error_kind": result["error_kind"],
            "error": result.get("error"),
            "duration_s": round(time.perf_counter() - started, 1),
        }
    return {
        "probe_id": probe["probe_id"],
        "task_id": probe["task_id"],
        "layer": probe["layer"],
        "severity": probe["severity"],
        "condition": "baseline",
        "label": classification["label"],
        "classification": classification["classification"],
        "execution_violation": classification["execution_violation"],
        "explicit_compliance": classification["explicit_compliance"],
        "ambiguous_reference": classification["ambiguous_reference"],
        "deferred_tool": classification["deferred_tool"],
        "tool_calls": classification["tool_calls"],
        "reply": str(result.get("reply") or ""),
        "duration_s": round(time.perf_counter() - started, 1),
    }


async def main_async(args: argparse.Namespace) -> int:
    layers = {item.strip() for item in args.layers.split(",") if item.strip()}
    probes = select_probes(load_probes(args.eval_file), layers)
    if args.limit:
        probes = probes[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "v10-probe-baseline-1",
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "eval_file": str(args.eval_file),
        "eval_sha256": sha256(args.eval_file),
        "probes": len(probes),
        "temperature": args.temperature,
        "seed": args.seed,
        "environment": {
            name: package_version(name)
            for name in ("torch", "transformers", "peft", "bitsandbytes", "pyarrow")
        },
    }
    provider: TransformersOfflineProvider | None = None
    try:
        provider = TransformersOfflineProvider(
            args.model_path,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            seed=args.seed,
            max_tokens=args.max_tokens,
            adapter_path=args.adapter_path,
            quantization="nf4",
        )
        if args.preflight_only:
            print(json.dumps({"status": "ok", "manifest": manifest}, ensure_ascii=False))
            return 0
        rows: list[dict[str, Any]] = []
        for index, probe in enumerate(probes, start=1):
            row = await _run_probe(probe, provider=provider, args=args)
            rows.append(row)
            print(
                f"[{index}/{len(probes)}] {row['probe_id']} "
                f"label={row.get('label', row.get('error_kind'))}",
                flush=True,
            )
        summary = summarize_layers(rows)
        report = {
            "version": "v10-probe-baseline-1",
            "manifest": manifest,
            "summary": summary,
            "gate": trigger_gate(summary, layers=("path", "edit")),
            "rows": rows,
        }
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"summary": summary, "gate": report["gate"]}, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - surface load failures in the result file
        error = {
            "status": "error",
            "error_kind": classify_exception(exc),
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "manifest": manifest,
        }
        args.out.write_text(
            json.dumps(error, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(error, ensure_ascii=False))
        return 2
    finally:
        if provider is not None:
            await provider.aclose()


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
