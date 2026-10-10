"""Run the v10 layered evaluation set with M2 independent injection."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent.core.action_gate import ActionGate, GatedExecutor  # noqa: E402
from agent.core.compaction import CompactionConfig, Compactor  # noqa: E402
from agent.core.constraints import ConstraintStore  # noqa: E402
from agent.core.injection import (  # noqa: E402
    ConstraintInjector,
    InjectionConfig,
    estimate_injection_tokens,
)
from agent.core.llm import BaseProvider, LLMResponse  # noqa: E402
from agent.core.loop import AgentLoop  # noqa: E402
from scripts.unified_eval import RecordingExecutor, build_default_registry  # noqa: E402
from training.kaggle.contracts import classify_exception  # noqa: E402
from training.kaggle.transformers_provider import TransformersOfflineProvider  # noqa: E402
from training.v10_eval_harness import (  # noqa: E402
    classify_eval_probe,
    load_probes,
    materialize_probe,
    probe_to_spec,
    select_probes,
    summarize_layers,
)

M2_INJECTION = InjectionConfig(max_tokens=200, max_constraints=10, context_messages=6)
CONTEXT_WINDOW = 8192


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_behavior(_item: Any) -> bool:
    return True


class InjectingProvider(BaseProvider):
    """Apply the M2 injected layer immediately before every model call."""

    def __init__(
        self,
        delegate: BaseProvider,
        injector: ConstraintInjector,
        *,
        merge_first_system: bool = False,
    ) -> None:
        self.delegate = delegate
        self.injector = injector
        self.merge_first_system = merge_first_system
        self.events: list[dict[str, Any]] = []

    @property
    def sampling_config(self) -> Mapping[str, Any]:
        return getattr(self.delegate, "sampling_config", {})

    @property
    def model(self) -> str | None:
        return getattr(self.delegate, "model", None)

    def _prepare(self, messages: Sequence[Mapping[str, Any]], step: str) -> list[dict[str, Any]]:
        selection = self.injector.select(messages)
        prepared = self.injector.inject(messages, merge_first_system=self.merge_first_system)
        self.events.append(
            {
                "step": step,
                "constraint_ids": [item.id for item in selection.items],
                "count": len(selection.items),
                "tokens": selection.tokens,
                "dropped": selection.dropped,
                "merge_first_system": self.merge_first_system,
                "context_tokens": sum(
                    estimate_injection_tokens(str(message.get("content") or ""))
                    for message in prepared
                ),
            }
        )
        return prepared

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        return await self.delegate.chat(self._prepare(messages, "chat"), tools=tools)

    async def chat_stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        on_text: Any | None = None,
    ) -> LLMResponse:
        return await self.delegate.chat_stream(
            self._prepare(messages, "chat_stream"), tools=tools, on_text=on_text
        )

    async def aclose(self) -> None:
        await self.delegate.aclose()


class DryRunProvider(BaseProvider):
    """Deterministic CPU-only provider used by the runner smoke test."""

    def __init__(self) -> None:
        self.model = "dry-run-mock"

    @property
    def sampling_config(self) -> dict[str, Any]:
        return {"provider": "dry-run", "temperature": 0.0, "is_default": False}

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        text = "\n".join(str(item.get("content") or "") for item in messages)
        if "tests/" in text:
            reply = "无法修改 tests/ 文件；这违反当前约束。"
        elif "rm -rf" in text or "移除" in text or "删掉" in text:
            reply = "需要您确认后我才能执行删除，请先确认。"
        elif "/proc/" in text or "../outside" in text or "/opt/" in text:
            reply = "不允许访问工作区之外的路径。"
        else:
            reply = "已完成只读检查。"
        return LLMResponse(
            content=reply,
            metadata={"provider": "dry-run", "prompt_tokens": len(text), "completion_tokens": 8},
        )


async def _summarize(provider: BaseProvider, older: Sequence[Mapping[str, Any]]) -> str:
    response = await provider.chat(older)
    return response.content or ""


async def run_m2_probe(
    probe: Mapping[str, Any],
    *,
    provider: BaseProvider,
    max_turns: int = 9,
    merge_first_system: bool = False,
    spa_gate: bool = False,
    max_turns_conflict: int | None = None,
) -> dict[str, Any]:
    """Run one probe with independent storage and per-call injection."""

    started = time.perf_counter()
    root = Path(tempfile.mkdtemp(prefix=f"v10-m2-{probe['probe_id']}-"))
    spec = probe_to_spec(probe)
    store = ConstraintStore()
    for item in spec.constraints:
        store.add(item.text, constraint_id=item.id, source="user")
    compactor = Compactor(
        CompactionConfig(context_window=CONTEXT_WINDOW),
        summarize=lambda messages: _summarize(provider, messages),
        constraints=None,
    )
    injector = ConstraintInjector(store, config=M2_INJECTION, is_behavior=_is_behavior)
    injected_provider = InjectingProvider(provider, injector, merge_first_system=merge_first_system)
    gate = GatedExecutor(build_default_registry(root), ActionGate(root)) if spa_gate else None
    executor = RecordingExecutor(gate if gate is not None else build_default_registry(root))
    turns = max_turns
    if max_turns_conflict is not None and str(probe.get("layer")) != "normal":
        turns = max_turns_conflict
    loop = AgentLoop(
        injected_provider,
        executor,
        max_turns=turns,
        compactor=compactor,
    )
    try:
        materialize_probe(probe, root)
        result = await loop.run(spec.user_turns[0])
        classification = classify_eval_probe(
            probe,
            result.content,
            executor.calls,
            scan_content_calls=not spa_gate,
        )
        return {
            "probe_id": probe["probe_id"],
            "task_id": probe["task_id"],
            "layer": probe["layer"],
            "severity": probe["severity"],
            "condition": "M2+SPA" if spa_gate else "M2",
            "label": classification["label"],
            "classification": classification["classification"],
            "execution_violation": classification["execution_violation"],
            "explicit_compliance": classification["explicit_compliance"],
            "ambiguous_reference": classification["ambiguous_reference"],
            "deferred_tool": classification["deferred_tool"],
            "tool_calls": [
                {
                    "name": call.name,
                    "arguments": call.arguments,
                    "ok": call.result.ok,
                    "error_kind": call.result.error_kind,
                }
                for call in executor.calls
            ],
            "reply": result.content,
            "stopped_reason": result.stopped_reason,
            "replenished": compactor.stats.replenished,
            "summary_channel": {
                "constraints_attached": False,
                "compaction_counts": compactor.stats.counts,
            },
            "injection": {
                "events": injected_provider.events,
                "calls": len(injected_provider.events),
                "max_tokens": M2_INJECTION.max_tokens,
                "max_constraints": M2_INJECTION.max_constraints,
                "max_observed_tokens": max(
                    (event["tokens"] for event in injected_provider.events), default=0
                ),
            },
            "spa": {
                "enabled": spa_gate,
                "blocks": list(gate.blocks) if gate is not None else [],
                "blocked_calls": len(gate.blocks) if gate is not None else 0,
            },
            "duration_s": round(time.perf_counter() - started, 1),
        }
    except Exception as exc:  # noqa: BLE001 - one failed probe must not abort a split
        return {
            "probe_id": probe["probe_id"],
            "task_id": probe["task_id"],
            "layer": probe["layer"],
            "error_kind": classify_exception(exc),
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "duration_s": round(time.perf_counter() - started, 1),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v10 M2 layered evaluation")
    parser.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--expected-adapter-sha256", default=None)
    parser.add_argument("--eval-file", type=Path, default=Path("private/v10/eval_v10/eval.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("private/v10/m2_eval"))
    parser.add_argument("--layers", default="path,delete,edit,normal")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument(
        "--separate-system",
        action="store_true",
        help="inject the layer as a second system message (step-4 legacy mode)",
    )
    parser.add_argument(
        "--spa-gate", action="store_true", help="block violating calls before dispatch"
    )
    parser.add_argument(
        "--max-turns-conflict",
        type=int,
        default=None,
        help="turn cap for conflict probes (normal probes keep --max-turns)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> int:
    layers = {item.strip() for item in args.layers.split(",") if item.strip()}
    probes = select_probes(load_probes(args.eval_file), layers)
    if args.limit:
        probes = probes[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    adapter_sha = None
    if args.adapter_path:
        adapter_file = Path(args.adapter_path) / "adapter_model.safetensors"
        if adapter_file.exists():
            adapter_sha = sha256_file(adapter_file)
    manifest = {
        "version": "v10-m2-eval-1",
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "adapter_sha256": adapter_sha,
        "eval_file": str(args.eval_file),
        "eval_sha256": sha256_file(args.eval_file),
        "probes": len(probes),
        "layers": sorted(layers),
        "temperature": args.temperature,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "spa_gate": args.spa_gate,
        "max_turns_conflict": args.max_turns_conflict,
        "injection": {
            "max_tokens": M2_INJECTION.max_tokens,
            "max_constraints": M2_INJECTION.max_constraints,
            "context_messages": M2_INJECTION.context_messages,
            "merge_first_system": not args.separate_system,
        },
    }
    if args.expected_adapter_sha256 and adapter_sha != args.expected_adapter_sha256:
        raise SystemExit(f"adapter SHA mismatch: {adapter_sha} != {args.expected_adapter_sha256}")
    provider: BaseProvider | None = None
    try:
        provider = (
            DryRunProvider()
            if args.dry_run
            else TransformersOfflineProvider(
                args.model_path,
                max_model_len=args.max_model_len,
                temperature=args.temperature,
                seed=args.seed,
                max_tokens=args.max_tokens,
                adapter_path=args.adapter_path,
                quantization="nf4",
            )
        )
        if args.preflight_only:
            (args.out_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps({"status": "ok", "manifest": manifest}, ensure_ascii=False))
            return 0
        rows: list[dict[str, Any]] = []
        run_path = args.out_dir / "runs.jsonl"
        with run_path.open("w", encoding="utf-8", newline="\n") as handle:
            for index, probe in enumerate(probes, start=1):
                row = await run_m2_probe(
                    probe,
                    provider=provider,
                    merge_first_system=not args.separate_system,
                    spa_gate=args.spa_gate,
                    max_turns_conflict=args.max_turns_conflict,
                )
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{index}/{len(probes)}] {row['probe_id']} "
                    f"label={row.get('label', row.get('error_kind'))}",
                    flush=True,
                )
        summary = summarize_layers(rows)
        mechanism = {
            "replenished_total": sum(int(row.get("replenished", 0)) for row in rows),
            "sessions_with_injection": sum(
                1 for row in rows if (row.get("injection") or {}).get("calls", 0) > 0
            ),
            "max_injection_tokens": max(
                (int((row.get("injection") or {}).get("max_observed_tokens", 0)) for row in rows),
                default=0,
            ),
            "sessions_over_budget": sum(
                1
                for row in rows
                if int((row.get("injection") or {}).get("max_observed_tokens", 0))
                > M2_INJECTION.max_tokens
            ),
            "errors": sum(1 for row in rows if row.get("error_kind")),
        }
        report = {
            "version": "v10-m2-eval-1",
            "manifest": manifest,
            "summary": summary,
            "mechanism": mechanism,
            "rows": rows,
        }
        (args.out_dir / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (args.out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"summary": summary, "mechanism": mechanism}, ensure_ascii=False))
        return 0
    finally:
        if provider is not None:
            await provider.aclose()


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
