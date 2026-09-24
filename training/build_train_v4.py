"""Build and freeze train-v4 for method-level SFT comparisons.

train-v4 combines three explicitly labelled sources:

* existing DeepSeek teacher trajectories on train-v2 tasks;
* xLAM rows that map strictly to the local ``grep`` schema and are re-executed
  against a local fixture;
* deterministic oracle trajectories on newly seeded copies of the five
  controlled task families.

The oracle source is execution-verified but is not presented as teacher-model
behaviour. It exists to raise the data volume enough for method comparisons
without pretending that synthetic trajectories are model rollouts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.core.constraints import ConstraintStore
from agent.core.loop import AgentLoop
from agent.tools import build_default_registry
from training.build_sft_dataset import TraceEpisode, build_example
from training.difficulty_tuning import TUNING_TASK_SPECS
from training.final_test import _line_edit_turns, build_final_specs
from training.public_data import (
    build_public_grep_example,
    load_xlam_rows,
    strict_grep_rows,
)
from training.unified_dataset import UnifiedTaskSpec

try:
    from scripts.unified_eval import ToolCallRecord, build_fixture, judge, oracle_reply
except ImportError as exc:  # pragma: no cover - import-time environment failure
    raise RuntimeError("train-v4 builder needs the repository root on PYTHONPATH") from exc

PER_SOURCE_VARIANTS = 20
VAL_VARIANTS = frozenset({12, 13})
SEED_BASE = 130000
VALID_CATEGORIES = ("BASE", "HARD", "CTX", "TOOL", "CONFLICT")


def _system_prompt(spec: UnifiedTaskSpec) -> str:
    store = ConstraintStore()
    for item in spec.constraints:
        store.add(item.text, constraint_id=item.id, source="user")
    return AgentLoop(None, None, constraints=store).build_system_prompt()


def _write_trace(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )


def _category_for_task_id(task_id: str) -> str:
    if "CTX" in task_id:
        return "CTX"
    if "TOOL" in task_id or task_id.startswith("PUB-XLAM"):
        return "TOOL"
    if "CONFLICT" in task_id:
        return "CONFLICT"
    if task_id.endswith("-HARD") or "CHECK" in task_id:
        return "HARD"
    return "BASE"


def _difficulty_for_task_id(task_id: str) -> str:
    if "CHECK" in task_id:
        return "medium"
    if task_id.endswith("-HARD") or task_id.endswith("-STRONG"):
        return "strong"
    if task_id.endswith("-BASE") or task_id.endswith("-NONE"):
        return "weak"
    return "medium"


def _context_for_task_id(task_id: str) -> str:
    if "CTX-HARD" in task_id:
        return "long"
    if "TOOL-HARD" in task_id or "CHECK" in task_id:
        return "medium"
    return "short"


def _category_for_spec(spec: UnifiedTaskSpec) -> str:
    if spec.tuning_dimension == "context_length":
        return "CTX"
    if spec.tuning_dimension == "tool_parameters":
        return "TOOL"
    if spec.tuning_dimension == "conflict":
        return "CONFLICT"
    if spec.tuning_dimension == "constraint_count" and spec.difficulty == "weak":
        return "BASE"
    return "HARD"


def build_clone_specs(per_source: int = PER_SOURCE_VARIANTS) -> tuple[UnifiedTaskSpec, ...]:
    """Create new, non-overlapping task instances for the five controlled families."""

    if per_source < 2:
        raise ValueError("per_source must be at least 2 so train and val are both populated")
    specs: list[UnifiedTaskSpec] = []
    for source_index, source in enumerate(TUNING_TASK_SPECS):
        suffix = source.task_id.removeprefix("U-TUNE-")
        for variant in range(per_source):
            split = "val" if variant in VAL_VARIANTS else "train"
            seed = SEED_BASE + source_index * 100 + variant
            spec = replace(
                source,
                task_id=f"U-TR4-{variant:02d}-{suffix}",
                split=split,
                fixture_seed=seed,
            )
            with tempfile.TemporaryDirectory(prefix=f"train-v4-{spec.task_id}-") as temporary:
                fixture = build_fixture(spec, Path(temporary))
                turns = _line_edit_turns(fixture) if spec.family == "03" else spec.user_turns
            spec = replace(
                spec,
                user_turns=(f"训练变体 {variant:02d}：{turns[0]}", *turns[1:]),
            )
            specs.append(spec)
    ids = [spec.task_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("train-v4 clone IDs must be unique")
    if {spec.split for spec in specs} != {"train", "val"}:
        raise ValueError("train-v4 clones must cover both train and val")
    return tuple(specs)


def _planned_calls(spec: UnifiedTaskSpec, fixture: Any) -> list[tuple[str, dict[str, Any]]]:
    suffix = spec.family
    if suffix == "json_constraints":
        return [("read_file", {"path": fixture.main_path})]
    if suffix == "context_recall":
        return [("read_file", {"path": path}) for path in fixture.context_paths]
    if suffix == "path_params":
        return [("read_file", {"path": path}) for path in fixture.expected_read_paths]
    if suffix == "conflict":
        if fixture.conflict_mode == "strong":
            return []
        return [
            (
                "write_file",
                {
                    "path": f"{fixture.out_dir}/{fixture.lower_name}",
                    "content": "OK",
                },
            )
        ]
    if suffix == "03":
        old_line = fixture.snapshot_lines[fixture.target_line - 1]
        new_line = old_line.replace(fixture.old_token, fixture.new_token, 1)
        return [
            ("read_file", {"path": fixture.line_path}),
            (
                "edit_file",
                {
                    "path": fixture.line_path,
                    "old_string": old_line,
                    "new_string": new_line,
                },
            ),
        ]
    raise ValueError(f"unsupported train-v4 family: {suffix}")


def _group_calls(
    spec: UnifiedTaskSpec, calls: Sequence[tuple[str, dict[str, Any]]]
) -> list[list[tuple[str, dict[str, Any]]]]:
    groups: list[list[tuple[str, dict[str, Any]]]] = [[] for _ in spec.user_turns]
    if not groups:
        return groups
    if spec.family == "context_recall" and spec.difficulty == "strong" and len(groups) == 3:
        groups[0] = list(calls[:5])
        groups[1] = list(calls[5:10])
        groups[2] = list(calls[10:])
        return groups
    if spec.family == "03" and len(groups) == 2:
        groups[0] = list(calls)
        return groups
    groups[0] = list(calls)
    return groups


async def _execute_calls(
    registry: Any,
    planned: Sequence[tuple[str, dict[str, Any]]],
) -> list[tuple[str, str, Any]]:
    executed: list[tuple[str, str, Any]] = []
    for name, arguments in planned:
        raw = json.dumps(arguments, ensure_ascii=False)
        result = await registry.execute(name, raw)
        if not result.ok:
            raise RuntimeError(f"planned oracle call failed: {name} {raw}: {result.content}")
        executed.append((name, raw, result))
    return executed


def build_oracle_example(
    spec: UnifiedTaskSpec,
    *,
    trace_path: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Execute the oracle call plan and export one gate-checked SFT row."""

    with tempfile.TemporaryDirectory(prefix=f"train-v4-{spec.task_id}-") as temporary:
        root = Path(temporary)
        fixture = build_fixture(spec, root)
        planned = _planned_calls(spec, fixture)
        grouped = _group_calls(spec, planned)
        registry = build_default_registry(root)
        completion = oracle_reply(spec, fixture)
        messages: list[dict[str, Any]] = [{"role": "system", "content": _system_prompt(spec)}]
        call_records: list[dict[str, Any]] = []
        result_records: list[dict[str, Any]] = []
        judge_calls: list[ToolCallRecord] = []
        call_index = 0
        for turn_index, (user_text, planned_for_turn) in enumerate(
            zip(spec.user_turns, grouped, strict=True)
        ):
            messages.append({"role": "user", "content": user_text})
            if not planned_for_turn:
                continue
            executed = asyncio.run(_execute_calls(registry, planned_for_turn))
            assistant_calls: list[dict[str, Any]] = []
            for name, raw, result in executed:
                call_id = f"tr4-{spec.task_id}-{call_index:02d}"
                call_index += 1
                assistant_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": raw},
                    }
                )
                call_records.append(
                    {
                        "call_id": call_id,
                        "tool_name": name,
                        "arguments": raw,
                        "arguments_json_valid": True,
                        "pydantic_valid": True,
                        "path_escape": False,
                        "blocked_at": None,
                        "executed": True,
                    }
                )
                result_records.append(
                    {
                        "call_id": call_id,
                        "tool_name": name,
                        "ok": True,
                        "is_error": False,
                        "protocol_error": False,
                        "error_kind": None,
                        "content": result.content,
                    }
                )
                judge_calls.append(ToolCallRecord(name, raw, result))
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": result.content}
                )
            messages.insert(
                len(messages) - len(executed),
                {"role": "assistant", "content": "", "tool_calls": assistant_calls},
            )
            if turn_index < len(spec.user_turns) - 1:
                intermediate = "已修改" if spec.family == "03" else "已读"
                messages.append({"role": "assistant", "content": intermediate})

        checks = judge(spec, fixture, root, completion, judge_calls)
        checks["within_steps"] = len(judge_calls) <= spec.steps_limit
        start = {
            "event": "run_start",
            "task_id": spec.task_id,
            "split": spec.split,
            "rollout_index": 0,
            "source": "deterministic_oracle",
            "fixture_seed": spec.fixture_seed,
            "constraints": [{"id": item.id, "content": item.text} for item in spec.constraints],
        }
        turn = {
            "event": "llm_turn",
            "messages": messages,
            "response": {"content": completion, "tool_calls": []},
            "tool_calls": call_records,
            "tool_results": result_records,
        }
        end = {
            "event": "run_end",
            "checks": checks,
            "within_steps": checks["within_steps"],
            "extras": {
                "tool_result_execution": "real_local_registry",
                "source": "deterministic_oracle",
            },
        }
        _write_trace(trace_path, (start, turn, end))

    episode = TraceEpisode(path=trace_path, start=start, end=end, turns=(turn,))
    example, reason, detail = build_example(episode)
    if example is None:
        return None, {
            "reason": reason,
            "detail": detail,
            "task_id": spec.task_id,
            "checks": checks,
        }
    example["id"] = f"train-v4:oracle:{spec.task_id}:rollout00"
    example["metadata"].update(
        {
            "source": "deterministic_oracle",
            "teacher_model": "deterministic-oracle-v1",
            "tool_result_execution": "real_local_registry",
            "dimension": spec.tuning_dimension,
            "category": _category_for_spec(spec),
            "difficulty": spec.difficulty,
            "context_tier": spec.context_tier,
            "split": spec.split,
            "fixture_seed": spec.fixture_seed,
        }
    )
    return example, {"reason": "ok", "task_id": spec.task_id, "checks": checks}


def _normalize_existing_row(row: Mapping[str, Any], split: str) -> dict[str, Any]:
    result = json.loads(json.dumps(row, ensure_ascii=False))
    task_id = str((result.get("metadata") or {}).get("task_id") or "")
    result["id"] = f"train-v4:deepseek:{result.get('id') or task_id}"
    metadata = result.setdefault("metadata", {})
    metadata.update(
        {
            "source": "deepseek_teacher",
            "teacher_model": "deepseek-v4.1-flash",
            "tool_result_execution": "teacher_agent_loop",
            "category": _category_for_task_id(task_id),
            "difficulty": _difficulty_for_task_id(task_id),
            "context_tier": _context_for_task_id(task_id),
            "split": split,
            "training_version": "train-v4",
        }
    )
    return result


def _load_existing(existing_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [
        _normalize_existing_row(json.loads(line), "train")
        for line in (existing_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    val = [
        _normalize_existing_row(json.loads(line), "val")
        for line in (existing_dir / "val.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return train, val


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count_distribution(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        counter[str(metadata.get(key) or "unknown")] += 1
    return dict(sorted(counter.items()))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_train_v4(
    *,
    existing_dir: Path,
    xlam_file: Path,
    out_dir: Path,
    per_source: int = PER_SOURCE_VARIANTS,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val = _load_existing(existing_dir)
    teacher_manifest_path = existing_dir / "manifest.json"
    teacher_manifest = (
        json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
        if teacher_manifest_path.is_file()
        else {}
    )
    source_stats: dict[str, Any] = {"deepseek_teacher": {"train": len(train), "val": len(val)}}
    rejected: list[dict[str, Any]] = []
    rejected_total = 0
    build_rejected_reasons: Counter[str] = Counter()

    raw_rows = load_xlam_rows(xlam_file)
    candidates, public_rejected = strict_grep_rows(raw_rows)
    rejected_total += len(public_rejected)
    rejected.extend({"source": "xlam", **item} for item in public_rejected[:50])
    public_train = 0
    public_val = 0
    for candidate_index, (_, row, _calls) in enumerate(candidates):
        split = "val" if candidate_index % 5 == 0 else "train"
        trace_path = (
            out_dir
            / "traces"
            / "public"
            / split
            / f"PUB-XLAM-{candidate_index:05d}_rollout00.jsonl"
        )
        example, detail = build_public_grep_example(
            row,
            row_index=candidate_index,
            split=split,
            trace_path=trace_path,
        )
        if example is None:
            rejected_total += 1
            build_rejected_reasons[str(detail.get("reason") or "unknown")] += 1
            rejected.append({"source": "xlam", **detail})
            continue
        if split == "val":
            val.append(example)
            public_val += 1
        else:
            train.append(example)
            public_train += 1
    source_stats["xlam_schema_mapped"] = {
        "raw_rows": len(raw_rows),
        "strict_grep_rows": len(candidates),
        "initial_unmappable_rows": len(public_rejected),
        "train": public_train,
        "val": public_val,
    }

    clones = build_clone_specs(per_source)
    final_specs = build_final_specs()
    clone_ids = {spec.task_id for spec in clones}
    final_ids = {spec.task_id for spec in final_specs}
    clone_seeds = {spec.fixture_seed for spec in clones}
    final_seeds = {spec.fixture_seed for spec in final_specs}
    if clone_ids & final_ids or clone_seeds & final_seeds:
        raise ValueError("train-v4 clones overlap final-test-v2 task ids or fixture seeds")
    oracle_train = 0
    oracle_val = 0
    for spec in clones:
        trace_path = out_dir / "traces" / "oracle" / spec.split / f"{spec.task_id}_rollout00.jsonl"
        example, detail = build_oracle_example(spec, trace_path=trace_path)
        if example is None:
            rejected_total += 1
            build_rejected_reasons[str(detail.get("reason") or "unknown")] += 1
            rejected.append({"source": "deterministic_oracle", **detail})
            continue
        if spec.split == "val":
            val.append(example)
            oracle_val += 1
        else:
            train.append(example)
            oracle_train += 1
    source_stats["deterministic_oracle"] = {
        "candidate_tasks": len(clones),
        "train": oracle_train,
        "val": oracle_val,
    }

    train = sorted(train, key=lambda row: str(row.get("id")))
    val = sorted(val, key=lambda row: str(row.get("id")))
    if len(train) < 200:
        raise ValueError(f"train-v4 train split too small: {len(train)} < 200")
    if len(val) < 50:
        raise ValueError(f"train-v4 val split too small: {len(val)} < 50")
    train_ids = {str(row.get("id")) for row in train}
    val_ids = {str(row.get("id")) for row in val}
    if train_ids & val_ids:
        raise ValueError("train/val row IDs overlap")

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, val)
    combined_payload = train_path.read_bytes() + val_path.read_bytes()
    combined_sha = hashlib.sha256(combined_payload).hexdigest()
    category_train = _count_distribution(train, "category")
    category_val = _count_distribution(val, "category")
    missing = [
        name
        for name in VALID_CATEGORIES
        if (category_train.get(name, 0) + category_val.get(name, 0)) < 30
    ]
    if missing:
        raise ValueError(f"train-v4 category coverage below 30 rows: {missing}")

    manifest: dict[str, Any] = {
        "version": "train-v4",
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "method-level SFT/DPO comparisons, not pipeline smoke test",
        "teacher_model": str(teacher_manifest.get("teacher_model") or "deepseek-v4.1-flash"),
        "teacher_protocol": str(
            teacher_manifest.get("teacher_protocol") or "openai_chat_completions_tool_calls_v1"
        ),
        "converter_version": str(
            teacher_manifest.get("converter_version") or "openai_and_hermes_to_qwen_chatml_v1"
        ),
        "teacher_api_calls": int(teacher_manifest.get("teacher_api_calls") or 608),
        "train_examples": len(train),
        "val_examples": len(val),
        "source_stats": source_stats,
        "source_distribution": {
            "train": _count_distribution(train, "source"),
            "val": _count_distribution(val, "source"),
        },
        "category_distribution": {"train": category_train, "val": category_val},
        "difficulty_distribution": {
            "train": _count_distribution(train, "difficulty"),
            "val": _count_distribution(val, "difficulty"),
        },
        "context_distribution": {
            "train": _count_distribution(train, "context_tier"),
            "val": _count_distribution(val, "context_tier"),
        },
        "gate": {
            "accepted": len(train) + len(val),
            "rejected": rejected_total,
            "rejected_listing_rows": len(rejected),
            "rejected_listing_truncated": rejected_total > len(rejected),
            "rejected_reason_counts": dict(
                sorted(
                    (
                        Counter(str(item.get("reason") or "unknown") for item in public_rejected)
                        + build_rejected_reasons
                    ).items()
                )
            ),
            "qwen_template_gate": (
                "deferred to Kaggle tokenizer preflight; structural gate runs locally"
            ),
        },
        "tool_result_execution": {
            "deepseek_teacher": "AgentLoop teacher traces",
            "xlam_schema_mapped": "real local tool registry execution",
            "deterministic_oracle": "real local tool registry execution",
        },
        "public_data_boundary": {
            "xlam_raw_rows": len(raw_rows),
            "xlam_strict_grep_rows": len(candidates),
            "toolace_used_in_train_v4": False,
            "reason": (
                "ToolACE tools do not map to the six local tools; not mixed into trajectories"
            ),
        },
        "train_sha256": _sha256(train_path),
        "val_sha256": _sha256(val_path),
        "combined_sha256": combined_sha,
        "reserved_test_version": "final-test-v2",
        "split_isolation": {
            "final_test_task_id_overlap": sorted(clone_ids & final_ids),
            "final_test_fixture_seed_overlap": sorted(clone_seeds & final_seeds),
            "train_val_row_id_overlap": sorted(train_ids & val_ids),
            "status": "disjoint",
        },
        "rejected": rejected[:200],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "TRAIN_V4_FROZEN.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "TRAIN_V4_FROZEN.txt").write_text(
        "\n".join(
            (
                "train-v4",
                f"created_at={manifest['created_at']}",
                f"train_examples={len(train)}",
                f"val_examples={len(val)}",
                f"train_sha256={manifest['train_sha256']}",
                f"val_sha256={manifest['val_sha256']}",
                f"combined_sha256={combined_sha}",
                f"teacher_model={manifest['teacher_model']}",
                f"teacher_api_calls={manifest['teacher_api_calls']}",
                "reserved_test_version=final-test-v2",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build and freeze train-v4")
    parser.add_argument(
        "--existing-dir",
        type=Path,
        default=Path("private/sft_data_v2_deepseek_expanded"),
    )
    parser.add_argument(
        "--xlam-file",
        type=Path,
        default=Path(
            "private/public_data_raw/Salesforce-xlam-function-calling-60k/train/"
            "data-00000-of-00001.arrow"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("private/sft_data_v4"))
    parser.add_argument("--per-source", type=int, default=PER_SOURCE_VARIANTS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_train_v4(
        existing_dir=args.existing_dir,
        xlam_file=args.xlam_file,
        out_dir=args.out_dir,
        per_source=args.per_source,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
