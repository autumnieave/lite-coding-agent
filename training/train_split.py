"""Build a train split that is disjoint from dev, tuning and final-test-v2.

The train split reuses the five controlled tuning families but creates new task
IDs, new fixture seeds and materialised fixtures. It is intentionally small
(22 tasks) for the first training-pipeline validation; it is not the full
100-200 task Route B corpus.

``train-v2`` keeps the same IDs, seeds and checks as ``train-v1`` but clarifies
the two prompts that interact ambiguously with the agent harness (multi-turn
context reads and strong-conflict file creation). ``train-v3`` keeps v2 and
additionally makes the C1 hard-task path literal: ``project/tuning/config_c1.ini``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from training.difficulty_tuning import TUNING_TASK_SPECS
from training.final_test import (
    _fixture_workspace,
    _line_edit_turns,
    _validate_prompt_references,
    build_final_specs,
)
from training.unified_dataset import TASK_SPECS, UnifiedTaskSpec

TRAIN_VARIANTS = ("A", "B")
TRAIN_SEED_START = 96000
TRAIN_VERSION = "train-v1"
TRAIN_V2 = "train-v2"
TRAIN_V3 = "train-v3"
TRAIN_VERSIONS = (TRAIN_VERSION, TRAIN_V2, TRAIN_V3)

_CTX_HARD_V2_TURNS = (
    "先记住：最终只输出 CODE 值，不得附加解释。然后在本轮内一次性读取 "
    "project/tuning/context/part01.txt 到 part05.txt（可在同一条回复中并行调用 "
    "read_file），全部读完后只回复「已读」。",
    "继续在本轮内一次性读取 project/tuning/context/part06.txt 到 part10.txt"
    "（可在同一条回复中并行调用 read_file），全部读完后只回复「已读」。",
    "现在输出最终 CODE 值，严格只输出该值。",
)

_CONFLICT_STRONG_V2_TURNS = (
    "在 project/tuning/out 创建 target.txt，内容按约束执行。"
    "若约束无法同时满足，请不要创建或写入 target.txt，只明确指出冲突，"
    "不要自行选择其中一个。",
)

_C1_HARD_V3_TURNS = (
    "读取 project/tuning/config_c1.ini。输出严格 JSON，字段 path 的值必须原样写成 "
    '"project/tuning/config_c1.ini"，保留 project/ 前缀；不要改写成 '
    "tuning/config_c1.ini。value 为文件去掉首尾空白后的完整内容。不要添加其他文字。",
)


def _apply_train_v2_prompt_clarifications(spec: UnifiedTaskSpec) -> UnifiedTaskSpec:
    """Clarify two prompts that interact badly with the harness semantics.

    The v1 prompts are kept untouched for the historical frozen split. The v2
    prompts remove ambiguity without changing fixtures, seeds, constraints or
    checks: the context task still requires ten reads, and the conflict task
    still requires detecting an unsatisfiable pair.
    """

    if spec.family == "context_recall" and spec.difficulty == "strong":
        return replace(spec, user_turns=_CTX_HARD_V2_TURNS)
    if spec.family == "conflict" and spec.difficulty == "strong":
        return replace(spec, user_turns=_CONFLICT_STRONG_V2_TURNS)
    return spec


def _apply_train_v3_prompt_clarifications(spec: UnifiedTaskSpec) -> UnifiedTaskSpec:
    """Apply v2 clarifications and make the C1 hard-task path literal."""

    spec = _apply_train_v2_prompt_clarifications(spec)
    if spec.family == "json_constraints" and spec.difficulty == "strong":
        return replace(spec, user_turns=_C1_HARD_V3_TURNS)
    return spec


def build_train_specs(version: str = TRAIN_VERSION) -> tuple[UnifiedTaskSpec, ...]:
    if version not in TRAIN_VERSIONS:
        raise ValueError(f"unknown train split version: {version}")
    specs: list[UnifiedTaskSpec] = []
    seed = TRAIN_SEED_START
    for variant in TRAIN_VARIANTS:
        for source in TUNING_TASK_SPECS:
            suffix = source.task_id.removeprefix("U-TUNE-")
            spec = replace(
                source,
                task_id=f"U-TR-{variant}-{suffix}",
                split="train",
                fixture_seed=seed,
            )
            with _fixture_workspace(spec) as (fixture, root):
                if spec.family == "03":
                    spec = replace(spec, user_turns=_line_edit_turns(fixture))
                if version == TRAIN_V3:
                    spec = _apply_train_v3_prompt_clarifications(spec)
                elif version == TRAIN_V2:
                    spec = _apply_train_v2_prompt_clarifications(spec)
                _validate_prompt_references(spec, fixture, root)
            specs.append(spec)
            seed += 1
    result = tuple(specs)
    validate_train_specs(result, version=version)
    return result


def validate_train_specs(
    specs: tuple[UnifiedTaskSpec, ...], *, version: str = TRAIN_VERSION
) -> None:
    if len(specs) < 20:
        raise ValueError("train split must contain at least 20 tasks")
    ids = [spec.task_id for spec in specs]
    seeds = [spec.fixture_seed for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("train task IDs must be unique")
    if len(seeds) != len(set(seeds)):
        raise ValueError("train fixture seeds must be unique")
    reserved_ids = (
        {spec.task_id for spec in TASK_SPECS}
        | {spec.task_id for spec in TUNING_TASK_SPECS}
        | {spec.task_id for spec in build_final_specs()}
    )
    reserved_seeds = (
        {spec.fixture_seed for spec in TASK_SPECS}
        | {spec.fixture_seed for spec in TUNING_TASK_SPECS}
        | {spec.fixture_seed for spec in build_final_specs()}
    )
    if set(ids) & reserved_ids:
        raise ValueError(f"train IDs overlap: {sorted(set(ids) & reserved_ids)}")
    if set(seeds) & reserved_seeds:
        raise ValueError(f"train seeds overlap: {sorted(set(seeds) & reserved_seeds)}")
    if {spec.tuning_dimension for spec in specs} != {
        "constraint_count",
        "context_length",
        "tool_parameters",
        "conflict",
        "check_depth",
    }:
        raise ValueError("train split must cover all five dimensions")
    if {spec.difficulty for spec in specs} != {"weak", "medium", "strong"}:
        raise ValueError("train split must cover weak/medium/strong")
    if {spec.context_tier for spec in specs} != {"short", "medium", "long"}:
        raise ValueError("train split must cover short/medium/long contexts")
    for spec in specs:
        if not spec.checks:
            raise ValueError(f"{spec.task_id} has no checks")
        if spec.min_tool_calls > spec.steps_limit:
            raise ValueError(f"{spec.task_id} min_tool_calls exceeds steps_limit")
        if set(spec.required_tools) - set(spec.allowed_tools):
            raise ValueError(f"{spec.task_id} required_tools outside allowed_tools")
    if version == TRAIN_V2:
        for spec in specs:
            prompt = "\n".join(spec.user_turns)
            if (
                spec.family == "context_recall"
                and spec.difficulty == "strong"
                and "一次性读取" not in prompt
            ):
                raise ValueError(f"{spec.task_id} missing v2 batch-read clarification")
            if (
                spec.family == "conflict"
                and spec.difficulty == "strong"
                and "不要创建或写入" not in prompt
            ):
                raise ValueError(f"{spec.task_id} missing v2 no-write clarification")


def canonical_payload(specs: tuple[UnifiedTaskSpec, ...]) -> str:
    rows = [spec.to_dict() for spec in specs]
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def freeze_train_split(out_dir: Path, version: str = TRAIN_VERSION) -> dict[str, object]:
    specs = build_train_specs(version)
    payload = canonical_payload(specs)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.jsonl").write_text(payload, encoding="utf-8", newline="\n")
    metadata: dict[str, object] = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "task_count": len(specs),
        "sha256": digest,
        "seed_range": [TRAIN_SEED_START, TRAIN_SEED_START + len(specs) - 1],
        "variants": list(TRAIN_VARIANTS),
        "reserved_test_version": "final-test-v2",
        "source_task_map": {
            source.task_id: [
                f"U-TR-{variant}-{source.task_id.removeprefix('U-TUNE-')}"
                for variant in TRAIN_VARIANTS
            ]
            for source in TUNING_TASK_SPECS
        },
        "purpose": "first training-pipeline validation, not full Route B corpus",
    }
    (out_dir / "TRAIN_FROZEN.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (out_dir / "TRAIN_FROZEN.txt").write_text(
        "\n".join(
            (
                version,
                f"created_at={metadata['created_at']}",
                f"task_count={len(specs)}",
                f"sha256={digest}",
                f"seed_range={TRAIN_SEED_START}-{TRAIN_SEED_START + len(specs) - 1}",
                "reserved_test_version=final-test-v2",
            )
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return metadata
