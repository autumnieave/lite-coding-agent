"""Build and freeze the final held-out test manifest.

The final test is cloned from the dev-only tuning families but uses new task
IDs and fixture seeds. It is intentionally kept out of ``TASK_SPECS`` so the
pilot dev/test manifests cannot accidentally include it. No rollout is run by
this module; it only writes the pre-registered manifest and its SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.difficulty_tuning import TUNING_TASK_SPECS
from training.unified_dataset import TASK_SPECS, UnifiedTaskSpec

FINAL_ID_BY_SOURCE: dict[str, str] = {
    "U-TUNE-C1-BASE": "U-FT01-BASE",
    "U-TUNE-C1-HARD": "U-FT02-HARD",
    "U-TUNE-CTX-BASE": "U-FT03-BASE",
    "U-TUNE-CTX-HARD": "U-FT04-HARD",
    "U-TUNE-TOOL-BASE": "U-FT05-BASE",
    "U-TUNE-TOOL-HARD": "U-FT06-HARD",
    "U-TUNE-CONFLICT-NONE": "U-FT07-NONE",
    "U-TUNE-CONFLICT-WEAK": "U-FT08-WEAK",
    "U-TUNE-CONFLICT-STRONG": "U-FT09-STRONG",
    "U-TUNE-CHECK-BASE": "U-FT10-BASE",
    "U-TUNE-CHECK-HARD": "U-FT11-HARD",
}

FINAL_SEED_BY_SOURCE: dict[str, int] = {
    "U-TUNE-C1-BASE": 95001,
    "U-TUNE-C1-HARD": 95002,
    "U-TUNE-CTX-BASE": 95003,
    "U-TUNE-CTX-HARD": 95004,
    "U-TUNE-TOOL-BASE": 95005,
    "U-TUNE-TOOL-HARD": 95006,
    "U-TUNE-CONFLICT-NONE": 95007,
    "U-TUNE-CONFLICT-WEAK": 95008,
    "U-TUNE-CONFLICT-STRONG": 95009,
    "U-TUNE-CHECK-BASE": 95010,
    "U-TUNE-CHECK-HARD": 95011,
}

SAMPLING_CONFIG: dict[str, object] = {
    "temperature": 0.0,
    "top_p": None,
    "seed": 20261019,
}

ROLLOUTS_PER_TASK_PER_GROUP = 10
GROUPS = ("baseline", "system", "training", "system+training")
PRE_REGISTERED_METRICS = (
    "constraint_compliance_rate",
    "tool_call_success_rate",
    "task_completion_rate",
    "stable_violation_rate",
    "trajectory_completeness_rate",
)
CHECKER_FILES = (
    "scripts/difficulty_tuning.py",
    "scripts/four_group_eval.py",
    "scripts/unified_eval.py",
    "training/difficulty_tuning.py",
    "training/final_test.py",
    "training/system_variants.py",
    "training/unified_dataset.py",
)

PROJECT_PATH = re.compile(r"project/[A-Za-z0-9_./-]+")
FIX_TOKEN = re.compile(r"FIX-[A-Z0-9]+")
LINE_REFERENCE = re.compile(r"第\s*(\d+)\s*行")


@contextmanager
def _fixture_workspace(spec: UnifiedTaskSpec) -> Any:
    """Build the test fixture in a throwaway workspace for prompt validation."""

    from scripts.unified_eval import build_fixture

    root = Path(tempfile.mkdtemp(prefix=f"freeze-{spec.task_id}-"))
    try:
        yield build_fixture(spec, root), root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _line_edit_turns(fixture: Any) -> tuple[str, str]:
    return (
        f"读取 {fixture.line_path}，把第 {fixture.target_line} 行（从 1 开始计数）这一行的 token "
        f"改成 {fixture.new_token}，且只改这一行。",
        f"完成后输出严格 JSON，字段 changed（布尔 true）和 line（整数 {fixture.target_line}）。",
    )


def _validate_prompt_references(spec: UnifiedTaskSpec, fixture: Any, root: Path) -> None:
    """Fail freezing when a prompt points at a dev-only input.

    The validator checks the prompt against the *test* fixture that will actually
    be materialised at rollout time. It deliberately allows output directories
    that the task asks the model to create, but not missing input files.
    """

    text = "\n".join(spec.user_turns)
    for reference in sorted(set(PROJECT_PATH.findall(text))):
        path = root / reference
        if path.exists():
            continue
        out_dir = str(getattr(fixture, "out_dir", "") or "")
        if out_dir and (reference == out_dir or reference.startswith(out_dir.rstrip("/") + "/")):
            continue
        raise ValueError(f"{spec.task_id} prompt references missing test fixture path: {reference}")

    if spec.family == "03":
        if fixture.line_path not in text:
            raise ValueError(f"{spec.task_id} prompt does not reference {fixture.line_path}")
        line_matches = {int(value) for value in LINE_REFERENCE.findall(text)}
        if fixture.target_line not in line_matches:
            raise ValueError(
                f"{spec.task_id} prompt line reference {sorted(line_matches)} "
                f"does not match fixture line {fixture.target_line}"
            )
        tokens = set(FIX_TOKEN.findall(text))
        if tokens != {fixture.new_token}:
            raise ValueError(
                f"{spec.task_id} prompt FIX token {sorted(tokens)} "
                f"does not match fixture token {fixture.new_token}"
            )
        target = fixture.snapshot_lines[fixture.target_line - 1]
        if fixture.old_token not in target:
            raise ValueError(
                f"{spec.task_id} fixture line {fixture.target_line} lacks {fixture.old_token}"
            )


def build_final_specs() -> tuple[UnifiedTaskSpec, ...]:
    by_id = {spec.task_id: spec for spec in TUNING_TASK_SPECS}
    missing = set(FINAL_ID_BY_SOURCE) - set(by_id)
    if missing:
        raise ValueError(f"unknown tuning task ids: {sorted(missing)}")
    specs_list: list[UnifiedTaskSpec] = []
    for source_id, final_id in FINAL_ID_BY_SOURCE.items():
        spec = replace(
            by_id[source_id],
            task_id=final_id,
            split="test",
            fixture_seed=FINAL_SEED_BY_SOURCE[source_id],
        )
        with _fixture_workspace(spec) as (fixture, root):
            if spec.family == "03":
                spec = replace(spec, user_turns=_line_edit_turns(fixture))
            _validate_prompt_references(spec, fixture, root)
        specs_list.append(spec)
    specs = tuple(specs_list)
    validate_final_specs(specs)
    return specs


def validate_final_specs(specs: tuple[UnifiedTaskSpec, ...]) -> None:
    ids = [spec.task_id for spec in specs]
    seeds = [spec.fixture_seed for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("final test task IDs must be unique")
    if len(seeds) != len(set(seeds)):
        raise ValueError("final test fixture seeds must be unique")
    existing_ids = {spec.task_id for spec in TASK_SPECS} | {
        spec.task_id for spec in TUNING_TASK_SPECS
    }
    existing_seeds = {spec.fixture_seed for spec in TASK_SPECS} | {
        spec.fixture_seed for spec in TUNING_TASK_SPECS
    }
    if set(ids) & existing_ids:
        raise ValueError(f"final test IDs overlap: {sorted(set(ids) & existing_ids)}")
    if set(seeds) & existing_seeds:
        raise ValueError(f"final test seeds overlap: {sorted(set(seeds) & existing_seeds)}")
    dimensions = {spec.tuning_dimension for spec in specs}
    if dimensions != {
        "constraint_count",
        "context_length",
        "tool_parameters",
        "conflict",
        "check_depth",
    }:
        raise ValueError(f"final test dimensions incomplete: {sorted(dimensions)}")
    if {spec.difficulty for spec in specs} != {"weak", "medium", "strong"}:
        raise ValueError("final test must cover weak/medium/strong")
    if {spec.context_tier for spec in specs} != {"short", "medium", "long"}:
        raise ValueError("final test must cover short/medium/long contexts")
    for spec in specs:
        if not spec.checks:
            raise ValueError(f"{spec.task_id} has no checks")
        if spec.min_tool_calls > spec.steps_limit:
            raise ValueError(f"{spec.task_id} min_tool_calls exceeds steps_limit")
        if set(spec.required_tools) - set(spec.allowed_tools):
            raise ValueError(f"{spec.task_id} required_tools outside allowed_tools")


def canonical_payload(specs: tuple[UnifiedTaskSpec, ...]) -> str:
    rows = [spec.to_dict() for spec in specs]
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_final_test(out_dir: Path, *, root: Path | None = None) -> dict[str, object]:
    root = root or Path.cwd()
    specs = build_final_specs()
    payload = canonical_payload(specs)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "final_test.jsonl"
    manifest_path.write_text(payload, encoding="utf-8", newline="\n")
    metadata: dict[str, object] = {
        "version": "final-test-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "task_count": len(specs),
        "sha256": digest,
        "sampling_config": SAMPLING_CONFIG,
        "rollouts_per_task_per_group": ROLLOUTS_PER_TASK_PER_GROUP,
        "groups": list(GROUPS),
        "run_policy": "test runs once per group after this manifest is frozen",
        "pilot_test_excluded": True,
        "train_seed_policy": "reserve 96000+ for train; never reuse 95001-95011",
        "pre_registered_metrics": list(PRE_REGISTERED_METRICS),
        "source_task_map": FINAL_ID_BY_SOURCE,
        "checker_sha256": {
            name: _sha256_file(root / name) for name in CHECKER_FILES if (root / name).is_file()
        },
        "no_rollout_yet": True,
        "prompts_validated_against_test_fixture": True,
        "invalidates": "final-test-v1",
    }
    (out_dir / "FINAL_TEST_FROZEN.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (out_dir / "FINAL_TEST_FROZEN.txt").write_text(
        "\n".join(
            (
                "final-test-v2",
                f"created_at={metadata['created_at']}",
                f"task_count={len(specs)}",
                f"sha256={digest}",
                "sampling=temperature=0.0,top_p=None,seed=20261019",
                "rollouts_per_task_per_group=10",
                "groups=baseline,system,training,system+training",
                "pilot_test_excluded=true",
                "train_seed_policy=96000+",
                "no_rollout_yet=true",
                "prompts_validated_against_test_fixture=true",
                "invalidates=final-test-v1",
            )
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return metadata
