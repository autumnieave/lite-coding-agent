"""Unified dev/test baseline runner.

Runs the task manifests from ``training/unified_dataset.py`` against the real
agent.  This module intentionally does not import ``scripts/benchmark.py``:
the old benchmark is a historical regression baseline, while this runner owns
the new split's fixtures, checkers and stability table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from trace_utils import (  # noqa: E402
    TraceRecorder,
    TracingCompactor,
    TracingProvider,
    TracingRegistry,
)

from agent.core.compaction import CompactionConfig, Compactor  # noqa: E402
from agent.core.config import find_env_file, load_env_file  # noqa: E402
from agent.core.constraints import ConstraintStore  # noqa: E402
from agent.core.llm import BaseProvider, OpenAICompatProvider  # noqa: E402
from agent.core.loop import AgentLoop  # noqa: E402
from agent.tools import build_default_registry  # noqa: E402
from agent.tools.base import ToolResult  # noqa: E402
from training.kaggle.contracts import classify_exception  # noqa: E402
from training.system_variants import get_variant, render_prompt_suffix  # noqa: E402
from training.unified_dataset import (  # noqa: E402
    TASK_SPECS,
    UnifiedTaskSpec,
    select_tasks,
    validate_splits,
)

CONTEXT_WINDOW = 12_000
CONTEXT_LINES_PER_FILE = 25
STABLE_HITS = 6
STABLE_MIN_TOTAL = 10
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")
EMOJI = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\u2b00-\u2bff\ufe0f\u200d]")
DEFAULT_OUT_DIR = Path("private/unified_dataset")
DEFAULT_TRACE_DIR = Path("private/traces/unified")
PUBLIC_GREP_FIXTURE_PATH = "project/public/knowledge.txt"

CHECK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "json_valid": ("合法 JSON",),
    "json_answer": ("合法 JSON", "JSON"),
    "snake_case_keys": ("snake_case",),
    "no_code_fence": ("代码围栏",),
    "no_fence": ("代码围栏",),
    "no_emoji": ("emoji",),
    "sorted": ("字母序",),
    "files_exact": ("字母序",),
    "value_correct": ("raw_value", "内容"),
    "heading_exact": ("标题",),
    "null_explicit": ("null",),
    "edit_file_used": ("edit_file",),
    "only_target_line": ("只允许改动目标行", "只改"),
    "token_replaced": ("只允许改动目标行", "token"),
    "context_read": ("读取",),
    "exact_lowercase_target": ("小写",),
    "content_exact": ("内容",),
    "reason_present": ("小写", "大小写"),
    "write_refused": ("write_file", "拒绝"),
    "edit_after_refusal": ("edit_file", "拒绝"),
    "answer_exact": ("前后缀",),
    "missing_null": ("null",),
    "exact_keys": ("额外字段",),
    "ids_exact": ("升序", "id"),
    "values_exact": ("字母序", "value"),
    "total_correct": ("id",),
    "source_count_correct": ("行数",),
    "policy_applied": ("策略", "最后", "FIRST_ACTIVE", "MODE"),
    "no_excluded_read": ("排除",),
    "selected_count_correct": ("数量", "文件"),
    "grep_used": ("grep",),
    "no_archive_read": ("archive",),
    "count_correct": ("count", "数量"),
    "line_count_preserved": ("只允许改动目标行", "行"),
    "verified_after_edit": ("重新读取", "验证"),
    "resolved_exact": ("REV", "冲突"),
    "conflicts_exact": ("冲突",),
    "sources_exact": ("读取",),
    "conflict_count_correct": ("冲突",),
    "no_old_read": ("_old",),
    "matches_complete": ("matches", "pattern", "line"),
}


def task_family(spec: UnifiedTaskSpec) -> str:
    """Return the explicit checker family, falling back to the legacy suffix."""

    return spec.family or spec.task_id[-2:]


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    name: str
    arguments: str
    result: ToolResult


@dataclass(slots=True)
class Fixture:
    task_id: str
    kind: str
    main_path: str = ""
    main_content: str = ""
    files: tuple[str, ...] = ()
    line_path: str = ""
    snapshot_lines: tuple[str, ...] = ()
    target_line: int = 0
    old_token: str = ""
    new_token: str = ""
    heading_path: str = ""
    heading: str = ""
    context_paths: tuple[str, ...] = ()
    out_dir: str = ""
    requested_name: str = ""
    lower_name: str = ""
    content_code: str = ""
    notes_path: str = ""
    final_token: str = ""
    data_paths: tuple[str, ...] = ()
    expected_ids: tuple[int, ...] = ()
    expected_values: tuple[str, ...] = ()
    expected_total: int = 0
    source_count: int = 0
    expected_read_paths: tuple[str, ...] = ()
    decoy_paths: tuple[str, ...] = ()
    conflict_mode: str = ""
    expected_json: dict[str, Any] | None = None
    primary_key: str = ""


class RecordingExecutor:
    """Record raw calls and results for checkers without changing loop semantics."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[ToolCallRecord] = []

    def specs(self) -> list[dict[str, Any]]:
        return self.inner.specs()

    async def execute(self, name: str, arguments: str) -> ToolResult:
        result = await self.inner.execute(name, arguments)
        self.calls.append(ToolCallRecord(name=name, arguments=arguments, result=result))
        return result


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _context_names(count: int = 10) -> tuple[str, ...]:
    return tuple(f"part{index:02d}.txt" for index in range(1, count + 1))


def build_fixture(spec: UnifiedTaskSpec, root: Path) -> Fixture:
    """Build one deterministic workspace for ``spec``."""

    suffix = task_family(spec)
    seed = spec.fixture_seed
    rng = random.Random(seed)

    if suffix == "policy_extract":
        rel = spec.json_fixture_path or "project/s15/extract/settings.ini"
        case = spec.task_id.rsplit("-", 1)[-1]
        if case == "E":
            lines = (
                "# app config",
                "TIMEOUT=30",
                "# TIMEOUT=10",
                "TIMEOUT=45",
            )
            expected = {
                "timeout": 45,
            }
            primary_key = "timeout"
        elif case == "M":
            lines = (
                "POLICY=FIRST_ACTIVE",
                "LIMIT=120",
                "# LIMIT=80",
                "LIMIT=150",
                "# LIMIT=240",
            )
            expected = {"policy": "FIRST_ACTIVE", "winner": 120, "active_count": 2}
            primary_key = "winner"
        elif case == "H":
            lines = (
                "MODE=SAFE",
                "SAFE_CODE=green-7",
                "FAST_CODE=red-9",
                "# both code keys are present; MODE decides the winner",
            )
            expected = {"mode": "SAFE", "code": "green-7", "rule": "SAFE_CODE"}
            primary_key = "code"
        else:
            raise ValueError(f"unknown policy_extract case: {case}")
        _write(root, rel, "\n".join(lines) + "\n")
        return Fixture(
            task_id=spec.task_id,
            kind="policy_extract",
            main_path=rel,
            main_content=str(expected[primary_key]),
            expected_json=expected,
            primary_key=primary_key,
        )

    if suffix == "selective_aggregate":
        directory = spec.path_directory or "project/s15/select"
        case = spec.task_id.rsplit("-", 1)[-1]
        if case == "E":
            selected = (f"{directory}/alpha.log", f"{directory}/beta.log")
            excluded = (f"{directory}/notes.txt",)
            payloads = {
                selected[0]: "10\n20\n",
                selected[1]: "5\n",
                excluded[0]: "999\n",
            }
            expected_total = 35
        elif case == "M":
            selected = (f"{directory}/app.log", f"{directory}/worker.log")
            excluded = (
                f"{directory}/app.draft.log",
                f"{directory}/notes.txt",
                f"{directory}/_index.txt",
            )
            payloads = {
                selected[0]: "10\n20\n",
                selected[1]: "5\n7\n",
                excluded[0]: "9000\n",
                excluded[1]: "8000\n",
                excluded[2]: "7000\n",
            }
            expected_total = 42
        elif case == "H":
            selected = (f"{directory}/a.csv", f"{directory}/b.csv", f"{directory}/d.csv")
            excluded = (f"{directory}/tmp_a.csv", f"{directory}/notes.txt")
            payloads = {
                selected[0]: "4\n9\n",
                selected[1]: "13\n2\n",
                selected[2]: "8\n",
                excluded[0]: "1000\n",
                excluded[1]: "ignored\n",
            }
            expected_total = 36
        else:
            raise ValueError(f"unknown selective_aggregate case: {case}")
        for rel, text in payloads.items():
            _write(root, rel, text)
        return Fixture(
            task_id=spec.task_id,
            kind="selective_aggregate",
            files=selected,
            expected_read_paths=selected,
            decoy_paths=excluded,
            expected_total=expected_total,
            source_count=len(selected),
        )

    if suffix == "search_collect":
        directory = spec.path_directory or "project/s15/search"
        case = spec.task_id.rsplit("-", 1)[-1]
        archive: str | None
        if case == "E":
            marker = "NEEDLE-"
            hits = (f"{directory}/hits.txt",)
            payloads = {
                hits[0]: "NEEDLE-11\nNEEDLE-13\n",
                f"{directory}/other.txt": "no matching marker\n",
            }
            expected_ids = (11, 13)
            archive = None
        elif case == "M":
            marker = "NEEDLE-"
            hits = (f"{directory}/alpha.txt", f"{directory}/beta.txt")
            payloads = {
                hits[0]: "NEEDLE-104\nNEEDLE-108\n",
                hits[1]: "NEEDLE-108\nNEEDLE-115\n",
                f"{directory}/gamma.txt": "no matching marker\n",
            }
            expected_ids = (104, 108, 115)
            archive = f"{directory}/archive/old.txt"
            payloads[archive] = f"{marker}999\n{marker}1000\n"
        elif case == "H":
            marker = "KEY-"
            hits = (f"{directory}/one.txt", f"{directory}/two.txt")
            payloads = {
                hits[0]: "KEY-7\n KEY-700\n",
                hits[1]: "KEY-7\nKEY-12\n",
                f"{directory}/three.txt": "prefix KEY-900 is not at line start\n",
            }
            expected_ids = (7, 12)
            archive = f"{directory}/archive/old.txt"
            payloads[archive] = f"{marker}999\n{marker}1000\n"
        else:
            raise ValueError(f"unknown search_collect case: {case}")
        for rel, text in payloads.items():
            _write(root, rel, text)
        return Fixture(
            task_id=spec.task_id,
            kind="search_collect",
            files=hits,
            expected_ids=expected_ids,
            expected_read_paths=hits,
            decoy_paths=(archive,) if archive else (),
        )

    if suffix == "edit_verify":
        rel = spec.edit_path or "project/s15/edit/part01.txt"
        target = spec.edit_target_line or 17
        new_token = spec.edit_new_token or "FIX-S15"
        case = spec.task_id.rsplit("-", 1)[-1]
        old_token = f"OLD-S15{case}"
        line_count = {"E": 20, "M": 60, "H": 140}[case]
        lines = [f"row-{index:03d}: token-{index:03d}" for index in range(1, line_count + 1)]
        lines[target - 1] = f"row-{target:03d}: {old_token}"
        _write(root, rel, "\n".join(lines) + "\n")
        return Fixture(
            task_id=spec.task_id,
            kind="edit_verify",
            line_path=rel,
            snapshot_lines=tuple(lines),
            target_line=target,
            old_token=old_token,
            new_token=new_token,
        )

    if suffix == "audit_reconcile":
        directory = spec.path_directory or "project/s15/audit"
        case = spec.task_id.rsplit("-", 1)[-1]
        shards: dict[str, str]
        if case == "E":
            shards = {
                "shard01.txt": "ALPHA=a1;REV=1\nBETA=b1;REV=2\n",
                "shard02.txt": "GAMMA=g1;REV=3\nBETA=b2;REV=4\n",
            }
            resolved = {"ALPHA": "a1", "BETA": "b2", "GAMMA": "g1"}
            conflicts: list[str] = []
            old_name = "shard01.txt"
        elif case == "M":
            shards = {
                "shard01.txt": "ALPHA=a1;REV=1\nBETA=b1;REV=4\nGAMMA=g1;REV=2\n",
                "shard02.txt": "ALPHA=a2;REV=5\nBETA=b2;REV=4\nDELTA=d1;REV=3\n",
                "shard03.txt": "ALPHA=a2;REV=5\nGAMMA=g2;REV=6\n",
            }
            resolved = {"ALPHA": "a2", "GAMMA": "g2", "DELTA": "d1"}
            conflicts = ["BETA"]
            old_name = "shard01.txt"
        elif case == "H":
            shards = {
                "shard01.txt": "ETA=e1;REV=1\nIOTA=i1;REV=5\nKAPPA=k1;REV=2\n",
                "shard02.txt": "ETA=e2;REV=1\nIOTA=i2;REV=5\nLAMBDA=l1;REV=4\n",
                "shard03.txt": "KAPPA=k2;REV=3\nMU=m1;REV=6\n",
                "shard04.txt": "THETA=t1;REV=4\nZETA=z1;REV=2\n",
                "shard05.txt": "ZETA=z1;REV=2\n",
            }
            resolved = {
                "KAPPA": "k2",
                "LAMBDA": "l1",
                "MU": "m1",
                "THETA": "t1",
                "ZETA": "z1",
            }
            conflicts = ["ETA", "IOTA"]
            old_name = "shard03.txt"
        else:
            raise ValueError(f"unknown audit_reconcile case: {case}")
        sources = tuple(sorted(f"{directory}/{name}" for name in shards))
        for name, text in shards.items():
            _write(root, f"{directory}/{name}", text)
        old_path = f"{directory}/_old/{old_name}"
        _write(root, old_path, "ALPHA=stale;REV=99\nETA=stale;REV=99\nIOTA=stale;REV=99\n")
        _write(root, f"{directory}/notes.txt", "not a shard\n")
        expected = {
            "resolved": resolved,
            "conflicts": sorted(conflicts),
            "sources": list(sources),
            "conflict_count": len(conflicts),
        }
        return Fixture(
            task_id=spec.task_id,
            kind="audit_reconcile",
            files=sources,
            expected_read_paths=sources,
            decoy_paths=(old_path, f"{directory}/notes.txt"),
            expected_total=len(conflicts),
            source_count=len(sources),
            expected_json=expected,
        )

    if suffix == "json_constraints":
        rel = spec.json_fixture_path or "project/tuning/config_c1.ini"
        content = f"mode = shared-{seed}\n"
        _write(root, rel, content)
        return Fixture(
            task_id=spec.task_id,
            kind="json_constraints",
            main_path=rel,
            main_content=content.strip(),
        )

    if suffix == "context_recall":
        count = spec.context_file_count or (1 if spec.task_id.endswith("-BASE") else 10)
        directory = spec.context_directory or "project/tuning/context"
        lines_per_file = spec.context_lines_per_file or CONTEXT_LINES_PER_FILE
        paths: list[str] = []
        final_token = f"CTX-{seed}"
        for index in range(1, count + 1):
            lines = [
                f"filler-{index:02d}-{line:03d}: {seed + line}"
                for line in range(1, lines_per_file + 1)
            ]
            if index == count:
                lines.append(f"CODE={final_token}")
            rel = f"{directory}/part{index:02d}.txt"
            _write(root, rel, "\n".join(lines) + "\n")
            paths.append(rel)
        return Fixture(
            task_id=spec.task_id,
            kind="context_recall",
            context_paths=tuple(paths),
            final_token=final_token,
        )

    if suffix == "path_params":
        directory = spec.path_directory or "project/tuning/files"
        if spec.path_targets:
            expected = tuple(spec.path_targets)
            lengths = {rel: 5 + index * 2 for index, rel in enumerate(expected, start=1)}
            decoys = tuple(f"{directory}_old/{Path(rel).name}" for rel in expected)
        elif spec.task_id.endswith("-BASE"):
            expected = (f"{directory}/part03.txt",)
            lengths = {expected[0]: 7}
            decoys: tuple[str, ...] = ()
        else:
            expected = (f"{directory}/part03.txt", f"{directory}/part07.txt")
            lengths = {expected[0]: 7, expected[1]: 9}
            decoys = ("project/tuning/files_old/part03.txt", "project/tuning/files_old/part07.txt")
        for rel, line_count in lengths.items():
            _write(
                root,
                rel,
                "\n".join(f"entry-{index:02d}-{seed}" for index in range(1, line_count + 1)) + "\n",
            )
        for rel in decoys:
            _write(root, rel, "DECOY\n" * 12)
        return Fixture(
            task_id=spec.task_id,
            kind="path_params",
            expected_read_paths=expected,
            decoy_paths=decoys,
            expected_total=sum(lengths.values()),
        )

    if suffix == "conflict":
        mode = (
            "none"
            if spec.task_id.endswith("-NONE")
            else ("weak" if spec.task_id.endswith("-WEAK") else "strong")
        )
        out_dir = "project/tuning/out"
        (root / out_dir).mkdir(parents=True, exist_ok=True)
        requested = (
            "report.txt" if mode == "none" else ("Report.txt" if mode == "weak" else "target.txt")
        )
        return Fixture(
            task_id=spec.task_id,
            kind="conflict",
            out_dir=out_dir,
            requested_name=requested,
            lower_name=requested.lower(),
            content_code="OK" if mode != "strong" else "ALPHA/BETA",
            conflict_mode=mode,
        )

    if suffix == "01":
        rel = spec.json_fixture_path or (
            "project/runtime.ini" if spec.split == "dev" else "project/config.txt"
        )
        content = f"mode = steady-{seed}\n"
        _write(root, rel, content)
        return Fixture(
            task_id=spec.task_id,
            kind="json_file",
            main_path=rel,
            main_content=content.strip(),
        )

    if suffix == "02":
        directory = spec.list_directory or (
            "project/assets" if spec.split == "dev" else "project/inputs"
        )
        names = ["zeta.log", "alpha.log", "mike.log", "beta.log"]
        rng.shuffle(names)
        for index, name in enumerate(names):
            _write(root, f"{directory}/{name}", f"entry-{index}-{seed}\n")
        return Fixture(task_id=spec.task_id, kind="json_list", files=tuple(names))

    if suffix == "03":
        rel = spec.edit_path or (
            "project/notes/part02.txt" if spec.split == "dev" else "project/notes/part07.txt"
        )
        target = spec.edit_target_line or (37 if spec.split == "dev" else 53)
        new_token = spec.edit_new_token or ("FIX-D03" if spec.split == "dev" else "FIX-T03")
        old_token = f"OLD-{new_token.split('-')[-1]}"
        lines = [f"row-{line:02d}: token-{line:02d}" for line in range(1, 61)]
        lines[target - 1] = f"row-{target:02d}: {old_token}"
        _write(root, rel, "\n".join(lines) + "\n")
        return Fixture(
            task_id=spec.task_id,
            kind="line_edit",
            line_path=rel,
            snapshot_lines=tuple(lines),
            target_line=target,
            old_token=old_token,
            new_token=new_token,
        )

    if suffix == "04":
        rel = "project/readme.md" if spec.split == "dev" else "project/guide.md"
        target = 2 if spec.split == "dev" else 3
        headings = ("Environment", "Deployment notes", "Rollback plan", "Appendix")
        body = ["# Project guide", ""]
        for heading in headings:
            body.extend([f"## {heading}", "", f"Dummy content for {heading}.", ""])
        _write(root, rel, "\n".join(body))
        return Fixture(
            task_id=spec.task_id,
            kind="heading",
            heading_path=rel,
            heading=headings[target - 1],
        )

    if suffix == "05":
        directory = "project/context" if spec.split == "dev" else "project/atlas"
        names = _context_names()
        for index, name in enumerate(names, start=1):
            lines = [f"filler-{index:02d}-{line:03d}: {seed + line}" for line in range(1, 161)]
            lines.append(f"PART={index:02d}")
            _write(root, f"{directory}/{name}", "\n".join(lines) + "\n")
        requested = "Report.md" if spec.split == "dev" else "Summary.md"
        return Fixture(
            task_id=spec.task_id,
            kind="lowercase_create",
            context_paths=tuple(f"{directory}/{name}" for name in names),
            out_dir="project/out",
            requested_name=requested,
            lower_name=requested.lower(),
            content_code="U-C41" if spec.split == "dev" else "U-C101",
        )

    if suffix == "06":
        rel = "project/notes/part05.txt" if spec.split == "dev" else "project/notes/part09.txt"
        _write(root, rel, "alpha\nbeta\ngamma\n")
        return Fixture(
            task_id=spec.task_id,
            kind="recover_edit",
            notes_path=rel,
            final_token="RECOVER-D06" if spec.split == "dev" else "RECOVER-T06",
        )

    if suffix == "07":
        directory = "project/data" if spec.split == "dev" else "project/records"
        prefix = "alpha" if spec.split == "dev" else "beta"
        smaller = "aaa-07" if spec.split == "dev" else "aab-07"
        mapping = {index: f"{prefix}-{index:02d}" for index in range(1, 13)}
        mapping[7] = smaller
        paths: list[str] = []
        for index in range(1, 13):
            lines = [f"id={index};value={mapping[index]}"]
            if index == 10:
                lines.insert(0, f"id=7;value={smaller}")
            if index == 12:
                lines.insert(0, "id=x;value=bad")
            rel = f"{directory}/part{index:02d}.txt"
            _write(root, rel, "\n".join(lines) + "\n")
            paths.append(rel)
        return Fixture(
            task_id=spec.task_id,
            kind="aggregate",
            data_paths=tuple(paths),
            expected_ids=tuple(sorted(mapping)),
            expected_values=tuple(sorted(mapping.values())),
            expected_total=sum(mapping),
            source_count=13,
        )

    if suffix == "public_grep":
        patterns = tuple(str(pattern) for pattern in spec.public_patterns)
        if not patterns:
            raise ValueError(f"{spec.task_id} has no public grep patterns")
        lines = [f"record-{index:03d}: {pattern}" for index, pattern in enumerate(patterns)]
        _write(root, PUBLIC_GREP_FIXTURE_PATH, "\n".join(lines) + "\n")
        matches = []
        for pattern in patterns:
            line_index = next(
                (index for index, line in enumerate(lines) if pattern in line),
                None,
            )
            if line_index is None:
                raise ValueError(f"{spec.task_id} pattern is missing from fixture: {pattern}")
            matches.append(
                {
                    "pattern": pattern,
                    "line": (
                        f"{PUBLIC_GREP_FIXTURE_PATH}:{line_index + 1}:"
                        f"record-{line_index:03d}: {pattern}"
                    ),
                }
            )
        return Fixture(
            task_id=spec.task_id,
            kind="public_grep",
            main_path=PUBLIC_GREP_FIXTURE_PATH,
            expected_json={"matches": matches},
        )

    raise ValueError(f"未知任务族：{spec.task_id}")


def parse_json(text: str) -> Any | None:
    try:
        return json.loads(text.strip())
    except (TypeError, ValueError):
        return None


def collect_keys(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        keys = [str(key) for key in payload]
        for value in payload.values():
            keys.extend(collect_keys(value))
        return keys
    if isinstance(payload, list):
        nested: list[str] = []
        for item in payload:
            nested.extend(collect_keys(item))
        return nested
    return []


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _normalize_rel(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _successful_calls(calls: Sequence[ToolCallRecord], name: str) -> list[ToolCallRecord]:
    return [call for call in calls if call.name == name and call.result.ok]


def _read_paths(calls: Sequence[ToolCallRecord]) -> set[str]:
    found: set[str] = set()
    for call in calls:
        if call.name != "read_file" or not call.result.ok:
            continue
        try:
            payload = json.loads(call.arguments)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("path"), str):
            found.add(_normalize_rel(payload["path"]))
    return found


def _call_path(call: ToolCallRecord) -> str:
    try:
        payload = json.loads(call.arguments)
    except (TypeError, ValueError):
        return ""
    if isinstance(payload, dict) and isinstance(payload.get("path"), str):
        return _normalize_rel(payload["path"])
    return ""


def judge(
    spec: UnifiedTaskSpec,
    fixture: Fixture,
    root: Path,
    reply: str,
    calls: Sequence[ToolCallRecord],
) -> dict[str, bool]:
    """Apply the manifest checks to one rollout's observable state."""

    payload = parse_json(reply)
    checks: dict[str, bool] = {}
    suffix = task_family(spec)

    if suffix == "policy_extract":
        expected = fixture.expected_json or {}
        primary_key = fixture.primary_key
        if not primary_key:
            raise ValueError(f"{spec.task_id} fixture is missing primary_key")
        checks["json_valid"] = isinstance(payload, dict)
        checks["exact_keys"] = isinstance(payload, dict) and set(payload) == set(expected)
        checks["value_correct"] = bool(
            isinstance(payload, dict) and payload.get(primary_key) == expected.get(primary_key)
        )
        checks["policy_applied"] = isinstance(payload, dict) and payload == expected
        checks["no_fence"] = "```" not in reply

    elif suffix == "selective_aggregate":
        files = payload.get("files") if isinstance(payload, dict) else None
        file_list = [str(item) for item in files] if isinstance(files, list) else None
        checks["json_valid"] = isinstance(payload, dict)
        checks["files_exact"] = bool(
            file_list is not None
            and len(file_list) == len(fixture.files)
            and set(file_list) == set(fixture.files)
        )
        checks["total_correct"] = bool(
            isinstance(payload, dict) and payload.get("total") == fixture.expected_total
        )
        checks["selected_count_correct"] = bool(
            isinstance(payload, dict) and payload.get("selected_count") == fixture.source_count
        )
        checks["sorted"] = bool(
            file_list is not None and file_list == sorted(file_list, key=str.lower)
        )
        checks["no_excluded_read"] = not (_read_paths(calls) & set(fixture.decoy_paths))

    elif suffix == "search_collect":
        ids = payload.get("ids") if isinstance(payload, dict) else None
        files = payload.get("files") if isinstance(payload, dict) else None
        file_list = [str(item) for item in files] if isinstance(files, list) else None
        checks["json_valid"] = isinstance(payload, dict)
        checks["ids_exact"] = bool(ids == list(fixture.expected_ids))
        checks["files_exact"] = bool(
            file_list is not None
            and len(file_list) == len(fixture.expected_read_paths)
            and set(file_list) == set(fixture.expected_read_paths)
        )
        checks["sorted"] = bool(
            isinstance(ids, list)
            and ids == sorted(ids)
            and file_list is not None
            and file_list == sorted(file_list, key=str.lower)
        )
        checks["count_correct"] = bool(
            isinstance(payload, dict) and payload.get("count") == len(fixture.expected_ids)
        )
        checks["grep_used"] = bool(_successful_calls(calls, "grep"))
        checks["no_archive_read"] = not (_read_paths(calls) & set(fixture.decoy_paths))

    elif suffix == "edit_verify":
        current = _read_lines(root / fixture.line_path)
        target_index = fixture.target_line - 1
        target_text = current[target_index] if target_index < len(current) else ""
        checks["edit_file_used"] = bool(_successful_calls(calls, "edit_file"))
        checks["only_target_line"] = bool(
            len(current) == len(fixture.snapshot_lines)
            and all(
                current[index] == fixture.snapshot_lines[index]
                for index in range(len(current))
                if index != target_index
            )
        )
        checks["token_replaced"] = (
            fixture.new_token in target_text and fixture.old_token not in target_text
        )
        checks["line_count_preserved"] = len(current) == len(fixture.snapshot_lines)
        verified = False
        for index, call in enumerate(calls):
            if (
                call.name != "edit_file"
                or not call.result.ok
                or _call_path(call) != fixture.line_path
            ):
                continue
            if any(
                later.name == "read_file"
                and later.result.ok
                and _call_path(later) == fixture.line_path
                for later in calls[index + 1 :]
            ):
                verified = True
                break
        checks["verified_after_edit"] = verified
        checks["json_answer"] = bool(
            isinstance(payload, dict)
            and payload.get("changed") is True
            and payload.get("line") == fixture.target_line
            and payload.get("verified") is True
        )

    elif suffix == "audit_reconcile":
        expected = fixture.expected_json or {}
        checks["json_valid"] = isinstance(payload, dict)
        checks["resolved_exact"] = bool(
            isinstance(payload, dict) and payload.get("resolved") == expected.get("resolved")
        )
        checks["conflicts_exact"] = bool(
            isinstance(payload, dict) and payload.get("conflicts") == expected.get("conflicts")
        )
        checks["sources_exact"] = bool(
            isinstance(payload, dict) and payload.get("sources") == expected.get("sources")
        )
        checks["conflict_count_correct"] = bool(
            isinstance(payload, dict)
            and payload.get("conflict_count") == expected.get("conflict_count")
        )
        checks["no_old_read"] = not any(
            "/_old/" in path or path.startswith("_old/") for path in _read_paths(calls)
        )

    elif suffix == "json_constraints":
        keys = collect_keys(payload)
        checks["json_valid"] = isinstance(payload, dict)
        checks["snake_case_keys"] = bool(keys) and all(SNAKE_CASE.fullmatch(key) for key in keys)
        checks["no_fence"] = "```" not in reply
        checks["exact_keys"] = bool(isinstance(payload, dict) and set(payload) == {"path", "value"})
        checks["value_correct"] = bool(
            isinstance(payload, dict)
            and _normalize_rel(str(payload.get("path") or "")) == fixture.main_path
            and payload.get("value") == fixture.main_content
        )

    elif suffix == "context_recall":
        checks["context_read"] = set(fixture.context_paths) <= _read_paths(calls)
        checks["answer_exact"] = reply.strip() == fixture.final_token

    elif suffix == "path_params":
        read_paths = _read_paths(calls)
        checks["json_valid"] = isinstance(payload, dict)
        checks["value_correct"] = bool(
            isinstance(payload, dict) and payload.get("total_lines") == fixture.expected_total
        )
        checks["read_paths_exact"] = read_paths == set(fixture.expected_read_paths)
        checks["no_decoy_read"] = not (read_paths & set(fixture.decoy_paths))

    elif suffix == "conflict":
        if fixture.conflict_mode in {"none", "weak"}:
            target = root / fixture.out_dir / fixture.lower_name
            checks["file_created_correct"] = target.is_file()
            checks["content_exact"] = bool(
                target.is_file()
                and target.read_text(encoding="utf-8").strip() == fixture.content_code
            )
            if fixture.conflict_mode == "weak":
                checks["reason_present"] = any(
                    word in reply for word in ("大小写", "小写", "冲突", "lowercase")
                )
        else:
            checks["conflict_declared"] = any(
                word in reply
                for word in (
                    "冲突",
                    "互斥",
                    "无法同时",
                    "不可能同时",
                    "不能同时满足",
                    "unresolved",
                    "矛盾",
                )
            )
            requested_path = root / fixture.out_dir / fixture.requested_name
            checks["no_arbitrary_file"] = not requested_path.exists()

    elif suffix == "01":
        path_key = "file_path" if spec.split == "dev" else "path"
        value_key = "raw_value" if spec.split == "dev" else "value"
        checks["json_valid"] = isinstance(payload, dict)
        keys = collect_keys(payload)
        checks["snake_case_keys"] = bool(keys) and all(SNAKE_CASE.fullmatch(key) for key in keys)
        checks["value_correct"] = bool(
            isinstance(payload, dict)
            and _normalize_rel(str(payload.get(path_key) or "")) == fixture.main_path
            and payload.get(value_key) == fixture.main_content
        )

    elif suffix == "02":
        files = payload.get("files") if isinstance(payload, dict) else None
        checks["files_exact"] = bool(
            isinstance(files, list)
            and len(files) == len(fixture.files)
            and {str(item) for item in files} == set(fixture.files)
        )
        checks["sorted"] = bool(
            isinstance(files, list)
            and [str(item) for item in files] == sorted(map(str, files), key=str.lower)
        )
        checks["no_emoji"] = EMOJI.search(reply) is None

    elif suffix == "03":
        checks["edit_file_used"] = bool(_successful_calls(calls, "edit_file"))
        current = _read_lines(root / fixture.line_path)
        target_index = fixture.target_line - 1
        checks["only_target_line"] = bool(
            len(current) == len(fixture.snapshot_lines)
            and all(
                current[index] == fixture.snapshot_lines[index]
                for index in range(len(current))
                if index != target_index
            )
        )
        target_text = current[target_index] if target_index < len(current) else ""
        checks["token_replaced"] = (
            fixture.new_token in target_text and fixture.old_token not in target_text
        )
        checks["json_answer"] = bool(
            isinstance(payload, dict)
            and payload.get("changed") is True
            and payload.get("line") == fixture.target_line
        )

    elif suffix == "04":
        checks["heading_exact"] = bool(
            isinstance(payload, dict) and payload.get("heading") == fixture.heading
        )
        checks["json_valid"] = isinstance(payload, dict)
        checks["no_fence"] = "```" not in reply
        checks["null_explicit"] = bool(
            isinstance(payload, dict)
            and "missing_note" in payload
            and payload.get("missing_note") is None
        )

    elif suffix == "05":
        expected = set(fixture.context_paths)
        checks["context_read"] = expected <= _read_paths(calls)
        out_dir = root / fixture.out_dir
        entries = {entry.name: entry for entry in out_dir.iterdir()} if out_dir.is_dir() else {}
        lower_path = out_dir / fixture.lower_name
        has_lower = lower_path.is_file()
        checks["exact_lowercase_target"] = bool(
            has_lower
            and fixture.lower_name in entries
            and not any(
                name.casefold() == fixture.lower_name.casefold() and name != fixture.lower_name
                for name in entries
            )
        )
        checks["content_exact"] = bool(
            has_lower and lower_path.read_text(encoding="utf-8").strip() == fixture.content_code
        )
        reason = payload.get("reason") if isinstance(payload, dict) else None
        checks["reason_present"] = bool(
            isinstance(reason, str)
            and reason.strip()
            and any(word in reason for word in ("小写", "大小写", "lowercase"))
        )
        checks["json_valid"] = isinstance(payload, dict)

    elif suffix == "06":
        refusal_index: int | None = None
        for index, call in enumerate(calls):
            refused = not call.result.ok and (
                "拒绝覆盖" in call.result.content or "存在" in call.result.content
            )
            if call.name == "write_file" and refused:
                refusal_index = index
                break
        checks["write_refused"] = refusal_index is not None
        checks["edit_after_refusal"] = bool(
            refusal_index is not None
            and any(
                call.name == "edit_file" and call.result.ok for call in calls[refusal_index + 1 :]
            )
        )
        notes = root / fixture.notes_path
        checks["content_exact"] = bool(
            notes.is_file() and notes.read_text(encoding="utf-8").strip() == fixture.final_token
        )
        checks["answer_exact"] = reply.strip() == fixture.final_token

    elif suffix == "07":
        keys = collect_keys(payload)
        checks["json_valid"] = isinstance(payload, dict)
        checks["snake_case_keys"] = bool(keys) and all(SNAKE_CASE.fullmatch(key) for key in keys)
        checks["no_fence"] = "```" not in reply
        checks["missing_null"] = bool(
            isinstance(payload, dict) and "missing" in payload and payload.get("missing") is None
        )
        checks["exact_keys"] = bool(
            isinstance(payload, dict)
            and set(payload) == {"ids", "values", "total", "missing", "source_count"}
        )
        checks["ids_exact"] = bool(
            isinstance(payload, dict) and payload.get("ids") == list(fixture.expected_ids)
        )
        checks["values_exact"] = bool(
            isinstance(payload, dict) and payload.get("values") == list(fixture.expected_values)
        )
        checks["total_correct"] = bool(
            isinstance(payload, dict) and payload.get("total") == fixture.expected_total
        )
        checks["source_count_correct"] = bool(
            isinstance(payload, dict) and payload.get("source_count") == fixture.source_count
        )

    elif suffix == "public_grep":
        keys = collect_keys(payload)
        checks["json_valid"] = isinstance(payload, dict)
        checks["snake_case_keys"] = bool(keys) and all(SNAKE_CASE.fullmatch(key) for key in keys)
        checks["no_fence"] = "```" not in reply
        checks["matches_complete"] = bool(
            isinstance(payload, dict) and payload == fixture.expected_json
        )

    else:
        raise ValueError(f"未知任务族：{spec.task_id}")

    used = {call.name for call in calls}
    checks["required_tools_used"] = set(spec.required_tools) <= used
    checks["tools_allowed"] = used <= set(spec.allowed_tools)
    allowed = {check.name for check in spec.checks}
    return {name: ok for name, ok in checks.items() if name in allowed}


def constraint_ids_for_check(spec: UnifiedTaskSpec, check: str) -> list[str]:
    keywords = CHECK_KEYWORDS.get(check, ())
    return [item.id for item in spec.constraints if any(word in item.text for word in keywords)]


def build_evidence(spec: UnifiedTaskSpec, checks: Mapping[str, bool]) -> list[dict[str, Any]]:
    label = f"user_turn_{len(spec.user_turns)}"
    return [
        {
            "check": check,
            "step_label": label,
            "constraint_ids": constraint_ids_for_check(spec, check),
        }
        for check, ok in checks.items()
        if not ok and check != "within_steps"
    ]


def _summarizer(provider: BaseProvider) -> Any:
    async def summarize(messages: object) -> str:
        response = await provider.chat(messages)  # type: ignore[arg-type]
        return response.content or ""

    return summarize


async def run_once(
    spec: UnifiedTaskSpec,
    *,
    provider: BaseProvider,
    rollout_index: int,
    trace_dir: Path | str | None = DEFAULT_TRACE_DIR,
    sampling_config: Mapping[str, Any] | None = None,
    system_variant: str | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(tempfile.mkdtemp(prefix=f"unified-{spec.task_id}-"))
    started = time.perf_counter()
    recorder: TraceRecorder | None = None
    trace_path: Path | None = None
    try:
        fixture = build_fixture(spec, root)
        variant_name = (
            system_variant
            if isinstance(system_variant, str)
            else (system_variant or {}).get("id", "s0")
        )
        variant = get_variant(str(variant_name))
        effective_sampling = dict(
            sampling_config
            if sampling_config is not None
            else getattr(provider, "sampling_config", {"is_default": True})
        )
        if trace_dir is not None:
            trace_path = (
                Path(trace_dir) / spec.split / f"{spec.task_id}_rollout{rollout_index:02d}.jsonl"
            )
            recorder = TraceRecorder(
                trace_path,
                metadata={
                    "experiment": "difficulty_tuning"
                    if spec.tuning_dimension
                    else "unified_dataset",
                    "sample_set": f"difficulty_tuning_{spec.tuning_dimension}"
                    if spec.tuning_dimension
                    else f"baseline_{spec.split}",
                    "task_id": spec.task_id,
                    "split": spec.split,
                    "difficulty": spec.difficulty,
                    "context_tier": spec.context_tier,
                    "tuning_dimension": spec.tuning_dimension or None,
                    "rollout_index": rollout_index,
                    "fixture_seed": spec.fixture_seed,
                    "max_turns": spec.steps_limit + 3,
                    "system_variant": variant.id,
                    "system_variant_config": variant.to_dict(),
                },
                workspace=root,
                constraints=[(item.id, item.text) for item in spec.constraints],
                sampling_config=effective_sampling,
            )

        store = ConstraintStore()
        for item in spec.constraints:
            store.add(item.text, constraint_id=item.id, source="user")
        traced_provider: Any = (
            TracingProvider(provider, recorder) if recorder is not None else provider
        )
        compactor = Compactor(
            CompactionConfig(context_window=CONTEXT_WINDOW),
            summarize=_summarizer(traced_provider),
            constraints=store,
        )
        traced_compactor: Any = (
            TracingCompactor(compactor, recorder) if recorder is not None else compactor
        )
        registry: Any = build_default_registry(root)
        if recorder is not None:
            registry = TracingRegistry(registry, recorder)
        executor = RecordingExecutor(registry)
        loop = AgentLoop(
            traced_provider,
            executor,
            max_turns=spec.steps_limit + 3,
            compactor=traced_compactor,
            constraints=store,
        )

        history: list[dict[str, Any]] = []
        replies: list[str] = []
        stopped: list[str] = []
        prompt_suffix = render_prompt_suffix(variant, spec)
        for index, turn in enumerate(spec.user_turns, start=1):
            if recorder is not None:
                recorder.set_step(index, f"user_turn_{index}")
            turn_text = turn
            if prompt_suffix and index == len(spec.user_turns):
                turn_text = f"{turn}\n\n{prompt_suffix}"
            result = await loop.run(turn_text, history=history)
            history = [dict(item) for item in result.messages]
            replies.append(result.content)
            stopped.append(result.stopped_reason)
            if recorder is not None:
                recorder.end_step()

        reply = replies[-1] if replies else ""
        checks = judge(spec, fixture, root, reply, executor.calls)
        checks["within_steps"] = len(executor.calls) <= spec.steps_limit
        used = [call.name for call in executor.calls]
        params_ok = all(isinstance(parse_json(call.arguments), dict) for call in executor.calls)
        duration = round(time.perf_counter() - started, 1)
        trace_display = ""
        if trace_path is not None:
            try:
                trace_display = trace_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
            except ValueError:
                trace_display = trace_path.as_posix()
        if recorder is not None:
            recorder.end_run(
                checks=checks,
                evidence=build_evidence(spec, checks),
                step_replies={
                    f"user_turn_{index}": text for index, text in enumerate(replies, start=1)
                },
                duration_s=duration,
                extras={
                    "tools_used": used,
                    "tool_selection_ok": set(spec.required_tools)
                    <= set(used)
                    <= set(spec.allowed_tools),
                    "params_ok": params_ok,
                    "within_steps": checks["within_steps"],
                    "stopped_reasons": sorted(set(stopped)),
                },
            )
        return {
            "task_id": spec.task_id,
            "split": spec.split,
            "difficulty": spec.difficulty,
            "context_tier": spec.context_tier,
            "rollout_index": rollout_index,
            "fixture_seed": spec.fixture_seed,
            "sampling_config": effective_sampling,
            "system_variant": variant.id,
            "model": getattr(provider, "model", None),
            "trace_path": trace_display,
            "checks": checks,
            "violated": sum(1 for key, ok in checks.items() if not ok and key != "within_steps"),
            "task_success": bool(checks) and all(checks.values()),
            "within_steps": checks["within_steps"],
            "steps": len(executor.calls),
            "tools_used": used,
            "tool_selection_ok": set(spec.required_tools) <= set(used) <= set(spec.allowed_tools),
            "params_ok": params_ok,
            "reply": reply,
            "stopped_reasons": sorted(set(stopped)),
            "duration_s": duration,
        }
    except Exception as exc:  # noqa: BLE001 - one failed rollout must not abort the split
        return {
            "task_id": spec.task_id,
            "split": spec.split,
            "difficulty": spec.difficulty,
            "context_tier": spec.context_tier,
            "rollout_index": rollout_index,
            "fixture_seed": spec.fixture_seed,
            "sampling_config": dict(sampling_config or {}),
            "system_variant": str(
                system_variant
                if isinstance(system_variant, str)
                else (system_variant or {}).get("id", "s0")
            ),
            "model": getattr(provider, "model", None),
            "trace_path": str(trace_path) if trace_path else "",
            "error_kind": classify_exception(exc),
            "error": f"{type(exc).__name__}: {exc}",
            "checks": {},
            "violated": 0,
            "task_success": False,
            "within_steps": False,
            "steps": 0,
            "tools_used": [],
            "tool_selection_ok": False,
            "params_ok": False,
            "reply": "",
            "stopped_reasons": [],
            "duration_s": round(time.perf_counter() - started, 1),
        }
    finally:
        if recorder is not None:
            recorder.close()
        shutil.rmtree(root, ignore_errors=True)


def stability(hits: int, total: int) -> str:
    if total < STABLE_MIN_TOTAL:
        return "insufficient"
    if hits == 0:
        return "not_observed"
    if hits >= STABLE_HITS:
        return "stable"
    return "intermittent"


def aggregate(
    records: Sequence[Mapping[str, Any]], specs: Sequence[UnifiedTaskSpec]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    check_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for spec in specs:
        rows = [row for row in records if row.get("task_id") == spec.task_id and "error" not in row]
        total = len(rows)
        stable_count = 0
        intermittent_count = 0
        for check in spec.checks:
            hits = sum(1 for row in rows if not row["checks"].get(check.name, False))
            status = stability(hits, total)
            if status == "stable":
                stable_count += 1
            elif status == "intermittent":
                intermittent_count += 1
            check_rows.append(
                {
                    "task_id": spec.task_id,
                    "split": spec.split,
                    "difficulty": spec.difficulty,
                    "context_tier": spec.context_tier,
                    "check": check.name,
                    "kind": check.kind,
                    "hits": hits,
                    "total": total,
                    "stability": status,
                }
            )
        task_rows.append(
            {
                "task_id": spec.task_id,
                "split": spec.split,
                "difficulty": spec.difficulty,
                "context_tier": spec.context_tier,
                "rollouts": total,
                "errors": sum(
                    1 for row in records if row.get("task_id") == spec.task_id and "error" in row
                ),
                "task_success": sum(1 for row in rows if row.get("task_success")),
                "stable_checks": stable_count,
                "intermittent_checks": intermittent_count,
                "within_steps_failures": sum(
                    1 for row in rows if not row.get("within_steps", True)
                ),
                "mean_steps": round(sum(int(row.get("steps", 0)) for row in rows) / total, 2)
                if total
                else 0.0,
            }
        )
    return check_rows, task_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(columns)]
    for row in rows:
        values: list[str] = []
        for column in columns:
            value = str(row.get(column, ""))
            if any(char in value for char in ',\n"'):
                value = '"' + value.replace('"', '""') + '"'
            values.append(value)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    out_dir: Path,
    records: Sequence[Mapping[str, Any]],
    specs: Sequence[UnifiedTaskSpec],
    *,
    split: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / f"baseline_{split}.jsonl"
    run_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
        encoding="utf-8",
    )
    check_rows, task_rows = aggregate(records, specs)
    write_csv(
        out_dir / f"baseline_{split}_checks.csv",
        check_rows,
        (
            "task_id",
            "split",
            "difficulty",
            "context_tier",
            "check",
            "kind",
            "hits",
            "total",
            "stability",
        ),
    )
    write_csv(
        out_dir / f"baseline_{split}_tasks.csv",
        task_rows,
        (
            "task_id",
            "split",
            "difficulty",
            "context_tier",
            "rollouts",
            "errors",
            "task_success",
            "stable_checks",
            "intermittent_checks",
            "within_steps_failures",
            "mean_steps",
        ),
    )
    stable = sum(1 for row in check_rows if row["stability"] == "stable")
    return {
        "split": split,
        "runs": len(records),
        "checks": len(check_rows),
        "stable_checks": stable,
        "stable_rate": round(stable / len(check_rows), 4) if check_rows else 0.0,
        "run_path": str(run_path),
    }


def oracle_reply(spec: UnifiedTaskSpec, fixture: Fixture) -> str:
    suffix = task_family(spec)
    if suffix == "policy_extract":
        return json.dumps(fixture.expected_json or {}, ensure_ascii=False)
    if suffix == "selective_aggregate":
        return json.dumps(
            {
                "files": sorted(fixture.files, key=str.lower),
                "total": fixture.expected_total,
                "selected_count": fixture.source_count,
            },
            ensure_ascii=False,
        )
    if suffix == "search_collect":
        return json.dumps(
            {
                "ids": list(fixture.expected_ids),
                "files": sorted(fixture.expected_read_paths, key=str.lower),
                "count": len(fixture.expected_ids),
            },
            ensure_ascii=False,
        )
    if suffix == "edit_verify":
        return json.dumps(
            {"changed": True, "line": fixture.target_line, "verified": True},
            ensure_ascii=False,
        )
    if suffix == "audit_reconcile":
        return json.dumps(fixture.expected_json or {}, ensure_ascii=False)
    if suffix == "json_constraints":
        return json.dumps(
            {"path": fixture.main_path, "value": fixture.main_content},
            ensure_ascii=False,
        )
    if suffix == "context_recall":
        return fixture.final_token
    if suffix == "path_params":
        return json.dumps({"total_lines": fixture.expected_total}, ensure_ascii=False)
    if suffix == "conflict":
        if fixture.conflict_mode == "none":
            return "已创建 report.txt。"
        if fixture.conflict_mode == "weak":
            return "已创建 report.txt；因文件名含大写，按约束改为小写。"
        return "存在冲突：两条内容约束无法同时满足。"
    if suffix == "01":
        path_key = "file_path" if spec.split == "dev" else "path"
        value_key = "raw_value" if spec.split == "dev" else "value"
        return json.dumps(
            {path_key: fixture.main_path, value_key: fixture.main_content},
            ensure_ascii=False,
        )
    if suffix == "02":
        return json.dumps({"files": sorted(fixture.files, key=str.lower)}, ensure_ascii=False)
    if suffix == "03":
        return json.dumps({"changed": True, "line": fixture.target_line})
    if suffix == "04":
        return json.dumps({"heading": fixture.heading, "missing_note": None}, ensure_ascii=False)
    if suffix == "05":
        return json.dumps(
            {
                "created": f"{fixture.out_dir}/{fixture.lower_name}",
                "reason": "文件名含大写，已改为全小写。",
            },
            ensure_ascii=False,
        )
    if suffix == "06":
        return fixture.final_token
    if suffix == "07":
        return json.dumps(
            {
                "ids": list(fixture.expected_ids),
                "values": list(fixture.expected_values),
                "total": fixture.expected_total,
                "missing": None,
                "source_count": fixture.source_count,
            },
            ensure_ascii=False,
        )
    raise ValueError(spec.task_id)


def oracle_calls(spec: UnifiedTaskSpec, fixture: Fixture) -> list[ToolCallRecord]:
    ok = ToolResult.success("ok")
    suffix = task_family(spec)
    if suffix == "policy_extract":
        return [ToolCallRecord("read_file", json.dumps({"path": fixture.main_path}), ok)]
    if suffix == "selective_aggregate":
        calls = [ToolCallRecord("list_dir", json.dumps({"path": spec.path_directory}), ok)]
        calls.extend(
            ToolCallRecord("read_file", json.dumps({"path": path}), ok)
            for path in fixture.expected_read_paths
        )
        return calls
    if suffix == "search_collect":
        directory = spec.path_directory
        calls = [ToolCallRecord("grep", json.dumps({"pattern": ".", "path": directory}), ok)]
        calls.extend(
            ToolCallRecord("read_file", json.dumps({"path": path}), ok)
            for path in fixture.expected_read_paths
        )
        return calls
    if suffix == "edit_verify":
        return [
            ToolCallRecord("read_file", json.dumps({"path": fixture.line_path}), ok),
            ToolCallRecord("edit_file", json.dumps({"path": fixture.line_path}), ok),
            ToolCallRecord("read_file", json.dumps({"path": fixture.line_path}), ok),
        ]
    if suffix == "audit_reconcile":
        calls = [ToolCallRecord("list_dir", json.dumps({"path": spec.path_directory}), ok)]
        calls.extend(
            ToolCallRecord("read_file", json.dumps({"path": path}), ok)
            for path in fixture.expected_read_paths
        )
        return calls
    if suffix == "json_constraints":
        return [ToolCallRecord("read_file", json.dumps({"path": fixture.main_path}), ok)]
    if suffix == "context_recall":
        return [
            ToolCallRecord("read_file", json.dumps({"path": path}), ok)
            for path in fixture.context_paths
        ]
    if suffix == "path_params":
        return [
            ToolCallRecord("read_file", json.dumps({"path": path}), ok)
            for path in fixture.expected_read_paths
        ]
    if suffix == "conflict":
        if fixture.conflict_mode == "strong":
            return []
        return [
            ToolCallRecord(
                "write_file",
                json.dumps({"path": f"{fixture.out_dir}/{fixture.lower_name}"}),
                ok,
            )
        ]
    if suffix == "01":
        return [ToolCallRecord("read_file", json.dumps({"path": fixture.main_path}), ok)]
    if suffix == "02":
        return [ToolCallRecord("list_dir", json.dumps({"path": "project/assets"}), ok)]
    if suffix == "03":
        return [
            ToolCallRecord("read_file", json.dumps({"path": fixture.line_path}), ok),
            ToolCallRecord("edit_file", json.dumps({"path": fixture.line_path}), ok),
        ]
    if suffix == "04":
        return [ToolCallRecord("read_file", json.dumps({"path": fixture.heading_path}), ok)]
    if suffix == "05":
        calls = [
            ToolCallRecord("read_file", json.dumps({"path": path}), ok)
            for path in fixture.context_paths
        ]
        calls.append(
            ToolCallRecord(
                "write_file",
                json.dumps({"path": f"{fixture.out_dir}/{fixture.lower_name}"}),
                ok,
            )
        )
        return calls
    if suffix == "06":
        return [
            ToolCallRecord("read_file", json.dumps({"path": fixture.notes_path}), ok),
            ToolCallRecord(
                "write_file",
                json.dumps({"path": fixture.notes_path}),
                ToolResult.failure(
                    "文件已存在，拒绝覆盖",
                    error_kind="tool_error",
                    validation_stage="execution",
                ),
            ),
            ToolCallRecord("edit_file", json.dumps({"path": fixture.notes_path}), ok),
        ]
    if suffix == "07":
        return [
            ToolCallRecord("read_file", json.dumps({"path": path}), ok)
            for path in fixture.data_paths
        ]
    raise ValueError(spec.task_id)


def run_self_test(specs: Sequence[UnifiedTaskSpec] | None = None) -> int:
    target_specs = tuple(specs) if specs is not None else TASK_SPECS
    if specs is None:
        validate_splits()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for spec in target_specs:
            workspace = root / spec.task_id
            workspace.mkdir()
            fixture = build_fixture(spec, workspace)
            suffix = task_family(spec)
            if suffix == "03":
                lines = list(fixture.snapshot_lines)
                lines[fixture.target_line - 1] = (
                    f"row-{fixture.target_line:02d}: {fixture.new_token}"
                )
                (workspace / fixture.line_path).write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
            if suffix == "edit_verify":
                lines = list(fixture.snapshot_lines)
                lines[fixture.target_line - 1] = (
                    f"row-{fixture.target_line:03d}: {fixture.new_token}"
                )
                (workspace / fixture.line_path).write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
            if suffix == "05":
                out = workspace / fixture.out_dir
                out.mkdir(parents=True, exist_ok=True)
                (out / fixture.lower_name).write_text(fixture.content_code, encoding="utf-8")
            if suffix == "06":
                (workspace / fixture.notes_path).write_text(fixture.final_token, encoding="utf-8")
            if suffix == "conflict" and fixture.conflict_mode in {"none", "weak"}:
                target = workspace / fixture.out_dir / fixture.lower_name
                target.write_text(fixture.content_code, encoding="utf-8")
            checks = judge(
                spec,
                fixture,
                workspace,
                oracle_reply(spec, fixture),
                oracle_calls(spec, fixture),
            )
            assert all(checks.values()), (spec.task_id, checks)
    print(f"self-test 通过：{len(target_specs)} 个任务夹具与判定逻辑符合预期")
    return 0


async def main_async(args: argparse.Namespace) -> int:
    if args.self_test:
        return run_self_test()
    if args.rollouts < 1:
        print("--rollouts 必须 >= 1", file=sys.stderr)
        return 2
    specs = tuple(TASK_SPECS) if args.split == "both" else select_tasks(args.split)
    if args.tasks:
        wanted = {item.strip() for item in args.tasks.split(",") if item.strip()}
        specs = tuple(item for item in specs if item.task_id in wanted)
        if not specs:
            print("没有匹配的任务", file=sys.stderr)
            return 2

    env_path = find_env_file(Path.cwd())
    if env_path is not None:
        load_env_file(env_path)
    provider = OpenAICompatProvider.from_env(
        temperature=args.sampling_temperature,
        top_p=args.sampling_top_p,
        seed=args.sampling_seed,
    )
    jobs = [(task, rollout) for task in specs for rollout in range(args.rollouts)]
    semaphore = asyncio.Semaphore(args.concurrency)

    async def worker(task: UnifiedTaskSpec, rollout: int) -> dict[str, Any]:
        async with semaphore:
            return await run_once(
                task,
                provider=provider,
                rollout_index=rollout,
                trace_dir=args.trace_dir,
                sampling_config=provider.sampling_config,
                system_variant=args.system_variant,
            )

    records: list[dict[str, Any]] = []
    for coro in asyncio.as_completed([worker(*job) for job in jobs]):
        row = await coro
        records.append(row)
        print(
            f"[{len(records)}/{len(jobs)}] {row['task_id']}#{row['rollout_index']:02d} "
            f"success={row.get('task_success')} steps={row.get('steps')} "
            f"error={row.get('error', '')}",
            flush=True,
        )
    await provider.aclose()
    records.sort(key=lambda item: (item["task_id"], item["rollout_index"]))

    summary = []
    for split in sorted({spec.split for spec in specs}):
        split_specs = tuple(spec for spec in specs if spec.split == split)
        split_records = [row for row in records if row["split"] == split]
        summary.append(write_outputs(args.out_dir, split_records, split_specs, split=split))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统一数据集 baseline runner")
    parser.add_argument("--split", choices=("dev", "test", "both"), default="both")
    parser.add_argument("--tasks", default="", help="逗号分隔的任务 ID；默认全部")
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--sampling-temperature", type=float, default=0.0)
    parser.add_argument("--sampling-top-p", type=float, default=None)
    parser.add_argument("--sampling-seed", type=int, default=20261019)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--system-variant", choices=("s0", "s1", "s1t", "s2"), default="s0")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
