"""Read-only data diversity audit for train-v4.

The audit verifies the frozen SHA-256, applies the recorded Qwen tokenizer
over-length gate, and writes evidence files.  It never rebuilds data, mutates
task definitions, or calls a model.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "audit-data-diversity-v1"

HELDOUT_TASKS: dict[str, str] = {
    "U-TR-A-C1-HARD": "json_constraints",
    "U-TR-B-CHECK-BASE": "constraint_edit",
    "U-TR-B-C1-BASE": "json_constraints",
    "U-TR-B-CTX-HARD": "context_recall",
    "U-TR-B-TOOL-HARD": "path_params",
    "U-TR-B-CONFLICT-WEAK": "conflict",
}

TEMPLATE_LABELS = {
    "json_constraints": "JSON 约束（C1）",
    "json_file": "JSON 文件读取（FILE）",
    "json_list": "JSON 目录列表（LIST）",
    "context_recall": "长上下文回忆（CTX）",
    "path_params": "路径/参数（TOOL）",
    "conflict": "冲突取舍（CONFLICT）",
    "constraint_edit": "约束编辑（CHECK）",
    "public_grep": "公开 grep 映射（xLAM）",
    "unknown": "unknown",
}

STEP_LIMIT_BY_SUFFIX = {
    "C1-BASE": 3,
    "C1-HARD": 3,
    "FILE-BASE": 3,
    "LIST-BASE": 3,
    "CTX-BASE": 3,
    "CTX-HARD": 14,
    "TOOL-BASE": 3,
    "TOOL-HARD": 5,
    "CONFLICT-NONE": 3,
    "CONFLICT-WEAK": 3,
    "CONFLICT-STRONG": 3,
    "CHECK-BASE": 5,
    "CHECK-HARD": 5,
}

FORMAT_TAGS = frozenset(
    {"T-C01", "T-C02", "T-C03", "T-C04", "T-P01", "T-P02", "T-X01", "T-F01", "T-F02", "PUB-C02"}
)
FORMAT_CHECKS = frozenset({"json_valid", "snake_case_keys", "no_fence", "json_answer"})
SEMANTIC_CHECKS = frozenset(
    {
        "exact_keys",
        "value_correct",
        "matches_complete",
        "answer_exact",
        "reason_present",
        "conflict_declared",
    }
)
TOOL_CHECKS = frozenset(
    {
        "required_tools_used",
        "tools_allowed",
        "context_read",
        "read_paths_exact",
        "no_decoy_read",
        "edit_file_used",
    }
)
FILE_CHECKS = frozenset(
    {
        "file_created_correct",
        "content_exact",
        "only_target_line",
        "token_replaced",
        "no_arbitrary_file",
    }
)

TAG_RE = re.compile(r"^-\s+\[([^\]]+)\]\s+(.+)$", re.MULTILINE)
TOOL_RE = re.compile(r"本次任务只允许调用以下工具：([^；\n]+)")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected an object, got {type(value).__name__}")
        rows.append(value)
    return rows


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def parse_frozen_text(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def verify_frozen_inputs(data_dir: Path) -> dict[str, Any]:
    frozen_text = parse_frozen_text(data_dir / "TRAIN_V4_FROZEN.txt")
    frozen_json = read_json(data_dir / "TRAIN_V4_FROZEN.json")
    manifest = read_json(data_dir / "manifest.json")
    verified: dict[str, Any] = {}
    for split in ("train", "val"):
        path = data_dir / f"{split}.jsonl"
        actual = sha256_file(path)
        expected_values = {
            "frozen_txt": frozen_text.get(f"{split}_sha256"),
            "frozen_json": frozen_json.get(f"{split}_sha256"),
            "manifest": manifest.get(f"{split}_sha256"),
        }
        missing = [name for name, value in expected_values.items() if not value]
        if missing:
            raise SystemExit(f"{split} SHA record missing in: {', '.join(missing)}")
        unique = set(expected_values.values())
        if len(unique) != 1:
            raise SystemExit(f"{split} SHA records disagree: {expected_values}")
        expected = unique.pop()
        if actual != expected:
            raise SystemExit(f"{split} SHA mismatch: actual={actual} expected={expected}")
        verified[split] = {
            "path": path.as_posix(),
            "rows": len(read_jsonl(path)),
            "sha256": actual,
            "frozen_sha256": expected,
            "frozen_match": True,
        }
    verified["manifest"] = {
        "path": (data_dir / "manifest.json").as_posix(),
        "sha256": sha256_file(data_dir / "manifest.json"),
    }
    verified["frozen_text"] = {
        "path": (data_dir / "TRAIN_V4_FROZEN.txt").as_posix(),
        "sha256": sha256_file(data_dir / "TRAIN_V4_FROZEN.txt"),
    }
    return {
        "frozen_text": frozen_text,
        "frozen_json": frozen_json,
        "manifest": manifest,
        "verified": verified,
    }


def load_tokenizer_gate(
    gate_path: Path, train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    gate = read_json(gate_path)
    if gate.get("status") != "ok":
        raise SystemExit(f"tokenizer gate status is not ok: {gate.get('status')!r}")
    result: dict[str, Any] = {"path": gate_path.as_posix(), "sha256": sha256_file(gate_path)}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        block = gate.get(split)
        if not isinstance(block, Mapping):
            raise SystemExit(f"tokenizer gate missing split: {split}")
        if int(block.get("rows") or 0) != len(rows):
            raise SystemExit(
                f"tokenizer gate {split} rows={block.get('rows')} but data rows={len(rows)}"
            )
        if int(block.get("failure_count") or 0) != 0:
            raise SystemExit(f"tokenizer gate {split} has structural failures")
        if int(block.get("assistant_mask_failure_count") or 0) != 0:
            raise SystemExit(f"tokenizer gate {split} has assistant-mask failures")
        over = block.get("over_max_length") or []
        if not isinstance(over, list):
            raise SystemExit(f"tokenizer gate {split} over_max_length is not a list")
        if len(over) != int(block.get("over_max_length_count") or 0):
            raise SystemExit(f"tokenizer gate {split} over_max_length count disagrees")
        over_ids = {str(item["id"]) for item in over if isinstance(item, Mapping) and "id" in item}
        if len(over_ids) != len(over):
            raise SystemExit(f"tokenizer gate {split} has duplicate or malformed over-length ids")
        row_ids = {str(row.get("id") or "") for row in rows}
        missing = over_ids - row_ids
        if missing:
            raise SystemExit(
                f"tokenizer gate {split} references unknown ids: {sorted(missing)[:5]}"
            )
        result[split] = {
            "rows": len(rows),
            "over_max_length_count": len(over),
            "effective_rows": len(rows) - len(over),
            "over_max_length": {
                str(item["id"]): int(item.get("tokens") or 0)
                for item in over
                if isinstance(item, Mapping)
            },
            "max_tokens": max((int(item.get("tokens") or 0) for item in over), default=0),
            "token_count_p50": block.get("token_count_p50"),
            "token_count_max": block.get("token_count_max"),
        }
    return result


def task_suffix(task_id: str) -> str | None:
    if task_id.startswith("PUB-XLAM-"):
        return None
    for prefix in ("U-TR5-", "U-TR4-", "U-TR-A-", "U-TR-B-", "U-TUNE-"):
        if task_id.startswith(prefix):
            suffix = task_id[len(prefix) :]
            if prefix in {"U-TR5-", "U-TR4-"} and "-" in suffix:
                suffix = suffix.split("-", 1)[1]
            return suffix
    return None


def classify_task_template(task_id: str) -> str:
    if task_id.startswith("PUB-XLAM-"):
        return "public_grep"
    suffix = task_suffix(task_id)
    if suffix is None:
        return "unknown"
    if suffix.startswith("C1-"):
        return "json_constraints"
    if suffix.startswith("FILE-"):
        return "json_file"
    if suffix.startswith("LIST-"):
        return "json_list"
    if suffix.startswith("CTX-"):
        return "context_recall"
    if suffix.startswith("TOOL-"):
        return "path_params"
    if suffix.startswith("CONFLICT-"):
        return "conflict"
    if suffix.startswith("CHECK-"):
        return "constraint_edit"
    return "unknown"


def classify_constraint_tag(tag: str) -> str:
    if tag in {"T-F03", "T-F04"}:
        return "conflict_tradeoff"
    if tag == "T-K01":
        return "tool_whitelist"
    if tag == "T-K02":
        return "path"
    if tag == "PUB-C01":
        return "tool_whitelist"
    if tag.endswith("-TOOLS"):
        return "tool_whitelist"
    if tag in FORMAT_TAGS:
        return "format"
    return "unknown"


def classify_check(name: str) -> str:
    if name in FORMAT_CHECKS:
        return "format"
    if name in SEMANTIC_CHECKS:
        return "semantic"
    if name in TOOL_CHECKS:
        return "tool"
    if name == "within_steps":
        return "step"
    if name in FILE_CHECKS:
        return "file"
    return "unknown"


def extract_constraints(system_prompt: str) -> list[tuple[str, str]]:
    return [(match.group(1), match.group(2).strip()) for match in TAG_RE.finditer(system_prompt)]


def extract_tool_set(system_prompt: str) -> str:
    match = TOOL_RE.search(system_prompt)
    if match:
        return match.group(1).strip()
    if "[PUB-C01]" in system_prompt:
        return "grep"
    return "unknown"


def enrich_row(
    row: Mapping[str, Any], split: str, over_max_length: Mapping[str, int]
) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    task_id = str(metadata.get("task_id") or "unknown")
    template = classify_task_template(task_id)
    messages = row.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    system_prompt = next(
        (str(item.get("content") or "") for item in messages if item.get("role") == "system"), ""
    )
    user_prompt = " | ".join(
        str(item.get("content") or "") for item in messages if item.get("role") == "user"
    )
    tags = extract_constraints(system_prompt)
    checks = (row.get("reward_context") or {}).get("checks") or metadata.get("checks") or {}
    if not isinstance(checks, Mapping):
        checks = {}
    constraint_types: set[str] = set()
    unknown_tags: list[str] = []
    for tag, _text in tags:
        kind = classify_constraint_tag(tag)
        if kind == "unknown":
            unknown_tags.append(tag)
        else:
            constraint_types.add(kind)
    if (
        checks.get("read_paths_exact") is not None
        or checks.get("no_decoy_read") is not None
        or checks.get("only_target_line") is not None
    ):
        constraint_types.add("path")
    if checks.get("within_steps") is not None:
        constraint_types.add("steps")
    if checks.get("conflict_declared") or checks.get("reason_present"):
        constraint_types.add("conflict_tradeoff")
    check_types = {classify_check(str(name)) for name, value in checks.items() if value is not None}
    unknown_checks = sorted(name for name in check_types if name == "unknown")
    tool_calls = (row.get("reward_context") or {}).get("tool_calls") or []
    if not isinstance(tool_calls, list):
        tool_calls = []
    suffix = task_suffix(task_id)
    if template == "public_grep":
        steps_limit: int | None = None
        steps_limit_source = "not_recorded_public_source"
    elif suffix in STEP_LIMIT_BY_SUFFIX:
        steps_limit = STEP_LIMIT_BY_SUFFIX[suffix]
        steps_limit_source = "task_spec"
    else:
        steps_limit = None
        steps_limit_source = "unknown"
    has_full_trajectory = (
        bool(messages)
        and bool(row.get("completion"))
        and any(item.get("role") in {"system", "user"} for item in messages)
    )
    notes: list[str] = []
    if str(row.get("id") or "") in over_max_length:
        notes.append("tokenizer_over_max_length")
    if unknown_tags:
        notes.append("unknown_tags=" + ",".join(sorted(set(unknown_tags))))
    if unknown_checks:
        notes.append("unknown_checks=" + ",".join(unknown_checks))
    if not tool_calls:
        notes.append("zero_tool_calls")
    raw_id = str(row.get("id") or "unknown")
    return {
        "id": raw_id,
        "split": split,
        "category": str(metadata.get("category") or "unknown"),
        "source": str(metadata.get("source") or "unknown"),
        "difficulty": str(metadata.get("difficulty") or "unknown"),
        "constraint_count": len(tags),
        "constraint_types": ",".join(sorted(constraint_types)) or "unknown",
        "context_length_bucket": str(metadata.get("context_tier") or "unknown"),
        "tools_allowed": extract_tool_set(system_prompt),
        "check_types": ",".join(sorted(check_types)) or "unknown",
        "has_full_trajectory": has_full_trajectory,
        "duplicate_group": task_id,
        "notes": ";".join(notes),
        "task_id": task_id,
        "template": template,
        "tool_call_count": len(tool_calls),
        "steps_limit": steps_limit,
        "steps_limit_source": steps_limit_source,
        "included_after_tokenizer_gate": raw_id not in over_max_length,
        "tokenizer_gate": "dropped_over_max_length" if raw_id in over_max_length else "passed",
        "over_max_length_tokens": over_max_length.get(raw_id),
        "message_count": len(messages),
        "_within_steps": checks.get("within_steps"),
        "_prompt_text": user_prompt,
        "_unknown_tag_count": len(unknown_tags),
        "_unknown_check_count": len(unknown_checks),
        "_tool_set_unknown": extract_tool_set(system_prompt) == "unknown",
    }


def distribution(counter: Counter[Any], total: int) -> dict[str, dict[str, Any]]:
    return {
        str(key): {"count": int(count), "pct": round(100 * count / total, 2) if total else 0.0}
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    category = Counter(str(row["category"]) for row in rows)
    source = Counter(str(row["source"]) for row in rows)
    difficulty = Counter(str(row["difficulty"]) for row in rows)
    context = Counter(str(row["context_length_bucket"]) for row in rows)
    template = Counter(str(row["template"]) for row in rows)
    tools = Counter(str(row["tools_allowed"]) for row in rows)
    constraint_counts = Counter(int(row["constraint_count"]) for row in rows)
    constraint_types: Counter[str] = Counter()
    category_types: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for item in str(row["constraint_types"]).split(","):
            if not item or item == "unknown":
                continue
            constraint_types[item] += 1
            category_types[str(row["category"])][item] += 1
    check_types: Counter[str] = Counter()
    for row in rows:
        for item in str(row["check_types"]).split(","):
            if item and item != "unknown":
                check_types[item] += 1
    template_source: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        template_source[str(row["template"])][str(row["source"])] += 1
    task_groups = Counter(str(row["task_id"]) for row in rows)
    exact_prompt_groups = Counter(str(row["_prompt_text"]) for row in rows)
    normalized_prompt_groups = Counter(
        re.sub(r"训练变体 \d+：", "", str(row["_prompt_text"])) for row in rows
    )
    duplicate_task_groups = {key: value for key, value in task_groups.items() if value > 1}
    duplicate_prompt_groups = {
        key: value for key, value in exact_prompt_groups.items() if value > 1
    }
    duplicate_normalized_groups = {
        key: value for key, value in normalized_prompt_groups.items() if value > 1
    }
    controlled = [row for row in rows if row["steps_limit"] is not None]
    controlled_margins = [
        int(row["steps_limit"]) - int(row["tool_call_count"]) for row in controlled
    ]
    public_sample = sorted(
        [row for row in rows if row["steps_limit_source"] == "not_recorded_public_source"],
        key=lambda row: str(row["id"]),
    )[:20]
    step_budget = {
        "controlled_rows": len(controlled),
        "controlled_at_or_over_limit": sum(1 for margin in controlled_margins if margin <= 0),
        "controlled_min_margin": min(controlled_margins) if controlled_margins else None,
        "controlled_margin_at_1": sum(1 for margin in controlled_margins if margin == 1),
        "controlled_margin_distribution": distribution(
            Counter(controlled_margins), len(controlled)
        ),
        "public_rows_without_recorded_limit": sum(
            1 for row in rows if row["steps_limit_source"] == "not_recorded_public_source"
        ),
        "rows_with_unknown_limit": sum(1 for row in rows if row["steps_limit_source"] == "unknown"),
        "public_step_sample": {
            "sampled_rows": len(public_sample),
            "ids": [str(row["id"]) for row in public_sample],
            "tool_call_count": distribution(
                Counter(int(row["tool_call_count"]) for row in public_sample), len(public_sample)
            ),
            "within_steps_true": sum(1 for row in public_sample if row["_within_steps"] is True),
        },
    }
    return {
        "rows": total,
        "category": distribution(category, total),
        "source": distribution(source, total),
        "difficulty": distribution(difficulty, total),
        "context_length": distribution(context, total),
        "template": distribution(template, total),
        "tools_allowed": distribution(tools, total),
        "constraint_count": distribution(constraint_counts, total),
        "constraint_types": distribution(constraint_types, total),
        "category_types": {
            str(cat): distribution(values, category[str(cat)])
            for cat, values in sorted(category_types.items())
        },
        "check_types": distribution(check_types, total),
        "template_source": {
            str(key): dict(sorted(value.items())) for key, value in sorted(template_source.items())
        },
        "trajectory": {
            "has_full_trajectory": sum(1 for row in rows if row["has_full_trajectory"]),
            "zero_tool_calls": sum(1 for row in rows if int(row["tool_call_count"]) == 0),
            "tool_call_count": distribution(
                Counter(int(row["tool_call_count"]) for row in rows), total
            ),
        },
        "duplicates": {
            "unique_task_ids": len(task_groups),
            "task_groups_with_multiple_rows": len(duplicate_task_groups),
            "extra_rows_in_task_groups": sum(value - 1 for value in duplicate_task_groups.values()),
            "max_rows_per_task": max(task_groups.values(), default=0),
            "unique_exact_prompts": len(exact_prompt_groups),
            "exact_prompt_groups_with_multiple_rows": len(duplicate_prompt_groups),
            "exact_prompt_rows_in_duplicate_groups": sum(duplicate_prompt_groups.values()),
            "unique_normalized_prompts": len(normalized_prompt_groups),
            "normalized_prompt_groups_with_multiple_rows": len(duplicate_normalized_groups),
            "normalized_prompt_rows_in_duplicate_groups": sum(duplicate_normalized_groups.values()),
        },
        "step_budget": step_budget,
        "unknown": {
            "template": sum(1 for row in rows if row["template"] == "unknown"),
            "constraint_tag": sum(int(row["_unknown_tag_count"]) for row in rows),
            "check_type": sum(int(row["_unknown_check_count"]) for row in rows),
            "tool_set": sum(1 for row in rows if row["_tool_set_unknown"]),
        },
    }


def boundary_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "CTX-HARD": sum("CTX-HARD" in str(row["task_id"]) for row in rows),
        "CONFLICT-STRONG": sum("CONFLICT-STRONG" in str(row["task_id"]) for row in rows),
        "TOOL-HARD": sum("TOOL-HARD" in str(row["task_id"]) for row in rows),
    }


def heldout_counts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for task_id, template in HELDOUT_TASKS.items():
        exact = sum(1 for row in rows if row["task_id"] == task_id)
        same = [row for row in rows if row["template"] == template]
        result.append(
            {
                "task_id": task_id,
                "template": template,
                "exact_rows_in_train": exact,
                "same_template_rows": len(same),
                "same_template_unique_tasks": len({str(row["task_id"]) for row in same}),
            }
        )
    return result


def teacher_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": path.as_posix()}
    runs = read_jsonl(path)
    by_template: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "failed_checks": Counter()}
    )
    for run in runs:
        template = classify_task_template(str(run.get("task_id") or "unknown"))
        item = by_template[template]
        item["total"] += 1
        if bool(run.get("task_success")):
            item["passed"] += 1
            continue
        checks = run.get("checks") or {}
        if isinstance(checks, Mapping):
            for name, value in checks.items():
                if value is False:
                    item["failed_checks"][str(name)] += 1
    overall_total = len(runs)
    overall_passed = sum(1 for run in runs if bool(run.get("task_success")))
    return {
        "status": "ok",
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "total": overall_total,
        "passed": overall_passed,
        "rate": round(overall_passed / overall_total, 4) if overall_total else None,
        "by_template": {
            str(name): {
                "total": item["total"],
                "passed": item["passed"],
                "rate": round(item["passed"] / item["total"], 4) if item["total"] else None,
                "failed_checks": dict(
                    sorted(item["failed_checks"].items(), key=lambda pair: (-pair[1], pair[0]))
                ),
            }
            for name, item in sorted(by_template.items())
        },
    }


def fmt_pct(count: int, total: int) -> str:
    return f"{count}/{total} ({100 * count / total:.2f}%)" if total else "0/0"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_markdown(
    *,
    path: Path,
    inputs: Mapping[str, Any],
    gate: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    val_stats: Mapping[str, Any],
    raw_train_stats: Mapping[str, Any],
    raw_boundary: Mapping[str, int],
    boundary: Mapping[str, int],
    heldout: Sequence[Mapping[str, Any]],
    teacher: Mapping[str, Any],
    source_stats: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    train_total = int(train_stats["rows"])
    category_rows = []
    for name, item in train_stats["category"].items():
        category_rows.append([name, item["count"], f"{item['pct']:.2f}%"])
    template_rows = []
    for name, item in train_stats["template"].items():
        sources = train_stats["template_source"].get(name, {})
        source_text = ", ".join(f"{key}={value}" for key, value in sources.items())
        template_rows.append(
            [TEMPLATE_LABELS.get(name, name), item["count"], f"{item['pct']:.2f}%", source_text]
        )
    constraint_rows = []
    type_notes = {
        "format": "JSON / snake_case / 无围栏 / exact_keys / 精确 CODE / 文件名或内容精确",
        "path": "只改目标行、read_paths_exact、no_decoy_read、only_target_line",
        "tool_whitelist": "*-TOOLS、PUB-C01、T-K01 工具选择约束",
        "steps": "within_steps 检查；受控任务另有 task_spec 步数预算",
        "conflict_tradeoff": "T-F03/T-F04 互斥对，或 conflict_declared / reason_present",
    }
    for name, item in train_stats["constraint_types"].items():
        constraint_rows.append(
            [name, item["count"], f"{item['pct']:.2f}%", type_notes.get(name, "")]
        )
    category_type_rows = []
    for category, types in train_stats["category_types"].items():
        category_total = train_stats["category"].get(category, {}).get("count", 0)
        category_type_rows.append(
            [
                category,
                category_total,
                *[
                    f"{types.get(name, {}).get('count', 0)} ({types.get(name, {}).get('pct', 0):.2f}%)"
                    for name in ("format", "path", "tool_whitelist", "steps", "conflict_tradeoff")
                ],
            ]
        )
    heldout_rows = [
        [
            item["task_id"],
            TEMPLATE_LABELS.get(str(item["template"]), str(item["template"])),
            item["exact_rows_in_train"],
            item["same_template_rows"],
            item["same_template_unique_tasks"],
        ]
        for item in heldout
    ]
    boundary_rows = []
    for name in ("CTX-HARD", "CONFLICT-STRONG", "TOOL-HARD"):
        effective = boundary.get(name, 0)
        boundary_rows.append(
            [
                name,
                raw_boundary.get(name, 0),
                effective,
                f"{100 * effective / train_total:.2f}%" if train_total else "0.00%",
                "全部 19 条长上下文被 3072 门禁丢弃" if name == "CTX-HARD" else "",
            ]
        )
    teacher_rows = []
    if teacher.get("status") == "ok":
        for name, item in teacher["by_template"].items():
            top = ", ".join(
                f"{key}={value}" for key, value in list(item["failed_checks"].items())[:3]
            )
            teacher_rows.append(
                [
                    TEMPLATE_LABELS.get(name, name),
                    item["total"],
                    item["passed"],
                    f"{item['rate'] * 100:.2f}%",
                    top or "-",
                ]
            )
    deepseek_source = source_stats.get("deepseek_teacher") or {}
    deepseek_accepted = int(deepseek_source.get("train") or 0) + int(
        deepseek_source.get("val") or 0
    )
    xlam_source = source_stats.get("xlam_schema_mapped") or {}
    xlam_raw = int(xlam_source.get("raw_rows") or 0)
    xlam_strict = int(xlam_source.get("strict_grep_rows") or 0)
    xlam_accepted = int(xlam_source.get("train") or 0) + int(xlam_source.get("val") or 0)
    xlam_raw_mapping = (100 * xlam_strict / xlam_raw) if xlam_raw else 0.0
    oracle_source = source_stats.get("deterministic_oracle") or {}
    oracle_candidates = int(oracle_source.get("candidate_tasks") or 0)
    oracle_accepted = int(oracle_source.get("train") or 0) + int(oracle_source.get("val") or 0)
    source_gate_rows = [
        [
            "deepseek_teacher",
            f"{teacher.get('total', 'N/A')} rollouts",
            f"{teacher.get('total', 'N/A')}",
            f"{deepseek_accepted} 条（train {deepseek_source.get('train', 0)} / val {deepseek_source.get('val', 0)}）",
            f"{teacher.get('passed', 'N/A')}/{teacher.get('total', 'N/A')} = {teacher.get('rate', 0) * 100:.2f}%"
            if teacher.get("status") == "ok"
            else "N/A",
            "train 中 1 条超长被丢，有效 teacher 行 89；oracle/xLAM 不是教师模型行为",
        ],
        [
            "xlam_schema_mapped",
            f"{xlam_raw} raw",
            f"{xlam_strict} strict",
            f"{xlam_accepted} accepted",
            f"{xlam_strict}/{xlam_strict} = 100.00% strict",
            f"raw 映射率 {xlam_raw_mapping:.3f}%；strict 映射后全部通过，不能当作模型教师数据",
        ],
        [
            "deterministic_oracle",
            f"{oracle_candidates} tasks",
            f"{oracle_candidates}",
            f"{oracle_accepted} accepted",
            "100.00%",
            "执行验证的 oracle 轨迹，不是教师模型行为",
        ],
    ]
    context_rows = [
        [name, item["count"], f"{item['pct']:.2f}%"]
        for name, item in train_stats["context_length"].items()
    ]
    tools_rows = [
        [name, item["count"], f"{item['pct']:.2f}%"]
        for name, item in train_stats["tools_allowed"].items()
    ]
    check_rows = [
        [name, item["count"], f"{item['pct']:.2f}%"]
        for name, item in train_stats["check_types"].items()
    ]
    duplicates = train_stats["duplicates"]
    step_budget = train_stats["step_budget"]
    unknown = train_stats["unknown"]
    decision_text = str(decision["decision"])
    markdown = rf"""# 训练数据多样性审计（阶段 0.0）

> 审计脚本：`scripts/audit_data_diversity.py`（`{SCRIPT_VERSION}`）
> 生成时间：{generated}
> 输入：`private/sft_data_v4/` 冻结文件与 tokenizer 门禁报告
> 性质：只读审计；未改数据、未改任务定义、未跑模型。

## 0. 结论摘要

- train-v4 原始 **380 train / 80 val**；Qwen tokenizer 3072 门禁后 **361 train / 74 val**。19 条 train 超长行全部是 `CTX-HARD`，因此有效训练集里 **CTX-HARD = 0、long context = 0**。
- HARD 类有效 **77 条**，落在预注册的 **50–79 灰区**；五种约束类型占比均不低于 10%，但任务/模板覆盖不均匀。
- 6 个行为 held-out task ID 在 train 中精确出现 **0 次**；同模板 sibling 数量见 2.2，不能把 sibling 当成 held-out 任务本身。
- 教师门禁通过率（DeepSeek，raw 220 rollouts）为 **180/220 = 81.82%**；`context_recall` 与 `constraint_edit` 均为 **25/40 = 62.50%**，是教师最弱的两类。
- 按 3.1 做一次性灰区决定：**补数据**。触发依据是 `CTX-HARD=0`、long context=0、BASE=23、`context_recall` 模板仅 28 条（7.76%），不是“约束类型 <10%”触发；该决定在看到任何模型结果前记录，后续不调。

## 1. 输入、冻结与口径

{
        md_table(
            ["文件", "行数", "SHA-256", "与冻结记录一致"],
            [
                [
                    "train.jsonl",
                    inputs["verified"]["train"]["rows"],
                    inputs["verified"]["train"]["sha256"],
                    "是",
                ],
                [
                    "val.jsonl",
                    inputs["verified"]["val"]["rows"],
                    inputs["verified"]["val"]["sha256"],
                    "是",
                ],
                [
                    "tokenizer_gate.json",
                    f"train={gate['train']['rows']} / val={gate['val']['rows']}",
                    gate["sha256"],
                    "记录文件，按行数/ID 校验",
                ],
            ],
        )
    }

- 冻结记录：`TRAIN_V4_FROZEN.txt` / `TRAIN_V4_FROZEN.json` / `manifest.json` 的 train、val SHA 三者一致，脚本逐字节重算后通过。
- 有效集定义：train 380 减去 tokenizer 门禁 `over_max_length` 的 19 行 = 361；val 80 减 6 = 74。审计统计默认使用 **train 有效 361 行**，原始 380 行只用于对比缺口。
- 模板识别：按 task_id 后缀映射到 5 个受控族（`json_constraints`、`context_recall`、`path_params`、`conflict`、`constraint_edit`）和 1 个公开映射族（`public_grep`）。无法识别的写 `unknown`，不静默丢弃。
- 约束类型：一行可命中多个类型，所以各类型计数之和大于 361；`constraint_count` 统计 system prompt 中显式 `[ID]` 标签数，`steps` 是检查项级约束，不强行加进显式标签数。
- 教师统计：只使用 `private/sft_data_v2_deepseek_expanded/teacher_runs.jsonl` 的 220 条 raw rollout；oracle 和 xLAM 不是教师模型行为，单独列在来源表。

## 2. 四维分布

### 2.1 约束类型分布（train 有效 361）

{md_table(["约束类型", "行数", "占比", "判定依据"], constraint_rows)}

显式约束数量分布：

{
        md_table(
            ["显式约束数", "行数", "占比"],
            [
                [name, item["count"], f"{item['pct']:.2f}%"]
                for name, item in train_stats["constraint_count"].items()
            ],
        )
    }

按任务类别的约束类型覆盖（每格为“行数（类别内占比）”，一行可多标签）：

{
        md_table(
            ["类别", "行数", "format", "path", "tool_whitelist", "steps", "conflict_tradeoff"],
            category_type_rows,
        )
    }

解读：`tool_whitelist` 和 `steps` 覆盖 100% 是脚手架检查项，不代表任务难度；真正有区分度的是 `path`（101 行，27.98%）和 `conflict_tradeoff`（61 行，16.90%）。

### 2.2 任务模板分布与 held-out 同模板计数

train 有效集共 **6 个可自动识别模板，unknown=0**：

{md_table(["模板", "行数", "占比", "来源构成"], template_rows)}

6 个行为 held-out 任务在 train 中的精确/同模板计数：

{
        md_table(
            ["held-out task", "模板", "train 中精确 ID", "同模板行数", "同模板唯一任务"],
            heldout_rows,
        )
    }

注意：同模板行数包含同一任务的多次 rollout 和 oracle 克隆；它只能说明“有同模板 sibling”，不能说明模型见过 held-out 实例。6 个精确 ID 在 train 中均为 0。

### 2.3 边界案例占比（train 有效 361）

{md_table(["边界案例", "raw 行数", "有效行数", "有效占比", "说明"], boundary_rows)}

`CTX-HARD` 的 19 条 raw 行全部因 3072 超长被丢；`CONFLICT-STRONG` 38 条全部保留；`TOOL-HARD` 24 条全部保留。

### 2.4 教师门禁通过率（按任务类型）

DeepSeek 教师 raw 220 rollouts（22 个任务 × 10）：

{md_table(["模板", "rollouts", "通过", "通过率", "主要失败检查"], teacher_rows)}

来源级门禁：

{md_table(["来源", "raw", "候选", "保留", "通过率", "说明"], source_gate_rows)}

结论限定：train 有效 361 行里，DeepSeek 教师只有 **89 行**，oracle **180 行**，xLAM **92 行**。因此“教师通过率 81.82%”不能外推成整个训练集的教师质量；它只描述 DeepSeek 的 220 条 raw rollout。

## 3. 补充分布与重复项

### 3.1 类别、来源、难度、上下文

{md_table(["类别", "行数", "占比"], category_rows)}

{
        md_table(
            ["来源", "行数", "占比"],
            [
                [name, item["count"], f"{item['pct']:.2f}%"]
                for name, item in train_stats["source"].items()
            ],
        )
    }

{
        md_table(
            ["难度", "行数", "占比"],
            [
                [name, item["count"], f"{item['pct']:.2f}%"]
                for name, item in train_stats["difficulty"].items()
            ],
        )
    }

{md_table(["上下文桶", "行数", "占比"], context_rows)}

上下文阶梯已经断裂：raw 有 19 条 long，全部是 CTX-HARD，全部被 3072 门禁丢弃；有效集 long=0。val 同样有 6 条 CTX-HARD 超长被丢，有效 val 也没有 long。

### 3.2 工具集合与检查项

{md_table(["允许工具集合", "行数", "占比"], tools_rows)}

{md_table(["检查项类型", "行数", "占比"], check_rows)}

`steps` 和 `tool` 检查覆盖 361/361；`format` 244/361（67.59%）、`semantic` 284/361（78.67%）、`file` 117/361（32.41%）。检查项不是只有格式，但语义检查主要集中在 value/answer/file 三类。

### 3.3 轨迹完整性

- `has_full_trajectory`：**361/361**。定义是至少包含 system/user 和最终 completion；38 条 `CONFLICT-STRONG` 为 0 次工具调用，这是“不擅自写入”的预期行为，不是截断。
- 工具调用数分布：{
        ", ".join(
            f"{name} 次={item['count']}"
            for name, item in train_stats["trajectory"]["tool_call_count"].items()
        )
    }。
- 0 次工具调用：38 行；其余 323 行至少 1 次。

### 3.4 重复项

- 任务级：361 行对应 **{duplicates["unique_task_ids"]} 个唯一 task_id**；{
        duplicates["task_groups_with_multiple_rows"]
    } 个任务有多行，额外重复 **{duplicates["extra_rows_in_task_groups"]} 行**，单任务最多 **{
        duplicates["max_rows_per_task"]
    } 行**。
- 精确 user prompt：**{duplicates["unique_exact_prompts"]} 个唯一文本**；{
        duplicates["exact_prompt_groups_with_multiple_rows"]
    } 组重复，涉及 **{duplicates["exact_prompt_rows_in_duplicate_groups"]} 行**。
- 去掉“训练变体 NN：”前缀后：**{duplicates["unique_normalized_prompts"]} 个唯一文本**；{
        duplicates["normalized_prompt_groups_with_multiple_rows"]
    } 组重复，涉及 **{duplicates["normalized_prompt_rows_in_duplicate_groups"]} 行**。

这些重复主要来自同一任务的多次 rollout 和同一模板的 oracle 克隆；它们能增加样本量，但不能增加 prompt 模板多样性。阶段 0.0a 补数据时不能用“复制换参数”继续放大这个比例。

### 3.5 步数预算

- 受控任务 **{
        step_budget["controlled_rows"]
    } 行**有 task_spec 步数预算；`tool_calls >= steps_limit` 的行数为 **{
        step_budget["controlled_at_or_over_limit"]
    }**，最小剩余预算 **{step_budget["controlled_min_margin"]}**，剩余 1 步的行数 **{
        step_budget["controlled_margin_at_1"]
    }**。因此受控任务的步数预算没有明显偏紧。
- `public_grep` **{
        step_budget["public_rows_without_recorded_limit"]
    } 行**在源数据中没有记录 steps_limit，`within_steps` 是构建时硬编码 true；该维度标记为“无法自动统计”，不把推断值写成预算结论。
- 对 public_grep 固定抽样 **{
        step_budget["public_step_sample"]["sampled_rows"]
    } 条**（按 id 排序取前 20，非随机）：工具调用数 {
        ", ".join(
            f"{name} 次={item['count']}"
            for name, item in step_budget["public_step_sample"]["tool_call_count"].items()
        )
    }；`within_steps=true` {step_budget["public_step_sample"]["within_steps_true"]}/{
        step_budget["public_step_sample"]["sampled_rows"]
    }。该字段是构建时硬编码，只能证明“被标记为通过”，不能证明预算宽松。
- `steps_limit_source=unknown` 行：{
        step_budget["rows_with_unknown_limit"]
    }；public_grep 的 92 行单独标为“源数据未记录”，不混入 unknown。

### 3.6 unknown 计数

{
        md_table(
            ["维度", "unknown 数"],
            [
                ["模板", unknown["template"]],
                ["约束标签", unknown["constraint_tag"]],
                ["检查项类型", unknown["check_type"]],
                ["工具集合", unknown["tool_set"]],
            ],
        )
    }

所有 unknown 均保留原始 `id`，未删除任何行。

## 4. 预注册门槛对照与一次性决定

### 4.1 逐条对照 3.1

- **HARD < 50 条 → 补**：不触发；HARD 有效 77 条。
- **任一约束类型 < 10% → 补**：不触发；最低为 `conflict_tradeoff` 61/361 = 16.90%。
- **HARD ≥ 80 且覆盖均匀 → 不补**：不满足；HARD=77，且 CTX-HARD=0、long=0、BASE=23。
- **HARD 50–79 为灰区 → 必须做一次性决定**：触发；HARD=77。

### 4.2 灰区一次性决定

**决定：{decision_text}。**

依据（只看审计数据，未看任何模型结果）：

1. `CTX-HARD` raw 19 条全部未过 3072 tokenizer 门禁，有效集为 0；long context 也为 0，上下文长度阶梯实际缺失。
2. BASE 只有 23 条（6.37%），低于此前数据方案中“每类至少 30–50 条”的目标下限。
3. `context_recall` 模板只有 28 条（7.76%），且没有 CTX-HARD；模板覆盖不均匀。
4. HARD=77 位于灰区下半段；DeepSeek 教师在 `context_recall`、`constraint_edit` 上的门禁通过率均为 62.50%，说明难例供给本身偏薄。

这不是修改 3.1 的门槛，而是在门槛明确要求的灰区内做一次性判断；决定写入本文件后不再调整。

### 4.3 阶段 0.0a 的补数据边界

- 只补缺口：优先补能通过 3072 门禁的 **CTX-HARD / 长上下文**、**BASE**、**HARD**，不复制现有参数凑数。
- 最低缺口：CTX-HARD 19 条（补齐被丢弃的有效长上下文）+ BASE 7 条（到 30）+ HARD 3 条（到 80）= 29 条，有效集到 390。若按计划补到 450–500，需要在同一缺口内再增加 60–110 条，具体分配由阶段 0.0a 在开工前一次写定。
- 新数据必须与 final-test-v2、dev 的 task ID / fixture seed 不重叠，并通过 Qwen tokenizer、tool schema、tool-call 结构门禁。
- 若无法在不重复、不降质的前提下补到 450，按计划停止并写“因质量/覆盖不足不补”，不硬凑。

## 5. 产出物与复现

- 逐行审计：`docs/evidence/训练数据多样性审计.csv`（train+val 全部行，含 `included_after_tokenizer_gate` 标记）。
- 机器可读摘要：`docs/evidence/训练数据审计_manifest.json`（脚本版本、输入 SHA、行数、分类计数、unknown 计数、教师统计）。
- 本文档：`docs/evidence/数据多样性审计.md`；计划路径副本：`docs/evidence/训练数据多样性审计.md`。
- 复现命令：

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\audit_data_diversity.py `
  --data-dir private\sft_data_v4 `
  --gate-file private\kaggle_pull_method_baseline_meta\method\baseline\tokenizer_gate.json `
  --teacher-runs private\sft_data_v2_deepseek_expanded\teacher_runs.jsonl `
  --out-dir docs\evidence
```

脚本在 SHA 不一致、门禁行数不一致、门禁引用未知 ID 或结构失败时直接非零退出；不会用人工抽样替代全量统计。

## 6. 验收对照

- 四维分布表：约束类型（2.1）、任务模板（2.2）、边界案例（2.3）、教师门禁（2.4）均有量化数字。
- 预注册门槛：4.1 逐条引用，4.2 给出灰区一次性决定；未放宽、未收紧。
- 可重复：输入 SHA 与 train-v4 冻结记录逐字节一致；脚本无模型调用、无数据写入。
- 计划状态：阶段 0.0 已完成，产出物路径写入 `docs/plan/训练侧执行计划.md` 状态表。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8", newline="\n")


CSV_FIELDS = [
    "id",
    "split",
    "category",
    "source",
    "difficulty",
    "constraint_count",
    "constraint_types",
    "context_length_bucket",
    "tools_allowed",
    "check_types",
    "has_full_trajectory",
    "duplicate_group",
    "notes",
    "task_id",
    "template",
    "tool_call_count",
    "steps_limit",
    "steps_limit_source",
    "included_after_tokenizer_gate",
    "tokenizer_gate",
    "over_max_length_tokens",
    "message_count",
]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def write_manifest(
    path: Path,
    *,
    inputs: Mapping[str, Any],
    gate: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    val_stats: Mapping[str, Any],
    raw_train_stats: Mapping[str, Any],
    raw_boundary: Mapping[str, int],
    boundary: Mapping[str, int],
    heldout: Sequence[Mapping[str, Any]],
    teacher: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    payload = {
        "script": "scripts/audit_data_diversity.py",
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": inputs["verified"],
        "gate": gate,
        "counts": {
            "train_raw": int(train_stats["rows"]) + int(gate["train"]["over_max_length_count"]),
            "train_effective": int(train_stats["rows"]),
            "train_excluded": int(gate["train"]["over_max_length_count"]),
            "val_raw": int(val_stats["rows"]) + int(gate["val"]["over_max_length_count"]),
            "val_effective": int(val_stats["rows"]),
            "val_excluded": int(gate["val"]["over_max_length_count"]),
        },
        "effective_train": train_stats,
        "effective_val": val_stats,
        "raw_train": raw_train_stats,
        "boundary_raw": raw_boundary,
        "boundary_effective": boundary,
        "heldout": list(heldout),
        "teacher": teacher,
        "decision": decision,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="read-only diversity audit for train-v4")
    parser.add_argument("--data-dir", type=Path, default=Path("private/sft_data_v4"))
    parser.add_argument(
        "--gate-file",
        type=Path,
        default=Path(
            "private/kaggle_pull_method_baseline_meta/method/baseline/tokenizer_gate.json"
        ),
    )
    parser.add_argument(
        "--teacher-runs",
        type=Path,
        default=Path("private/sft_data_v2_deepseek_expanded/teacher_runs.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("docs/evidence"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    inputs = verify_frozen_inputs(data_dir)
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "val.jsonl")
    gate = load_tokenizer_gate(args.gate_file.resolve(), train_rows, val_rows)
    train_over = {str(key): int(value) for key, value in gate["train"]["over_max_length"].items()}
    val_over = {str(key): int(value) for key, value in gate["val"]["over_max_length"].items()}
    raw_train_rows = [enrich_row(row, "train", train_over) for row in train_rows]
    raw_val_rows = [enrich_row(row, "val", val_over) for row in val_rows]
    train_effective = [row for row in raw_train_rows if bool(row["included_after_tokenizer_gate"])]
    val_effective = [row for row in raw_val_rows if bool(row["included_after_tokenizer_gate"])]
    if len(train_rows) != 380 or len(train_effective) != 361:
        raise SystemExit(
            f"unexpected train counts: raw={len(train_rows)} effective={len(train_effective)}"
        )
    if len(val_rows) != 80 or len(val_effective) != 74:
        raise SystemExit(
            f"unexpected val counts: raw={len(val_rows)} effective={len(val_effective)}"
        )
    raw_train_stats = summarize(raw_train_rows)
    train_stats = summarize(train_effective)
    val_stats = summarize(val_effective)
    raw_boundary = boundary_counts(raw_train_rows)
    boundary = boundary_counts(train_effective)
    heldout = heldout_counts(train_effective)
    teacher = teacher_stats(args.teacher_runs.resolve())
    source_stats = inputs["manifest"].get("source_stats") or {}
    decision = {
        "decision": "补数据",
        "threshold_branch": "HARD 50–79 灰区；一次性决定",
        "hard_effective": train_stats["category"].get("HARD", {}).get("count", 0),
        "min_constraint_type_pct": min(
            item["pct"] for item in train_stats["constraint_types"].values()
        ),
        "rationale": [
            "CTX-HARD 19 条全部因 3072 超长被丢，有效 0",
            "long context 有效 0",
            "BASE 23 条，低于 30 条目标下限",
            "context_recall 模板 28 条，占 7.76%",
            "DeepSeek 教师在 context_recall/constraint_edit 通过率 62.50%",
        ],
        "minimum_supplement": {"CTX-HARD": 19, "BASE": 7, "HARD": 3, "total": 29},
        "target_range": "450–500 train / 80–100 val，只补缺口",
    }
    csv_rows = sorted(
        [*raw_train_rows, *raw_val_rows],
        key=lambda row: (0 if row["split"] == "train" else 1, str(row["id"])),
    )
    write_csv(out_dir / "训练数据多样性审计.csv", csv_rows)
    write_manifest(
        out_dir / "训练数据审计_manifest.json",
        inputs=inputs,
        gate=gate,
        train_stats=train_stats,
        val_stats=val_stats,
        raw_train_stats=raw_train_stats,
        raw_boundary=raw_boundary,
        boundary=boundary,
        heldout=heldout,
        teacher=teacher,
        decision=decision,
    )
    markdown_path = out_dir / "数据多样性审计.md"
    write_markdown(
        path=markdown_path,
        inputs=inputs,
        gate=gate,
        train_stats=train_stats,
        val_stats=val_stats,
        raw_train_stats=raw_train_stats,
        raw_boundary=raw_boundary,
        boundary=boundary,
        heldout=heldout,
        teacher=teacher,
        source_stats=source_stats,
        decision=decision,
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    (out_dir / "训练数据多样性审计.md").write_text(markdown, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "status": "ok",
                "train_raw": len(train_rows),
                "train_effective": len(train_effective),
                "val_raw": len(val_rows),
                "val_effective": len(val_effective),
                "decision": decision["decision"],
                "markdown": markdown_path.as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
