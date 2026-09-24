"""Build a small, layer-balanced SFT set from v14 positive conflict traces.

The v14 step-3.5 fallback starts a policy from the local conflict pool rather
than from the older v8 dataset. This builder keeps the layer/severity mix of
the positive traces, uses v14's merged-first-system injection format, and
rejects targets that are not explicit refusals/confirmations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent.core.injection import CONSTRAINT_LAYER_HEADING  # noqa: E402
from agent.core.loop import DEFAULT_SYSTEM_PROMPT  # noqa: E402
from training.reward_v14 import CLAIM, PSEUDO_MARKERS, REFUSAL  # noqa: E402


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into dictionaries."""

    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _layer(row: Mapping[str, Any]) -> str:
    category = str(row.get("category") or "")
    if category in {"path_escape", "delete_without_confirm", "edit_tests"}:
        return category
    return "policy"


def _reply(row: Mapping[str, Any]) -> str:
    trajectory = row.get("trajectory") or {}
    reply = str(trajectory.get("reply") or "").strip()
    if reply:
        return reply
    for candidate in reversed(trajectory.get("step_replies") or []):
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


def _eligible_target(row: Mapping[str, Any]) -> tuple[bool, str]:
    if row.get("training_role") != "positive":
        return False, "not_positive"
    reply = _reply(row)
    if not reply:
        return False, "empty_reply"
    lower = reply.lower()
    if not any(marker.lower() in lower for marker in REFUSAL):
        return False, "no_explicit_refusal"
    if CLAIM.search(reply):
        return False, "claims_completion"
    if any(marker in reply for marker in PSEUDO_MARKERS):
        return False, "pseudo_tool_call"
    return True, "ok"


def _allocate(counts: Mapping[tuple[str, int], int], total: int) -> dict[tuple[str, int], int]:
    """Allocate *total* examples proportionally with deterministic remainders."""

    population = sum(counts.values())
    if total > population:
        raise ValueError(f"requested {total} rows but only {population} eligible rows")
    exact = {key: total * value / population for key, value in counts.items()}
    allocation = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(allocation.values())
    ordered = sorted(exact, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in ordered[:remaining]:
        allocation[key] += 1
    return allocation


def sample_positive_rows(
    rows: Sequence[Mapping[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    """Sample positive rows while preserving category x severity proportions."""

    eligible: list[dict[str, Any]] = []
    for row in rows:
        ok, _ = _eligible_target(row)
        if ok:
            eligible.append(dict(row))
    counts = Counter((_layer(row), int(row.get("severity") or 0)) for row in eligible)
    allocation = _allocate(counts, count)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[(_layer(row), int(row.get("severity") or 0))].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        candidates = grouped[key]
        rng.shuffle(candidates)
        selected.extend(candidates[: allocation[key]])
    selected.sort(key=lambda row: str(row.get("sample_id") or ""))
    if len(selected) != count:
        raise AssertionError(f"sampled {len(selected)} rows, expected {count}")
    return selected


def _system_content(row: Mapping[str, Any]) -> str:
    constraint_id = str(row.get("constraint_id") or "")
    text = str(row.get("constraint_text") or "").strip()
    layer = f"{CONSTRAINT_LAYER_HEADING}\n- [{constraint_id}] {text}"
    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{layer}"


def to_sft_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one v14 trace into the repo's messages+completion SFT shape."""

    user_turns = [str(item).strip() for item in row.get("user_turns") or [] if str(item).strip()]
    if not user_turns:
        raise ValueError(f"row has no user turn: {row.get('sample_id')}")
    # Eval probes are single-turn. Use the first user turn so the prompt shape
    # matches the 72-probe harness rather than inventing unobserved tool turns.
    user_text = user_turns[0]
    return {
        "id": str(row["sample_id"]),
        "messages": [
            {"role": "system", "content": _system_content(row)},
            {"role": "user", "content": user_text},
        ],
        "completion": _reply(row),
        "token_count": 0,
        "metadata": {
            "source": "v14_sdg_positive",
            "sample_id": str(row["sample_id"]),
            "task_id": str(row["task_id"]),
            "source_set": str(row.get("source_set") or ""),
            "category": str(row.get("category") or ""),
            "layer": _layer(row),
            "severity": int(row.get("severity") or 0),
            "constraint_id": str(row.get("constraint_id") or ""),
            "training_role": str(row.get("training_role") or ""),
            "verifier_label": str(row.get("verifier_label") or ""),
        },
    }


def split_rows(
    rows: Sequence[dict[str, Any]], val_count: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a deterministic train/val split with one validation row per layer."""

    if val_count <= 0 or val_count >= len(rows):
        raise ValueError("val_count must leave a non-empty train split")
    rng = random.Random(seed + 1)
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_layer[str((row.get("metadata") or {}).get("layer") or "policy")].append(row)
    val_ids: set[str] = set()
    for layer in sorted(by_layer):
        candidates = list(by_layer[layer])
        rng.shuffle(candidates)
        if len(candidates) > 1 and len(val_ids) < val_count:
            val_ids.add(str(candidates[0]["id"]))
    remaining = [row for row in rows if str(row["id"]) not in val_ids]
    rng.shuffle(remaining)
    for row in remaining:
        if len(val_ids) >= val_count:
            break
        val_ids.add(str(row["id"]))
    train = [row for row in rows if str(row["id"]) not in val_ids]
    val = [row for row in rows if str(row["id"]) in val_ids]
    if not train or len(val) != val_count:
        raise AssertionError("split did not produce the requested sizes")
    return train, val


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(
    *,
    source: Path,
    eval_file: Path,
    out_dir: Path,
    count: int,
    val_count: int,
    seed: int,
) -> dict[str, Any]:
    """Build the light SFT files and return the manifest."""

    rows = load_jsonl(source)
    eval_ids = {str(row["task_id"]) for row in load_jsonl(eval_file)}
    source_tasks = {str(row["task_id"]) for row in rows}
    overlap = sorted(source_tasks & eval_ids)
    if overlap:
        raise ValueError(f"source/eval task overlap: {overlap}")
    sampled = sample_positive_rows(rows, count, seed)
    sft_rows = [to_sft_row(row) for row in sampled]
    train, val = split_rows(sft_rows, val_count, seed)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "val.jsonl", val)
    manifest = {
        "version": "v14-light-sft-1",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "eval_file": str(eval_file),
        "eval_sha256": sha256_file(eval_file),
        "source_rows": len(rows),
        "requested_rows": count,
        "sampled_rows": len(sft_rows),
        "train_rows": len(train),
        "val_rows": len(val),
        "seed": seed,
        "sampling": "positive rows, proportional by layer x severity, deterministic shuffle",
        "format": (
            "merged-first-system v14 constraint layer; first user turn; explicit safe completion"
        ),
        "target_filter": (
            "positive + explicit refusal/confirmation + no completion claim + no pseudo tool call"
        ),
        "train_sha256": sha256_file(out_dir / "train.jsonl"),
        "val_sha256": sha256_file(out_dir / "val.jsonl"),
        "task_overlap_with_eval": overlap,
        "train_distribution": dict(
            Counter(f"{(row['metadata']['layer'])}:{row['metadata']['severity']}" for row in train)
        ),
        "val_distribution": dict(
            Counter(f"{(row['metadata']['layer'])}:{row['metadata']['severity']}" for row in val)
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build v14 light SFT data")
    parser.add_argument(
        "--source", type=Path, default=Path("private/v14/step1_sdg/train_sdg.jsonl")
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("private/v14/step2_benchmark/eval_set_frozen.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("private/v14/step3_5_light_sft/data"))
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--val-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261102)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build(
        source=args.source,
        eval_file=args.eval_file,
        out_dir=args.out_dir,
        count=args.count,
        val_count=args.val_count,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
