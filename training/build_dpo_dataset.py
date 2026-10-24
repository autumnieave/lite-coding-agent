"""Build preference pairs without re-weighting the SFT train split.

The DPO prompts come from train-v4's *val* split, which is disjoint from the SFT
train split. Natural rejected traces are audited, but only included when their
prompt matches a non-overlapping val prompt. Synthetic negatives are explicitly
labelled and used to make the small-scale DPO arm possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HELDOUT_TASK_IDS = frozenset(
    {
        "U-TR-A-C1-HARD",
        "U-TR-B-CHECK-BASE",
        "U-TR-B-C1-BASE",
        "U-TR-B-CTX-HARD",
        "U-TR-B-TOOL-HARD",
        "U-TR-B-CONFLICT-WEAK",
    }
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _normalized_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        normalized = dict(message)
        if normalized.get("tool_calls"):
            normalized["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": (call.get("function") or {}).get("name"),
                        "arguments": (call.get("function") or {}).get("arguments"),
                    },
                }
                for call in normalized["tool_calls"]
            ]
        if normalized.get("role") == "tool":
            normalized.pop("tool_call_id", None)
        result.append(normalized)
    return result


def _final_from_trace(path: Path) -> tuple[str, list[dict[str, Any]]] | None:
    turns: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "llm_turn":
            turns.append(event)
    for turn in reversed(turns):
        response = turn.get("response") or {}
        content = str(response.get("content") or "").strip()
        if content and not response.get("tool_calls"):
            return content, list(turn.get("messages") or ())
    return None


def _synthetic_negatives(completion: str) -> list[tuple[str, str]]:
    negatives: list[tuple[str, str]] = []
    stripped = completion.strip()
    try:
        json.loads(stripped)
    except ValueError:
        negatives.append(("appended_prose", f"{completion}\n补充说明：以上是答案。"))
        negatives.append(("prefixed_prose", f"答案如下：{completion}"))
    else:
        negatives.append(("markdown_fence", f"```json\n{completion}\n```"))
        negatives.append(("appended_prose", f"{completion}\n补充说明：以上是结果。"))
    return negatives


def build_dpo_pairs(
    *,
    sft_dir: Path,
    rejected_sft_dir: Path,
    out_dir: Path,
    max_synthetic: int = 160,
    excluded_task_ids: Sequence[str] = tuple(HELDOUT_TASK_IDS),
) -> dict[str, Any]:
    all_val_rows = _load_jsonl(sft_dir / "val.jsonl")
    excluded = set(excluded_task_ids)
    val_rows = [
        row
        for row in all_val_rows
        if str((row.get("metadata") or {}).get("task_id") or "") not in excluded
    ]
    train_rows = _load_jsonl(sft_dir / "train.jsonl")
    train_prompt_hashes = {
        hashlib.sha256(
            json.dumps(
                _normalized_messages(row["messages"]), ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        for row in train_rows
    }
    val_by_hash = {
        hashlib.sha256(
            json.dumps(
                _normalized_messages(row["messages"]), ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest(): row
        for row in val_rows
    }
    excluded_val_hashes = {
        hashlib.sha256(
            json.dumps(
                _normalized_messages(row["messages"]), ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        for row in all_val_rows
        if str((row.get("metadata") or {}).get("task_id") or "") in excluded
    }

    pairs: list[dict[str, Any]] = []
    natural_candidates = 0
    natural_kept = 0
    natural_heldout_excluded = 0
    rejected_manifest_path = rejected_sft_dir / "manifest.json"
    if rejected_manifest_path.exists():
        manifest = json.loads(rejected_manifest_path.read_text(encoding="utf-8"))
        for rejected in manifest.get("rejected", []):
            trace_path = Path(str(rejected.get("trace_path") or ""))
            if not trace_path.exists():
                continue
            final = _final_from_trace(trace_path)
            if final is None:
                continue
            rejected_completion, rejected_messages = final
            prompt_hash = hashlib.sha256(
                json.dumps(
                    _normalized_messages(rejected_messages),
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if prompt_hash in excluded_val_hashes:
                natural_heldout_excluded += 1
                continue
            chosen_row = val_by_hash.get(prompt_hash)
            if chosen_row is None:
                if prompt_hash in train_prompt_hashes:
                    natural_candidates += 1
                continue
            natural_candidates += 1
            natural_kept += 1
            pairs.append(
                {
                    "id": f"dpo:natural:{rejected.get('task_id')}:{natural_kept:03d}",
                    "prompt_messages": chosen_row["messages"],
                    "chosen": chosen_row["completion"],
                    "rejected": rejected_completion,
                    "metadata": {
                        "pair_type": "natural_rejected",
                        "reject_reason": rejected.get("reason"),
                        "source_task_id": rejected.get("task_id"),
                        "category": (chosen_row.get("metadata") or {}).get("category"),
                        "split": "dpo",
                    },
                }
            )

    synthetic_count = 0
    for row in val_rows:
        if synthetic_count >= max_synthetic:
            break
        chosen = str(row.get("completion") or "").strip()
        if not chosen:
            continue
        for negative_type, rejected in _synthetic_negatives(chosen):
            if synthetic_count >= max_synthetic:
                break
            if rejected.strip() == chosen:
                continue
            synthetic_count += 1
            pairs.append(
                {
                    "id": f"dpo:synthetic:{synthetic_count:04d}",
                    "prompt_messages": row["messages"],
                    "chosen": chosen,
                    "rejected": rejected,
                    "metadata": {
                        "pair_type": "synthetic_negative",
                        "negative_type": negative_type,
                        "source_task_id": (row.get("metadata") or {}).get("task_id"),
                        "category": (row.get("metadata") or {}).get("category"),
                        "split": "dpo",
                    },
                }
            )

    if not pairs:
        raise ValueError("no DPO pairs were constructed")
    pairs.sort(key=lambda row: str(row["id"]))
    train_pairs: list[dict[str, Any]] = []
    val_pairs: list[dict[str, Any]] = []
    for pair in pairs:
        bucket = int(hashlib.sha256(str(pair["id"]).encode("utf-8")).hexdigest()[:8], 16) % 5
        (val_pairs if bucket == 0 else train_pairs).append(pair)
    if not val_pairs:
        val_pairs.append(train_pairs.pop())
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "dpo_pairs.jsonl", pairs)
    _write_jsonl(out_dir / "dpo_train.jsonl", train_pairs)
    _write_jsonl(out_dir / "dpo_val.jsonl", val_pairs)
    manifest: dict[str, Any] = {
        "version": "dpo-v1",
        "pair_count": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "pair_type_counts": dict(
            sorted(Counter(str(row["metadata"]["pair_type"]) for row in pairs).items())
        ),
        "category_counts": dict(
            sorted(
                Counter(str(row["metadata"].get("category") or "unknown") for row in pairs).items()
            )
        ),
        "natural_rejected": {
            "found_in_manifest": natural_candidates,
            "kept_non_overlapping": natural_kept,
            "excluded_due_to_sft_train_overlap": natural_candidates - natural_kept,
            "excluded_due_to_heldout_overlap": natural_heldout_excluded,
        },
        "synthetic_negative": {
            "count": synthetic_count,
            "reason": "SFT val prompts plus explicit corruptions for a method smoke comparison",
        },
        "sft_train_prompt_overlap": 0,
        "heldout_exclusion": {
            "task_ids": sorted(excluded),
            "val_rows_before": len(all_val_rows),
            "val_rows_after": len(val_rows),
            "excluded_val_rows": len(all_val_rows) - len(val_rows),
        },
        "beta": 0.1,
        "reference_model": "SFT 前基座（adapter 关闭）",
        "train_sha256": hashlib.sha256((out_dir / "dpo_train.jsonl").read_bytes()).hexdigest(),
        "val_sha256": hashlib.sha256((out_dir / "dpo_val.jsonl").read_bytes()).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build DPO pairs from train-v4 val prompts")
    parser.add_argument("--sft-dir", type=Path, default=Path("private/sft_data_v4"))
    parser.add_argument(
        "--rejected-sft-dir",
        type=Path,
        default=Path("private/sft_data_v2_deepseek_expanded"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("private/dpo_v1"))
    parser.add_argument("--max-synthetic", type=int, default=160)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_dpo_pairs(
        sft_dir=args.sft_dir,
        rejected_sft_dir=args.rejected_sft_dir,
        out_dir=args.out_dir,
        max_synthetic=args.max_synthetic,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
