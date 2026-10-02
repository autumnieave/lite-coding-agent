"""Build DPO pairs from train-v5 rejected traces without mixing SFT train data.

Only ``code_prefix_output`` rejects have an unambiguous correction: the model
answered ``CODE=CTX-...`` when the task required the bare ``CTX-...`` token.
Other rejects are retained as rejected behavior evidence but are not promoted
to preference pairs because their prompts have no clean positive trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.build_train_v5 import build_v5_specs
from training.sft_format import canonicalize_tool_arguments
from training.train_dpo import render_dpo_prompt

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


def _final_turn(path: Path) -> tuple[str, list[dict[str, Any]]] | None:
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


def _prompt_hash(messages: Sequence[Mapping[str, Any]]) -> str:
    normalized = canonicalize_tool_arguments(messages)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _render_prompt(tokenizer: Any, messages: Sequence[Mapping[str, Any]]) -> str:
    return render_dpo_prompt(tokenizer, messages)


def build_pairs(
    *,
    sft_dir: Path,
    out_dir: Path,
    tokenizer: Any,
    tokenizer_label: str,
) -> dict[str, Any]:
    manifest = json.loads((sft_dir / "manifest.json").read_text(encoding="utf-8"))
    rejected_rows = list((manifest.get("v5_teacher") or {}).get("rejected") or ())
    train_rows = _load_jsonl(sft_dir / "train.jsonl")
    val_rows = _load_jsonl(sft_dir / "val.jsonl")
    train_hashes = {_prompt_hash(row["messages"]) for row in train_rows}
    val_hashes = {_prompt_hash(row["messages"]) for row in val_rows}
    specs = build_v5_specs()
    seed_by_task = {spec.task_id: spec.fixture_seed for spec in specs}
    expected_tokens = {task_id: f"CTX-{seed}" for task_id, seed in seed_by_task.items()}

    pair_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    overlap_train = 0
    overlap_val = 0
    heldout = 0
    for rejected in rejected_rows:
        reason = str(rejected.get("reason") or "unknown")
        reason_counts[reason] += 1
        if reason != "code_prefix_output":
            skipped[f"not_clean_positive:{reason}"] += 1
            continue
        task_id = str(rejected.get("task_id") or "")
        if task_id in HELDOUT_TASK_IDS:
            heldout += 1
            skipped["heldout_task"] += 1
            continue
        trace_path = Path(str(rejected.get("trace_path") or ""))
        if not trace_path.is_file():
            skipped["missing_trace"] += 1
            continue
        final = _final_turn(trace_path)
        if final is None:
            skipped["missing_final_turn"] += 1
            continue
        rejected_text, messages = final
        if not rejected_text.startswith("CODE="):
            skipped["not_code_prefix"] += 1
            continue
        chosen = rejected_text.removeprefix("CODE=").strip()
        expected = expected_tokens.get(task_id)
        if expected is None or chosen != expected:
            skipped["chosen_not_equal_fixture_token"] += 1
            continue
        prompt_hash = _prompt_hash(messages)
        if prompt_hash in train_hashes:
            overlap_train += 1
            skipped["prompt_overlap_sft_train"] += 1
            continue
        if prompt_hash in val_hashes:
            overlap_val += 1
            skipped["prompt_overlap_sft_val"] += 1
            continue
        pair_rows.append(
            {
                "id": f"dpo:v2:code-prefix:{task_id}:{len(pair_rows):03d}",
                "prompt_messages": messages,
                "chosen": chosen,
                "rejected": rejected_text,
                "metadata": {
                    "pair_type": "code_prefix_fix",
                    "reject_reason": reason,
                    "source_task_id": task_id,
                    "category": "CTX",
                    "split": "dpo",
                    "fixture_seed": seed_by_task[task_id],
                },
            }
        )

    if not pair_rows:
        raise ValueError("no DPO v2 pairs were constructed")

    prompt_tokens: list[int] = []
    chosen_tokens: list[int] = []
    rejected_tokens: list[int] = []
    for pair in pair_rows:
        rendered = _render_prompt(tokenizer, pair["prompt_messages"])
        prompt_count = _token_len(tokenizer, rendered)
        chosen_count = _token_len(tokenizer, str(pair["chosen"]))
        rejected_count = _token_len(tokenizer, str(pair["rejected"]))
        prompt_tokens.append(prompt_count)
        chosen_tokens.append(chosen_count)
        rejected_tokens.append(rejected_count)
        pair["metadata"].update(
            {
                "prompt_tokens": prompt_count,
                "chosen_tokens": chosen_count,
                "rejected_tokens": rejected_count,
                "chosen_total_tokens": prompt_count + chosen_count,
                "rejected_total_tokens": prompt_count + rejected_count,
            }
        )

    limits = {"1536": 1536, "2048": 2048, "3072": 3072}
    truncation = {}
    for label, limit in limits.items():
        truncation[label] = {
            "prompt_over_limit": sum(count > limit for count in prompt_tokens),
            "chosen_total_over_limit": sum(
                pair["metadata"]["chosen_total_tokens"] > limit for pair in pair_rows
            ),
            "rejected_total_over_limit": sum(
                pair["metadata"]["rejected_total_tokens"] > limit for pair in pair_rows
            ),
        }

    pair_rows.sort(key=lambda row: str(row["id"]))
    train_pairs: list[dict[str, Any]] = []
    val_pairs: list[dict[str, Any]] = []
    for pair in pair_rows:
        bucket = int(hashlib.sha256(str(pair["id"]).encode("utf-8")).hexdigest()[:8], 16) % 5
        (val_pairs if bucket == 0 else train_pairs).append(pair)
    if (not train_pairs or not val_pairs) and len(pair_rows) >= 2:
        train_pairs = pair_rows[:-1]
        val_pairs = pair_rows[-1:]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "dpo_pairs.jsonl", pair_rows)
    _write_jsonl(out_dir / "dpo_train.jsonl", train_pairs)
    _write_jsonl(out_dir / "dpo_val.jsonl", val_pairs)
    manifest_out: dict[str, Any] = {
        "version": "dpo-v2-preflight",
        "teacher_model": manifest.get("teacher_model"),
        "source_train_version": manifest.get("version"),
        "source_train_sha256": manifest.get("train_sha256"),
        "tokenizer_label": tokenizer_label,
        "pair_count": len(pair_rows),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "pair_type_counts": dict(Counter(str(row["metadata"]["pair_type"]) for row in pair_rows)),
        "category_counts": dict(Counter(str(row["metadata"]["category"]) for row in pair_rows)),
        "rejected_reason_counts": dict(sorted(reason_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "overlap": {
            "sft_train_prompt_matches": overlap_train,
            "sft_val_prompt_matches": overlap_val,
            "heldout_task_matches": heldout,
            "sft_train_rows": len(train_rows),
            "sft_val_rows": len(val_rows),
        },
        "prompt_tokens": {
            "min": min(prompt_tokens),
            "p50": _percentile(prompt_tokens, 0.5),
            "max": max(prompt_tokens),
        },
        "completion_tokens": {
            "chosen_min": min(chosen_tokens),
            "chosen_max": max(chosen_tokens),
            "rejected_min": min(rejected_tokens),
            "rejected_max": max(rejected_tokens),
        },
        "truncation": truncation,
        "dpo_config": {
            "max_seq_length": 3072,
            "max_prompt_length": 2048,
            "beta": 0.1,
            "reference_model": "SFT 前基座（adapter 关闭）",
        },
        "caveat": (
            "Only code-prefix corrections are clean pairs. Behavior-defect rejects are not used."
        ),
        "train_sha256": hashlib.sha256((out_dir / "dpo_train.jsonl").read_bytes()).hexdigest(),
        "val_sha256": hashlib.sha256((out_dir / "dpo_val.jsonl").read_bytes()).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="preflight DPO v2 pairs from train-v5 rejects")
    parser.add_argument("--sft-dir", type=Path, default=Path("private/sft_data_v5"))
    parser.add_argument("--out-dir", type=Path, default=Path("private/dpo_v2"))
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-label", default="Qwen/Qwen2.5-7B-Instruct")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=bool(os.environ.get("HF_HUB_OFFLINE")),
    )
    result = build_pairs(
        sft_dir=args.sft_dir,
        out_dir=args.out_dir,
        tokenizer=tokenizer,
        tokenizer_label=args.tokenizer_label,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
