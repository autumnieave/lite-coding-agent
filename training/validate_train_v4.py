"""Kaggle-side Qwen chat-template gate for train-v4."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.build_sft_dataset import validate_qwen_render, validate_teacher_messages
from training.loss_masking import tokenize_training_row
from training.train_sft import default_tool_schemas, render_training_text


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
) -> dict[str, Any]:
    tools = default_tool_schemas()
    failures: list[dict[str, Any]] = []
    over_length: list[dict[str, Any]] = []
    assistant_mask_failures: list[dict[str, Any]] = []
    token_counts: list[int] = []
    for row in rows:
        row_id = str(row.get("id") or "")
        try:
            validate_teacher_messages(row["messages"])
            render_failure = validate_qwen_render(
                row["messages"], str(row["completion"]), tokenizer, tools=tools
            )
            if render_failure:
                raise ValueError(render_failure)
            text = render_training_text(tokenizer, row, tools=tools)
            count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            token_counts.append(count)
            if count > max_length:
                over_length.append({"id": row_id, "tokens": count})
                continue
            encoded = tokenize_training_row(
                tokenizer,
                row,
                loss_mode="assistant_only",
                max_length=max_length,
                tools=tools,
            )
            kept = sum(label != -100 for label in encoded["labels"])
            if kept == 0:
                assistant_mask_failures.append({"id": row_id, "kept_tokens": 0})
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            failures.append({"id": row_id, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "rows": len(rows),
        "passed": len(rows) - len(failures),
        "failures": failures[:100],
        "failure_count": len(failures),
        "over_max_length": over_length,
        "over_max_length_count": len(over_length),
        "assistant_mask_failures": assistant_mask_failures[:100],
        "assistant_mask_failure_count": len(assistant_mask_failures),
        "token_count_max": max(token_counts) if token_counts else 0,
        "token_count_p50": sorted(token_counts)[len(token_counts) // 2] if token_counts else 0,
        "failure_types": dict(
            sorted(Counter(str(item["error"]).split(":", 1)[0] for item in failures).items())
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="validate train-v4 with the Qwen tokenizer")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    result: dict[str, Any] = {}
    for split, path in (("train", args.train_file), ("val", args.val_file)):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result[split] = validate_rows(rows, tokenizer, max_length=args.max_length)
    result["status"] = (
        "ok"
        if all(
            part["failure_count"] == 0 and part["assistant_mask_failure_count"] == 0
            for part in result.values()
            if isinstance(part, dict)
        )
        else "failed"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
