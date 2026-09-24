"""Check that SFT rows render as Qwen2.5 chat-template text, not teacher-native text."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.sft_format import canonicalize_tool_arguments

DEEPSEEK_MARKERS = ("<|User|>", "<|Assistant|>", "<|tool▁calls▁begin|>", "<｜begin▁of▁sentence｜>")
TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _tool_call_count(row: Mapping[str, Any]) -> int:
    return sum(
        len(message.get("tool_calls") or [])
        for message in canonicalize_tool_arguments(row.get("messages") or [])
        if message.get("role") == "assistant"
    )


def _tool_message_count(row: Mapping[str, Any]) -> int:
    return sum(1 for message in row.get("messages") or [] if message.get("role") == "tool")


def check_rendered_text(
    text: str,
    row: Mapping[str, Any],
    *,
    extra_tool_call_blocks: int = 0,
) -> list[str]:
    """Return human-readable format failures for one rendered training row."""

    failures: list[str] = []
    if text.count("<|im_start|>") != text.count("<|im_end|>"):
        failures.append("im_start/im_end unbalanced")
    if text.count("<|im_start|>") == 0:
        failures.append("not a Qwen chat template (no <|im_start|>)")
    for marker in DEEPSEEK_MARKERS:
        if marker in text:
            failures.append(f"deepseek marker present: {marker}")
    expected_calls = _tool_call_count(row)
    expected_blocks = expected_calls + extra_tool_call_blocks
    blocks = TOOL_CALL_BLOCK.findall(text)
    if len(blocks) != expected_blocks:
        failures.append(f"tool_call blocks {len(blocks)} != expected {expected_blocks}")
    object_arguments = sum(1 for block in blocks if _arguments_is_object(block))
    if object_arguments != expected_calls:
        failures.append(
            f"tool_call arguments object {object_arguments} != expected {expected_calls}"
        )
    expected_results = _tool_message_count(row)
    if text.count("<tool_response>") != expected_results:
        failures.append(
            f"tool_response blocks {text.count('<tool_response>')} != expected {expected_results}"
        )
    return failures


def _arguments_is_object(block: str) -> bool:
    try:
        payload = json.loads(block)
    except ValueError:
        return False
    arguments = payload.get("arguments") if isinstance(payload, Mapping) else None
    return isinstance(arguments, (dict, list))


def _load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="check SFT rows render as Qwen chat template")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", type=Path, default=Path("private/sft_data/train.jsonl"))
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--out", type=Path, default=Path("private/sft_data/format_check.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer

    from training.train_sft import render_training_text

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = _load_rows(args.data, args.limit)
    results: list[dict[str, Any]] = []
    failed = 0
    for row in rows:
        text = render_training_text(tokenizer, row)
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        failures = check_rendered_text(text, row)
        if tokens > args.max_seq_length:
            failures.append(f"tokens {tokens} > max_seq_length {args.max_seq_length}")
        results.append(
            {
                "id": row.get("id"),
                "tokens": tokens,
                "tool_calls": _tool_call_count(row),
                "failures": failures,
            }
        )
        failed += bool(failures)
    summary = {
        "model": args.model,
        "data": args.data.as_posix(),
        "checked": len(results),
        "failed": failed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
