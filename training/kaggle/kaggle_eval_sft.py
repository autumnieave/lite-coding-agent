"""Held-out evaluation for the Kaggle SFT adapter.

This is a probe-only evaluation: it replays the held-out final prompt and checks output-format
constraints. Tool execution / file content judgement need a full-loop evaluation and are
reported as N/A rather than fabricated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.constrained_decoding import validate_output  # noqa: E402
from training.eval_metrics import (  # noqa: E402
    compare_summaries,
    summarize_generation_records,
)
from training.kaggle.local_prompt import build_chat_prompt  # noqa: E402
from training.train_sft import default_tool_schemas  # noqa: E402


def load_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def _generate(
    model: Any, tokenizer: Any, messages: Sequence[Mapping[str, Any]], max_new_tokens: int
) -> str:
    prompt = build_chat_prompt(tokenizer, messages, default_tool_schemas())
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output[0][inputs["input_ids"].shape[1] :]
    return str(tokenizer.decode(generated, skip_special_tokens=True))


def evaluate_rows(
    model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], max_new_tokens: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        started = time.perf_counter()
        output = _generate(model, tokenizer, row["messages"], max_new_tokens)
        validation = validate_output(output)
        validation.pop("parsed", None)
        records.append(
            {
                "id": row.get("id"),
                "run_id": (row.get("metadata") or {}).get("run_id"),
                "output": output,
                "validation": validation,
                "metadata": {"duration_s": round(time.perf_counter() - started, 2)},
            }
        )
    return records


def _content_judgement(
    records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    applicable = 0
    passed = 0
    for record, row in zip(records, rows, strict=True):
        expected = ((row.get("metadata") or {}).get("reward") or {}).get("checks") or {}
        if expected.get("file_written_correctly") is not None:
            continue
        applicable += 1
        validation = record.get("validation") or {}
        if all(
            validation.get(check) for check in ("json_valid", "snake_case_keys", "no_code_fence")
        ):
            passed += 1
    return {
        "value": round(passed / applicable, 4) if applicable else None,
        "hits": passed,
        "total": applicable,
        "reason": "val 行只有格式检查；文件内容判定需要全链路工具执行",
    }


def run_eval(
    *,
    model_path: Path,
    adapter_path: Path,
    val_file: Path,
    output_file: Path,
    max_new_tokens: int,
    limit: int,
    dry_run: bool,
) -> dict[str, Any]:
    rows = load_rows(val_file, limit)
    if dry_run:
        plan = {
            "status": "dry_run",
            "model_path": str(model_path),
            "adapter_path": str(adapter_path),
            "val_file": str(val_file),
            "rows": len(rows),
            "max_new_tokens": max_new_tokens,
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("没有 CUDA；SFT held-out 评测必须在 Kaggle T4 上执行。")
    started = time.perf_counter()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    with torch.inference_mode():
        before_records = evaluate_rows(model, tokenizer, rows, max_new_tokens)
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        after_records = evaluate_rows(model, tokenizer, rows, max_new_tokens)

    before = summarize_generation_records(before_records)
    after = summarize_generation_records(after_records)
    result = {
        "status": "completed",
        "model_path": str(model_path),
        "adapter_path": str(adapter_path),
        "val_file": str(val_file),
        "rollouts": len(rows),
        "sampling_config": {
            "temperature": 0.0,
            "top_p": None,
            "seed": 20261019,
            "do_sample": False,
            "is_default": False,
        },
        "constraint_compliance": {"before": before, "after": after},
        "content_judgement": _content_judgement(after_records, rows),
        "tool_call_success": {
            "value": None,
            "reason": "probe-only held-out 集不含可执行工具任务；需全链路 loop 评测",
        },
        "delta": compare_summaries(before, after),
        "metadata": {
            "model_version": getattr(model.config, "_name_or_path", None),
            "system_fingerprint": "N/A (local transformers; no API)",
            "response_id": None,
            "duration_s": round(time.perf_counter() - started, 1),
            "device": torch.cuda.get_device_name(0),
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kaggle SFT held-out probe eval")
    parser.add_argument("--model-path", default="/kaggle/input/qwen2.5-coder-7b-instruct")
    parser.add_argument("--adapter-path", default="/kaggle/working/sft_adapter")
    parser.add_argument("--val-file", default="/kaggle/input/lite-agent-training/val.jsonl")
    parser.add_argument("--output-file", default="/kaggle/working/sft_eval.json")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_eval(
        model_path=Path(args.model_path),
        adapter_path=Path(args.adapter_path),
        val_file=Path(args.val_file),
        output_file=Path(args.output_file),
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        dry_run=bool(args.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
