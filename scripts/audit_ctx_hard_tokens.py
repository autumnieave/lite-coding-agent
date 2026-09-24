"""Audit why CTX-HARD rows exceed the Qwen SFT token budget.

The audit is read-only and decomposes each selected row into fixture text,
tool-result text, system/user text, assistant tool-call JSON and chat-template
overhead. It does not modify train-v4 or the frozen evaluation fixtures.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.build_sft_dataset import canonicalize_tool_arguments
from training.train_sft import default_tool_schemas, render_training_text


def _load_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def _token_count(tokenizer: Any, text: object) -> int:
    return len(tokenizer(str(text), add_special_tokens=False)["input_ids"])


def analyze_row(
    row: Mapping[str, Any], tokenizer: Any, tools: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return an auditable token decomposition for one SFT row."""

    messages = canonicalize_tool_arguments(row["messages"])
    completion = str(row.get("completion") or "")
    rendered_messages = [*messages, {"role": "assistant", "content": completion}]
    with_tools = tokenizer.apply_chat_template(
        rendered_messages,
        tools=list(tools),
        tokenize=False,
        add_generation_prompt=False,
    )
    no_tools = tokenizer.apply_chat_template(
        rendered_messages,
        tools=[],
        tokenize=False,
        add_generation_prompt=False,
    )
    training_text = render_training_text(tokenizer, row, tools=tools)

    tool_contents = [
        str(item.get("content") or "") for item in messages if item.get("role") == "tool"
    ]
    system_user = "\n".join(
        str(item.get("content") or "")
        for item in messages
        if item.get("role") in {"system", "user"}
    )
    assistant_text = "\n".join(
        str(item.get("content") or "")
        for item in messages
        if item.get("role") == "assistant" and item.get("content")
    )
    call_arguments = [
        str(call["function"]["arguments"])
        for item in messages
        for call in item.get("tool_calls") or []
    ]
    call_json = [
        json.dumps(call, ensure_ascii=False)
        for item in messages
        for call in item.get("tool_calls") or []
    ]

    with_tools_tokens = _token_count(tokenizer, with_tools)
    no_tools_tokens = _token_count(tokenizer, no_tools)
    fixture_tokens = sum(_token_count(tokenizer, item) for item in tool_contents)
    fixture_unique_tokens = sum(
        _token_count(tokenizer, item) for item in dict.fromkeys(tool_contents)
    )
    system_user_tokens = _token_count(tokenizer, system_user)
    assistant_text_tokens = _token_count(tokenizer, assistant_text)
    call_argument_tokens = sum(_token_count(tokenizer, item) for item in call_arguments)
    call_json_tokens = sum(_token_count(tokenizer, item) for item in call_json)
    completion_tokens = _token_count(tokenizer, completion)
    tools_overhead = with_tools_tokens - no_tools_tokens
    raw_content_tokens = (
        fixture_tokens
        + system_user_tokens
        + assistant_text_tokens
        + call_json_tokens
        + completion_tokens
    )
    template_overhead = with_tools_tokens - raw_content_tokens
    return {
        "id": str(row.get("id") or ""),
        "messages": len(messages),
        "tool_messages": len(tool_contents),
        "completion_chars": len(completion),
        "rendered_tokens_with_tools": with_tools_tokens,
        "rendered_tokens_without_tools": no_tools_tokens,
        "training_render_tokens": _token_count(tokenizer, training_text),
        "training_render_matches_direct": training_text == with_tools,
        "tools_template_overhead": tools_overhead,
        "tools_template_share": round(tools_overhead / with_tools_tokens, 4)
        if with_tools_tokens
        else 0.0,
        "fixture_tokens": fixture_tokens,
        "fixture_unique_tokens": fixture_unique_tokens,
        "fixture_share": round(fixture_tokens / with_tools_tokens, 4) if with_tools_tokens else 0.0,
        "system_user_tokens": system_user_tokens,
        "assistant_text_tokens": assistant_text_tokens,
        "tool_call_argument_tokens": call_argument_tokens,
        "tool_call_json_tokens": call_json_tokens,
        "completion_tokens": completion_tokens,
        "template_overhead_tokens": template_overhead,
        "fixture_tokens_per_read": [_token_count(tokenizer, item) for item in tool_contents[:5]],
        "unique_tool_contents": len(set(tool_contents)),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    rows = payload["samples"]
    lines = [
        "# CTX-HARD 超长 token 审计",
        "",
        f"> 脚本：`{payload['script']}`",
        f"> 输入 train：`{payload['train_file']}`",
        f"> tokenizer：`{payload['tokenizer_path']}`",
        f"> 最大长度：{payload['max_length']}",
        "> 性质：只读；不修改任务定义、fixture 或冻结数据。",
        "",
        "## 逐样本分解",
        "",
        "| ID | 总 token | fixture | fixture 占比 | tools 模板 | tools 占比 | "
        "system/user | 工具调用 JSON | 最终回答 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {id} | {total} | {fixture} | {fixture_share:.2%} | {tools} | {tools_share:.2%} | "
            "{sys_user} | {calls} | {completion} |".format(
                id=row["id"],
                total=row["rendered_tokens_with_tools"],
                fixture=row["fixture_tokens"],
                fixture_share=row["fixture_share"],
                tools=row["tools_template_overhead"],
                tools_share=row["tools_template_share"],
                sys_user=row["system_user_tokens"],
                calls=row["tool_call_json_tokens"],
                completion=row["completion_tokens"],
            )
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            f"- 主因：**{payload['root_cause']}**。",
            f"- 证据：CTX-HARD 样本的 fixture 平均占 "
            f"{payload['summary']['fixture_share_mean']:.2%}；"
            f"`tools=` 模板平均占 {payload['summary']['tools_share_mean']:.2%}。",
            f"- 训练路径与直接 Qwen `apply_chat_template(..., tools=...)` 一致："
            f"{payload['summary']['training_render_matches_direct']}。",
            "- 修复边界：缩短新增 train-v5 CTX fixture 的每文件行数，保留 10 文件、多轮读取结构；"
            "不修改 final-test-v2 的 fixture，也不靠提高全局 `max_seq_length` 掩盖数据问题。",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="audit CTX-HARD token composition")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--tokenizer-gate", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=3072)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    tools = default_tool_schemas()
    rows = _load_rows((args.train_file, args.val_file))
    gate = json.loads(args.tokenizer_gate.read_text(encoding="utf-8"))
    over_ids = {
        str(item["id"])
        for split in ("train", "val")
        for item in (gate.get(split) or {}).get("over_max_length") or []
    }
    ctx_rows = [
        row
        for row in rows
        if str(row.get("id") or "") in over_ids and "CTX-HARD" in str(row.get("id") or "")
    ][: args.limit]
    controls = [
        row
        for row in rows
        if str(row.get("id") or "") not in over_ids and "C1-BASE" in str(row.get("id") or "")
    ][:1]
    selected = [*ctx_rows, *controls]
    if not ctx_rows:
        raise SystemExit("no over-length CTX-HARD rows found in the tokenizer gate")
    samples = [analyze_row(row, tokenizer, tools) for row in selected]
    ctx_samples = samples[: len(ctx_rows)]
    summary = {
        "ctx_rows": len(ctx_samples),
        "fixture_share_mean": sum(item["fixture_share"] for item in ctx_samples) / len(ctx_samples),
        "tools_share_mean": sum(item["tools_template_share"] for item in ctx_samples)
        / len(ctx_samples),
        "training_render_matches_direct": all(
            item["training_render_matches_direct"] for item in samples
        ),
    }
    root_cause = "fixture 太长"
    if summary["fixture_share_mean"] < 0.5 and summary["tools_share_mean"] > 0.4:
        root_cause = "渲染膨胀（tools= 模板）"
    payload = {
        "script": "scripts/audit_ctx_hard_tokens.py",
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
        "tokenizer_gate": str(args.tokenizer_gate),
        "tokenizer_path": str(args.tokenizer_path),
        "max_length": args.max_length,
        "root_cause": root_cause,
        "summary": summary,
        "samples": samples,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.out_md.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
