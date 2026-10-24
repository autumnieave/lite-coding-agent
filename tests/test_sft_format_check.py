"""Rendered-text checks for the Qwen chat template; no model or tokenizer download."""

from __future__ import annotations

from training.check_sft_format import check_rendered_text

ROW = {
    "messages": [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "value"},
    ]
}

VALID = (
    "<|im_start|>assistant\n<tool_call>\n"
    '{"name": "read_file", "arguments": {"path": "a.txt"}}\n'
    "</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nvalue\n"
    "</tool_response><|im_end|>\n"
)


def test_valid_qwen_render_passes() -> None:
    assert check_rendered_text(VALID, ROW) == []


def test_string_arguments_are_reported() -> None:
    text = VALID.replace(
        '"arguments": {"path": "a.txt"}',
        '"arguments": "{\\"path\\": \\"a.txt\\"}"',
    )
    failures = check_rendered_text(text, ROW)
    assert any("arguments object" in item for item in failures)


def test_deepseek_marker_is_reported() -> None:
    failures = check_rendered_text(VALID + "<|User|>", ROW)
    assert any("deepseek marker" in item for item in failures)


def test_missing_tool_response_is_reported() -> None:
    text = VALID.replace("<tool_response>\nvalue\n</tool_response>", "plain")
    failures = check_rendered_text(text, ROW)
    assert any("tool_response blocks" in item for item in failures)


def test_extra_schema_example_block_is_allowed_explicitly() -> None:
    text = VALID.replace(
        "<|im_start|>assistant\n",
        "<|im_start|>system\n<tool_call>{}</tool_call><|im_end|>\n<|im_start|>assistant\n",
        1,
    )
    assert check_rendered_text(text, ROW)
    assert check_rendered_text(text, ROW, extra_tool_call_blocks=1) == []
