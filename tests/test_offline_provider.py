"""Offline vLLM adapter tests; no GPU or model is loaded."""

from __future__ import annotations

import json
import os

import pytest

from training.kaggle import offline_provider as offline_module
from training.kaggle.offline_provider import (
    classify_wire_format,
    parse_tool_calls,
    should_use_guided_decoding,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
]


def test_parse_hermes_tool_call_and_strip_block() -> None:
    text = (
        "I will inspect it.\n"
        '<tool_call>{"name": "read_file", "arguments": {"path": "project/a.txt"}}</tool_call>'
    )
    content, calls = parse_tool_calls(text, TOOLS)
    assert content == "I will inspect it."
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert json.loads(calls[0].arguments) == {"path": "project/a.txt"}


def test_parse_fenced_json_tool_call() -> None:
    text = '```json\n{"name": "read_file", "arguments": {"path": "project/a.txt"}}\n```'
    content, calls = parse_tool_calls(text, TOOLS)
    assert content == ""
    assert calls[0].name == "read_file"


def test_plain_json_answer_is_not_misread_as_tool_call() -> None:
    text = '{"path": "project/a.txt", "value": "mode = x"}'
    content, calls = parse_tool_calls(text, TOOLS)
    assert calls == []
    assert content == text


def test_unknown_tool_name_is_left_as_content() -> None:
    text = '<tool_call>{"name": "shell", "arguments": {"cmd": "rm -rf /"}}</tool_call>'
    content, calls = parse_tool_calls(text, TOOLS)
    assert calls == []
    assert "shell" in content


def test_configure_vllm_env_sets_t4_defaults(monkeypatch) -> None:
    from training.kaggle.offline_provider import configure_vllm_env

    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)

    resolved = configure_vllm_env()

    assert resolved["VLLM_ATTENTION_BACKEND"] == "TRITON_ATTN"
    assert resolved["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert os.environ["VLLM_ATTENTION_BACKEND"] == "TRITON_ATTN"


def test_configure_vllm_env_keeps_explicit_override(monkeypatch) -> None:
    from training.kaggle.offline_provider import configure_vllm_env

    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASHINFER")
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)

    resolved = configure_vllm_env()

    assert resolved["VLLM_ATTENTION_BACKEND"] == "FLASHINFER"
    assert resolved["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_guided_decoding_only_on_final_json_self_check_turn() -> None:
    schema = {"type": "object"}
    tool_turn = [
        {"role": "system", "content": "工具约束"},
        {"role": "user", "content": "读取文件"},
    ]
    final_turn = [
        {"role": "system", "content": "工具约束"},
        {
            "role": "user",
            "content": "输出严格 JSON。\n\n输出前先自检：最终回复必须恰好是一个 JSON 对象。",
        },
    ]
    plain_turn = [
        {
            "role": "user",
            "content": "只输出该值。\n\n输出前先自检：最终回复只输出任务要求的最终答案。",
        }
    ]
    assert should_use_guided_decoding(tool_turn, schema) is False
    assert should_use_guided_decoding(final_turn, schema) is True
    assert should_use_guided_decoding(plain_turn, schema) is False
    assert should_use_guided_decoding(final_turn, None) is False


def test_load_guided_decoding_params_tries_vllm_layouts(monkeypatch) -> None:
    class FakeGuidedDecodingParams:
        pass

    class FakeModule:
        GuidedDecodingParams = FakeGuidedDecodingParams

    def fake_import(name: str):
        if name == "vllm.config":
            return FakeModule
        raise ImportError(name)

    monkeypatch.setattr(offline_module.importlib, "import_module", fake_import)
    assert offline_module.load_guided_decoding_params() is FakeGuidedDecodingParams


def test_load_guided_decoding_params_reports_unavailable(monkeypatch) -> None:
    def fake_import(name: str):
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(offline_module.importlib, "import_module", fake_import)
    with pytest.raises(RuntimeError, match="guided_decoding_unavailable"):
        offline_module.load_guided_decoding_params()


def test_classify_wire_format_distinguishes_qwen_and_fenced() -> None:
    calls = [object()]
    assert classify_wire_format("<tool_call>{}</tool_call>", calls) == "qwen_tool_call_xml"
    assert classify_wire_format("```json\n{}\n```", calls) == "fenced_json"
    assert classify_wire_format('{"name": "x"}', calls) == "json_object"
    assert classify_wire_format("plain answer", []) == "no_tool_call"
