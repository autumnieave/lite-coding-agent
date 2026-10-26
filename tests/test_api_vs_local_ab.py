from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.api_vs_local_ab import ApiUsageProvider, api_key_from_env, summarize_runs


class _Usage:
    def model_dump(self, *, exclude_none: bool = True) -> dict[str, Any]:
        assert exclude_none is True
        return {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
            "completion_tokens_details": {"reasoning_tokens": 2},
        }


class _Completions:
    def __init__(self, *, content: str = "", tool_calls: list[Any] | None = None) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.content = content
        self.tool_calls = tool_calls or []

    async def create(self, **payload: Any) -> Any:
        self.payloads.append(payload)
        message = SimpleNamespace(content=self.content, tool_calls=self.tool_calls)
        choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        return SimpleNamespace(
            choices=[choice],
            usage=_Usage(),
            model="deepseek-v4.1-flash",
            id="resp-1",
            request_id="req-1",
            created=123,
            system_fingerprint=None,
        )


def _provider(completions: _Completions) -> ApiUsageProvider:
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return ApiUsageProvider(
        client=client,  # type: ignore[arg-type]
        model="deepseek-v4.1-flash",
        base_url="https://example.invalid",
        temperature=0.7,
        top_p=0.9,
    )


def test_provider_preserves_native_tool_calls_and_usage() -> None:
    raw_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="grep", arguments='{"pattern":"x"}'),
    )
    completions = _Completions(tool_calls=[raw_call])
    provider = _provider(completions)

    response = asyncio.run(
        provider.chat([{"role": "user", "content": "search"}], tools=[{"type": "function"}])
    )

    assert response.content == ""
    assert [(call.id, call.name, call.arguments) for call in response.tool_calls] == [
        ("call-1", "grep", '{"pattern":"x"}')
    ]
    assert response.metadata["usage"]["total_tokens"] == 16
    assert completions.payloads[0]["temperature"] == 0.7
    assert completions.payloads[0]["top_p"] == 0.9
    assert "seed" not in completions.payloads[0]
    audit = provider.audit()
    assert audit["requests"] == 1
    assert audit["total_tokens"] == 16
    assert audit["reasoning_tokens"] == 2
    assert audit["api_tool_call_requests"] == 1
    assert audit["fenced_json_turns"] == 0


def test_provider_marks_fenced_json_content() -> None:
    provider = _provider(_Completions(content='```json\n{"matches": []}\n```'))

    response = asyncio.run(provider.chat([{"role": "user", "content": "answer"}]))

    assert response.tool_calls == ()
    assert provider.audit()["fenced_json_turns"] == 1


def test_api_key_environment_order_and_error() -> None:
    assert api_key_from_env({"LLM_API_KEY": "fallback", "DASHSCOPE_API_KEY": "primary"}) == (
        "primary",
        "DASHSCOPE_API_KEY",
    )
    assert api_key_from_env({"LLM_API_KEY": "fallback"}) == ("fallback", "LLM_API_KEY")
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        api_key_from_env({})


def test_summarize_runs_computes_equal_task_rate_and_stable_tasks() -> None:
    rows = []
    for task_id, success_count in (("A", 5), ("B", 10)):
        for index in range(10):
            rows.append(
                {
                    "task_id": task_id,
                    "rollout_index": index,
                    "task_success": index < success_count,
                    "within_steps": True,
                    "tool_selection_ok": True,
                    "duration_s": 2.0,
                    "api_audit": {
                        "requests": 2,
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "api_tool_call_requests": 1,
                        "fenced_json_turns": 0,
                        "request_latencies_s": [1.0, 1.0],
                        "model_versions": ["deepseek-v4.1-flash"],
                    },
                }
            )
    rows.append({"task_id": "B", "rollout_index": 10, "error": "boom", "error_kind": "api"})

    summary = summarize_runs(
        rows,
        task_ids=["A", "B"],
        model="deepseek-v4.1-flash",
        base_url="https://example.invalid",
        task_sha256="task-sha",
        source_sha256="source-sha",
    )

    assert summary["tasks_with_valid_rollouts"] == 2
    assert summary["rollouts_valid"] == 20
    assert summary["error_rows"] == 1
    assert summary["equal_task_success_rate"] == pytest.approx(0.75)
    assert summary["stable_tasks"] == 1
    assert summary["stable_rate"] == pytest.approx(0.5)
    assert summary["total_tokens"] == 300
    assert summary["api_requests"] == 40
    assert summary["api_tool_call_requests"] == 20
    assert summary["request_latency_median_s"] == pytest.approx(1.0)
