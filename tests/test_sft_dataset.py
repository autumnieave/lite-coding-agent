"""SFT dataset builder tests; no teacher API and no model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.build_sft_dataset import (
    DEFAULT_TEACHER_MODEL,
    TEACHER_CONVERTER_VERSION,
    TEACHER_PROTOCOL,
    VAL_TASK_IDS,
    _kind_counts,
    _pass_rates,
    a_c1_hard_regression,
    build_example,
    build_examples,
    deterministic_gate,
    load_episode,
    normalize_teacher_messages,
    split_for_task,
    task_split,
    trace_to_example,
    validate_teacher_messages,
    verify_no_final_test_overlap,
    verify_train_split,
    verify_train_v1,
)


def _write_trace(
    path: Path,
    *,
    within_steps: bool = True,
    bad_result: bool = False,
    task_id: str = "U-TR-A-C1-BASE",
    rollout_index: int = 0,
    completion: str = '{"path": "a.txt", "value": "value"}',
    tools_allowed: bool = True,
) -> None:
    run_start = {
        "event": "run_start",
        "task_id": task_id,
        "rollout_index": rollout_index,
        "split": "train",
        "constraints": [{"id": "T-C01", "content": "输出必须是合法 JSON"}],
    }
    tool_turn = {
        "event": "llm_turn",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "读取文件并输出 JSON"},
        ],
        "response": {
            "content": "",
            "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": '{"path": "a.txt"}'}],
        },
        "tool_calls": [
            {
                "tool_name": "read_file",
                "arguments": json.dumps({"path": "a.txt"}),
                "arguments_json_valid": True,
                "pydantic_valid": True,
                "path_escape": False,
                "blocked_at": None,
            }
        ],
        "tool_results": [
            {
                "tool_name": "read_file",
                "ok": not bad_result,
                "is_error": bad_result,
                "protocol_error": False,
                "error_kind": "tool_error" if bad_result else None,
            }
        ],
    }
    final_turn = {
        "event": "llm_turn",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "读取文件并输出 JSON"},
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
        ],
        "response": {"content": completion, "tool_calls": []},
    }
    run_end = {
        "event": "run_end",
        "checks": {
            "json_valid": True,
            "snake_case_keys": True,
            "value_correct": True,
            "required_tools_used": True,
            "tools_allowed": tools_allowed,
            "within_steps": within_steps,
        },
        "within_steps": within_steps,
    }
    path.write_text(
        "\n".join(
            json.dumps(event, ensure_ascii=False)
            for event in (run_start, tool_turn, final_turn, run_end)
        )
        + "\n",
        encoding="utf-8",
    )


def test_trace_to_example_keeps_tool_trajectory_and_completion(tmp_path: Path) -> None:
    trace = tmp_path / "train" / "U-TR-A-C1-BASE_rollout00.jsonl"
    trace.parent.mkdir(parents=True)
    _write_trace(trace)

    episode = load_episode(trace)
    assert episode is not None
    example = trace_to_example(episode)
    assert example is not None
    assert example["completion"] == '{"path": "a.txt", "value": "value"}'
    assert any(message.get("role") == "tool" for message in example["messages"])
    assert example["metadata"]["task_id"] == "U-TR-A-C1-BASE"
    assert example["metadata"]["split"] == "train"
    assert example["metadata"]["teacher_protocol"] == TEACHER_PROTOCOL
    assert example["metadata"]["converter_version"] == TEACHER_CONVERTER_VERSION
    assert example["metadata"]["converted_tool_calls"] == 0
    assert example["reward_context"]["expects_json"] is True


def test_trace_to_example_rejects_step_failure(tmp_path: Path) -> None:
    trace = tmp_path / "U-TR-A-C1-BASE_rollout00.jsonl"
    _write_trace(trace, within_steps=False)
    episode = load_episode(trace)
    assert episode is not None
    assert trace_to_example(episode) is None


def test_trace_to_example_rejects_bad_tool_result(tmp_path: Path) -> None:
    trace = tmp_path / "U-TR-A-C1-BASE_rollout00.jsonl"
    _write_trace(trace, bad_result=True)
    episode = load_episode(trace)
    assert episode is not None
    assert trace_to_example(episode) is None


def test_code_prefix_completion_is_rejected_with_specific_reason(tmp_path: Path) -> None:
    trace = tmp_path / "U-TR-A-CTX-HARD_rollout00.jsonl"
    _write_trace(
        trace,
        task_id="U-TR-A-CTX-HARD",
        completion="CODE=CTX-96003",
    )
    episode = load_episode(trace)
    assert episode is not None
    assert trace_to_example(episode) is None
    ok, reason = deterministic_gate(episode)
    assert not ok
    assert reason == "code_prefix_output"


def test_build_examples_keeps_multiple_rollouts_per_task(tmp_path: Path) -> None:
    for rollout in range(3):
        _write_trace(
            tmp_path / f"U-TR-A-C1-BASE_rollout{rollout:02d}.jsonl",
            rollout_index=rollout,
        )
    examples, rejected = build_examples(
        tmp_path,
        allowed_task_ids={"U-TR-A-C1-BASE"},
        rollouts=3,
        task_caps={"U-TR-A-C1-BASE": 2},
    )
    assert len(examples) == 2
    assert not rejected
    assert {row["metadata"]["rollout_index"] for row in examples} == {0, 1}


def test_task_split_is_disjoint_and_has_val() -> None:
    class Spec:
        def __init__(self, task_id: str) -> None:
            self.task_id = task_id

    specs = [Spec("U-TR-A-C1-BASE"), Spec("U-TR-B-C1-BASE")]
    train, val = task_split(specs)
    assert train.isdisjoint(val)
    assert val == {"U-TR-B-C1-BASE"}
    assert "U-TR-B-C1-BASE" in VAL_TASK_IDS


def test_split_for_task_matches_val_task_set() -> None:
    assert split_for_task("U-TR-B-C1-BASE") == "val"
    assert split_for_task("U-TR-A-C1-BASE") == "train"


def test_kind_counts_and_pass_rates_filter_by_split() -> None:
    difficulty = {"U-TR-A-BASE": "weak", "U-TR-B-HARD": "strong"}
    rows = [
        {"task_id": "U-TR-A-BASE", "split": "train", "task_success": True},
        {"task_id": "U-TR-B-HARD", "split": "val", "task_success": False},
    ]
    assert _kind_counts(rows, difficulty, split="train") == {"base": 1, "hard": 0}
    assert _kind_counts(rows, difficulty, split="val") == {"base": 0, "hard": 1}
    assert _pass_rates(rows, difficulty, split="train")["base"] == {
        "passed": 1,
        "total": 1,
        "rate": 1.0,
    }
    assert _pass_rates(rows, difficulty, split="val")["hard"] == {
        "passed": 0,
        "total": 1,
        "rate": 0.0,
    }


def test_normalize_hermes_fenced_tool_call_to_native_qwen_shape() -> None:
    messages = [
        {"role": "user", "content": "read a.txt"},
        {
            "role": "assistant",
            "content": '```json\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n```',
        },
        {"role": "tool", "content": "value"},
    ]
    normalized, converted = normalize_teacher_messages(messages)
    assert converted == 1
    call = normalized[1]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "a.txt"}
    assert normalized[1]["content"] == ""
    assert normalized[2]["role"] == "tool"
    assert normalized[2]["tool_call_id"] == call["id"]
    validate_teacher_messages(normalized)


def test_validate_teacher_messages_rejects_orphan_tool_result() -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        validate_teacher_messages([{"role": "tool", "tool_call_id": "missing", "content": "x"}])


def test_teacher_model_is_locked_and_protocol_recorded() -> None:
    assert DEFAULT_TEACHER_MODEL == "deepseek-flash"
    assert "openai_chat_completions" in TEACHER_PROTOCOL
    assert TEACHER_CONVERTER_VERSION.startswith("openai_and_hermes_to_qwen_chatml")


def test_train_v1_sha_check_and_final_test_overlap(tmp_path: Path) -> None:
    train_dir = tmp_path / "private" / "unified_dataset" / "train_v1"
    train_dir.mkdir(parents=True)
    payload = b'{"task_id": "U-TR-A-C1-BASE"}\n'
    (train_dir / "train.jsonl").write_bytes(payload)
    (train_dir / "TRAIN_FROZEN.json").write_text(
        json.dumps({"sha256": hashlib.sha256(payload).hexdigest()}),
        encoding="utf-8",
    )
    assert verify_train_v1(tmp_path) == hashlib.sha256(payload).hexdigest()

    final_dir = tmp_path / "private" / "unified_dataset" / "final_test_v2"
    final_dir.mkdir(parents=True)
    (final_dir / "final_test.jsonl").write_text(
        json.dumps({"task_id": "U-TR-A-C1-BASE"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap final-test-v2"):
        verify_no_final_test_overlap(tmp_path, {"U-TR-A-C1-BASE"})


def test_verify_train_split_checks_versioned_directory(tmp_path: Path) -> None:
    train_dir = tmp_path / "private" / "unified_dataset" / "train_v2"
    train_dir.mkdir(parents=True)
    payload = b'{"task_id": "U-TR-A-C1-BASE"}\n'
    (train_dir / "train.jsonl").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (train_dir / "TRAIN_FROZEN.json").write_text(
        json.dumps({"version": "train-v2", "sha256": digest}), encoding="utf-8"
    )
    assert verify_train_split(tmp_path, "train-v2") == digest
    train_v1 = tmp_path / "private" / "unified_dataset" / "train_v1"
    train_v1.mkdir(parents=True)
    (train_v1 / "train.jsonl").write_bytes(payload)
    (train_v1 / "TRAIN_FROZEN.json").write_text(
        json.dumps({"version": "train-v2", "sha256": digest}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="version mismatch"):
        verify_train_split(tmp_path, "train-v1")


def _write_loop_trace(
    path: Path,
    *,
    repeats: int = 3,
    answer_then_call: bool = False,
    continuation_after_answer: bool = False,
    task_id: str = "U-TR-A-C1-HARD",
) -> None:
    run_start = {
        "event": "run_start",
        "task_id": task_id,
        "rollout_index": 0,
        "constraints": [{"id": "T-C01", "content": "输出必须是合法 JSON"}],
    }
    turns = []
    for _ in range(repeats):
        turns.append(
            {
                "event": "llm_turn",
                "messages": [{"role": "user", "content": "read the file"}],
                "response": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "offline-0",
                            "name": "read_file",
                            "arguments": '{"path": "a.txt"}',
                        }
                    ],
                },
                "tool_calls": [
                    {
                        "tool_name": "read_file",
                        "arguments": '{"path": "a.txt"}',
                        "arguments_json_valid": True,
                        "pydantic_valid": True,
                        "path_escape": False,
                        "blocked_at": None,
                    }
                ],
                "tool_results": [
                    {
                        "tool_name": "read_file",
                        "ok": True,
                        "is_error": False,
                        "protocol_error": False,
                        "error_kind": None,
                    }
                ],
            }
        )
    turns.append(
        {
            "event": "llm_turn",
            "messages": [{"role": "user", "content": "read the file"}],
            "response": {"content": '{"path": "a.txt", "value": "v"}', "tool_calls": []},
        }
    )
    if answer_then_call:
        previous_messages = [
            {"role": "user", "content": "read the file"},
            {"role": "assistant", "content": '{"path": "a.txt", "value": "v"}'},
        ]
        turns.append(
            {
                "event": "llm_turn",
                "messages": previous_messages,
                "response": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "offline-0",
                            "name": "read_file",
                            "arguments": '{"path": "a.txt"}',
                        }
                    ],
                },
                "tool_calls": [
                    {
                        "tool_name": "read_file",
                        "arguments": '{"path": "a.txt"}',
                        "arguments_json_valid": True,
                        "pydantic_valid": True,
                        "path_escape": False,
                        "blocked_at": None,
                    }
                ],
                "tool_results": [
                    {
                        "tool_name": "read_file",
                        "ok": True,
                        "is_error": False,
                        "protocol_error": False,
                        "error_kind": None,
                    }
                ],
            }
        )
    if continuation_after_answer:
        turns.append(
            {
                "event": "llm_turn",
                "messages": [
                    {"role": "user", "content": "read the file"},
                    {"role": "assistant", "content": '{"path": "a.txt", "value": "v"}'},
                    {"role": "user", "content": "continue with the second batch"},
                ],
                "response": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "offline-1",
                            "name": "read_file",
                            "arguments": '{"path": "b.txt"}',
                        }
                    ],
                },
                "tool_calls": [
                    {
                        "tool_name": "read_file",
                        "arguments": '{"path": "b.txt"}',
                        "arguments_json_valid": True,
                        "pydantic_valid": True,
                        "path_escape": False,
                        "blocked_at": None,
                    }
                ],
                "tool_results": [
                    {
                        "tool_name": "read_file",
                        "ok": True,
                        "is_error": False,
                        "protocol_error": False,
                        "error_kind": None,
                    }
                ],
            }
        )
    run_end = {
        "event": "run_end",
        "checks": {"json_valid": True, "within_steps": True},
        "within_steps": True,
    }
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in [run_start, *turns, run_end])
        + "\n",
        encoding="utf-8",
    )


def test_deterministic_gate_rejects_repeated_tool_call(tmp_path: Path) -> None:
    trace = tmp_path / "loop.jsonl"
    _write_loop_trace(trace, repeats=3)
    episode = load_episode(trace)
    assert episode is not None
    ok, reason = deterministic_gate(episode)
    assert not ok
    assert reason == "repeated_tool_call"


def test_deterministic_gate_rejects_call_after_final_answer(tmp_path: Path) -> None:
    trace = tmp_path / "after_answer.jsonl"
    _write_loop_trace(trace, repeats=1, answer_then_call=True)
    episode = load_episode(trace)
    assert episode is not None
    ok, reason = deterministic_gate(episode)
    assert not ok
    assert reason == "tool_call_after_answer"


def test_deterministic_gate_allows_tool_call_after_new_user_message(tmp_path: Path) -> None:
    trace = tmp_path / "continuation.jsonl"
    _write_loop_trace(trace, repeats=1, continuation_after_answer=True)
    episode = load_episode(trace)
    assert episode is not None
    ok, reason = deterministic_gate(episode)
    assert ok
    assert reason == "ok"


def test_deterministic_gate_rejects_disallowed_tools(tmp_path: Path) -> None:
    trace = tmp_path / "tools.jsonl"
    _write_trace(trace, tools_allowed=False)
    episode = load_episode(trace)
    assert episode is not None
    ok, reason = deterministic_gate(episode)
    assert not ok
    assert reason == "tools_not_allowed"


def test_validate_teacher_messages_rejects_malformed_arguments() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{not-json"},
                }
            ],
        }
    ]
    with pytest.raises(ValueError, match="parseable JSON"):
        validate_teacher_messages(messages)


def test_validate_teacher_messages_accepts_content_before_tool_call() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "I'll read the file.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                }
            ],
        }
    ]
    validate_teacher_messages(messages)


def test_build_example_reports_format_rejection_reason(tmp_path: Path) -> None:
    trace = tmp_path / "bad_format.jsonl"
    _write_trace(trace)
    payload = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    payload[2]["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{not-json}"
    trace.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in payload) + "\n",
        encoding="utf-8",
    )
    episode = load_episode(trace)
    assert episode is not None
    example, reason, detail = build_example(episode)
    assert example is None
    assert reason == "format_invalid"
    assert "parseable JSON" in detail


def test_build_example_rejects_non_qwen_render(tmp_path: Path) -> None:
    trace = tmp_path / "wrong_template.jsonl"
    _write_trace(trace)
    episode = load_episode(trace)
    assert episode is not None

    class PlainTokenizer:
        def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt):
            return "plain prompt without qwen tool schema"

    example, reason, detail = build_example(
        episode,
        tokenizer=PlainTokenizer(),
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )
    assert example is None
    assert reason == "qwen_template_invalid"
    assert "not a Qwen chat template" in detail
    assert "tool schema missing" in detail


def test_a_c1_hard_regression_gate_flags_loop(tmp_path: Path) -> None:
    trace = tmp_path / "a_c1_hard.jsonl"
    _write_loop_trace(trace, repeats=6)
    episode = load_episode(trace)
    assert episode is not None
    ok, reason = a_c1_hard_regression(episode)
    assert not ok
    assert reason == "a_c1_hard_loop"
