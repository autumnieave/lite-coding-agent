"""Constrained decoding 本地测试：不加载模型、不联网。"""

from __future__ import annotations

from training.constrained_decoding import (
    DEFAULT_REQUIRED_KEYS,
    _generate_one,
    build_summary_schema,
    generate_records,
    record_generation,
    run_with_outlines,
    schema_accepts,
    validate_output,
)


def _valid_payload() -> dict[str, object]:
    payload: dict[str, object] = {key: "x" for key in DEFAULT_REQUIRED_KEYS}
    payload["files_read"] = []
    payload["constraint_codes"] = []
    payload["files_read_count"] = 0
    payload["lines_per_file"] = 0
    payload["constraint_codes_count"] = 0
    payload["key_findings"] = {"content_nature": "x"}
    return payload


def test_schema_accepts_only_snake_case_keys() -> None:
    schema = build_summary_schema()
    payload = _valid_payload()
    assert schema_accepts(payload, schema)

    payload["fieldBad"] = 1
    assert not schema_accepts(payload, schema)


def test_nested_known_object_also_requires_snake_case() -> None:
    schema = build_summary_schema()
    payload = _valid_payload()
    payload["key_findings"] = {"badKey": 1}
    assert not schema_accepts(payload, schema)


def test_schema_requires_contract_keys() -> None:
    schema = build_summary_schema()
    payload = _valid_payload()
    payload.pop("session_summary")
    assert not schema_accepts(payload, schema)


def test_validate_output_rejects_trailing_text_and_fence() -> None:
    assert validate_output('{"a_b": 1}')["json_valid"] is True
    trailing = validate_output('{"a_b": 1}\n说明')
    assert trailing["json_valid"] is False
    assert trailing["snake_case_keys"] is False
    assert validate_output('```json\n{"a_b": 1}\n```')["no_code_fence"] is False


def test_generate_one_prefers_outlines_max_tokens_kwarg() -> None:
    calls: list[tuple[object, ...]] = []

    def generator(prompt: str, max_tokens: int | None = None) -> str:
        calls.append((prompt, max_tokens))
        return "ok"

    assert _generate_one(generator, "p", {"type": "object"}, 32) == "ok"
    assert calls == [("p", 32)]


def test_generate_one_supports_max_new_tokens_api() -> None:
    calls: list[tuple[object, ...]] = []

    def generator(prompt: str, max_new_tokens: int | None = None) -> str:
        calls.append((prompt, max_new_tokens))
        return "ok"

    assert _generate_one(generator, "p", {"type": "object"}, 32) == "ok"
    assert calls == [("p", 32)]


def test_generate_one_falls_back_for_legacy_generator() -> None:
    def generator(prompt: str, schema: object, max_tokens: int | None = None) -> str:
        assert schema == {"type": "object"}
        assert max_tokens == 32
        return "legacy"

    assert _generate_one(generator, "p", {"type": "object"}, 32) == "legacy"


def test_run_with_outlines_emits_records_incrementally(monkeypatch) -> None:
    monkeypatch.setattr(
        "training.constrained_decoding._load_outlines_generator",
        lambda model_path, schema, quantize_4bit=False: (object(), None),
    )
    monkeypatch.setattr(
        "training.constrained_decoding._generate_one",
        lambda generator, prompt, schema, max_new_tokens: "{}",
    )
    emitted: list[dict[str, object]] = []
    prompts = [
        {"run_id": 2, "rollout_index": 0, "prompt": "a"},
        {"run_id": 10, "rollout_index": 0, "prompt": "b"},
    ]
    records = run_with_outlines(prompts, model_path="/model", on_record=emitted.append)
    assert len(records) == 2
    assert [record["run_id"] for record in emitted] == [2, 10]


def test_generate_records_turns_unexpected_errors_into_rows() -> None:
    def generator(prompt: str, max_tokens: int | None = None) -> str:
        if "bad" in prompt:
            raise ValueError("adapter exploded")
        return "{}"

    emitted: list[dict[str, object]] = []
    records = generate_records(
        generator,
        [
            {"run_id": 2, "rollout_index": 0, "prompt": "bad"},
            {"run_id": 2, "rollout_index": 1, "prompt": "good"},
        ],
        schema={"type": "object"},
        model_path="/model",
        max_new_tokens=16,
        on_record=emitted.append,
    )
    assert len(records) == 2
    assert records[0]["error_kind"] == "unknown"
    assert "adapter exploded" in records[0]["error"]
    assert records[1].get("error_kind") is None
    assert [row["rollout_index"] for row in emitted] == [0, 1]


def test_record_generation_keeps_metadata_and_schema_verdict() -> None:
    schema = build_summary_schema()
    output = '{"session_summary": "x"}'
    record = record_generation(
        run_id=2,
        rollout_index=0,
        prompt="输出 JSON",
        output=output,
        schema=schema,
        model_path="/kaggle/input/model",
        duration_s=1.234,
        model_version="local-test",
    )
    assert record["validation"]["json_valid"] is True
    assert record["schema_accepts"] is False
    assert record["metadata"]["system_fingerprint"] == "N/A (local outlines)"
    assert record["sampling_config"]["seed"] == 20261019
