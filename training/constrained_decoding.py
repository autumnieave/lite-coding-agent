"""Constrained decoding baseline 的本地可测部分。

本地只做 JSON Schema 构造、输出校验和结果记录；Outlines / transformers 的模型加载与
推理在 `run_with_outlines` 中延迟导入，单测不会加载模型、不会联网。
"""

from __future__ import annotations

import inspect
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from training.kaggle.contracts import classify_exception, time_limit

SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")
DEFAULT_REQUIRED_KEYS = (
    "session_summary",
    "timestamp_utc",
    "operation_mode",
    "files_read",
    "files_read_count",
    "lines_per_file",
    "key_findings",
    "constraint_codes",
    "constraint_codes_count",
    "next_action",
)
NESTED_OBJECT_KEYS = ("key_findings", "unresolved_conflicts")


def collect_keys(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        keys = [str(key) for key in payload]
        for value in payload.values():
            keys.extend(collect_keys(value))
        return keys
    if isinstance(payload, list):
        keys: list[str] = []
        for item in payload:
            keys.extend(collect_keys(item))
        return keys
    return []


def build_summary_schema(
    required_keys: Sequence[str] = DEFAULT_REQUIRED_KEYS,
    *,
    additional_properties: bool = True,
) -> dict[str, Any]:
    """构造 summary JSON Schema。

    顶层和已知嵌套对象都用 `propertyNames.pattern` 约束 snake_case；`required` 来自
    任务契约中固定的字段。Outlines 对 `propertyNames` 的支持取决于版本，Kaggle 脚本会
    在启动时打印实际 schema 并记录后端版本。
    """

    properties: dict[str, Any] = {key: {} for key in required_keys}
    for key in NESTED_OBJECT_KEYS:
        if key in properties:
            properties[key] = {
                "type": "object",
                "propertyNames": {"pattern": SNAKE_CASE.pattern},
                "additionalProperties": True,
            }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "propertyNames": {"pattern": SNAKE_CASE.pattern},
        "required": list(required_keys),
        "additionalProperties": additional_properties,
        "properties": properties,
    }


def validate_output(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text.strip())
    except (TypeError, ValueError):
        payload = None
    keys = collect_keys(payload)
    snake_ok = bool(keys) and all(SNAKE_CASE.fullmatch(key) for key in keys)
    return {
        "json_valid": payload is not None,
        "snake_case_keys": snake_ok,
        "no_code_fence": "```" not in text,
        "parsed": payload,
    }


def schema_accepts(payload: Any, schema: Mapping[str, Any]) -> bool:
    """校验本模块用到的 JSON Schema 子集，不冒充完整 draft 实现。"""

    if schema.get("type") == "object":
        if not isinstance(payload, dict):
            return False
        pattern = (schema.get("propertyNames") or {}).get("pattern")
        if pattern and any(SNAKE_CASE.fullmatch(str(key)) is None for key in payload):
            return False
        required = schema.get("required") or []
        if any(key not in payload for key in required):
            return False
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False and set(payload) - set(properties):
            return False
        for key, child_schema in properties.items():
            if (
                key in payload
                and isinstance(child_schema, Mapping)
                and child_schema.get("type") == "object"
                and not schema_accepts(payload[key], child_schema)
            ):
                return False
    return True


def record_generation(
    *,
    run_id: int,
    rollout_index: int,
    prompt: str,
    output: str,
    schema: Mapping[str, Any],
    model_path: str,
    duration_s: float,
    model_version: str | None = None,
    response_id: str | None = None,
    system_fingerprint: str | None = None,
    backend: str = "outlines",
) -> dict[str, Any]:
    validation = validate_output(output)
    parsed = validation.pop("parsed")
    return {
        "run_id": run_id,
        "rollout_index": rollout_index,
        "backend": backend,
        "prompt": prompt,
        "output": output,
        "validation": validation,
        "schema_accepts": schema_accepts(parsed, schema) if parsed is not None else False,
        "sampling_config": {
            "temperature": 0.0,
            "top_p": None,
            "seed": 20261019,
            "is_default": False,
        },
        "metadata": {
            "model_path": model_path,
            "model_version": model_version,
            "system_fingerprint": system_fingerprint or "N/A (local outlines)",
            "response_id": response_id,
            "duration_s": round(duration_s, 2),
        },
    }


def _load_outlines_generator(
    model_path: str,
    schema: Mapping[str, Any],
    *,
    quantize_4bit: bool = False,
) -> tuple[Any, Any]:
    try:
        import outlines
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "缺少 Outlines/transformers；Kaggle 环境请安装 training/requirements-kaggle.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        # 1.5B fits on a 16 GB T4 without bitsandbytes; fp16 avoids an extra
        # runtime dependency and keeps the baseline reproducible.
        load_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if hasattr(outlines, "from_transformers"):
        outlines_model = outlines.from_transformers(model, tokenizer)
        try:
            generator = outlines.generate.json(outlines_model, schema)
        except (AttributeError, TypeError):
            generator = outlines_model.json(schema)
    else:
        from outlines.models.transformers import Transformers

        generator = outlines.generate.json(Transformers(model, tokenizer), schema)
    return generator, tokenizer


def _generation_kwargs(generator: Any, max_new_tokens: int) -> dict[str, int]:
    """Map the token cap to the kwarg name used by the installed Outlines."""
    try:
        parameters = inspect.signature(generator).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "max_tokens" in parameters:
        return {"max_tokens": max_new_tokens}
    if "max_new_tokens" in parameters:
        return {"max_new_tokens": max_new_tokens}
    return {"max_tokens": max_new_tokens}


def _generate_one(
    generator: Any, prompt: str, schema: Mapping[str, Any], max_new_tokens: int
) -> str:
    # Outlines binds the schema when the generator is built. Passing the schema
    # again as a positional argument can be swallowed as an inference kwarg,
    # leaving generation without a token cap. Try the bound-schema form first;
    # only fall back for older generators that require an explicit schema.
    kwargs = _generation_kwargs(generator, max_new_tokens)
    try:
        return str(generator(prompt, **kwargs))
    except TypeError:
        return str(generator(prompt, schema, **kwargs))


def load_outlines_generator(
    model_path: str,
    schema: Mapping[str, Any],
    *,
    quantize_4bit: bool = False,
) -> tuple[Any, Any]:
    """Load the Outlines generator; separated so callers can time the load."""

    return _load_outlines_generator(model_path, schema, quantize_4bit=quantize_4bit)


def generate_records(
    generator: Any,
    prompts: Sequence[Mapping[str, Any]],
    *,
    schema: Mapping[str, Any],
    model_path: str,
    max_new_tokens: int,
    on_record: Callable[[dict[str, Any]], None] | None = None,
    progress_every: int = 0,
    inference_timeout_s: float | int | None = 0,
) -> list[dict[str, Any]]:
    """Generate records one by one and preserve a row when one prompt times out."""

    records: list[dict[str, Any]] = []
    total = len(prompts)
    for index, item in enumerate(prompts, start=1):
        started = time.perf_counter()
        try:
            with time_limit(inference_timeout_s, "inference_timeout"):
                output = _generate_one(generator, str(item["prompt"]), schema, max_new_tokens)
        except Exception as exc:  # noqa: BLE001 - one bad prompt must not hide later rows
            record = {
                "run_id": int(item["run_id"]),
                "rollout_index": int(item["rollout_index"]),
                "backend": "outlines",
                "prompt": str(item["prompt"]),
                "output": "",
                "validation": {
                    "json_valid": False,
                    "snake_case_keys": False,
                    "no_code_fence": False,
                },
                "schema_accepts": False,
                "sampling_config": {
                    "temperature": 0.0,
                    "top_p": None,
                    "seed": 20261019,
                    "is_default": False,
                },
                "error_kind": classify_exception(exc),
                "error": str(exc),
                "metadata": {
                    "model_path": model_path,
                    "duration_s": round(time.perf_counter() - started, 2),
                },
            }
        else:
            record = record_generation(
                run_id=int(item["run_id"]),
                rollout_index=int(item["rollout_index"]),
                prompt=str(item["prompt"]),
                output=output,
                schema=schema,
                model_path=model_path,
                duration_s=time.perf_counter() - started,
            )
        records.append(record)
        if on_record is not None:
            on_record(record)
        if progress_every and (index % progress_every == 0 or index == total):
            print(f"[constrained] {index}/{total} prompts processed", flush=True)
    return records


def run_with_outlines(
    prompts: Sequence[Mapping[str, Any]],
    *,
    model_path: str,
    max_new_tokens: int = 512,
    quantize_4bit: bool = False,
    on_record: Callable[[dict[str, Any]], None] | None = None,
    load_timeout_s: float | int | None = 0,
    inference_timeout_s: float | int | None = 0,
    progress_every: int = 0,
) -> list[dict[str, Any]]:
    """在 Kaggle 上执行 Outlines constrained decoding 的入口，本地不调用。"""

    schema = build_summary_schema()
    with time_limit(load_timeout_s, "model_load_timeout"):
        generator, _ = load_outlines_generator(model_path, schema, quantize_4bit=quantize_4bit)
    return generate_records(
        generator,
        prompts,
        schema=schema,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        on_record=on_record,
        progress_every=progress_every,
        inference_timeout_s=inference_timeout_s,
    )
