"""In-process vLLM provider used when the OpenAI-compatible server cannot bind.

The core agent only depends on ``BaseProvider``; this adapter keeps the vLLM
runtime inside ``training/`` and never changes ``core/loop.py``. It is a
fallback for Kaggle staging, not a replacement for the OpenAI-compatible
provider in the main experiment.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from agent.core.llm import BaseProvider, LLMResponse, ToolCall
from training.kaggle.local_prompt import build_chat_prompt

TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FENCED_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
SELF_CHECK_MARKER = "输出前先自检"

# Kaggle's T4 image (compute capability 7.5) cannot build FlashInfer's JIT
# kernels, and vLLM 0.12 otherwise selects FlashInfer for this model. The
# server path already pins the same fallback; keep the offline path identical
# so both providers exercise the same attention backend.
VLLM_ENV_DEFAULTS = {
    "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}


def classify_wire_format(text: str, calls: Sequence[Any]) -> str:
    """Classify how the model expressed tool calls in its raw output."""

    if not calls:
        return "no_tool_call"
    if "<tool_call>" in text:
        return "qwen_tool_call_xml"
    if "```" in text:
        return "fenced_json"
    return "json_object"


GUIDED_DECODING_MODULES = (
    "vllm.sampling_params",
    "vllm.config",
    "vllm.model_executor.guided_decoding",
    "vllm.entrypoints.openai.protocol",
)


def configure_vllm_env(environ: Any | None = None) -> dict[str, str]:
    """Apply T4-safe vLLM defaults without overriding explicit settings."""

    target = os.environ if environ is None else environ
    for key, value in VLLM_ENV_DEFAULTS.items():
        target.setdefault(key, value)
    return {key: target[key] for key in VLLM_ENV_DEFAULTS}


def load_guided_decoding_params() -> Any:
    """Load vLLM's GuidedDecodingParams across 0.6-0.12 layout changes."""

    errors: list[str] = []
    for module_name in GUIDED_DECODING_MODULES:
        try:
            module = importlib.import_module(module_name)
            return module.GuidedDecodingParams
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise RuntimeError("guided_decoding_unavailable: " + "; ".join(errors))


configure_vllm_env()


def should_use_guided_decoding(
    messages: Sequence[Mapping[str, Any]],
    guided_json: Mapping[str, Any] | None,
) -> bool:
    """Apply guided JSON only on the final self-check turn of a JSON task.

    Tool-call turns must remain unconstrained, otherwise the decoder would
    force every turn to be a JSON object and the agent could never call a tool.
    The task-aware suffix is appended only to the final user turn and contains
    the word JSON for JSON-output tasks.
    """

    if not guided_json:
        return False
    last_user = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            last_user = str(message.get("content") or "")
            break
    return SELF_CHECK_MARKER in last_user and "json" in last_user.lower()


def _tool_payload(payload: Any, allowed_names: set[str]) -> tuple[str, str] | None:
    if not isinstance(payload, Mapping):
        return None
    function = payload.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments", function.get("parameters", {}))
    else:
        name = payload.get("name") or payload.get("tool")
        arguments = payload.get(
            "arguments",
            payload.get("parameters", payload.get("input", {})),
        )
    if not isinstance(name, str) or name not in allowed_names:
        return None
    if isinstance(arguments, str):
        serialized = arguments
    else:
        serialized = json.dumps(arguments, ensure_ascii=False)
    return name, serialized


def parse_tool_calls(
    text: str,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, list[ToolCall]]:
    """Extract Hermes-style tool calls and remove them from visible content.

    Qwen2.5-Instruct emits ``<tool_call>...</tool_call>`` when the chat template
    exposes tools. A fenced JSON fallback is accepted only when the object names
    an allowed tool, so ordinary JSON answers are not misread as tool calls.
    """

    allowed_names = {
        str(tool.get("function", {}).get("name"))
        for tool in (tools or ())
        if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
    }
    calls: list[ToolCall] = []

    def add(payload: Any) -> None:
        parsed = _tool_payload(payload, allowed_names)
        if parsed is None:
            return
        name, arguments = parsed
        calls.append(ToolCall(id=f"offline-{len(calls)}", name=name, arguments=arguments))

    valid_spans: list[tuple[int, int]] = []
    for match in TOOL_CALL_BLOCK.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        before = len(calls)
        add(payload)
        if len(calls) > before:
            valid_spans.append(match.span())
    content = text
    for start, end in reversed(valid_spans):
        content = content[:start] + content[end:]
    content = content.strip()

    if not calls:
        for match in FENCED_BLOCK.finditer(text):
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            add(payload)
        if calls:
            content = FENCED_BLOCK.sub("", text).strip()

    if not calls:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = None
        add(payload)
        if calls:
            content = ""
    return content, calls


class VLLMOfflineProvider(BaseProvider):
    """Run Qwen/vLLM in-process without an HTTP server."""

    def __init__(
        self,
        model: str,
        *,
        max_model_len: int = 8192,
        dtype: str = "float16",
        gpu_memory_utilization: float = 0.85,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        max_tokens: int = 512,
        enforce_eager: bool = True,
        guided_json: Mapping[str, Any] | None = None,
        adapter_path: str | None = None,
        quantization: str | None = None,
        load_format: str | None = None,
    ) -> None:
        configure_vllm_env()
        from vllm import LLM, SamplingParams

        sampling_kwargs: dict[str, Any] = {"max_tokens": max_tokens}
        if temperature is not None:
            sampling_kwargs["temperature"] = temperature
        if top_p is not None:
            sampling_kwargs["top_p"] = top_p
        if seed is not None:
            sampling_kwargs["seed"] = seed
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.guided_json = dict(guided_json) if guided_json else None
        self.adapter_path = adapter_path
        self.quantization = quantization
        self._sampling = SamplingParams(**sampling_kwargs)
        self._guided_sampling = None
        if self.guided_json:
            GuidedDecodingParams = load_guided_decoding_params()
            self._guided_sampling = SamplingParams(
                **sampling_kwargs,
                guided_decoding=GuidedDecodingParams(json=self.guided_json),
            )
        self._lora_request = None
        if adapter_path:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("sft", 1, adapter_path)
        llm_kwargs: dict[str, Any] = {
            "model": model,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "trust_remote_code": True,
            "enable_lora": bool(adapter_path),
            "max_lora_rank": 64,
        }
        if quantization:
            llm_kwargs["quantization"] = quantization
        if load_format:
            llm_kwargs["load_format"] = load_format
        self._llm = LLM(**llm_kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        self._lock = asyncio.Lock()

    @property
    def sampling_config(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "is_default": self.temperature is None and self.top_p is None and self.seed is None,
            "provider": "vllm-offline",
            "guided_json": bool(self.guided_json),
            "adapter_path": self.adapter_path,
            "quantization": self.quantization,
        }

    def _build_prompt(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
    ) -> str:
        return build_chat_prompt(self._tokenizer, messages, tools)

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        prompt = self._build_prompt(messages, tools)
        sampling = (
            self._guided_sampling
            if self._guided_sampling is not None
            and should_use_guided_decoding(messages, self.guided_json)
            else self._sampling
        )
        generate_kwargs: dict[str, Any] = {"use_tqdm": False}
        if self._lora_request is not None:
            generate_kwargs["lora_request"] = self._lora_request
        async with self._lock:
            outputs = await asyncio.to_thread(
                self._llm.generate,
                [prompt],
                sampling,
                **generate_kwargs,
            )
        request_output = outputs[0]
        text = request_output.outputs[0].text
        content, calls = parse_tool_calls(text, tools)
        metadata: dict[str, Any] = {
            "provider": "vllm-offline",
            "model_version": self.model,
            "prompt_tokens": len(getattr(request_output, "prompt_token_ids", ()) or ()),
            "completion_tokens": len(getattr(request_output.outputs[0], "token_ids", ()) or ()),
            "sampling_config": self.sampling_config,
            "wire_format": classify_wire_format(text, calls),
            "raw_reply_head": text[:400],
        }
        return LLMResponse(content=content, tool_calls=tuple(calls), metadata=metadata)

    async def aclose(self) -> None:
        return None
