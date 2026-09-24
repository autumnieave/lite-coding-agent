"""Transformers + bitsandbytes NF4 provider for 7B held-out runs on T4.

The vLLM provider remains the fast path. This provider exists so baseline and
adapter evaluation can use the same NF4 quantization path that QLoRA training
uses; applying a QLoRA adapter to an AWQ base would mix quantization schemes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from agent.core.llm import BaseProvider, LLMResponse
from training.kaggle.local_prompt import build_chat_prompt
from training.kaggle.offline_provider import classify_wire_format, parse_tool_calls


class TransformersOfflineProvider(BaseProvider):
    """In-process transformers generation with optional NF4 and a PEFT adapter."""

    def __init__(
        self,
        model: str,
        *,
        max_model_len: int = 8192,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        max_tokens: int = 512,
        adapter_path: str | None = None,
        quantization: str = "nf4",
        device_map: str | dict[str, int] | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.adapter_path = adapter_path
        self.quantization = quantization
        load_kwargs: dict[str, Any] = {
            # Keep inference on one GPU so it matches single-GPU QLoRA training
            # and avoids device_map sharding on Kaggle T4 x2 sessions.
            "device_map": device_map if device_map is not None else {"": 0},
            "trust_remote_code": True,
        }
        if quantization == "nf4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        loaded = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        if adapter_path:
            from peft import PeftModel

            loaded = PeftModel.from_pretrained(loaded, adapter_path)
        loaded.eval()
        self._model = loaded
        self._torch = torch
        self._lock = asyncio.Lock()

    @property
    def sampling_config(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "is_default": self.temperature is None and self.top_p is None and self.seed is None,
            "provider": "transformers-offline",
            "quantization": self.quantization,
            "adapter_path": self.adapter_path,
        }

    def _build_prompt(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
    ) -> str:
        return build_chat_prompt(self._tokenizer, messages, tools)

    def _generate(self, prompt: str) -> tuple[str, int, int]:
        torch = self._torch
        if self.seed is not None:
            torch.manual_seed(self.seed)
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        if prompt_tokens >= self.max_model_len:
            raise ValueError(f"prompt_tokens={prompt_tokens} >= max_model_len={self.max_model_len}")
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": min(self.max_tokens, self.max_model_len - prompt_tokens),
            "do_sample": bool(self.temperature and self.temperature > 0),
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = float(self.temperature or 0.0)
            if self.top_p is not None:
                generate_kwargs["top_p"] = float(self.top_p)
        with torch.inference_mode():
            output = self._model.generate(**inputs, **generate_kwargs)
        new_tokens = output[0][prompt_tokens:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, prompt_tokens, int(new_tokens.shape[-1])

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        prompt = self._build_prompt(messages, tools)
        async with self._lock:
            text, prompt_tokens, completion_tokens = await asyncio.to_thread(self._generate, prompt)
        content, calls = parse_tool_calls(text, tools)
        metadata = {
            "provider": "transformers-offline",
            "model_version": self.model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "sampling_config": self.sampling_config,
            "wire_format": classify_wire_format(text, calls),
            "raw_reply_head": text[:400],
            "request_id": None,
            "system_fingerprint": None,
        }
        return LLMResponse(content=content, tool_calls=tuple(calls), metadata=metadata)

    async def aclose(self) -> None:
        return None
