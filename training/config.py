"""QLoRA 配置校验；不导入 torch/transformers，方便本地单测。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_config(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_t4_config(config: Mapping[str, Any]) -> None:
    required = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "gradient_checkpointing": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"T4 配置要求 {key}={expected!r}，实际为 {config.get(key)!r}")
    max_seq_length = int(config.get("max_seq_length") or 0)
    if max_seq_length <= 0 or max_seq_length > 4096:
        raise ValueError(f"max_seq_length 必须在 1..4096，实际为 {max_seq_length}")
