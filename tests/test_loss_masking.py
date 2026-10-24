"""Loss-mask tests use a character tokenizer; no model download is required."""

from __future__ import annotations

from typing import Any

import pytest

from training.loss_masking import (
    assistant_token_mask,
    labels_for_loss,
    pad_token_rows,
    tokenize_training_row,
)


class _CharTokenizer:
    """Encode each character as one token and return a fixed ChatML transcript."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = False,
        **_: Any,
    ) -> dict[str, list[int]]:
        values = self.encode(text)
        if truncation and max_length is not None:
            values = values[:max_length]
        return {"input_ids": values, "attention_mask": [1] * len(values)}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        assert tools
        parts: list[str] = []
        for message in messages:
            role = message["role"]
            content = message.get("content") or ""
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            for call in message.get("tool_calls") or []:
                parts.append(f"CALL {call['function']['name']} {call['function']['arguments']}\n")
        return "".join(parts)


def _text() -> str:
    return (
        "<|im_start|>system\nsystem-only<|im_end|>\n"
        "<|im_start|>user\nuser-only<|im_end|>\n"
        "<|im_start|>assistant\nCALL read_file args<|im_end|>\n"
        "<|im_start|>tool\ntool-output-secret<|im_end|>\n"
        "<|im_start|>assistant\nFINAL-ANSWER<|im_end|>\n"
    )


def test_full_loss_keeps_every_token() -> None:
    tokenizer = _CharTokenizer()
    ids = tokenizer.encode(_text())
    assert labels_for_loss(ids, tokenizer, "full") == ids


def test_assistant_only_masks_prompt_and_tool_result() -> None:
    tokenizer = _CharTokenizer()
    ids = tokenizer.encode(_text())
    labels = labels_for_loss(ids, tokenizer, "assistant_only")
    kept = "".join(chr(token) for token, label in zip(ids, labels, strict=True) if label != -100)

    assert "system-only" not in kept
    assert "user-only" not in kept
    assert "tool-output-secret" not in kept
    assert "CALL read_file" in kept
    assert "FINAL-ANSWER" in kept
    assert labels[-2] != -100


def test_assistant_token_mask_returns_one_span_per_assistant_message() -> None:
    tokenizer = _CharTokenizer()
    ids = tokenizer.encode(_text())
    mask = assistant_token_mask(ids, tokenizer)
    decoded = "".join(chr(token) for token, keep in zip(ids, mask, strict=True) if keep)
    assert decoded.count("FINAL-ANSWER") == 1
    assert decoded.count("CALL read_file") == 1
    assert "tool-output-secret" not in decoded


def test_tokenize_training_row_rejects_row_without_assistant_span() -> None:
    class NoAssistantTokenizer(_CharTokenizer):
        def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
            return "<|im_start|>system\nonly system<|im_end|>\n"

    row = {
        "messages": [{"role": "user", "content": "x"}],
        "completion": "y",
    }
    with pytest.raises(ValueError, match="no assistant tokens"):
        tokenize_training_row(
            NoAssistantTokenizer(),
            row,
            loss_mode="assistant_only",
            max_length=128,
        )


def test_pad_token_rows_preserves_assistant_only_labels() -> None:
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [4, 5]},
    ]
    batch = pad_token_rows(features, pad_token_id=0, label_pad_token_id=-100)

    assert batch["input_ids"] == [[1, 2, 3], [4, 5, 0]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"] == [[-100, 2, 3], [4, 5, -100]]


def test_pad_token_rows_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        pad_token_rows(
            [{"input_ids": [1, 2], "attention_mask": [1], "labels": [1, 2]}],
            pad_token_id=0,
        )


def test_tokenize_training_row_reports_full_length_for_filtering() -> None:
    tokenizer = _CharTokenizer()
    row = {
        "messages": [{"role": "user", "content": "x" * 200}],
        "completion": "FINAL",
    }

    encoded = tokenize_training_row(
        tokenizer,
        row,
        loss_mode="assistant_only",
        max_length=20,
    )

    assert len(encoded["input_ids"]) == 20
    assert encoded["token_count"] > 20
    assert encoded["labels"] == [-100] * 20
