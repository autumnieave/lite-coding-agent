"""Token-level loss masks for Qwen ChatML SFT rows.

``full`` trains on every token. ``assistant_only`` keeps only tokens generated
after an assistant header, including native tool-call JSON and the final answer;
system, user and tool-result tokens are masked with ``-100``. Masking tool
results prevents the model from learning to recite observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

ASSISTANT_HEADER = "<|im_start|>assistant\n"
MESSAGE_END = "<|im_end|>"
IGNORE_INDEX = -100
LOSS_MODES = ("full", "assistant_only")


def pad_token_rows(
    features: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
    label_pad_token_id: int = IGNORE_INDEX,
) -> dict[str, list[list[int]]]:
    """Right-pad pre-tokenized rows while preserving their existing labels.

    ``transformers.DataCollatorForLanguageModeling`` rebuilds ``labels`` from
    ``input_ids``, which silently turns assistant-only training into full
    sequence training. This helper keeps caller-provided labels intact.
    """

    if not features:
        raise ValueError("features must not be empty")
    max_length = max(len(row["input_ids"]) for row in features)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    for row in features:
        row_ids = [int(value) for value in row["input_ids"]]
        row_mask = [int(value) for value in row.get("attention_mask") or [1] * len(row_ids)]
        row_labels = [int(value) for value in row["labels"]] if "labels" in row else list(row_ids)
        if not (len(row_ids) == len(row_mask) == len(row_labels)):
            raise ValueError("input_ids, attention_mask and labels must have equal length")
        padding = max_length - len(row_ids)
        input_ids.append(row_ids + [pad_token_id] * padding)
        attention_mask.append(row_mask + [0] * padding)
        labels.append(row_labels + [label_pad_token_id] * padding)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class LabelPreservingCollator:
    """Collator for pre-tokenized SFT rows with explicit loss masks."""

    def __init__(self, pad_token_id: int, label_pad_token_id: int = IGNORE_INDEX) -> None:
        self.pad_token_id = int(pad_token_id)
        self.label_pad_token_id = int(label_pad_token_id)

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        batch = pad_token_rows(
            features,
            pad_token_id=self.pad_token_id,
            label_pad_token_id=self.label_pad_token_id,
        )
        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}


def _find_sequence(haystack: Sequence[int], needle: Sequence[int], start: int) -> int:
    if not needle:
        return -1
    last = len(haystack) - len(needle)
    for index in range(start, last + 1):
        if list(haystack[index : index + len(needle)]) == list(needle):
            return index
    return -1


def assistant_token_mask(input_ids: Sequence[int], tokenizer: Any) -> list[bool]:
    """Return a mask for generated assistant spans in a ChatML token sequence."""

    header = list(tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False))
    end = list(tokenizer.encode(MESSAGE_END, add_special_tokens=False))
    if not header:
        raise ValueError("tokenizer produced an empty assistant header")
    mask = [False] * len(input_ids)
    cursor = 0
    while cursor <= len(input_ids) - len(header):
        if list(input_ids[cursor : cursor + len(header)]) != header:
            cursor += 1
            continue
        content_start = cursor + len(header)
        end_start = _find_sequence(input_ids, end, content_start)
        content_end = len(input_ids) if end_start < 0 else end_start + len(end)
        for index in range(content_start, content_end):
            mask[index] = True
        cursor = max(content_end, cursor + 1)
    return mask


def labels_for_loss(input_ids: Sequence[int], tokenizer: Any, loss_mode: str) -> list[int]:
    """Build labels for one already-tokenized row."""

    if loss_mode not in LOSS_MODES:
        raise ValueError(f"unknown loss_mode: {loss_mode}")
    ids = [int(value) for value in input_ids]
    if loss_mode == "full":
        return ids
    mask = assistant_token_mask(ids, tokenizer)
    if not any(mask):
        raise ValueError("assistant_only loss has no assistant tokens")
    return [token if keep else IGNORE_INDEX for token, keep in zip(ids, mask, strict=True)]


def tokenize_training_row(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    loss_mode: str,
    max_length: int,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render one SFT row and return input ids, labels and attention mask."""

    from training.train_sft import render_training_text

    text = render_training_text(tokenizer, row, tools=tools)
    encoded = tokenizer(text, add_special_tokens=False)
    full_ids = [int(value) for value in encoded["input_ids"]]
    input_ids = full_ids[:max_length]
    try:
        labels = labels_for_loss(input_ids, tokenizer, loss_mode)
    except ValueError:
        if len(full_ids) <= max_length:
            raise
        # The row will be filtered out by its full token count. Keep a
        # placeholder mask so dataset.map does not fail on truncation.
        labels = [IGNORE_INDEX] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "token_count": len(full_ids),
    }
