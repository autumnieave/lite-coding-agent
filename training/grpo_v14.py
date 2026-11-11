"""Pure helpers for the v14 action-layer GRPO experiment."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from agent.core.injection import CONSTRAINT_LAYER_HEADING
from agent.core.loop import DEFAULT_SYSTEM_PROMPT
from training.kaggle.local_prompt import build_chat_prompt
from training.kaggle.offline_provider import parse_tool_calls
from training.reward_v14 import score_reply

MAX_LOG_RATIO = 20.0


def completion_token_mask(
    token_ids: Sequence[int],
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> list[bool]:
    """Return a mask that keeps the first EOS and drops trailing padding.

    Qwen commonly uses the same id for EOS and padding. Applying a naive
    ``token_id != pad_token_id`` mask would therefore remove the first EOS as
    well, so padding is only filtered explicitly when the two ids differ.
    """

    if not token_ids:
        raise ValueError("token_ids must not be empty")
    mask = [True] * len(token_ids)
    if eos_token_id is not None:
        for index, token_id in enumerate(token_ids):
            if int(token_id) == int(eos_token_id):
                for trailing in range(index + 1, len(mask)):
                    mask[trailing] = False
                break
    pad_differs_from_eos = pad_token_id is not None and (
        eos_token_id is None or int(pad_token_id) != int(eos_token_id)
    )
    if pad_token_id is not None and pad_differs_from_eos:
        mask = [
            keep and int(token_id) != int(pad_token_id)
            for keep, token_id in zip(mask, token_ids, strict=True)
        ]
    if not any(mask):
        mask[0] = True
    return mask


def completion_loss_mask(
    token_ids: Sequence[int],
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
    enabled: bool = True,
) -> list[bool]:
    """Return the completion loss mask, optionally including trailing padding.

    ``enabled=True`` is the current C2 behaviour: keep the first EOS and drop
    trailing padding. ``enabled=False`` reproduces the v16-era denominator where
    padding tokens participate in the token mean.
    """

    if not token_ids:
        raise ValueError("token_ids must not be empty")
    if not enabled:
        return [True] * len(token_ids)
    return completion_token_mask(
        token_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )


def split_minibatches(indices: Sequence[int], mini_batch_size: int) -> list[list[int]]:
    """Split ordered sample indices into fixed-size minibatches."""

    if mini_batch_size <= 0:
        raise ValueError(f"mini_batch_size must be positive, got {mini_batch_size}")
    values = list(indices)
    return [
        values[start : start + mini_batch_size] for start in range(0, len(values), mini_batch_size)
    ]


def _clip_bounds(clip_eps: float) -> tuple[float, float]:
    if not 0.0 < clip_eps < 1.0:
        raise ValueError(f"clip_eps must be in (0, 1), got {clip_eps}")
    return 1.0 - clip_eps, 1.0 + clip_eps


def _mask_weights(length: int, mask: Sequence[bool] | None) -> list[float]:
    if mask is None:
        return [1.0] * length
    if len(mask) != length:
        raise ValueError(f"mask length {len(mask)} != logprob length {length}")
    weights = [1.0 if item else 0.0 for item in mask]
    if not any(weights):
        raise ValueError("mask must keep at least one token")
    return weights


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    if total <= 0:
        raise ValueError("mask must keep at least one token")
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total


def clipped_policy_loss(
    new_logps: Sequence[float],
    old_logps: Sequence[float],
    advantage: float,
    *,
    mask: Sequence[bool] | None = None,
    clip_eps: float = 0.2,
    enabled: bool = True,
) -> dict[str, float]:
    """Pure-Python PPO clipped surrogate for one completion.

    Returns the token-masked policy loss plus ratio diagnostics. Keeping this
    helper torch-free lets the loss formula be tested without a GPU runtime.
    """

    if len(new_logps) != len(old_logps):
        raise ValueError("new_logps and old_logps must have the same length")
    if not new_logps:
        raise ValueError("logprobs must not be empty")
    weights = _mask_weights(len(new_logps), mask)
    lower, upper = _clip_bounds(clip_eps)
    ratios: list[float] = []
    active_weights: list[float] = []
    surrogates: list[float] = []
    for new_logp, old_logp, weight in zip(new_logps, old_logps, weights, strict=True):
        if weight == 0.0:
            continue
        log_ratio = min(MAX_LOG_RATIO, max(-MAX_LOG_RATIO, new_logp - old_logp))
        ratio = math.exp(log_ratio)
        clipped = min(upper, max(lower, ratio))
        surr1 = ratio * advantage
        surr2 = clipped * advantage
        ratios.append(ratio)
        active_weights.append(weight)
        surrogates.append(min(surr1, surr2) if enabled else surr1)
    return {
        "policy_loss": -_weighted_mean(surrogates, active_weights),
        "ratio_mean": _weighted_mean(ratios, active_weights),
        "clip_fraction": (
            sum(not lower <= ratio <= upper for ratio in ratios) / len(ratios) if enabled else 0.0
        ),
        "active_tokens": float(len(ratios)),
    }


def legacy_reinforce_loss(
    new_logps: Sequence[float],
    advantage: float,
    *,
    mask: Sequence[bool] | None = None,
) -> float:
    """Return the v16 direct policy-gradient loss ``-A * mean(logp)``.

    This is the pre-PPO-clipping objective used by v16 best. It is kept as a
    pure helper so the mask-only attribution arm can be tested without torch.
    """

    if not new_logps:
        raise ValueError("logprobs must not be empty")
    weights = _mask_weights(len(new_logps), mask)
    return -float(advantage) * _weighted_mean(new_logps, weights)


def kl_penalty(
    new_logps: Sequence[float],
    reference_logps: Sequence[float],
    *,
    mask: Sequence[bool] | None = None,
) -> float:
    """Return the original v14 KL form: masked mean(policy_logp - ref_logp)."""

    if len(new_logps) != len(reference_logps):
        raise ValueError("new_logps and reference_logps must have the same length")
    weights = _mask_weights(len(new_logps), mask)
    differences = [
        new_logp - reference_logp
        for new_logp, reference_logp in zip(new_logps, reference_logps, strict=True)
    ]
    return _weighted_mean(differences, weights)


def grpo_loss(
    new_logps: Sequence[float],
    old_logps: Sequence[float],
    reference_logps: Sequence[float],
    advantage: float,
    *,
    beta: float,
    mask: Sequence[bool] | None = None,
    clip_eps: float = 0.2,
    enabled: bool = True,
) -> dict[str, float]:
    """Combine the clipped policy objective with the original KL penalty."""

    if beta < 0.0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    policy = clipped_policy_loss(
        new_logps,
        old_logps,
        advantage,
        mask=mask,
        clip_eps=clip_eps,
        enabled=enabled,
    )
    kl = kl_penalty(new_logps, reference_logps, mask=mask)
    return {**policy, "kl": kl, "beta": beta, "loss": policy["policy_loss"] + beta * kl}


def torch_masked_mean(values: Any, mask: Any, *, torch: Any) -> Any:
    """Return a token-masked mean for a one-dimensional torch tensor."""

    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def torch_clipped_policy_loss(
    new_logps: Any,
    old_logps: Any,
    advantage: float,
    *,
    mask: Any,
    clip_eps: float,
    enabled: bool,
    torch: Any,
) -> tuple[Any, dict[str, float]]:
    """Torch implementation used by the GRPO-lite training loop."""

    lower, upper = _clip_bounds(clip_eps)
    log_ratio = (new_logps - old_logps).clamp(-MAX_LOG_RATIO, MAX_LOG_RATIO)
    ratio = log_ratio.exp()
    clipped = ratio.clamp(lower, upper)
    surr1 = ratio * advantage
    surr2 = clipped * advantage
    surrogate = torch.minimum(surr1, surr2) if enabled else surr1
    weights = mask.to(dtype=ratio.dtype)
    active = weights.sum().clamp_min(1.0)
    policy_loss = -(surrogate * weights).sum() / active
    with torch.no_grad():
        ratio_mean = float((ratio * weights).sum().cpu() / active.cpu())
        out_of_bounds = ((ratio < lower) | (ratio > upper)).to(dtype=ratio.dtype)
        clip_fraction = (
            float((out_of_bounds * weights).sum().cpu() / active.cpu()) if enabled else 0.0
        )
        active_tokens = float(weights.sum().cpu())
    return policy_loss, {
        "ratio_mean": ratio_mean,
        "clip_fraction": clip_fraction,
        "active_tokens": active_tokens,
    }


def group_advantages(rewards: Sequence[float]) -> tuple[list[float], bool]:
    """Return group-normalized GRPO advantages and a zero-variance flag."""

    if not rewards:
        return [], True
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std < 1e-6:
        return [0.0 for _ in rewards], True
    return [(reward - mean) / std for reward in rewards], False


def task_messages(task: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build the same merged-first-system prompt used by the v14 harness."""

    constraint_id = str(task.get("constraint_id") or "")
    constraint_text = str(task.get("constraint_text") or "").strip()
    system = (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n{CONSTRAINT_LAYER_HEADING}\n"
        f"- [{constraint_id}] {constraint_text}"
    )
    user_turns = [str(item).strip() for item in task.get("user_turns") or [] if str(item).strip()]
    if not user_turns:
        raise ValueError(f"{task.get('task_id')}: no user turn")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_turns[0]},
    ]


def render_prompt(
    tokenizer: Any, task: Mapping[str, Any], tools: Sequence[Mapping[str, Any]]
) -> str:
    """Render one task with the same chat template and tool schema as inference."""

    return build_chat_prompt(tokenizer, task_messages(task), tools)


def score_completion(
    task: Mapping[str, Any],
    completion: str,
    tools: Sequence[Mapping[str, Any]],
    *,
    severity_weighted: bool,
    delete_split: bool = False,
) -> dict[str, Any]:
    """Parse generated tool calls and score one completion without executing tools."""

    _content, calls = parse_tool_calls(completion, tools)
    result = score_reply(task, completion, calls, from_text=False, delete_split=delete_split)
    if not severity_weighted and result["bucket"] in {
        "execution_violation",
        "claimed_completion",
    }:
        result = {**result, "reward": -1.0}
    return {
        **result,
        "tool_calls": [{"name": call.name, "arguments": call.arguments} for call in calls],
    }
