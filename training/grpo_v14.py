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
) -> dict[str, Any]:
    """Parse generated tool calls and score one completion without executing tools."""

    _content, calls = parse_tool_calls(completion, tools)
    result = score_reply(task, completion, calls, from_text=False)
    if not severity_weighted and result["bucket"] in {
        "execution_violation",
        "claimed_completion",
    }:
        result = {**result, "reward": -1.0}
    return {
        **result,
        "tool_calls": [{"name": call.name, "arguments": call.arguments} for call in calls],
    }
