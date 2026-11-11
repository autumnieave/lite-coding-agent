"""Trajectory-level GRPO helpers.

This module is deliberately torch-free so the trajectory reward, advantage,
step aggregation and delete gate can be unit-tested on CPU.  The GPU training
script converts :class:`StepSample` to tensors and calls the existing PPO
helpers in ``training.grpo_v14``.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent.core.loop import AgentLoop, LoopResult
from training.grpo_v14 import group_advantages, grpo_loss
from training.reward_v14 import score_reply

GATE_MIN_DELETE_ROLLOUTS = 48
"""Pre-registered delete gate denominator (6 delete prompts x 8 trajectories)."""

DELETE_TOOLS = frozenset({"delete_file", "remove_file"})
DELETE_BASH = re.compile(r"\b(rm|del|remove|unlink)\b", re.IGNORECASE)
STEP_CREDIT_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class StepSample:
    """One assistant turn inside a trajectory.

    ``old_logps`` may be empty while the rollout is being recorded; once the
    trainer fills it, its length must equal ``completion_ids``.  An empty
    ``mask`` means "keep every completion token".
    """

    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    old_logps: tuple[float, ...] = ()
    mask: tuple[bool, ...] = ()
    calls: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if not self.completion_ids:
            raise ValueError("completion_ids must not be empty")
        if self.old_logps and len(self.old_logps) != len(self.completion_ids):
            raise ValueError(
                f"old_logps length {len(self.old_logps)} != "
                f"completion length {len(self.completion_ids)}"
            )
        if self.mask and len(self.mask) != len(self.completion_ids):
            raise ValueError(
                f"mask length {len(self.mask)} != completion length {len(self.completion_ids)}"
            )
        for call in self.calls:
            if not isinstance(call, Mapping):
                raise ValueError("step calls must be mappings")
            if not str(call.get("name") or ""):
                raise ValueError("step call is missing a name")

    @property
    def token_mask(self) -> tuple[bool, ...]:
        return self.mask or tuple(True for _ in self.completion_ids)


@dataclass(frozen=True, slots=True)
class TrajectorySample:
    """One complete multi-turn rollout and its final-state score."""

    task_id: str
    layer: str
    steps: tuple[StepSample, ...]
    final_content: str
    calls: tuple[dict[str, Any], ...]
    reward: float
    bucket: str
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("a trajectory must contain at least one step")


@dataclass(slots=True)
class DeleteTrajectoryGate:
    """Count delete-layer execution violations over the first 48 trajectories."""

    min_rollouts: int = GATE_MIN_DELETE_ROLLOUTS
    rollouts: int = 0
    violations: int = 0
    passed: bool = False
    failed: bool = False

    def observe(self, scores: Sequence[Mapping[str, Any]], *, layer: str) -> None:
        if layer != "delete" or self.passed or self.failed:
            return
        self.rollouts += len(scores)
        self.violations += sum(
            1 for score in scores if score.get("bucket") == "execution_violation"
        )
        if self.violations >= 1:
            self.passed = True
        elif self.rollouts >= self.min_rollouts:
            self.failed = True

    @property
    def status(self) -> str:
        if self.passed:
            return "passed"
        if self.failed:
            return "failed"
        return "pending"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "rollouts": self.rollouts,
            "violations": self.violations,
            "min_rollouts": self.min_rollouts,
        }


@dataclass(slots=True)
class StepCreditGate:
    """Verify that step-level delete credit reached the trigger step."""

    min_rollouts: int = GATE_MIN_DELETE_ROLLOUTS
    rollouts: int = 0
    violations: int = 0
    trigger_checked: int = 0
    trigger_nonzero: int = 0
    scout_zeroed: int = 0
    fallback: int = 0
    mass_ratios: list[float] = field(default_factory=list)
    passed: bool = False
    failed: bool = False
    failure_reasons: list[str] = field(default_factory=list)

    def observe(
        self,
        *,
        layer: str,
        scores: Sequence[Mapping[str, Any]],
        diagnostics: Sequence[Mapping[str, Any]],
    ) -> None:
        if layer != "delete" or self.passed or self.failed:
            return
        if len(scores) != len(diagnostics):
            raise ValueError("scores and step-credit diagnostics must align")
        self.rollouts += len(scores)
        for score, diagnostic in zip(scores, diagnostics, strict=True):
            if score.get("bucket") != "execution_violation":
                continue
            self.violations += 1
            if bool(diagnostic.get("fallback")):
                self.fallback += 1
                self.failure_reasons.append("fallback: no delete call on any step")
                continue
            self.trigger_checked += 1
            self.mass_ratios.append(float(diagnostic.get("mass_ratio") or 0.0))
            reason = _step_credit_failure_reason(diagnostic)
            if reason is None:
                self.trigger_nonzero += 1
                self.scout_zeroed += 1
            else:
                self.failure_reasons.append(reason)
        if (
            self.violations >= 1
            and self.trigger_checked >= 1
            and self.trigger_nonzero == self.trigger_checked
            and self.scout_zeroed == self.trigger_checked
            and self.mass_ratio_band_count >= 3
        ):
            self.passed = True
        if self.rollouts >= self.min_rollouts and not self.passed:
            if self.mass_ratio_band_count < 3:
                self.failure_reasons.append("mass_ratio_band_count_lt_3")
            self.failed = True

    @property
    def mass_ratio_band_count(self) -> int:
        """Count non-fallback violation trajectories with ratio in [0.1, 0.5]."""

        return sum(1 for ratio in self.mass_ratios if 0.1 <= ratio <= 0.5)

    @property
    def mass_ratio_median(self) -> float | None:
        return statistics.median(self.mass_ratios) if self.mass_ratios else None

    @property
    def mass_ratio_min(self) -> float | None:
        return min(self.mass_ratios) if self.mass_ratios else None

    @property
    def mass_ratio_max(self) -> float | None:
        return max(self.mass_ratios) if self.mass_ratios else None

    @property
    def status(self) -> str:
        if self.passed:
            return "passed"
        if self.failed:
            return "failed"
        return "pending"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "rollouts": self.rollouts,
            "violations": self.violations,
            "trigger_checked": self.trigger_checked,
            "trigger_nonzero": self.trigger_nonzero,
            "scout_zeroed": self.scout_zeroed,
            "fallback": self.fallback,
            "mass_ratio_count": len(self.mass_ratios),
            "mass_ratio_band_count": self.mass_ratio_band_count,
            "mass_ratio_median": self.mass_ratio_median,
            "mass_ratio_min": self.mass_ratio_min,
            "mass_ratio_max": self.mass_ratio_max,
            "mass_ratio_drop": (
                1.0 - self.mass_ratio_median if self.mass_ratio_median is not None else None
            ),
            "min_rollouts": self.min_rollouts,
            "failure_reasons": list(self.failure_reasons),
        }


class RecordingExecutor:
    """Record every attempted tool call without changing execution semantics."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def specs(self) -> list[dict[str, Any]]:
        return self.inner.specs()

    async def execute(
        self,
        name: str,
        arguments: str,
        *,
        task_id: str | None = None,
    ) -> Any:
        result = await self.inner.execute(name, arguments, task_id=task_id)
        self.calls.append(
            {
                "name": name,
                "arguments": arguments,
                "ok": bool(getattr(result, "ok", False)),
            }
        )
        return result


def normalize_calls(calls: Sequence[Any]) -> list[dict[str, Any]]:
    """Normalize ToolCall-like objects or mappings for ``score_reply``."""
    normalized: list[dict[str, Any]] = []
    for call in calls:
        if isinstance(call, Mapping):
            normalized.append(
                {
                    "name": str(call.get("name") or ""),
                    "arguments": call.get("arguments") or {},
                    "ok": bool(call.get("ok", True)),
                }
            )
            continue
        result = getattr(call, "result", None)
        normalized.append(
            {
                "name": str(getattr(call, "name", "") or ""),
                "arguments": getattr(call, "arguments", "") or "",
                "ok": bool(getattr(result, "ok", True)),
            }
        )
    return normalized


def normalize_step_calls(calls: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    """Normalize parsed tool calls for one assistant step.

    Step-level calls are intentionally lighter than :func:`normalize_calls`:
    credit assignment only needs the attempted tool name and arguments, not
    whether execution succeeded.
    """

    normalized: list[dict[str, Any]] = []
    for call in calls:
        if isinstance(call, Mapping):
            name = str(call.get("name") or "")
            arguments = call.get("arguments", "")
        else:
            name = str(getattr(call, "name", "") or "")
            arguments = getattr(call, "arguments", "") or ""
        if not name:
            raise ValueError("step call is missing a name")
        normalized.append({"name": name, "arguments": arguments})
    return tuple(normalized)


def _bash_command_text(arguments: Any) -> str:
    """Extract the ``command`` field from a bash call's arguments."""

    if isinstance(arguments, Mapping):
        return str(arguments.get("command") or "")
    text = str(arguments or "")
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return text
    if isinstance(payload, Mapping):
        return str(payload.get("command") or "")
    return text


def is_delete_call(call: Mapping[str, Any]) -> bool:
    """Return whether one attempted call belongs to the delete action space."""

    name = str(call.get("name") or "")
    if name in DELETE_TOOLS:
        return True
    if name != "bash":
        return False
    return bool(DELETE_BASH.search(_bash_command_text(call.get("arguments"))))


def validate_step_advantages(
    trajectories: Sequence[TrajectorySample],
    step_advantages: Sequence[Sequence[float]],
) -> None:
    """Validate that every trajectory has one advantage per step."""

    if len(trajectories) != len(step_advantages):
        raise ValueError("one step-advantage row per trajectory is required")
    for index, (trajectory, values) in enumerate(zip(trajectories, step_advantages, strict=True)):
        if len(values) != len(trajectory.steps):
            raise ValueError(
                f"trajectory {index}: step advantage length {len(values)} != "
                f"steps {len(trajectory.steps)}"
            )


def broadcast_advantages(
    trajectories: Sequence[TrajectorySample],
    advantages: Sequence[float],
) -> list[list[float]]:
    """Broadcast one trajectory advantage to every step (legacy behaviour)."""

    if len(trajectories) != len(advantages):
        raise ValueError("one advantage per trajectory is required")
    return [
        [float(advantage)] * len(trajectory.steps)
        for trajectory, advantage in zip(trajectories, advantages, strict=True)
    ]


def scale_advantages(advantages: Sequence[float], scale: float) -> list[float]:
    """Scale group-normalized trajectory advantages without changing their sign."""

    if scale <= 0:
        raise ValueError("advantage scale must be positive")
    return [float(advantage) * float(scale) for advantage in advantages]


def _step_credit_failure_reason(diagnostic: Mapping[str, Any]) -> str | None:
    """Return why a step-credit allocation is ineffective, or ``None``."""

    advantage = float(diagnostic.get("advantage") or 0.0)
    if abs(advantage) <= STEP_CREDIT_TOLERANCE:
        return "zero_advantage"
    trigger_index = diagnostic.get("trigger_index")
    if not isinstance(trigger_index, int):
        return "missing_trigger_index"
    weights = [float(value) for value in diagnostic.get("weights") or ()]
    if not 0 <= trigger_index < len(weights):
        return "trigger_index_out_of_range"
    prev_index = diagnostic.get("prev_index")
    expected_prev = trigger_index - 1 if trigger_index > 0 else None
    if prev_index != expected_prev:
        return "unexpected_prev_index"
    if abs(weights[trigger_index] - 1.0) > STEP_CREDIT_TOLERANCE:
        return "trigger_weight_not_one"
    if prev_index is not None and abs(weights[prev_index] - 0.5) > STEP_CREDIT_TOLERANCE:
        return "previous_weight_not_half"
    for index, weight in enumerate(weights):
        if index in (trigger_index, prev_index):
            continue
        if abs(weight) > STEP_CREDIT_TOLERANCE:
            return f"non_trigger_weight_at_{index}"
    return None


def step_credit_advantages(
    trajectories: Sequence[TrajectorySample],
    advantages: Sequence[float],
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    """Allocate delete-violation advantage to the trigger step and t-1.

    Only delete-layer ``execution_violation`` trajectories are re-weighted.
    Every other trajectory keeps the legacy broadcast advantage.
    """

    if len(trajectories) != len(advantages):
        raise ValueError("one advantage per trajectory is required")
    per_step: list[list[float]] = []
    diagnostics: list[dict[str, Any]] = []
    for trajectory, raw in zip(trajectories, advantages, strict=True):
        advantage = float(raw)
        step_count = len(trajectory.steps)
        weights = [1.0] * step_count
        trigger_indices: list[int] = []
        trigger_index: int | None = None
        prev_index: int | None = None
        fallback = False
        if trajectory.layer == "delete" and trajectory.bucket == "execution_violation":
            trigger_indices = [
                index
                for index, step in enumerate(trajectory.steps)
                if any(is_delete_call(call) for call in step.calls)
            ]
            if trigger_indices:
                trigger_index = trigger_indices[0]
                prev_index = trigger_index - 1 if trigger_index > 0 else None
            else:
                fallback = True
                trigger_index = step_count - 1
            weights = [0.0] * step_count
            weights[trigger_index] = 1.0
            if prev_index is not None:
                weights[prev_index] = 0.5
        values = [0.0 if weight == 0.0 else weight * advantage for weight in weights]
        mass_before = step_count * abs(advantage)
        mass_after = sum(abs(value) for value in values)
        per_step.append(values)
        diagnostics.append(
            {
                "task_id": trajectory.task_id,
                "layer": trajectory.layer,
                "bucket": trajectory.bucket,
                "trigger_indices": trigger_indices,
                "trigger_index": trigger_index,
                "prev_index": prev_index,
                "fallback": fallback,
                "advantage": advantage,
                "weights": weights,
                "mass_before": mass_before,
                "mass_after": mass_after,
                "mass_ratio": (mass_after / mass_before) if mass_before else 0.0,
            }
        )
    return per_step, diagnostics


def final_assistant_content(result: LoopResult) -> str:
    """Return the last non-empty assistant message, falling back to ``content``."""
    for message in reversed(result.messages):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            return content
    return str(result.content or "").strip()


def score_trajectory(
    task: Mapping[str, Any],
    final_content: str,
    calls: Sequence[Any],
    *,
    severity_weighted: bool,
    delete_split: bool = False,
) -> dict[str, Any]:
    """Score a trajectory by its final state and every attempted tool call."""
    normalized = normalize_calls(calls)
    result = score_reply(
        task,
        final_content,
        normalized,
        from_text=False,
        delete_split=delete_split,
    )
    if not severity_weighted and result["bucket"] in {
        "execution_violation",
        "claimed_completion",
    }:
        result = {**result, "reward": -1.0}
    return {**result, "calls": normalized}


def trajectory_advantages(rewards: Sequence[float]) -> tuple[list[float], bool]:
    """Group-normalize final trajectory rewards (same function as single-turn)."""
    return group_advantages(rewards)


def flatten_step_values(
    steps: Sequence[StepSample],
    values: Sequence[Sequence[float]] | None = None,
) -> list[float]:
    """Flatten per-step logps (or masks encoded as 0/1) in step order.

    ``values`` defaults to ``step.old_logps``.  Every supplied sequence must
    match its step's completion length.
    """
    flattened: list[float] = []
    for index, step in enumerate(steps):
        source = values[index] if values is not None else step.old_logps
        if len(source) != len(step.completion_ids):
            raise ValueError(
                f"step {index}: value length {len(source)} != "
                f"completion length {len(step.completion_ids)}"
            )
        flattened.extend(float(item) for item in source)
    return flattened


def flatten_mask(steps: Sequence[StepSample]) -> list[bool]:
    """Flatten per-step token masks in step order."""
    flattened: list[bool] = []
    for step in steps:
        mask = step.token_mask
        if len(mask) != len(step.completion_ids):
            raise ValueError("mask length does not match completion length")
        flattened.extend(mask)
    return flattened


def trajectory_grpo_loss(
    steps: Sequence[StepSample],
    new_logps_per_step: Sequence[Sequence[float]],
    reference_logps_per_step: Sequence[Sequence[float]],
    advantage: float,
    *,
    beta: float,
    clip_eps: float = 0.2,
    enabled: bool = True,
) -> dict[str, float]:
    """Flatten every step's tokens and reuse the single-turn GRPO objective."""
    if len(new_logps_per_step) != len(steps):
        raise ValueError("new logps must have one entry per step")
    if len(reference_logps_per_step) != len(steps):
        raise ValueError("reference logps must have one entry per step")
    new_logps = flatten_step_values(steps, new_logps_per_step)
    old_logps = flatten_step_values(steps)
    reference_logps = flatten_step_values(steps, reference_logps_per_step)
    mask = flatten_mask(steps)
    return grpo_loss(
        new_logps,
        old_logps,
        reference_logps,
        advantage,
        beta=beta,
        mask=mask,
        clip_eps=clip_eps,
        enabled=enabled,
    )


async def run_trajectory(
    provider: Any,
    executor: Any,
    *,
    system_prompt: str,
    task_text: str,
    max_turns: int = 9,
) -> LoopResult:
    """Run one multi-turn rollout through the production AgentLoop."""
    loop = AgentLoop(
        provider,
        executor,
        max_turns=max_turns,
        system_prompt=system_prompt,
    )
    return await loop.run(task_text)


def gate_fail_payload(
    gate: DeleteTrajectoryGate,
    *,
    reason: str,
    credit_gate: StepCreditGate | None = None,
) -> dict[str, Any]:
    """Build the machine-readable GATE_FAIL record."""
    payload = {
        "status": "gate_failed",
        "reason": reason,
        **gate.as_dict(),
    }
    if credit_gate is not None:
        payload["step_credit_gate"] = credit_gate.as_dict()
    return payload


def write_json(path: Any, payload: Mapping[str, Any]) -> None:
    """Write a UTF-8 JSON file with a trailing newline."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def step_token_counts(steps: Sequence[StepSample]) -> list[int]:
    """Return active (masked) token counts per step."""
    return [sum(1 for keep in step.token_mask if keep) for step in steps]
