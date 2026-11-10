"""CPU-only smoke test for step-level delete credit allocation.

No torch, no 7B model, no adapter: it exercises the pure allocation rules,
mass accounting, the gate-b counter, and a finite-difference loss check.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.v14.train_grpo_v14 import summarize_step_credit  # noqa: E402
from training.grpo_v14 import grpo_loss  # noqa: E402
from training.trajectory_grpo import (  # noqa: E402
    StepCreditGate,
    StepSample,
    TrajectorySample,
    step_credit_advantages,
)


def _call_step(command: str) -> StepSample:
    return StepSample(
        (1, 2),
        (11, 12),
        calls=({"name": "bash", "arguments": json.dumps({"command": command})},),
    )


def _trajectory(
    *steps: StepSample,
    layer: str = "delete",
    bucket: str = "execution_violation",
    task_id: str = "SMOKE-STEP-CREDIT",
) -> TrajectorySample:
    return TrajectorySample(
        task_id=task_id,
        layer=layer,
        steps=tuple(steps),
        final_content="",
        calls=(),
        reward=-1.0 if bucket == "execution_violation" else 0.0,
        bucket=bucket,
    )


def _loss_steps() -> tuple[StepSample, StepSample]:
    trigger = StepSample(
        (1, 2),
        (11, 12, 13),
        old_logps=(0.0, -0.05, 0.1),
        mask=(True, True, False),
        calls=({"name": "bash", "arguments": json.dumps({"command": "rm -rf tmp"})},),
    )
    follow_up = StepSample(
        (1, 2),
        (21, 22),
        old_logps=(0.02, -0.03),
        mask=(True, True),
        calls=({"name": "bash", "arguments": json.dumps({"command": "ls"})},),
    )
    return trigger, follow_up


def _per_step_losses(
    steps: tuple[StepSample, StepSample],
    new_logps: list[list[float]],
    reference_logps: list[list[float]],
    advantages: tuple[float, float],
    *,
    beta: float = 0.01,
) -> tuple[float, list[dict[str, float]]]:
    parts: list[dict[str, float]] = []
    total = 0.0
    active_total = 0.0
    for index, step in enumerate(steps):
        result = grpo_loss(
            new_logps[index],
            list(step.old_logps),
            reference_logps[index],
            advantages[index],
            beta=beta,
            mask=list(step.token_mask),
        )
        parts.append(result)
        active = float(result["active_tokens"])
        active_total += active
        total += float(result["loss"]) * active
    return total / active_total, parts


def _finite_difference_gradients(
    steps: tuple[StepSample, StepSample],
    base: list[list[float]],
    reference: list[list[float]],
    advantages: tuple[float, float],
) -> list[float]:
    delta = 1e-6
    gradients: list[float] = []
    for step_index, row in enumerate(base):
        for token_index in range(len(row)):
            plus = [list(values) for values in base]
            minus = [list(values) for values in base]
            plus[step_index][token_index] += delta
            minus[step_index][token_index] -= delta
            plus_loss, _ = _per_step_losses(steps, plus, reference, advantages)
            minus_loss, _ = _per_step_losses(steps, minus, reference, advantages)
            gradients.append((plus_loss - minus_loss) / (2.0 * delta))
    return gradients


def run_smoke() -> dict[str, Any]:
    # Path 1: trigger at the last step of a 9-step trajectory.
    trigger_steps = tuple(_call_step("ls") for _ in range(8)) + (_call_step("rm -rf tmp"),)
    trigger_trajectory = _trajectory(*trigger_steps, task_id="TRIGGER")
    trigger_per_step, trigger_diagnostics = step_credit_advantages([trigger_trajectory], [-1.0])
    trigger_diagnostic = trigger_diagnostics[0]

    # Path 2: delete violation without a per-step delete call -> fallback.
    fallback_trajectory = _trajectory(_call_step("ls"), _call_step("cat a"), task_id="FALLBACK")
    fallback_per_step, fallback_diagnostics = step_credit_advantages([fallback_trajectory], [-1.0])
    fallback_diagnostic = fallback_diagnostics[0]
    fallback_gate = StepCreditGate(min_rollouts=1)
    fallback_gate.observe(
        layer="delete",
        scores=[{"bucket": "execution_violation"}],
        diagnostics=fallback_diagnostics,
    )

    # Path 3: no delete attempt -> bucket is not execution_violation.
    no_delete_trajectory = _trajectory(
        _call_step("ls"), _call_step("cat a"), bucket="no_stance", task_id="NO-DELETE"
    )
    no_delete_per_step, no_delete_diagnostics = step_credit_advantages(
        [no_delete_trajectory], [-0.7]
    )
    no_delete_diagnostic = no_delete_diagnostics[0]

    # Path 4: zero-variance group -> A=0, mass is zero.
    zero_trajectory = _trajectory(
        _call_step("ls"), _call_step("rm -rf tmp"), task_id="ZERO-VARIANCE"
    )
    zero_per_step, zero_diagnostics = step_credit_advantages([zero_trajectory], [0.0])
    zero_diagnostic = zero_diagnostics[0]
    zero_gate = StepCreditGate(min_rollouts=1)
    zero_gate.observe(
        layer="delete",
        scores=[{"bucket": "execution_violation"}],
        diagnostics=zero_diagnostics,
    )

    # The aggregation must count only delete-layer execution violations.
    with tempfile.TemporaryDirectory(prefix="step-credit-smoke-") as directory:
        log_path = Path(directory) / "train_log.jsonl"
        rows = [
            {"step_credit": {"diagnostics": [trigger_diagnostic]}},
            {"step_credit": {"diagnostics": [fallback_diagnostic]}},
            {"step_credit": {"diagnostics": [no_delete_diagnostic]}},
            {"step_credit": {"diagnostics": [zero_diagnostic]}},
        ]
        log_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        aggregate = summarize_step_credit(log_path)

    # Loss path: trigger at step 0, step 1 has advantage 0 and should be KL-only.
    steps = _loss_steps()
    new = [[0.01, -0.02, 0.03], [0.0, 0.04]]
    reference = [[0.0, 0.0, 0.0], [0.0, 0.0]]
    advantages = (-1.0, 0.0)
    total_loss, parts = _per_step_losses(steps, new, reference, advantages)
    gradients = _finite_difference_gradients(steps, new, reference, advantages)

    trigger_ratio = float(trigger_diagnostic["mass_ratio"])
    expected_trigger_ratio = 1.5 / 9.0
    below_point_two = trigger_ratio < 0.2
    warning = (
        "梯度总幅砍太狠，信用分配与梯度变小混淆严重，阶段 3 报告必须标注"
        if below_point_two
        else None
    )
    checks = {
        "trigger_weights": trigger_diagnostic["weights"] == [0.0] * 7 + [0.5, 1.0],
        "trigger_mass_after": math.isclose(
            float(trigger_diagnostic["mass_after"]), 1.5, rel_tol=1e-12, abs_tol=1e-12
        ),
        "trigger_mass_ratio": math.isclose(
            trigger_ratio, expected_trigger_ratio, rel_tol=1e-12, abs_tol=1e-12
        ),
        "fallback_flagged": bool(fallback_diagnostic["fallback"]),
        "fallback_weights": fallback_diagnostic["weights"] == [0.0, 1.0],
        "fallback_not_gate_b": fallback_gate.failed and fallback_gate.trigger_checked == 0,
        "fallback_mass": math.isclose(
            float(fallback_diagnostic["mass_after"]), 1.0, rel_tol=1e-12, abs_tol=1e-12
        ),
        "no_delete_unchanged": no_delete_per_step == [[-0.7, -0.7]],
        "no_delete_not_aggregated": aggregate["trajectories"] == 3,
        "zero_variance_mass_zero": (
            float(zero_diagnostic["mass_before"]) == 0.0
            and float(zero_diagnostic["mass_after"]) == 0.0
            and float(zero_diagnostic["mass_ratio"]) == 0.0
        ),
        "zero_variance_gate_b": zero_gate.failed and "zero_advantage" in zero_gate.failure_reasons,
        "aggregate_mass": (
            math.isclose(aggregate["mass_before_total"], 11.0, abs_tol=1e-9)
            and math.isclose(aggregate["mass_after_total"], 2.5, abs_tol=1e-9)
        ),
        "loss_finite": math.isfinite(total_loss),
        "gradients_finite": all(math.isfinite(value) for value in gradients),
        "masked_gradient_zero": abs(gradients[2]) < 1e-9,
        "zero_advantage_kl_only": math.isclose(float(parts[1]["policy_loss"]), 0.0, abs_tol=1e-12)
        and math.isclose(
            float(parts[1]["loss"]),
            0.01 * float(parts[1]["kl"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
    }
    return {
        "status": "ok" if all(checks.values()) else "failed",
        "boundary": "synthetic data; non-autograd finite differences; no 7B; no adapter",
        "paths": {
            "trigger": {
                "steps": len(trigger_steps),
                "per_step": trigger_per_step[0],
                "diagnostic": trigger_diagnostic,
            },
            "fallback": {
                "per_step": fallback_per_step[0],
                "diagnostic": fallback_diagnostic,
                "gate": fallback_gate.as_dict(),
            },
            "no_delete_attempt": {
                "per_step": no_delete_per_step[0],
                "diagnostic": no_delete_diagnostic,
            },
            "zero_variance": {
                "per_step": zero_per_step[0],
                "diagnostic": zero_diagnostic,
                "gate": zero_gate.as_dict(),
            },
        },
        "mass": {
            "trigger_mass_before": trigger_diagnostic["mass_before"],
            "trigger_mass_after": trigger_diagnostic["mass_after"],
            "trigger_mass_ratio": trigger_ratio,
            "expected_trigger_ratio": expected_trigger_ratio,
            "below_0_2": below_point_two,
            "warning": warning,
            "aggregate": aggregate,
        },
        "loss": {
            "total_loss": total_loss,
            "parts": parts,
            "gradients": gradients,
        },
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU smoke for step-level delete credit")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_smoke()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
