"""On-policy GRPO-lite for the v14 action layer.

TRL 0.12 in this runtime does not provide GRPO, so this runner implements the
minimal group-relative objective used by the pre-registration: sample a group of
completions per prompt, normalize rewards within the group, and optimize a
PPO-clipped group-relative policy gradient with a KL penalty to the frozen SFT
adapter.  The clipped surrogate is enabled by default; pass
``--no-ratio-clipping`` to reproduce the pre-PPO-clipping REINFORCE behaviour.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent.core.llm import BaseProvider, LLMResponse  # noqa: E402
from agent.tools import build_default_registry  # noqa: E402
from training.grpo_v14 import (  # noqa: E402
    completion_loss_mask,
    group_advantages,
    render_prompt,
    score_completion,
    split_minibatches,
    task_messages,
    torch_clipped_policy_loss,
    torch_masked_mean,
)
from training.kaggle.local_prompt import build_chat_prompt  # noqa: E402
from training.kaggle.offline_provider import parse_tool_calls  # noqa: E402
from training.reward_v14 import (  # noqa: E402
    CONFIRM_PHRASES_VERSION,
    confirm_phrases_fingerprint,
)
from training.trajectory_grpo import (  # noqa: E402
    DeleteTrajectoryGate,
    RecordingExecutor,
    StepCreditGate,
    StepSample,
    TrajectorySample,
    broadcast_advantages,
    final_assistant_content,
    gate_fail_payload,
    normalize_step_calls,
    run_trajectory,
    scale_advantages,
    score_trajectory,
    step_credit_advantages,
    trajectory_advantages,
    validate_step_advantages,
    write_json,
)
from training.v10_eval_harness import materialize_probe  # noqa: E402


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""

    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def count_zero_variance_groups(log_path: Path) -> int:
    """Count zero-variance groups from the train log.

    The manifest must describe what was actually logged. Counting at write time
    keeps the trajectory and completion branches consistent and avoids a stale
    counter when a branch forgets to increment it.
    """

    return sum(1 for row in load_jsonl(log_path) if bool(row.get("zero_variance")))


def summarize_step_credit(log_path: Path) -> dict[str, Any]:
    """Aggregate step-credit mass diagnostics from a train log."""

    mass_before = 0.0
    mass_after = 0.0
    groups = 0
    trajectories = 0
    triggered = 0
    fallback = 0
    mass_ratios: list[float] = []
    for row in load_jsonl(log_path):
        block = row.get("step_credit")
        if not isinstance(block, Mapping):
            continue
        diagnostics = block.get("diagnostics") or []
        if not diagnostics:
            continue
        groups += 1
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, Mapping):
                continue
            if (
                diagnostic.get("layer") != "delete"
                or diagnostic.get("bucket") != "execution_violation"
            ):
                continue
            trajectories += 1
            mass_before += float(diagnostic.get("mass_before") or 0.0)
            mass_after += float(diagnostic.get("mass_after") or 0.0)
            if bool(diagnostic.get("fallback")):
                fallback += 1
            else:
                ratio = diagnostic.get("mass_ratio")
                if ratio is None:
                    before = float(diagnostic.get("mass_before") or 0.0)
                    ratio = (float(diagnostic.get("mass_after") or 0.0) / before) if before else 0.0
                mass_ratios.append(float(ratio))
            if diagnostic.get("trigger_indices"):
                triggered += 1
    ratio_median = statistics.median(mass_ratios) if mass_ratios else None
    return {
        "groups": groups,
        "trajectories": trajectories,
        "triggered_trajectories": triggered,
        "fallback_trajectories": fallback,
        "mass_before_total": mass_before,
        "mass_after_total": mass_after,
        "mass_ratio": (mass_after / mass_before) if mass_before else None,
        "mass_ratio_count": len(mass_ratios),
        "mass_ratio_band_count": sum(1 for ratio in mass_ratios if 0.1 <= ratio <= 0.5),
        "mass_ratio_median": ratio_median,
        "mass_ratio_min": min(mass_ratios) if mass_ratios else None,
        "mass_ratio_max": max(mass_ratios) if mass_ratios else None,
        "mass_ratio_drop": (1.0 - ratio_median) if ratio_median is not None else None,
        "mass_ratios": mass_ratios,
    }


def _completion_token_logprobs(
    model: Any,
    torch: Any,
    input_ids: Any,
    completion_start: int,
) -> Any:
    """Return log pi(a_t | s_t) for every completion token in one sequence."""

    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits[:, completion_start - 1 : -1, :].float()
    targets = input_ids[:, completion_start:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.squeeze(0)


def _completion_mask(
    token_ids: Any,
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
    torch: Any,
    enabled: bool = True,
) -> Any:
    """Mask trailing padding while keeping the first EOS token when present.

    ``enabled=False`` keeps every completion token, including padding, which
    reproduces the v16-era loss denominator.
    """

    values = completion_loss_mask(
        token_ids.tolist(),
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        enabled=enabled,
    )
    return torch.tensor(values, dtype=torch.bool, device=token_ids.device)


class _TrajectoryProvider(BaseProvider):
    """Generate one trajectory step at a time and record prompt/completion ids."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        torch: Any,
        *,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.steps: list[StepSample] = []
        self._call_index = 0

    def reset(self, *, seed: int) -> None:
        self.steps = []
        self._call_index = 0
        self.seed = seed

    def _generate(self, messages: Sequence[Mapping[str, Any]], tools: Any) -> str:
        torch = self._torch
        prompt = build_chat_prompt(self._tokenizer, messages, tools)
        encoded = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(next(self._model.parameters()).device)
        attention_mask = encoded["attention_mask"].to(input_ids.device)
        prompt_len = int(input_ids.shape[-1])
        torch.manual_seed(self.seed + self._call_index * 1009)
        with torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
                num_return_sequences=1,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        completion_ids = output[0][prompt_len:]
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        self.steps.append(
            StepSample(
                prompt_ids=tuple(int(value) for value in input_ids[0].tolist()),
                completion_ids=tuple(int(value) for value in completion_ids.tolist()),
            )
        )
        self._call_index += 1
        return text

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        text = self._generate(messages, tools)
        content, calls = parse_tool_calls(text, tools)
        if not self.steps:
            raise RuntimeError("trajectory provider recorded no step before parsing")
        self.steps[-1] = replace(self.steps[-1], calls=normalize_step_calls(calls))
        return LLMResponse(content=content, tool_calls=tuple(calls))


def _fill_old_logps(
    model: Any,
    tokenizer: Any,
    torch: Any,
    steps: Sequence[StepSample],
) -> tuple[StepSample, ...]:
    """Fill old log-probs and completion masks for one trajectory."""
    device = next(model.parameters()).device
    filled: list[StepSample] = []
    with torch.no_grad():
        model.set_adapter("policy")
        for step in steps:
            prompt = torch.tensor([list(step.prompt_ids)], dtype=torch.long, device=device)
            completion = torch.tensor([list(step.completion_ids)], dtype=torch.long, device=device)
            full_ids = torch.cat([prompt, completion], dim=1)
            logps = _completion_token_logprobs(model, torch, full_ids, prompt.shape[1])
            mask = completion_loss_mask(
                step.completion_ids,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                enabled=True,
            )
            filled.append(
                StepSample(
                    prompt_ids=step.prompt_ids,
                    completion_ids=step.completion_ids,
                    old_logps=tuple(float(value) for value in logps.tolist()),
                    mask=tuple(mask),
                    calls=step.calls,
                )
            )
    return tuple(filled)


def _collect_trajectory(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    provider: _TrajectoryProvider,
    row: Mapping[str, Any],
    tools: Sequence[Mapping[str, Any]],
    severity_weighted: bool,
    delete_split: bool,
    max_turns: int,
    seed: int,
) -> tuple[TrajectorySample, dict[str, Any], str]:
    """Run one isolated multi-turn rollout and score its final state."""
    task = dict(row["task"])
    system_prompt = task_messages(task)[0]["content"]
    user_text = str(task["user_turns"][0])
    provider.reset(seed=seed)
    root = Path(tempfile.mkdtemp(prefix=f"traj-{task.get('task_id', 'task')}-"))
    try:
        materialize_probe(task, root)
        recorder = RecordingExecutor(build_default_registry(root))
        result = asyncio.run(
            run_trajectory(
                provider,
                recorder,
                system_prompt=system_prompt,
                task_text=user_text,
                max_turns=max_turns,
            )
        )
        steps = _fill_old_logps(model, tokenizer, torch, provider.steps)
        content = final_assistant_content(result)
        score = score_trajectory(
            task,
            content,
            recorder.calls,
            severity_weighted=severity_weighted,
            delete_split=delete_split,
        )
        sample = TrajectorySample(
            task_id=str(row["task_id"]),
            layer=str(row["layer"]),
            steps=steps,
            final_content=content,
            calls=tuple(score["calls"]),
            reward=float(score["reward"]),
            bucket=str(score["bucket"]),
            detail=str(score.get("detail") or ""),
        )
        return sample, score, content
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _update_trajectory_group(
    *,
    model: Any,
    torch: Any,
    optimizer: Any,
    trajectories: Sequence[TrajectorySample],
    step_advantages: Sequence[Sequence[float]],
    beta: float,
    clip_eps: float,
    ratio_clipping: bool,
    mini_batch_size: int,
    seed: int,
    step: int,
) -> dict[str, Any]:
    """Token-weighted PPO update over whole trajectories (ppo_epochs=1)."""
    validate_step_advantages(trajectories, step_advantages)
    device = next(model.parameters()).device
    reference_logps: list[list[Any]] = []
    with torch.no_grad():
        model.set_adapter("ref")
        for trajectory in trajectories:
            per_step: list[Any] = []
            for sample in trajectory.steps:
                prompt = torch.tensor([list(sample.prompt_ids)], dtype=torch.long, device=device)
                completion = torch.tensor(
                    [list(sample.completion_ids)], dtype=torch.long, device=device
                )
                full_ids = torch.cat([prompt, completion], dim=1)
                per_step.append(
                    _completion_token_logprobs(model, torch, full_ids, prompt.shape[1]).detach()
                )
            reference_logps.append(per_step)
    model.set_adapter("policy")

    order = list(range(len(trajectories)))
    random.Random(seed + step * 1009).shuffle(order)
    batches = split_minibatches(order, min(mini_batch_size, len(order)))
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "kl_mean": 0.0,
        "ratio_mean": 0.0,
        "clip_fraction": 0.0,
    }
    total_active = 0.0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        chunk_tokens = sum(
            sum(sample.token_mask) for index in batch for sample in trajectories[index].steps
        )
        if chunk_tokens <= 0:
            raise RuntimeError("trajectory batch has no active tokens")
        for index in batch:
            trajectory = trajectories[index]
            for step_index, sample in enumerate(trajectory.steps):
                advantage = float(step_advantages[index][step_index])
                prompt = torch.tensor([list(sample.prompt_ids)], dtype=torch.long, device=device)
                completion = torch.tensor(
                    [list(sample.completion_ids)], dtype=torch.long, device=device
                )
                full_ids = torch.cat([prompt, completion], dim=1)
                new_logp = _completion_token_logprobs(model, torch, full_ids, prompt.shape[1])
                mask = torch.tensor(sample.token_mask, dtype=torch.bool, device=device)
                old_logp = torch.tensor(sample.old_logps, dtype=torch.float32, device=device)
                policy_loss, metrics = torch_clipped_policy_loss(
                    new_logp,
                    old_logp,
                    advantage,
                    mask=mask,
                    clip_eps=clip_eps,
                    enabled=ratio_clipping,
                    torch=torch,
                )
                reference = reference_logps[index][step_index].to(device)
                kl = torch_masked_mean(new_logp, mask, torch=torch) - torch_masked_mean(
                    reference, mask, torch=torch
                )
                step_loss = policy_loss + beta * kl
                active = float(metrics["active_tokens"])
                (step_loss * (active / chunk_tokens)).backward()
                total_active += active
                totals["loss"] += float(step_loss.detach().cpu()) * active
                totals["policy_loss"] += float(policy_loss.detach().cpu()) * active
                totals["kl_mean"] += float(kl.detach().cpu()) * active
                totals["ratio_mean"] += metrics["ratio_mean"] * active
                totals["clip_fraction"] += metrics["clip_fraction"] * active
        optimizer.step()
    if total_active <= 0:
        raise RuntimeError("trajectory group produced no active tokens")
    return {
        "loss": totals["loss"] / total_active,
        "policy_loss": totals["policy_loss"] / total_active,
        "kl_mean": totals["kl_mean"] / total_active,
        "ratio_mean": totals["ratio_mean"] / total_active,
        "clip_fraction": totals["clip_fraction"] / total_active,
        "optimizer_updates": len(batches),
        "ppo_epochs": 1,
    }


def _run_trajectory_prompt(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    optimizer: Any,
    provider: _TrajectoryProvider,
    row: Mapping[str, Any],
    tools: Sequence[Mapping[str, Any]],
    group_size: int,
    seed: int,
    step: int,
    beta: float,
    clip_eps: float,
    ratio_clipping: bool,
    weighting: str,
    delete_split: bool,
    max_turns: int,
    mini_batch_size: int,
    gate: DeleteTrajectoryGate,
    step_credit: bool = False,
    credit_gate: StepCreditGate | None = None,
    advantage_scale: float = 1.0,
) -> tuple[dict[str, Any], int, bool]:
    """Roll out and update one prompt with trajectory-level scoring."""
    if step_credit and credit_gate is None:
        raise RuntimeError("step credit requires a StepCreditGate")
    model.config.use_cache = True
    model.eval()
    trajectories: list[TrajectorySample] = []
    scores: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    try:
        for trajectory_index in range(group_size):
            sample, score, content = _collect_trajectory(
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                provider=provider,
                row=row,
                tools=tools,
                severity_weighted=weighting == "severity",
                delete_split=delete_split,
                max_turns=max_turns,
                seed=seed + step * 1009 + trajectory_index * 7919,
            )
            trajectories.append(sample)
            scores.append(score)
            details.append(
                {
                    "steps": len(sample.steps),
                    "reward": sample.reward,
                    "bucket": sample.bucket,
                    "detail": sample.detail,
                    "final_content": content,
                    "calls": [
                        {
                            "name": call["name"],
                            "arguments": call["arguments"],
                            "ok": call["ok"],
                        }
                        for call in sample.calls
                    ],
                }
            )
    finally:
        model.train()
        model.config.use_cache = False
        model.set_adapter("policy")

    rewards = [sample.reward for sample in trajectories]
    advantages, zero_variance = trajectory_advantages(rewards)
    advantages = scale_advantages(advantages, advantage_scale)
    layer = str(row["layer"])
    gate.observe(scores, layer=layer)
    if step_credit:
        step_advantages, credit_diagnostics = step_credit_advantages(trajectories, advantages)
        if credit_gate is None:
            raise RuntimeError("step credit requires a StepCreditGate")
        credit_gate.observe(layer=layer, scores=scores, diagnostics=credit_diagnostics)
    else:
        step_advantages = broadcast_advantages(trajectories, advantages)
        credit_diagnostics = []
    gate_failed = gate.failed or (credit_gate.failed if credit_gate is not None else False)
    record: dict[str, Any] = {
        "task_id": row["task_id"],
        "layer": row["layer"],
        "scoring": "trajectory",
        "rewards": rewards,
        "mean_reward": sum(rewards) / len(rewards),
        "advantages": advantages,
        "advantage_scale": advantage_scale,
        "zero_variance": zero_variance,
        "buckets": dict(Counter(sample.bucket for sample in trajectories)),
        "delete_execution_violations": sum(
            1 for sample in trajectories if sample.bucket == "execution_violation"
        ),
        "trajectories": details,
        "delete_gate": gate.as_dict(),
        "step_credit": (
            {"enabled": True, "diagnostics": credit_diagnostics} if step_credit else None
        ),
        "step_credit_gate": credit_gate.as_dict() if credit_gate is not None else None,
    }
    if zero_variance:
        return record, 0, gate_failed
    metrics = _update_trajectory_group(
        model=model,
        torch=torch,
        optimizer=optimizer,
        trajectories=trajectories,
        step_advantages=step_advantages,
        beta=beta,
        clip_eps=clip_eps,
        ratio_clipping=ratio_clipping,
        mini_batch_size=mini_batch_size,
        seed=seed,
        step=step,
    )
    record.update(metrics)
    return record, int(metrics["optimizer_updates"]), gate_failed


def _generate_group(
    model: Any,
    tokenizer: Any,
    torch: Any,
    prompt: str,
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    completion_mask: bool = True,
    compute_old_logps: bool = True,
) -> tuple[Any, list[Any], list[str], list[Any], list[Any]]:
    model.config.use_cache = True
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    attention_mask = encoded["attention_mask"].to(input_ids.device)
    prompt_len = int(input_ids.shape[-1])
    torch.manual_seed(seed)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=group_size,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completions: list[Any] = []
    texts: list[str] = []
    for row in outputs:
        completion_ids = row[prompt_len:]
        completions.append(completion_ids)
        texts.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
    masks = [
        _completion_mask(
            completion,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            torch=torch,
            enabled=completion_mask,
        )
        for completion in completions
    ]
    old_token_logps: list[Any] = []
    if compute_old_logps:
        with torch.no_grad():
            for completion in completions:
                full_ids = torch.cat([input_ids[0], completion]).unsqueeze(0)
                old_token_logps.append(
                    _completion_token_logprobs(model, torch, full_ids, prompt_len).detach()
                )
    model.config.use_cache = False
    model.train()
    return input_ids[0], completions, texts, masks, old_token_logps


def _prepare_model(
    *,
    model_path: Path,
    init_adapter: Path,
    torch: Any,
    keep_dropout: bool = False,
) -> tuple[Any, Any]:
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
    )
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(
        base,
        str(init_adapter),
        adapter_name="policy",
        is_trainable=True,
    )
    model.load_adapter(str(init_adapter), adapter_name="ref", is_trainable=False)
    model.set_adapter("policy")
    model.config.use_cache = False
    model.train()
    # PPO compares sampler log-probs with recomputed log-probs, so active
    # dropout would inject noise into the ratio before the first update. The
    # SFT adapter ships lora_dropout=0.05; neutralise it by default while
    # staying in train mode, which gradient checkpointing requires. The
    # mask-only attribution arm opts back into the v16-era 0.05 dropout.
    if not keep_dropout:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    return model, tokenizer


def validate_completion_options(
    *,
    completion_mask: bool,
    legacy_reinforce: bool,
    ratio_clipping: bool,
    ppo_epochs: int,
    mini_batch_size: int,
    group_size: int,
    trajectory_scoring: bool,
) -> int:
    """Validate the v16-compatible mask attribution switches.

    Returns the resolved mini-batch size. The compatibility switches are not
    experimental variables; they only reconstruct the frozen v16 baseline so
    the treatment arm differs by completion mask alone.
    """

    resolved = mini_batch_size or group_size
    if trajectory_scoring and not completion_mask:
        raise ValueError("--no-completion-mask only supports completion scoring")
    if legacy_reinforce:
        if ratio_clipping:
            raise ValueError("--legacy-reinforce requires --no-ratio-clipping")
        if ppo_epochs != 1:
            raise ValueError("--legacy-reinforce requires --ppo-epochs 1")
        if resolved != group_size:
            raise ValueError("--legacy-reinforce requires a single full-group batch")
    return resolved


def _run_arm(
    *,
    prompts: Sequence[Mapping[str, Any]],
    model_path: Path,
    init_adapter: Path,
    out_dir: Path,
    name: str,
    beta: float,
    group_size: int,
    learning_rate: float,
    weighting: str,
    epochs: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    limit: int,
    clip_eps: float,
    ratio_clipping: bool,
    ppo_epochs: int,
    mini_batch_size: int,
    delete_split: bool = False,
    baseline_commit: str = "",
    trajectory_scoring: bool = False,
    trajectory_max_turns: int = 9,
    step_credit: bool = False,
    advantage_scale: float = 1.0,
    completion_mask: bool = True,
    keep_dropout: bool = False,
    legacy_reinforce: bool = False,
) -> dict[str, Any]:
    import torch

    if weighting not in {"severity", "equal"}:
        raise ValueError(f"unknown weighting: {weighting}")
    if not 0.0 < clip_eps < 1.0:
        raise ValueError(f"clip_eps must be in (0, 1), got {clip_eps}")
    if ppo_epochs < 1:
        raise ValueError(f"ppo_epochs must be at least 1, got {ppo_epochs}")
    if mini_batch_size < 0:
        raise ValueError(f"mini_batch_size must be non-negative, got {mini_batch_size}")
    if trajectory_scoring and ppo_epochs != 1:
        raise ValueError("轨迹级评分固定 ppo_epochs=1，避免双变量")
    if trajectory_scoring and delete_split:
        raise ValueError("轨迹级评分必须使用 legacy reward，不允许 delete split")
    if step_credit and not trajectory_scoring:
        raise ValueError("--step-credit 必须与 --trajectory-scoring 同时使用")
    if advantage_scale <= 0:
        raise ValueError("--advantage-scale 必须为正数")
    if step_credit and advantage_scale != 1.0:
        raise ValueError("步骤级信用分配实验不允许同时缩放 advantage")
    if trajectory_max_turns < 1:
        raise ValueError("trajectory_max_turns must be >= 1")
    resolved_mini_batch_size = validate_completion_options(
        completion_mask=completion_mask,
        legacy_reinforce=legacy_reinforce,
        ratio_clipping=ratio_clipping,
        ppo_epochs=ppo_epochs,
        mini_batch_size=mini_batch_size,
        group_size=group_size,
        trajectory_scoring=trajectory_scoring,
    )
    selected = list(prompts[:limit] if limit else prompts)
    tools = build_default_registry().specs()
    model, tokenizer = _prepare_model(
        model_path=model_path,
        init_adapter=init_adapter,
        torch=torch,
        keep_dropout=keep_dropout,
    )
    provider = (
        _TrajectoryProvider(
            model,
            tokenizer,
            torch,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        if trajectory_scoring
        else None
    )
    gate = DeleteTrajectoryGate()
    credit_gate = StepCreditGate() if trajectory_scoring and step_credit else None
    gate_failed_flag = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    started = time.perf_counter()
    step = 0
    completed_steps = 0
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        for epoch in range(epochs):
            order = list(selected)
            random.Random(seed + epoch).shuffle(order)
            for index, row in enumerate(order, start=1):
                if trajectory_scoring:
                    if provider is None:
                        raise RuntimeError("trajectory provider was not initialised")
                    record, updates, gate_failed = _run_trajectory_prompt(
                        model=model,
                        tokenizer=tokenizer,
                        torch=torch,
                        optimizer=optimizer,
                        provider=provider,
                        row=row,
                        tools=tools,
                        group_size=group_size,
                        seed=seed,
                        step=step,
                        beta=beta,
                        clip_eps=clip_eps,
                        ratio_clipping=ratio_clipping,
                        weighting=weighting,
                        delete_split=delete_split,
                        max_turns=trajectory_max_turns,
                        mini_batch_size=resolved_mini_batch_size,
                        gate=gate,
                        step_credit=step_credit,
                        credit_gate=credit_gate,
                        advantage_scale=advantage_scale,
                    )
                    record["epoch"] = epoch
                    record["step"] = step
                    record["prompt_index"] = index
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    step += 1
                    completed_steps += updates
                    print(
                        f"[{name}] step={step} task={row['task_id']} "
                        f"scoring=trajectory mean_r={record['mean_reward']:.3f} "
                        f"buckets={record['buckets']} gate={gate.status}",
                        flush=True,
                    )
                    if gate_failed:
                        gate_failed_flag = True
                        break
                    continue
                task = dict(row["task"])
                prompt = render_prompt(tokenizer, task, tools)
                prompt_ids, completion_ids, texts, completion_masks, old_token_logps = (
                    _generate_group(
                        model,
                        tokenizer,
                        torch,
                        prompt,
                        group_size=group_size,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        seed=seed + step * 1009,
                        completion_mask=completion_mask,
                        compute_old_logps=not legacy_reinforce,
                    )
                )
                scores = [
                    score_completion(
                        task,
                        text,
                        tools,
                        severity_weighted=weighting == "severity",
                        delete_split=delete_split,
                    )
                    for text in texts
                ]
                rewards = [float(item["reward"]) for item in scores]
                advantages, zero_variance = group_advantages(rewards)
                buckets = Counter(str(item["bucket"]) for item in scores)
                record: dict[str, Any] = {
                    "epoch": epoch,
                    "step": step,
                    "task_id": row["task_id"],
                    "layer": row["layer"],
                    "prompt_index": index,
                    "rewards": rewards,
                    "mean_reward": sum(rewards) / len(rewards),
                    "advantages": advantages,
                    "zero_variance": zero_variance,
                    "buckets": dict(buckets),
                    "completions": texts,
                    "tool_calls": [item["tool_calls"] for item in scores],
                }
                if zero_variance:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    step += 1
                    continue
                # Gradient checkpointing only runs in train mode; eval() here
                # would retain every activation and OOM at 7B. Dropout is
                # neutralised in _prepare_model, so train/eval log-probs match.
                model.train()
                # The reference adapter is frozen, so its log-probs are reused
                # across every inner epoch. Only the policy log-probs change.
                reference_token_logps: list[Any] = []
                with torch.no_grad():
                    model.set_adapter("ref")
                    for completion in completion_ids:
                        full_ids = torch.cat(
                            [prompt_ids, completion.to(prompt_ids.device)]
                        ).unsqueeze(0)
                        reference_token_logps.append(
                            _completion_token_logprobs(
                                model, torch, full_ids, prompt_ids.numel()
                            ).detach()
                        )
                    model.set_adapter("policy")
                effective_mini_batch_size = min(resolved_mini_batch_size, len(completion_ids))
                inner_epoch_records: list[dict[str, Any]] = []
                total_active = 0.0
                total_policy_loss = 0.0
                total_kl = 0.0
                total_loss = 0.0
                total_ratio = 0.0
                total_clip = 0.0
                prompt_optimizer_updates = 0
                for inner_epoch in range(ppo_epochs):
                    order = list(range(len(completion_ids)))
                    random.Random(seed + step * 1009 + inner_epoch * 7919).shuffle(order)
                    batches = split_minibatches(order, effective_mini_batch_size)
                    epoch_active = 0.0
                    epoch_policy_loss = 0.0
                    epoch_kl = 0.0
                    epoch_loss = 0.0
                    epoch_ratio = 0.0
                    epoch_clip = 0.0
                    for batch_indices in batches:
                        optimizer.zero_grad(set_to_none=True)
                        for sample_index in batch_indices:
                            completion = completion_ids[sample_index]
                            advantage = advantages[sample_index]
                            mask = completion_masks[sample_index]
                            old_logp = (
                                old_token_logps[sample_index] if not legacy_reinforce else None
                            )
                            reference_logp = reference_token_logps[sample_index]
                            full_ids = torch.cat(
                                [prompt_ids, completion.to(prompt_ids.device)]
                            ).unsqueeze(0)
                            model.set_adapter("policy")
                            new_logp = _completion_token_logprobs(
                                model, torch, full_ids, prompt_ids.numel()
                            )
                            if legacy_reinforce:
                                policy_loss = -float(advantage) * torch_masked_mean(
                                    new_logp, mask, torch=torch
                                )
                                ratio_metrics = {
                                    "ratio_mean": 1.0,
                                    "clip_fraction": 0.0,
                                    "active_tokens": float(mask.sum().detach().cpu()),
                                }
                            else:
                                policy_loss, ratio_metrics = torch_clipped_policy_loss(
                                    new_logp,
                                    old_logp,
                                    float(advantage),
                                    mask=mask,
                                    clip_eps=clip_eps,
                                    enabled=ratio_clipping,
                                    torch=torch,
                                )
                            new_mean = torch_masked_mean(new_logp, mask, torch=torch)
                            reference_mean = torch_masked_mean(reference_logp, mask, torch=torch)
                            kl = new_mean - reference_mean
                            sample_loss = policy_loss + beta * kl
                            (sample_loss / len(batch_indices)).backward()
                            active = ratio_metrics["active_tokens"]
                            epoch_active += active
                            epoch_policy_loss += float(policy_loss.detach().cpu()) * active
                            epoch_kl += float(kl.detach().cpu()) * active
                            epoch_loss += float(sample_loss.detach().cpu()) * active
                            epoch_ratio += ratio_metrics["ratio_mean"] * active
                            epoch_clip += ratio_metrics["clip_fraction"] * active
                        optimizer.step()
                        prompt_optimizer_updates += 1
                        completed_steps += 1
                    if epoch_active <= 0:
                        raise RuntimeError("PPO inner epoch produced no active tokens")
                    inner_epoch_records.append(
                        {
                            "inner_epoch": inner_epoch,
                            "policy_loss": epoch_policy_loss / epoch_active,
                            "kl_mean": epoch_kl / epoch_active,
                            "loss": epoch_loss / epoch_active,
                            "ratio_mean": epoch_ratio / epoch_active,
                            "clip_fraction": epoch_clip / epoch_active,
                            "optimizer_updates": len(batches),
                        }
                    )
                    total_active += epoch_active
                    total_policy_loss += epoch_policy_loss
                    total_kl += epoch_kl
                    total_loss += epoch_loss
                    total_ratio += epoch_ratio
                    total_clip += epoch_clip
                model.train()
                record["ppo_epochs"] = ppo_epochs
                record["mini_batch_size"] = effective_mini_batch_size
                record["optimizer_updates"] = prompt_optimizer_updates
                record["inner_epochs"] = inner_epoch_records
                record["loss"] = total_loss / total_active if total_active else None
                record["policy_loss"] = total_policy_loss / total_active if total_active else None
                record["kl_mean"] = total_kl / total_active if total_active else None
                record["ratio_mean"] = total_ratio / total_active if total_active else None
                record["clip_fraction"] = total_clip / total_active if total_active else None
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                step += 1
                if completed_steps % 1 == 0:
                    print(
                        f"[{name}] step={step} task={row['task_id']} "
                        f"mean_r={record['mean_reward']:.3f} buckets={dict(buckets)}",
                        flush=True,
                    )
                torch.cuda.empty_cache()
            if gate_failed_flag:
                break
    if gate_failed_flag:
        reason = "delete execution_violation 0/48"
        if credit_gate is not None and credit_gate.failed and gate.passed:
            reason = "step-credit allocation gate failed"
        fail_payload = gate_fail_payload(gate, reason=reason, credit_gate=credit_gate)
        credit_summary = summarize_step_credit(log_path) if step_credit else None
        fail_manifest = {
            **fail_payload,
            "name": name,
            "scoring": "trajectory",
            "step_credit": step_credit,
            "completion_mask": completion_mask,
            "dropout_disabled": not keep_dropout,
            "legacy_reinforce": legacy_reinforce,
            "advantage_scale": advantage_scale,
            "step_credit_rule": (
                "trigger=1.0,prev=0.5,others=0;delete_execution_violation_only"
                if step_credit
                else None
            ),
            "step_credit_gate": credit_gate.as_dict() if credit_gate is not None else None,
            "step_credit_summary": credit_summary,
            "mass_before_total": (
                credit_summary["mass_before_total"] if credit_summary is not None else None
            ),
            "mass_after_total": (
                credit_summary["mass_after_total"] if credit_summary is not None else None
            ),
            "mass_ratio": credit_summary["mass_ratio"] if credit_summary is not None else None,
            "mass_ratio_count": (
                credit_summary["mass_ratio_count"] if credit_summary is not None else None
            ),
            "mass_ratio_band_count": (
                credit_summary["mass_ratio_band_count"] if credit_summary is not None else None
            ),
            "mass_ratio_median": (
                credit_summary["mass_ratio_median"] if credit_summary is not None else None
            ),
            "mass_ratio_min": (
                credit_summary["mass_ratio_min"] if credit_summary is not None else None
            ),
            "mass_ratio_max": (
                credit_summary["mass_ratio_max"] if credit_summary is not None else None
            ),
            "mass_ratio_drop": (
                credit_summary["mass_ratio_drop"] if credit_summary is not None else None
            ),
            "mass_ratios": (credit_summary["mass_ratios"] if credit_summary is not None else None),
            "train_log": str(log_path),
            "prompts": len(selected),
            "zero_variance_groups": count_zero_variance_groups(log_path),
            "duration_s": round(time.perf_counter() - started, 1),
        }
        write_json(out_dir / "GATE_FAIL.json", fail_manifest)
        (out_dir / "run_manifest.json").write_text(
            json.dumps(fail_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(fail_manifest, ensure_ascii=False, indent=2), flush=True)
        return fail_manifest
    model.set_adapter("policy")
    model.save_pretrained(str(out_dir), selected_adapters=["policy"], safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    saved_adapter_dir = out_dir / "policy"
    weight_path = saved_adapter_dir / "adapter_model.safetensors"
    credit_summary = summarize_step_credit(log_path) if step_credit else None
    manifest = {
        "status": "completed",
        "name": name,
        "scoring": "trajectory" if trajectory_scoring else "completion",
        "trajectory_max_turns": trajectory_max_turns if trajectory_scoring else None,
        "delete_gate": gate.as_dict() if trajectory_scoring else None,
        "step_credit": step_credit,
        "advantage_scale": advantage_scale,
        "step_credit_rule": (
            "trigger=1.0,prev=0.5,others=0;delete_execution_violation_only" if step_credit else None
        ),
        "step_credit_gate": credit_gate.as_dict() if credit_gate is not None else None,
        "step_credit_summary": credit_summary,
        "mass_before_total": (
            credit_summary["mass_before_total"] if credit_summary is not None else None
        ),
        "mass_after_total": (
            credit_summary["mass_after_total"] if credit_summary is not None else None
        ),
        "mass_ratio": credit_summary["mass_ratio"] if credit_summary is not None else None,
        "mass_ratio_count": (
            credit_summary["mass_ratio_count"] if credit_summary is not None else None
        ),
        "mass_ratio_band_count": (
            credit_summary["mass_ratio_band_count"] if credit_summary is not None else None
        ),
        "mass_ratio_median": (
            credit_summary["mass_ratio_median"] if credit_summary is not None else None
        ),
        "mass_ratio_min": (
            credit_summary["mass_ratio_min"] if credit_summary is not None else None
        ),
        "mass_ratio_max": (
            credit_summary["mass_ratio_max"] if credit_summary is not None else None
        ),
        "mass_ratio_drop": (
            credit_summary["mass_ratio_drop"] if credit_summary is not None else None
        ),
        "mass_ratios": credit_summary["mass_ratios"] if credit_summary is not None else None,
        "beta": beta,
        "group_size": group_size,
        "learning_rate": learning_rate,
        "weighting": weighting,
        "delete_reward": "split_v1" if delete_split else "legacy",
        "confirm_phrases": CONFIRM_PHRASES_VERSION,
        "confirm_phrases_sha256": confirm_phrases_fingerprint(),
        "baseline_commit": baseline_commit or None,
        "ratio_clipping": ratio_clipping,
        "clip_eps": clip_eps,
        "epochs": epochs,
        "ppo_epochs": ppo_epochs,
        "mini_batch_size": resolved_mini_batch_size,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "completion_mask": completion_mask,
        "dropout_disabled": not keep_dropout,
        "legacy_reinforce": legacy_reinforce,
        "loss_mode": "legacy_reinforce" if legacy_reinforce else "ppo_clipped",
        "seed": seed,
        "prompts": len(selected),
        "optimizer_updates": completed_steps,
        "zero_variance_groups": count_zero_variance_groups(log_path),
        "init_adapter": str(init_adapter),
        "init_adapter_sha256": sha256_file(init_adapter / "adapter_model.safetensors"),
        "adapter_dir": str(saved_adapter_dir),
        "adapter_sha256": sha256_file(weight_path) if weight_path.is_file() else None,
        "train_log": str(log_path),
        "duration_s": round(time.perf_counter() - started, 1),
        "device": torch.cuda.get_device_name(0),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v14 GRPO-lite action training")
    parser.add_argument(
        "--prompts", type=Path, default=Path("private/v14/step5_grpo/prompts/prompts.jsonl")
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path("/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    )
    parser.add_argument(
        "--init-adapter",
        type=Path,
        default=Path("private/v14/step3_5_light_sft/adapter"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("private/v14/step5_grpo/arms"))
    parser.add_argument("--name", default="weighted_beta0.01_g8_lr5e-6")
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=1,
        help="inner PPO epochs reusing each generated group (default: 1)",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=0,
        help="completions per optimizer step; 0 uses the full group",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weighting", choices=("severity", "equal"), default="severity")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=0.2,
        help="PPO ratio clipping epsilon (default: 0.2)",
    )
    parser.add_argument(
        "--no-ratio-clipping",
        dest="ratio_clipping",
        action="store_false",
        help="disable PPO ratio clipping for backward-compatible REINFORCE runs",
    )
    parser.set_defaults(ratio_clipping=True)
    parser.add_argument(
        "--completion-mask",
        dest="completion_mask",
        action="store_true",
        help="mask trailing padding in completion loss (default: enabled)",
    )
    parser.add_argument(
        "--no-completion-mask",
        dest="completion_mask",
        action="store_false",
        help="include trailing padding in completion loss (v16-compatible)",
    )
    parser.set_defaults(completion_mask=True)
    parser.add_argument(
        "--keep-dropout",
        action="store_true",
        help="keep adapter dropout active during training (v16-compatible)",
    )
    parser.add_argument(
        "--legacy-reinforce",
        action="store_true",
        help="use the v16 direct -advantage*logp loss instead of the PPO ratio",
    )
    parser.add_argument(
        "--delete-confirm-split",
        dest="delete_split",
        action="store_true",
        help="delete layer: confirm request +1.5, refusal +1.0, execution -2*severity",
    )
    parser.set_defaults(delete_split=False)
    parser.add_argument(
        "--baseline-commit",
        default="",
        help="pre-registration commit recorded in the run manifest",
    )
    parser.add_argument(
        "--trajectory-scoring",
        action="store_true",
        help="score complete multi-turn trajectories instead of single completions",
    )
    parser.add_argument(
        "--trajectory-max-turns",
        type=int,
        default=9,
        help="max turns per trajectory rollout (default: 9)",
    )
    parser.add_argument(
        "--step-credit",
        action="store_true",
        help="delete layer: trigger step 1.0x, previous step 0.5x, others 0",
    )
    parser.add_argument(
        "--advantage-scale",
        type=float,
        default=1.0,
        help="multiply group-normalized trajectory advantages (default: 1.0)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompts = load_jsonl(args.prompts)
    if not args.init_adapter.is_dir():
        raise FileNotFoundError(f"missing init adapter: {args.init_adapter}")
    if args.ppo_epochs < 1:
        raise ValueError(f"ppo_epochs must be at least 1, got {args.ppo_epochs}")
    if args.mini_batch_size < 0:
        raise ValueError(f"mini_batch_size must be non-negative, got {args.mini_batch_size}")
    if args.trajectory_scoring and args.ppo_epochs != 1:
        raise ValueError("轨迹级评分固定 ppo_epochs=1，避免双变量")
    if args.trajectory_scoring and args.delete_split:
        raise ValueError("轨迹级评分必须使用 legacy reward，不允许 delete split")
    if args.trajectory_max_turns < 1:
        raise ValueError("trajectory_max_turns must be >= 1")
    if args.step_credit and not args.trajectory_scoring:
        raise ValueError("--step-credit 必须与 --trajectory-scoring 同时使用")
    if args.advantage_scale <= 0:
        raise ValueError("--advantage-scale 必须为正数")
    if args.step_credit and args.advantage_scale != 1.0:
        raise ValueError("步骤级信用分配实验不允许同时缩放 advantage")
    resolved_mini_batch_size = validate_completion_options(
        completion_mask=args.completion_mask,
        legacy_reinforce=args.legacy_reinforce,
        ratio_clipping=args.ratio_clipping,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        group_size=args.group_size,
        trajectory_scoring=args.trajectory_scoring,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "prompts": len(prompts),
                    "by_layer": dict(Counter(row["layer"] for row in prompts)),
                    "name": args.name,
                    "beta": args.beta,
                    "group_size": args.group_size,
                    "ppo_epochs": args.ppo_epochs,
                    "mini_batch_size": resolved_mini_batch_size,
                    "learning_rate": args.learning_rate,
                    "weighting": args.weighting,
                    "delete_reward": "split_v1" if args.delete_split else "legacy",
                    "confirm_phrases": CONFIRM_PHRASES_VERSION,
                    "confirm_phrases_sha256": confirm_phrases_fingerprint(),
                    "baseline_commit": args.baseline_commit or None,
                    "scoring": "trajectory" if args.trajectory_scoring else "completion",
                    "step_credit": args.step_credit,
                    "advantage_scale": args.advantage_scale,
                    "trajectory_max_turns": (
                        args.trajectory_max_turns if args.trajectory_scoring else None
                    ),
                    "ratio_clipping": args.ratio_clipping,
                    "clip_eps": args.clip_eps,
                    "completion_mask": args.completion_mask,
                    "keep_dropout": args.keep_dropout,
                    "dropout_disabled": not args.keep_dropout,
                    "legacy_reinforce": args.legacy_reinforce,
                    "loss_mode": ("legacy_reinforce" if args.legacy_reinforce else "ppo_clipped"),
                    "init_adapter_sha256": sha256_file(
                        args.init_adapter / "adapter_model.safetensors"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    out_dir = args.out_dir / args.name
    _run_arm(
        prompts=prompts,
        model_path=args.model_path,
        init_adapter=args.init_adapter,
        out_dir=out_dir,
        name=args.name,
        beta=args.beta,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        weighting=args.weighting,
        epochs=args.epochs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        limit=args.limit,
        clip_eps=args.clip_eps,
        ratio_clipping=args.ratio_clipping,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        delete_split=args.delete_split,
        baseline_commit=args.baseline_commit,
        trajectory_scoring=args.trajectory_scoring,
        trajectory_max_turns=args.trajectory_max_turns,
        step_credit=args.step_credit,
        advantage_scale=args.advantage_scale,
        completion_mask=args.completion_mask,
        keep_dropout=args.keep_dropout,
        legacy_reinforce=args.legacy_reinforce,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
