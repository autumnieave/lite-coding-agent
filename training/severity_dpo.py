"""Per-sample severity-weighted DPO trainer (TRL 0.12 DPO).

TRL 0.12 does not expose per-example weights, so this subclass re-implements
`get_batch_loss_metrics` and multiplies each DPO loss by the weight of the
constraint found in the batch prompt (severity 3/2/1 for path/delete/edit).
Weights are normalised per batch to keep the average loss scale unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from trl import DPOTrainer

from training.severity_weights import normalised_weights, weight_for_prompt


class SeverityWeightedDPOTrainer(DPOTrainer):
    """DPOTrainer with per-example weights keyed by constraint id in the prompt."""

    weights_by_constraint: Mapping[str, float] = {}

    def set_weights(self, weights: Mapping[str, float]) -> None:
        self.weights_by_constraint = dict(weights)

    def _weight_for(self, prompt: str) -> float:
        return weight_for_prompt(prompt, self.weights_by_constraint)

    def get_batch_loss_metrics(
        self,
        model: Any,
        batch: dict[str, Any],
        train_eval: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        metrics: dict[str, torch.Tensor] = {}
        model_output = self.concatenated_forward(model, batch)
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen = batch["ref_chosen_logps"]
            ref_rejected = batch["ref_rejected_logps"]
        else:
            ref_chosen, ref_rejected = self.compute_ref_log_probs(batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen,
            ref_rejected,
        )
        prompts = self.tokenizer.batch_decode(batch["prompt_input_ids"], skip_special_tokens=True)
        scaled = normalised_weights([self._weight_for(text) for text in prompts])
        weights = torch.tensor(scaled, device=losses.device, dtype=losses.dtype)
        weighted = losses * weights

        prefix = "eval_" if train_eval == "eval" else ""
        rewards_accuracies = (chosen_rewards > rejected_rewards).float()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().detach().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().detach().cpu()
        metrics[f"{prefix}rewards/accuracies"] = rewards_accuracies.mean().detach().cpu()
        metrics[f"{prefix}rewards/margins"] = (
            (chosen_rewards - rejected_rewards).mean().detach().cpu()
        )
        metrics[f"{prefix}logps/chosen"] = model_output["chosen_logps"].detach().mean().cpu()
        metrics[f"{prefix}logps/rejected"] = model_output["rejected_logps"].detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = model_output["mean_chosen_logits"].detach().cpu()
        metrics[f"{prefix}logits/rejected"] = model_output["mean_rejected_logits"].detach().cpu()
        metrics[f"{prefix}weights/mean"] = weights.mean().detach().cpu()
        metrics[f"{prefix}weights/max"] = weights.max().detach().cpu()
        self.store_metrics(metrics, train_eval=train_eval)
        return weighted.mean(), metrics
