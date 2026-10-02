"""On-policy GRPO-lite for the v14 action layer.

TRL 0.12 in this runtime does not provide GRPO, so this runner implements the
minimal group-relative objective used by the pre-registration: sample a group of
completions per prompt, normalize rewards within the group, and optimize the
group-relative policy gradient with a KL penalty to the frozen SFT adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent.tools import build_default_registry  # noqa: E402
from training.grpo_v14 import (  # noqa: E402
    group_advantages,
    render_prompt,
    score_completion,
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""

    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _completion_logprob(
    model: Any,
    torch: Any,
    input_ids: Any,
    completion_start: int,
) -> Any:
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits[:, completion_start - 1 : -1, :].float()
    targets = input_ids[:, completion_start:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.mean()


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
) -> tuple[Any, list[Any], list[str]]:
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
    model.config.use_cache = False
    model.train()
    return input_ids[0], completions, texts


def _prepare_model(
    *,
    model_path: Path,
    init_adapter: Path,
    torch: Any,
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
    return model, tokenizer


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
) -> dict[str, Any]:
    import torch

    if weighting not in {"severity", "equal"}:
        raise ValueError(f"unknown weighting: {weighting}")
    selected = list(prompts[:limit] if limit else prompts)
    tools = build_default_registry().specs()
    model, tokenizer = _prepare_model(
        model_path=model_path,
        init_adapter=init_adapter,
        torch=torch,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    started = time.perf_counter()
    step = 0
    completed_steps = 0
    skipped = 0
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        for epoch in range(epochs):
            order = list(selected)
            random.Random(seed + epoch).shuffle(order)
            for index, row in enumerate(order, start=1):
                task = dict(row["task"])
                prompt = render_prompt(tokenizer, task, tools)
                prompt_ids, completion_ids, texts = _generate_group(
                    model,
                    tokenizer,
                    torch,
                    prompt,
                    group_size=group_size,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    seed=seed + step * 1009,
                )
                scores = [
                    score_completion(
                        task,
                        text,
                        tools,
                        severity_weighted=weighting == "severity",
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
                    skipped += 1
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    step += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                losses: list[float] = []
                for completion, advantage in zip(completion_ids, advantages, strict=True):
                    full_ids = torch.cat([prompt_ids, completion.to(prompt_ids.device)]).unsqueeze(
                        0
                    )
                    model.set_adapter("policy")
                    policy_logp = _completion_logprob(model, torch, full_ids, prompt_ids.numel())
                    model.set_adapter("ref")
                    with torch.no_grad():
                        reference_logp = _completion_logprob(
                            model, torch, full_ids, prompt_ids.numel()
                        )
                    model.set_adapter("policy")
                    kl = policy_logp - reference_logp
                    loss = (-float(advantage) * policy_logp + beta * kl) / len(completion_ids)
                    loss.backward()
                    losses.append(float(loss.detach().cpu()))
                optimizer.step()
                completed_steps += 1
                record["loss"] = sum(losses)
                record["policy_logp_mean"] = None
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
    model.set_adapter("policy")
    model.save_pretrained(str(out_dir), selected_adapters=["policy"], safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    saved_adapter_dir = out_dir / "policy"
    weight_path = saved_adapter_dir / "adapter_model.safetensors"
    manifest = {
        "status": "completed",
        "name": name,
        "beta": beta,
        "group_size": group_size,
        "learning_rate": learning_rate,
        "weighting": weighting,
        "epochs": epochs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "seed": seed,
        "prompts": len(selected),
        "optimizer_updates": completed_steps,
        "zero_variance_groups": skipped,
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
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weighting", choices=("severity", "equal"), default="severity")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompts = load_jsonl(args.prompts)
    if not args.init_adapter.is_dir():
        raise FileNotFoundError(f"missing init adapter: {args.init_adapter}")
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
                    "learning_rate": args.learning_rate,
                    "weighting": args.weighting,
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
