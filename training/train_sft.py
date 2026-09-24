"""QLoRA SFT 骨架；实际执行放到 Kaggle T4。"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.adapter_artifacts import validate_saved_adapter
from training.config import load_config, validate_t4_config
from training.loss_masking import LOSS_MODES, LabelPreservingCollator, tokenize_training_row
from training.sft_format import canonicalize_tool_arguments

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def lora_target_modules(config: Mapping[str, Any]) -> str | list[str]:
    """Return configured PEFT target modules, defaulting to Qwen attention/MLP layers."""

    target = config.get("target_modules")
    if isinstance(target, str):
        return target
    if isinstance(target, list) and target:
        return [str(name) for name in target]
    return list(DEFAULT_TARGET_MODULES)


def over_length_rows(rows: Sequence[Mapping[str, Any]], max_length: int) -> list[dict[str, Any]]:
    """List rows whose rendered token count exceeds the training budget."""

    return [
        {"id": row.get("id"), "tokens": int(row.get("token_count") or 0)}
        for row in rows
        if int(row.get("token_count") or 0) > max_length
    ]


def _ensure_src_on_path() -> None:
    """Make the in-repo ``agent`` package importable outside an installed env."""

    src = Path(__file__).resolve().parents[1] / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def default_tool_schemas() -> list[dict[str, Any]]:
    """Return the same default tool menu used by the Agent inference loop."""

    _ensure_src_on_path()
    from agent.tools import build_default_registry

    return build_default_registry().specs()


def render_training_text(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Render one SFT row with the same Qwen tools schema used at inference time."""

    messages = canonicalize_tool_arguments(row["messages"])
    messages.append({"role": "assistant", "content": row["completion"]})
    tool_schemas = list(tools) if tools is not None else default_tool_schemas()
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=tool_schemas,
            tokenize=False,
            add_generation_prompt=False,
        )
    except TypeError as exc:
        raise RuntimeError(
            "tokenizer.apply_chat_template 不接受 tools=；训练与推理的工具协议会不一致"
        ) from exc


def _dataclass_kwargs(cls: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    fields = getattr(cls, "__dataclass_fields__", None)
    if not fields:
        return dict(values)
    return {key: value for key, value in values.items() if key in fields}


def _load_trainer_classes() -> tuple[Any, Any]:
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Kaggle 环境缺少 trl；先安装 training/requirements-kaggle.txt") from exc
    return SFTConfig, SFTTrainer


def run_sft(
    *,
    config_path: Path,
    model_path: Path,
    train_file: Path,
    val_file: Path,
    output_dir: Path,
    dry_run: bool,
    loss_mode: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_t4_config(config)
    selected_loss_mode = str(loss_mode or config.get("loss_mode") or "full")
    if selected_loss_mode not in LOSS_MODES:
        raise ValueError(f"loss_mode must be one of {LOSS_MODES}, got {selected_loss_mode!r}")
    for path, label in ((model_path, "model"), (train_file, "train"), (val_file, "val")):
        if not path.exists():
            raise FileNotFoundError(f"{label} 路径不存在：{path}")
    manifest: dict[str, Any] = {
        "status": "dry_run" if dry_run else "completed",
        "started_at": _now(),
        "model_path": str(model_path),
        "train_file": str(train_file),
        "val_file": str(val_file),
        "output_dir": str(output_dir),
        "config": config,
        "loss_mode": selected_loss_mode,
        "seed": seed if seed is not None else config.get("seed"),
        "request_id": None,
        "system_fingerprint": None,
    }
    if dry_run:
        manifest["duration_s"] = 0.0
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest

    # QLoRA training is single-GPU; hide any second T4 so SFTTrainer does not
    # wrap the quantized model in DataParallel.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle 环境缺少训练依赖；先安装 training/requirements-kaggle.txt"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("没有 CUDA。SFT 主实验必须在 Kaggle T4 上执行。")
    SFTConfig, SFTTrainer = _load_trainer_classes()
    started = time.perf_counter()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        # Kaggle T4 x2 sessions expose two GPUs. ``auto`` shards 7B across both,
        # which puts logits on cuda:1 and labels on cuda:0. Training must stay on
        # one GPU; the NF4 7B weights fit comfortably in a single 16 GB T4.
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if config.get("use_liger"):
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        except ImportError as exc:
            raise RuntimeError(
                "use_liger=true 但 Kaggle 环境缺少 liger-kernel；"
                "先安装 training/requirements-kaggle.txt"
            ) from exc
        apply_liger_kernel_to_qwen2(model=model)
    lora_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules(config),
    )

    max_length = int(config["max_seq_length"])

    def format_row(row: Mapping[str, Any]) -> dict[str, Any]:
        if selected_loss_mode == "assistant_only":
            return tokenize_training_row(
                tokenizer,
                row,
                loss_mode=selected_loss_mode,
                max_length=max_length,
            )
        text = render_training_text(tokenizer, row)
        tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        return {"text": text, "token_count": tokens}

    train_all = load_dataset("json", data_files=str(train_file), split="train").map(format_row)
    val_all = load_dataset("json", data_files=str(val_file), split="train").map(format_row)
    dropped = {
        "train": over_length_rows(list(train_all), max_length),
        "val": over_length_rows(list(val_all), max_length),
    }
    train_ds = train_all.filter(lambda row: row["token_count"] <= max_length)
    val_ds = val_all.filter(lambda row: row["token_count"] <= max_length)
    manifest["over_max_length"] = dropped
    config_values = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": int(config["num_train_epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "logging_steps": 1 if len(train_ds) < 32 else 10,
        "save_strategy": "no",
        # Per-epoch eval materialises full-vocab logits at 4096 tokens and OOMs on T4.
        # The real held-out comparison is a separate full-loop run after training.
        "eval_strategy": "no",
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_length": max_length,
        "max_seq_length": max_length,
        "dataset_text_field": "text" if selected_loss_mode == "full" else None,
        "report_to": "none",
    }
    if seed is not None:
        config_values["seed"] = int(seed)
    if config.get("optim"):
        config_values["optim"] = str(config["optim"])
    args = SFTConfig(**_dataclass_kwargs(SFTConfig, config_values))
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "peft_config": lora_config,
    }
    if selected_loss_mode == "assistant_only":
        # TRL's default DataCollatorForLanguageModeling rebuilds labels from
        # input_ids and would erase the assistant-only mask.
        trainer_kwargs["data_collator"] = LabelPreservingCollator(
            pad_token_id=int(tokenizer.pad_token_id or tokenizer.eos_token_id)
        )
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    manifest["trainable_params"] = sum(
        parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    manifest["total_params"] = sum(parameter.numel() for parameter in trainer.model.parameters())
    train_result = trainer.train()
    manifest["train_metrics"] = dict(getattr(train_result, "metrics", {}) or {})
    optimizer_steps = int(getattr(trainer.state, "global_step", 0) or 0)
    manifest["optimizer_steps"] = optimizer_steps
    manifest["optimizer_steps_ok"] = optimizer_steps >= 20
    manifest["steps_per_epoch"] = round(
        optimizer_steps / max(1, int(config["num_train_epochs"])), 2
    )
    manifest["effective_batch_size"] = 1 * 8
    manifest["train_loss"] = [
        row["loss"] for row in trainer.state.log_history if row.get("loss") is not None
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    adapter_validation = validate_saved_adapter(output_dir)
    manifest["adapter_files"] = adapter_validation["files"]
    manifest["adapter_total_bytes"] = adapter_validation["total_bytes"]
    manifest["adapter_validation"] = adapter_validation
    manifest.update(
        {
            "duration_s": round(time.perf_counter() - started, 1),
            "device": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "train_rows": len(train_ds),
            "val_rows": len(val_ds),
            "adapter_dir": str(output_dir),
        }
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA SFT（Kaggle T4）")
    parser.add_argument("--config", default="training/configs/qlora_1_5b_t4.json")
    parser.add_argument("--model-path", default="/kaggle/input/qwen2.5-1.5b-instruct")
    parser.add_argument("--train-file", default="/kaggle/input/lite-agent-training/train.jsonl")
    parser.add_argument("--val-file", default="/kaggle/input/lite-agent-training/val.jsonl")
    parser.add_argument("--output-dir", default="/kaggle/working/lora_adapter")
    parser.add_argument("--dry-run", action="store_true", help="只校验配置和路径，不加载模型")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_sft(
        config_path=Path(args.config),
        model_path=Path(args.model_path),
        train_file=Path(args.train_file),
        val_file=Path(args.val_file),
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
