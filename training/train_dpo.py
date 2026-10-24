"""QLoRA DPO entry point for the method-comparison arm."""

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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def default_tool_schemas() -> list[dict[str, Any]]:
    _ensure_src_on_path()
    from agent.tools import build_default_registry

    return build_default_registry().specs()


def render_dpo_prompt(tokenizer: Any, messages: Sequence[Mapping[str, Any]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            list(messages),
            tools=default_tool_schemas(),
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "tokenizer.apply_chat_template 不接受 tools=，DPO prompt 协议不一致"
        ) from exc


def _dataclass_kwargs(cls: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    fields = getattr(cls, "__dataclass_fields__", None)
    if not fields:
        return dict(values)
    return {key: value for key, value in values.items() if key in fields}


def run_dpo(
    *,
    config_path: Path,
    model_path: Path,
    train_file: Path,
    val_file: Path,
    output_dir: Path,
    dry_run: bool,
    init_adapter: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_t4_config(config)
    for path, label in ((model_path, "model"), (train_file, "train"), (val_file, "val")):
        if not path.exists():
            raise FileNotFoundError(f"{label} 路径不存在：{path}")
    if init_adapter is not None and not (init_adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"init adapter 缺少 adapter_config.json：{init_adapter}")
    manifest: dict[str, Any] = {
        "status": "dry_run" if dry_run else "completed",
        "started_at": _now(),
        "model_path": str(model_path),
        "train_file": str(train_file),
        "val_file": str(val_file),
        "output_dir": str(output_dir),
        "config": config,
        "beta": float(config.get("beta", 0.1)),
        "reference_model": "base model with adapter disabled",
        "policy_init": "sft_adapter" if init_adapter is not None else "base_model",
        "init_adapter": str(init_adapter) if init_adapter is not None else None,
    }
    if dry_run:
        manifest["duration_s"] = 0.0
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle 环境缺少 DPO 依赖；先安装 training/requirements-kaggle.txt"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("没有 CUDA；DPO 主实验必须在 Kaggle T4 上执行")

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_config: LoraConfig | None = None
    if init_adapter is None:
        lora_config = LoraConfig(
            r=int(config["lora_r"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.get("target_modules", "all-linear"),
        )
    else:
        # Continue from the SFT policy. DPOTrainer uses the adapter-disabled
        # base model as the reference when given a PeftModel policy.
        model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    def format_pair(row: Mapping[str, Any]) -> dict[str, str]:
        return {
            "prompt": render_dpo_prompt(tokenizer, row["prompt_messages"]),
            "chosen": str(row["chosen"]),
            "rejected": str(row["rejected"]),
        }

    train_ds = load_dataset("json", data_files=str(train_file), split="train").map(format_pair)
    val_ds = load_dataset("json", data_files=str(val_file), split="train").map(format_pair)
    config_values = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": int(config.get("num_train_epochs", 1)),
        "learning_rate": float(config.get("learning_rate", 5e-5)),
        "beta": float(config.get("beta", 0.1)),
        "max_length": int(config["max_seq_length"]),
        "max_prompt_length": int(config.get("max_prompt_length", 2048)),
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 1,
        "save_strategy": "no",
        "eval_strategy": "no",
        "report_to": "none",
    }
    if config.get("optim"):
        config_values["optim"] = str(config["optim"])
    if config.get("precompute_ref_log_probs"):
        config_values["precompute_ref_log_probs"] = True
        config_values["precompute_ref_batch_size"] = int(config.get("precompute_ref_batch_size", 1))
    args = DPOConfig(**_dataclass_kwargs(DPOConfig, config_values))
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
    }
    if lora_config is not None:
        trainer_kwargs["peft_config"] = lora_config
    signature = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" in signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)
    train_result = trainer.train()
    manifest["train_metrics"] = dict(getattr(train_result, "metrics", {}) or {})
    manifest["optimizer_steps"] = int(getattr(trainer.state, "global_step", 0) or 0)
    manifest["train_pairs"] = len(train_ds)
    manifest["val_pairs"] = len(val_ds)
    manifest["loss"] = [
        row["loss"] for row in trainer.state.log_history if row.get("loss") is not None
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    validation = validate_saved_adapter(output_dir)
    manifest.update(
        {
            "adapter_files": validation["files"],
            "adapter_total_bytes": validation["total_bytes"],
            "adapter_validation": validation,
            "duration_s": round(time.perf_counter() - started, 1),
            "device": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        }
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA DPO（Kaggle T4）")
    parser.add_argument("--config", default="training/configs/dpo_7b_t4.json")
    parser.add_argument("--model-path", default="/kaggle/input/qwen2.5-7b-instruct")
    parser.add_argument("--train-file", default="/kaggle/input/lite-agent-dpo/dpo_train.jsonl")
    parser.add_argument("--val-file", default="/kaggle/input/lite-agent-dpo/dpo_val.jsonl")
    parser.add_argument("--output-dir", default="/kaggle/working/dpo_adapter")
    parser.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help="SFT LoRA adapter used as the DPO policy initialisation",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dpo(
        config_path=Path(args.config),
        model_path=Path(args.model_path),
        train_file=Path(args.train_file),
        val_file=Path(args.val_file),
        output_dir=Path(args.output_dir),
        dry_run=bool(args.dry_run),
        init_adapter=Path(args.init_adapter) if args.init_adapter else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
