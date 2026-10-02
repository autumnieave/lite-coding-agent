"""TRL GRPOTrainer 接入骨架；奖励来自 training/reward.py 的行为信号。"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.config import load_config, validate_t4_config
from training.reward import grpo_reward


def _dataclass_kwargs(cls: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    fields = getattr(cls, "__dataclass_fields__", None)
    if not fields:
        return dict(values)
    return {key: value for key, value in values.items() if key in fields}


def run_grpo(
    *,
    config_path: Path,
    model_path: Path,
    train_file: Path,
    output_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_t4_config(config)
    for path, label in ((model_path, "model"), (train_file, "train")):
        if not path.exists():
            raise FileNotFoundError(f"{label} 路径不存在：{path}")
    manifest: dict[str, Any] = {
        "status": "dry_run" if dry_run else "completed",
        "started_at": datetime.now(UTC).isoformat(),
        "model_path": str(model_path),
        "train_file": str(train_file),
        "output_dir": str(output_dir),
        "config": config,
        "request_id": None,
        "system_fingerprint": None,
    }
    if dry_run:
        manifest["duration_s"] = 0.0
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle 环境缺少 GRPO 依赖；先安装 training/requirements-kaggle.txt"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("没有 CUDA。GRPO 验证必须在 Kaggle T4 上执行。")
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
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    dataset = load_dataset("json", data_files=str(train_file), split="train")

    def format_row(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "prompt": row["messages"],
            "reward_context": row["reward_context"],
        }

    dataset = dataset.map(format_row)
    config_values = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_generations": 4,
        "max_prompt_length": 2048,
        "max_completion_length": 1024,
        "learning_rate": 1e-5,
        "num_train_epochs": 1,
        "logging_steps": 10,
        "save_strategy": "no",
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": True,
        "temperature": 0.8,
        "report_to": "none",
    }
    args = GRPOConfig(**_dataclass_kwargs(GRPOConfig, config_values))
    trainer_kwargs = {
        "model": model,
        "reward_funcs": [grpo_reward],
        "args": args,
        "train_dataset": dataset,
        "peft_config": lora_config,
    }
    if "processing_class" in inspect.signature(GRPOTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    manifest.update(
        {
            "duration_s": round(time.perf_counter() - started, 1),
            "device": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "train_rows": len(dataset),
            "adapter_dir": str(output_dir),
        }
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA GRPO 骨架（Kaggle T4）")
    parser.add_argument("--config", default="training/configs/qlora_t4.json")
    parser.add_argument("--model-path", default="/kaggle/input/qwen2.5-coder-7b-instruct")
    parser.add_argument("--train-file", default="/kaggle/input/lite-agent-training/train.jsonl")
    parser.add_argument("--output-dir", default="/kaggle/working/grpo_adapter")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_grpo(
        config_path=Path(args.config),
        model_path=Path(args.model_path),
        train_file=Path(args.train_file),
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
