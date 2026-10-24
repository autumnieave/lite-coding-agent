"""QLoRA DPO entry point for the method-comparison arm."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.adapter_artifacts import validate_saved_adapter
from training.config import load_config, validate_t4_config
from training.sft_format import canonicalize_tool_arguments

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _adapter_weight_path(adapter_dir: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter_dir / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(f"init adapter 缺少非空权重文件：{adapter_dir}")


def _verify_expected(path: Path, actual_sha256: str, expected: str | None, label: str) -> None:
    if expected is None:
        return
    expected = expected.strip().lower()
    if actual_sha256 != expected:
        raise ValueError(
            f"{label} SHA-256 不匹配：expected={expected}, actual={actual_sha256}, path={path}"
        )


def _token_stats(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"min": 0, "p50": 0, "max": 0}
    return {"min": ordered[0], "p50": ordered[len(ordered) // 2], "max": ordered[-1]}


def _write_manifest(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def default_tool_schemas() -> list[dict[str, Any]]:
    _ensure_src_on_path()
    from agent.tools import build_default_registry

    return build_default_registry().specs()


def render_dpo_prompt(tokenizer: Any, messages: Sequence[Mapping[str, Any]]) -> str:
    messages = canonicalize_tool_arguments(messages)
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


def _resolved_max_steps(config: Mapping[str, Any], explicit: int | None) -> int | None:
    value = explicit if explicit is not None else config.get("max_steps")
    if value is None:
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"max_steps 必须为正整数，实际为 {resolved}")
    return resolved


def _run_dpo_training(
    *,
    config: Mapping[str, Any],
    model_path: Path,
    train_file: Path,
    val_file: Path,
    output_dir: Path,
    init_adapter: Path | None,
    max_steps: int | None,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
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
    max_prompt_length = int(config.get("max_prompt_length", 2048))
    max_length = int(config["max_seq_length"])

    prompt_lengths = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in train_ds["prompt"]
    ]
    chosen_totals = [
        prompt_len + len(tokenizer.encode(completion, add_special_tokens=False))
        for prompt_len, completion in zip(prompt_lengths, train_ds["chosen"], strict=True)
    ]
    rejected_totals = [
        prompt_len + len(tokenizer.encode(completion, add_special_tokens=False))
        for prompt_len, completion in zip(prompt_lengths, train_ds["rejected"], strict=True)
    ]
    observed_prompt_max = max(prompt_lengths, default=0)
    observed_total_max = max(chosen_totals + rejected_totals, default=0)
    effective_max_length = min(max_length, observed_total_max)
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    manifest["dataset_token_estimate"] = {
        "prompt": _token_stats(prompt_lengths),
        "chosen_total": _token_stats(chosen_totals),
        "rejected_total": _token_stats(rejected_totals),
        "prompts_over_max_prompt_length": sum(
            length > max_prompt_length for length in prompt_lengths
        ),
        "pairs_over_max_length": sum(
            length > max_length for length in chosen_totals + rejected_totals
        ),
    }
    manifest["precompute_ref_input_shape"].update(
        {
            "observed_prompt_tokens_max": observed_prompt_max,
            "observed_total_tokens_max": observed_total_max,
            "effective_max_sequence_tokens": effective_max_length,
            "vocab_size": vocab_size,
            "max_logits_elements_per_batch": 2 * effective_max_length * vocab_size,
        }
    )
    manifest["train_pairs"] = len(train_ds)
    manifest["val_pairs"] = len(val_ds)

    config_values = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": int(config.get("num_train_epochs", 1)),
        "learning_rate": float(config.get("learning_rate", 5e-5)),
        "beta": float(config.get("beta", 0.1)),
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
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
    if max_steps is not None:
        config_values["max_steps"] = max_steps
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
    manifest["status"] = "training"
    manifest["torch"] = torch.__version__
    manifest["cuda_device"] = torch.cuda.get_device_name(0)
    manifest["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    _write_manifest(output_dir, manifest)
    train_result = trainer.train()
    manifest["train_metrics"] = dict(getattr(train_result, "metrics", {}) or {})
    manifest["optimizer_steps"] = int(getattr(trainer.state, "global_step", 0) or 0)
    manifest["loss"] = [
        row["loss"] for row in trainer.state.log_history if row.get("loss") is not None
    ]
    trainer.model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    validation = validate_saved_adapter(output_dir)
    manifest.update(
        {
            "status": "completed",
            "adapter_files": validation["files"],
            "adapter_total_bytes": validation["total_bytes"],
            "adapter_validation": validation,
            "duration_s": round(time.perf_counter() - started, 1),
        }
    )
    _write_manifest(output_dir, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def run_dpo(
    *,
    config_path: Path,
    model_path: Path,
    train_file: Path,
    val_file: Path,
    output_dir: Path,
    dry_run: bool,
    init_adapter: Path | None = None,
    max_steps: int | None = None,
    expected_train_sha256: str | None = None,
    expected_val_sha256: str | None = None,
    expected_init_adapter_sha256: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_t4_config(config)
    for path, label in ((model_path, "model"), (train_file, "train"), (val_file, "val")):
        if not path.exists():
            raise FileNotFoundError(f"{label} 路径不存在：{path}")
    adapter_weight: Path | None = None
    if init_adapter is not None:
        if not (init_adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"init adapter 缺少 adapter_config.json：{init_adapter}")
        adapter_weight = _adapter_weight_path(init_adapter)
    elif expected_init_adapter_sha256 is not None:
        raise ValueError("提供了 expected_init_adapter_sha256，但没有 --init-adapter")

    train_sha256 = _sha256_file(train_file)
    val_sha256 = _sha256_file(val_file)
    adapter_sha256 = _sha256_file(adapter_weight) if adapter_weight is not None else None
    _verify_expected(train_file, train_sha256, expected_train_sha256, "train")
    _verify_expected(val_file, val_sha256, expected_val_sha256, "val")
    if adapter_weight is not None and adapter_sha256 is not None:
        _verify_expected(
            adapter_weight, adapter_sha256, expected_init_adapter_sha256, "init adapter"
        )

    resolved_max_steps = _resolved_max_steps(config, max_steps)
    data_manifest_path = train_file.parent / "manifest.json"
    data_manifest = _read_json_object(data_manifest_path)
    data_summary = {
        key: data_manifest[key]
        for key in (
            "version",
            "teacher_model",
            "source_train_version",
            "source_train_sha256",
            "pair_count",
            "train_pairs",
            "val_pairs",
            "train_sha256",
            "val_sha256",
            "truncation",
        )
        if key in data_manifest
    }
    manifest: dict[str, Any] = {
        "status": "dry_run" if dry_run else "initializing",
        "started_at": _now(),
        "model_path": str(model_path),
        "train_file": str(train_file),
        "val_file": str(val_file),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "config": config,
        "beta": float(config.get("beta", 0.1)),
        "reference_model": "base model with adapter disabled",
        "policy_init": "sft_adapter" if init_adapter is not None else "base_model",
        "init_adapter": str(init_adapter) if init_adapter is not None else None,
        "init_adapter_weight": str(adapter_weight) if adapter_weight is not None else None,
        "init_adapter_sha256": adapter_sha256,
        "init_adapter_bytes": adapter_weight.stat().st_size if adapter_weight is not None else None,
        "train_sha256": train_sha256,
        "val_sha256": val_sha256,
        "expected_sha256": {
            "train": expected_train_sha256,
            "val": expected_val_sha256,
            "init_adapter": expected_init_adapter_sha256,
        },
        "sha256_verified": {
            "train": expected_train_sha256 is not None,
            "val": expected_val_sha256 is not None,
            "init_adapter": expected_init_adapter_sha256 is not None,
        },
        "data_source": str(train_file.parent),
        "data_manifest_path": str(data_manifest_path),
        "data_manifest": data_summary,
        "max_steps": resolved_max_steps,
        "precompute_ref_log_probs": bool(config.get("precompute_ref_log_probs")),
        "precompute_ref_batch_size": int(config.get("precompute_ref_batch_size", 1)),
        "precompute_ref_input_shape": {
            "configured_max_prompt_length": int(config.get("max_prompt_length", 2048)),
            "configured_max_length": int(config["max_seq_length"]),
            "rows_per_pair_batch": 2,
            "note": (
                "TRL concatenates chosen and rejected; with precompute_ref_batch_size=1 "
                "the reference forward still sees 2 rows and the effective sequence cap."
            ),
        },
    }
    if dry_run:
        manifest["duration_s"] = 0.0
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(output_dir, manifest)
    try:
        return _run_dpo_training(
            config=config,
            model_path=model_path,
            train_file=train_file,
            val_file=val_file,
            output_dir=output_dir,
            init_adapter=init_adapter,
            max_steps=resolved_max_steps,
            manifest=manifest,
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "error",
                "error_kind": type(exc).__name__,
                "error": str(exc)[:2000],
                "traceback": traceback.format_exc()[-5000:],
                "failed_at": _now(),
            }
        )
        _write_manifest(output_dir, manifest)
        raise


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
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--expected-train-sha256", default=None)
    parser.add_argument("--expected-val-sha256", default=None)
    parser.add_argument("--expected-init-adapter-sha256", default=None)
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
        max_steps=args.max_steps,
        expected_train_sha256=args.expected_train_sha256,
        expected_val_sha256=args.expected_val_sha256,
        expected_init_adapter_sha256=args.expected_init_adapter_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
