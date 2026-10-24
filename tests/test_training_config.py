"""T4 配置和 dry-run 参数的本地测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from training.config import load_config, validate_t4_config
from training.train_dpo import run_dpo
from training.train_sft import (
    default_tool_schemas,
    lora_target_modules,
    over_length_rows,
    render_training_text,
)


def test_checked_in_t4_config_is_valid() -> None:
    validate_t4_config(load_config(Path("training/configs/qlora_t4.json")))
    validate_t4_config(load_config(Path("training/configs/qlora_1_5b_t4.json")))
    validate_t4_config(load_config(Path("training/configs/qlora_7b_t4.json")))


def test_1_5b_config_uses_all_linear_and_supervised_completion() -> None:
    config = load_config(Path("training/configs/qlora_1_5b_t4.json"))
    assert config["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert int(config["lora_r"]) == 64
    assert lora_target_modules(config) == "all-linear"
    assert not config.get("use_liger")


def test_only_7b_config_enables_liger_memory_kernel() -> None:
    config = load_config(Path("training/configs/qlora_7b_t4.json"))
    assert config.get("use_liger") is True


def test_7b_config_uses_memory_safe_sequence_and_optimizer() -> None:
    config = load_config(Path("training/configs/qlora_7b_t4.json"))
    assert int(config["max_seq_length"]) == 3072
    assert config["optim"] == "paged_adamw_8bit"
    assert int(config["lora_r"]) == 64
    # The 1.5B path keeps the previous default optimizer behavior.
    assert "optim" not in load_config(Path("training/configs/qlora_1_5b_t4.json"))


def test_dpo_config_precomputes_reference_log_probs_for_t4_memory() -> None:
    config = load_config(Path("training/configs/dpo_7b_t4.json"))
    assert config.get("precompute_ref_log_probs") is True
    assert int(config["precompute_ref_batch_size"]) == 1
    assert int(config["max_seq_length"]) == 3072

    captured: dict[str, object] = {}

    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, object]],
            *,
            tools: list[dict[str, object]],
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            captured["messages"] = messages
            captured["tools"] = tools
            captured["tokenize"] = tokenize
            captured["add_generation_prompt"] = add_generation_prompt
            return "rendered"

    row = {
        "messages": [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "value"},
        ],
        "completion": '{"value": "value"}',
    }
    assert render_training_text(FakeTokenizer(), row) == "rendered"
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[-1] == {"role": "assistant", "content": '{"value": "value"}'}
    assert any(message.get("role") == "tool" for message in messages)
    assistant = next(message for message in messages if message.get("tool_calls"))
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"path": "a.txt"}
    assert [tool["function"]["name"] for tool in captured["tools"]] == [
        tool["function"]["name"] for tool in default_tool_schemas()
    ]


def test_t4_config_rejects_wrong_batch_and_long_sequence() -> None:
    config = load_config(Path("training/configs/qlora_t4.json"))
    config["per_device_train_batch_size"] = 2
    with pytest.raises(ValueError, match="per_device_train_batch_size"):
        validate_t4_config(config)

    config = load_config(Path("training/configs/qlora_t4.json"))
    config["max_seq_length"] = 8192
    with pytest.raises(ValueError, match="max_seq_length"):
        validate_t4_config(config)


def test_over_length_rows_reports_ids_and_token_counts() -> None:
    rows = [
        {"id": "a", "token_count": 100},
        {"id": "b", "token_count": 5000},
        {"id": "c", "token_count": 4096},
    ]
    assert over_length_rows(rows, 4096) == [{"id": "b", "tokens": 5000}]


def test_default_tool_schemas_bootstraps_repo_src(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kaggle runs add only the repo root, not ``repo/src``, to ``sys.path``."""

    src = Path(__file__).resolve().parents[1] / "src"
    for name in [name for name in sys.modules if name == "agent" or name.startswith("agent.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != str(src)])
    assert str(src) not in sys.path

    names = [tool["function"]["name"] for tool in default_tool_schemas()]

    assert "read_file" in names
    assert str(src) in sys.path


def test_dpo_dry_run_records_sft_adapter_initialisation(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    train_file.write_text("{}\n", encoding="utf-8")
    val_file.write_text("{}\n", encoding="utf-8")
    adapter_dir = tmp_path / "sft_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    manifest = run_dpo(
        config_path=Path("training/configs/dpo_7b_t4.json"),
        model_path=model_dir,
        train_file=train_file,
        val_file=val_file,
        output_dir=tmp_path / "dpo",
        dry_run=True,
        init_adapter=adapter_dir,
    )

    assert manifest["status"] == "dry_run"
    assert manifest["policy_init"] == "sft_adapter"
    assert manifest["reference_model"] == "base model with adapter disabled"
    assert manifest["init_adapter"] == str(adapter_dir)
