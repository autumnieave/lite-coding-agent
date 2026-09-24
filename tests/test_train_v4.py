"""Contract tests for train-v4, public mapping, mixing and DPO pair construction."""

from __future__ import annotations

import json
from pathlib import Path

from training.build_dpo_dataset import build_dpo_pairs
from training.build_train_v4 import build_clone_specs, build_oracle_example
from training.data_mix import build_dolly_rows, mix_train_file
from training.data_mix import main as data_mix_main
from training.public_data import build_public_grep_example


def test_public_grep_row_is_executed_and_gate_checked(tmp_path: Path) -> None:
    row = {
        "query": "search two topics",
        "answers": [
            {"name": "search_content", "arguments": {"query": "alpha topic"}},
            {"name": "search_content", "arguments": {"query": "beta topic"}},
        ],
    }
    example, detail = build_public_grep_example(
        row,
        row_index=7,
        split="train",
        trace_path=tmp_path / "trace.jsonl",
    )
    assert detail["reason"] == "ok"
    assert example is not None
    assert example["metadata"]["category"] == "TOOL"
    assert example["metadata"]["tool_result_execution"] == "real_local_registry"
    assert json.loads(example["completion"])["matches"][0]["pattern"] == "alpha topic"
    assert any(message.get("role") == "tool" for message in example["messages"])


def test_train_v4_clones_cover_all_families_and_splits() -> None:
    specs = build_clone_specs(20)
    assert len(specs) == 220
    assert {spec.split for spec in specs} == {"train", "val"}
    assert {spec.tuning_dimension for spec in specs} == {
        "constraint_count",
        "context_length",
        "tool_parameters",
        "conflict",
        "check_depth",
    }
    assert len({spec.task_id for spec in specs}) == len(specs)
    assert len({spec.fixture_seed for spec in specs}) == len(specs)


def test_oracle_examples_pass_each_family(tmp_path: Path) -> None:
    specs = build_clone_specs(20)
    seen: set[str] = set()
    for spec in specs:
        if spec.family in seen:
            continue
        seen.add(spec.family)
        example, detail = build_oracle_example(spec, trace_path=tmp_path / f"{spec.task_id}.jsonl")
        assert detail["reason"] == "ok", detail
        assert example is not None
        assert example["metadata"]["tool_result_execution"] == "real_local_registry"
    assert len(seen) == 5


def test_dolly_loading_and_mix_manifest(tmp_path: Path) -> None:
    dolly = tmp_path / "dolly.jsonl"
    dolly.write_text(
        "\n".join(
            json.dumps({"instruction": f"q{i}", "response": f"a{i}", "context": ""})
            for i in range(10)
        )
        + "\n",
        encoding="utf-8",
    )
    general_rows = build_dolly_rows(dolly, limit=10)
    assert len(general_rows) == 10
    assert all(row["metadata"]["category"] == "GENERAL" for row in general_rows)
    general_file = tmp_path / "general.jsonl"
    general_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in general_rows) + "\n",
        encoding="utf-8",
    )
    train_file = tmp_path / "train.jsonl"
    base_rows = [{"id": f"base-{index}", "messages": [], "completion": "x"} for index in range(20)]
    train_file.write_text("\n".join(json.dumps(row) for row in base_rows) + "\n", encoding="utf-8")
    manifest = mix_train_file(
        base_train_file=train_file,
        general_file=general_file,
        out_file=tmp_path / "mixed.jsonl",
        general_ratio=0.15,
        seed=7,
    )
    assert manifest["general_sampled"] == 4
    assert manifest["total_rows"] == 24


def test_data_mix_cli_reuses_general_file_with_kernel_flags(tmp_path: Path) -> None:
    general_file = tmp_path / "general.jsonl"
    general_file.write_text(
        "\n".join(json.dumps({"id": f"general-{index}"}) for index in range(6)) + "\n",
        encoding="utf-8",
    )
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(
        "\n".join(json.dumps({"id": f"base-{index}"}) for index in range(17)) + "\n",
        encoding="utf-8",
    )
    out_file = tmp_path / "mixed.jsonl"

    exit_code = data_mix_main(
        [
            "--base-train-file",
            str(train_file),
            "--general-file",
            str(general_file),
            "--out-file",
            str(out_file),
            "--general-ratio",
            "0.15",
            "--seed",
            "7",
        ]
    )

    assert exit_code == 0
    assert len(out_file.read_text(encoding="utf-8").splitlines()) == 20
    manifest = json.loads((tmp_path / "mix_manifest.json").read_text(encoding="utf-8"))
    assert manifest["general_rows"] == 6
    assert manifest["general_sampled"] == 3


def test_dpo_pairs_use_val_prompts_and_do_not_overlap_train(tmp_path: Path) -> None:
    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    train_rows = [
        {
            "id": "train-1",
            "messages": [{"role": "user", "content": "train prompt"}],
            "completion": '{"ok": true}',
            "metadata": {"category": "BASE"},
        }
    ]
    val_rows = [
        {
            "id": "val-1",
            "messages": [{"role": "user", "content": "val prompt"}],
            "completion": '{"ok": true}',
            "metadata": {"category": "BASE"},
        },
        {
            "id": "val-2",
            "messages": [{"role": "user", "content": "ctx prompt"}],
            "completion": "TOKEN-1",
            "metadata": {"category": "CTX"},
        },
    ]
    (sft_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(row) for row in train_rows) + "\n", encoding="utf-8"
    )
    (sft_dir / "val.jsonl").write_text(
        "\n".join(json.dumps(row) for row in val_rows) + "\n", encoding="utf-8"
    )
    rejected_dir = tmp_path / "rejected"
    rejected_dir.mkdir()
    (rejected_dir / "manifest.json").write_text('{"rejected": []}', encoding="utf-8")

    manifest = build_dpo_pairs(
        sft_dir=sft_dir,
        rejected_sft_dir=rejected_dir,
        out_dir=tmp_path / "dpo",
        max_synthetic=3,
    )
    assert manifest["pair_count"] == 3
    assert manifest["sft_train_prompt_overlap"] == 0
    pairs = [
        json.loads(line)
        for line in (tmp_path / "dpo" / "dpo_pairs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(pair["prompt_messages"][0]["content"] != "train prompt" for pair in pairs)


def test_dpo_pairs_exclude_behavior_heldout_tasks(tmp_path: Path) -> None:
    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    (sft_dir / "train.jsonl").write_text(
        json.dumps({"id": "train-1", "messages": [], "completion": "x"}) + "\n",
        encoding="utf-8",
    )
    val_rows = [
        {
            "id": "val-keep",
            "messages": [{"role": "user", "content": "keep prompt"}],
            "completion": '{"ok": true}',
            "metadata": {"category": "BASE", "task_id": "U-TR4-12-C1-BASE"},
        },
        {
            "id": "val-heldout",
            "messages": [{"role": "user", "content": "heldout prompt"}],
            "completion": '{"ok": true}',
            "metadata": {"category": "HARD", "task_id": "U-TR-A-C1-HARD"},
        },
    ]
    (sft_dir / "val.jsonl").write_text(
        "\n".join(json.dumps(row) for row in val_rows) + "\n", encoding="utf-8"
    )
    rejected_dir = tmp_path / "rejected"
    rejected_dir.mkdir()
    (rejected_dir / "manifest.json").write_text('{"rejected": []}', encoding="utf-8")

    manifest = build_dpo_pairs(
        sft_dir=sft_dir,
        rejected_sft_dir=rejected_dir,
        out_dir=tmp_path / "dpo",
        max_synthetic=10,
    )

    assert manifest["heldout_exclusion"]["excluded_val_rows"] == 1
    assert manifest["pair_count"] == 2
    pairs = [
        json.loads(line)
        for line in (tmp_path / "dpo" / "dpo_pairs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(pair["metadata"]["source_task_id"] != "U-TR-A-C1-HARD" for pair in pairs)
