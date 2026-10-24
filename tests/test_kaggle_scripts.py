"""Kaggle 入口的本地 dry-run 测试：不加载模型、不联网。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import training.kaggle.kaggle_constrained_decoding as constrained_module
from training.kaggle.contracts import ContractError
from training.kaggle.kaggle_constrained_decoding import (
    apply_staging_defaults,
    build_parser,
    load_prompts,
    model_version_from_path,
    resolve_model_path,
)
from training.kaggle.kaggle_constrained_decoding import main as constrained_main
from training.kaggle.kaggle_eval_sft import load_rows, run_eval
from training.kaggle.kaggle_sft_qlora import main as sft_main


def test_constrained_decoding_defaults_to_fp16_on_t4() -> None:
    args = build_parser().parse_args([])
    assert args.quantize_4bit is False
    assert build_parser().parse_args(["--quantize-4bit"]).quantize_4bit is True


def test_constrained_decoding_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"run_id": 2, "rollout_index": 0, "prompt": "a"},
                {"run_id": 10, "rollout_index": 0, "prompt": "b"},
            )
        ),
        encoding="utf-8",
    )
    assert constrained_main(["--prompts-file", str(prompts), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert '"status": "dry_run"' in output
    assert '"prompts": 2' in output


def test_load_prompts_rejects_missing_prompt(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"run_id": 2, "rollout_index": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt"):
        load_prompts(prompts)


def test_resolve_model_path_finds_mounted_config(tmp_path: Path) -> None:
    model = tmp_path / "dataset" / "Qwen2.5-Coder-7B-Instruct"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    resolved = resolve_model_path(tmp_path / "missing", search_root=tmp_path)
    assert resolved == model


def test_resolve_model_path_finds_mounted_1_5b_config(tmp_path: Path) -> None:
    model = (
        tmp_path
        / "kaggle"
        / "input"
        / "qwen2-5-coder-1.5b-instruct"
        / "Qwen2.5-Coder-1.5B-Instruct"
    )
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    resolved = resolve_model_path(tmp_path / "missing", search_root=tmp_path)
    assert resolved == model
    assert (resolved / "config.json").is_file()


def test_resolve_model_path_finds_qwen2_5_instruct_1_5b_config(tmp_path: Path) -> None:
    model = tmp_path / "kaggle" / "input" / "qwen2-5-1-5b-instruct" / "qwen2.5-1.5b-instruct"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    resolved = resolve_model_path(
        tmp_path / "missing",
        search_root=tmp_path,
        preferred_model="Qwen2.5-1.5B-Instruct",
    )
    assert resolved == model
    assert model_version_from_path(resolved) == "qwen2.5-1.5b-instruct"


def test_resolve_model_path_can_select_small_model_when_both_are_mounted(tmp_path: Path) -> None:
    names = ("Qwen2.5-Coder-7B-Instruct", "Qwen2.5-Coder-1.5B-Instruct")
    models = []
    for name in names:
        model = tmp_path / "dataset" / name
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        models.append(model)
    resolved = resolve_model_path(
        tmp_path / "missing",
        search_root=tmp_path,
        preferred_model="Qwen2.5-Coder-1.5B-Instruct",
    )
    assert resolved == models[1]


def test_resolve_model_path_raises_when_no_config_is_mounted(tmp_path: Path) -> None:
    root = tmp_path / "kaggle" / "input"
    root.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="没有 config.json"):
        resolve_model_path(root / "missing", search_root=root)


def test_resolve_model_path_rejects_ambiguous_mount(tmp_path: Path) -> None:
    for name in ("a", "b"):
        model = tmp_path / name
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="多个候选"):
        resolve_model_path(tmp_path / "missing", search_root=tmp_path)


def test_model_version_from_path_keeps_the_actual_mounted_variant(tmp_path: Path) -> None:
    model = tmp_path / "Qwen2.5-Coder-1.5B-Instruct"
    assert model_version_from_path(model) == "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def test_kaggle_jsonl_loaders_accept_utf8_bom(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts_bom.jsonl"
    prompts.write_text(
        json.dumps({"run_id": 2, "rollout_index": 0, "prompt": "x"}),
        encoding="utf-8-sig",
    )
    assert load_prompts(prompts)[0]["run_id"] == 2

    val = tmp_path / "val_bom.jsonl"
    val.write_text(json.dumps({"id": 1, "messages": []}), encoding="utf-8-sig")
    assert load_rows(val)[0]["id"] == 1


def test_eval_dry_run_reads_limit(tmp_path: Path) -> None:
    val = tmp_path / "val.jsonl"
    val.write_text(
        "\n".join(json.dumps({"id": index, "messages": [], "metadata": {}}) for index in range(3)),
        encoding="utf-8",
    )
    assert len(load_rows(val, limit=2)) == 2
    plan = run_eval(
        model_path=tmp_path / "model",
        adapter_path=tmp_path / "adapter",
        val_file=val,
        output_file=tmp_path / "out.json",
        max_new_tokens=8,
        limit=2,
        dry_run=True,
    )
    assert plan["status"] == "dry_run"
    assert plan["rows"] == 2


def test_sft_runner_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = tmp_path / "model"
    model.mkdir()
    train = tmp_path / "train.jsonl"
    train.write_text("{}", encoding="utf-8")
    val = tmp_path / "val.jsonl"
    val.write_text("{}", encoding="utf-8")
    config = Path("training/configs/qlora_t4.json").resolve()
    code = sft_main(
        [
            "--repo",
            str(Path.cwd()),
            "--config",
            str(config),
            "--model-path",
            str(model),
            "--train-file",
            str(train),
            "--val-file",
            str(val),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert code == 0
    assert '"status": "dry_run"' in capsys.readouterr().out


def _constrained_inputs(tmp_path: Path, *, prompt_count: int = 2) -> tuple[Path, Path]:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join(
            json.dumps({"run_id": 2, "rollout_index": index, "prompt": f"p{index}"})
            for index in range(prompt_count)
        )
        + "\n",
        encoding="utf-8",
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return prompts, model


def test_constrained_staging_defaults_are_bounded() -> None:
    args = apply_staging_defaults(build_parser().parse_args(["--staging"]))
    assert args.limit == 3
    assert args.max_new_tokens == 256


def test_constrained_preflight_only_writes_report(tmp_path: Path) -> None:
    prompts, model = _constrained_inputs(tmp_path)
    report = tmp_path / "preflight.json"
    summary = tmp_path / "summary.json"
    code = constrained_main(
        [
            "--prompts-file",
            str(prompts),
            "--model-path",
            str(model),
            "--search-root",
            str(tmp_path),
            "--preflight-only",
            "--preflight-out",
            str(report),
            "--summary-out",
            str(summary),
        ]
    )
    assert code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["prompts"] == 2
    assert payload["model_version"] == "model"


def test_constrained_preflight_rejects_empty_prompts(tmp_path: Path) -> None:
    prompts, model = _constrained_inputs(tmp_path, prompt_count=0)
    report = tmp_path / "preflight.json"
    summary = tmp_path / "summary.json"
    code = constrained_main(
        [
            "--prompts-file",
            str(prompts),
            "--model-path",
            str(model),
            "--search-root",
            str(tmp_path),
            "--preflight-only",
            "--preflight-out",
            str(report),
            "--summary-out",
            str(summary),
        ]
    )
    assert code == 1
    assert json.loads(report.read_text(encoding="utf-8"))["error_kind"] == "input_empty"
    assert json.loads(summary.read_text(encoding="utf-8"))["error_kind"] == "input_empty"


def test_constrained_records_model_load_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts, model = _constrained_inputs(tmp_path)
    summary = tmp_path / "summary.json"

    def fail_load(*args: object, **kwargs: object) -> list[dict[str, object]]:
        raise ContractError("model_load_timeout", "load too slow")

    monkeypatch.setattr(constrained_module, "run_with_outlines", fail_load)
    code = constrained_main(
        [
            "--prompts-file",
            str(prompts),
            "--model-path",
            str(model),
            "--search-root",
            str(tmp_path),
            "--out",
            str(tmp_path / "out.jsonl"),
            "--summary-out",
            str(summary),
            "--preflight-out",
            str(tmp_path / "preflight.json"),
        ]
    )
    assert code == 1
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["error_kind"] == "model_load_timeout"


def test_constrained_resume_skips_completed_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts, model = _constrained_inputs(tmp_path)
    out = tmp_path / "out.jsonl"
    out.write_text(
        json.dumps({"run_id": 2, "rollout_index": 0, "output": "done"}) + "\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    seen: list[dict[str, object]] = []

    def fake_run(pending: list[dict[str, object]], **kwargs: object) -> list[dict[str, object]]:
        seen.extend(pending)
        record = {"run_id": 2, "rollout_index": 1, "validation": {}, "schema_accepts": False}
        callback = kwargs["on_record"]
        assert callable(callback)
        callback(record)
        return [record]

    monkeypatch.setattr(constrained_module, "run_with_outlines", fake_run)
    code = constrained_main(
        [
            "--prompts-file",
            str(prompts),
            "--model-path",
            str(model),
            "--search-root",
            str(tmp_path),
            "--out",
            str(out),
            "--summary-out",
            str(summary),
            "--preflight-out",
            str(tmp_path / "preflight.json"),
        ]
    )
    assert code == 0
    assert [row["rollout_index"] for row in seen] == [1]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["prompts_completed"] == 2


def test_constrained_resume_retries_error_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts, model = _constrained_inputs(tmp_path)
    out = tmp_path / "out.jsonl"
    out.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "run_id": 2,
                        "rollout_index": 0,
                        "error_kind": "inference_timeout",
                    }
                ),
                json.dumps({"run_id": 2, "rollout_index": 1, "output": "done"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    seen: list[dict[str, object]] = []

    def fake_run(pending: list[dict[str, object]], **kwargs: object) -> list[dict[str, object]]:
        seen.extend(pending)
        record = {"run_id": 2, "rollout_index": 0, "validation": {}, "schema_accepts": False}
        callback = kwargs["on_record"]
        assert callable(callback)
        callback(record)
        return [record]

    monkeypatch.setattr(constrained_module, "run_with_outlines", fake_run)
    code = constrained_main(
        [
            "--prompts-file",
            str(prompts),
            "--model-path",
            str(model),
            "--search-root",
            str(tmp_path),
            "--out",
            str(out),
            "--summary-out",
            str(summary),
            "--preflight-out",
            str(tmp_path / "preflight.json"),
        ]
    )
    assert code == 0
    assert [row["rollout_index"] for row in seen] == [0]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["error_kind"] is None
    assert payload["prompts_completed"] == 2
