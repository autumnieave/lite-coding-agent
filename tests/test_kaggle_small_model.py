"""Small-model Kaggle wrapper tests; no GPU or model loading."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.kaggle import kaggle_small_model_tuning
from training.kaggle.contracts import ContractError
from training.kaggle.kaggle_small_model_tuning import (
    DEFAULT_ATTENTION_BACKEND,
    DEFAULT_GENERATION_CONFIG,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MODEL,
    DEFAULT_TORCH_BACKEND,
    DEFAULT_VLLM_VERSION,
    _install_runtime,
    _probe_tools,
    _run_tool_protocol_probe,
    _server_command,
    _server_env,
    _staging_check,
    _trace_has_tool_call,
    _wait_for_server,
    apply_staging_defaults,
    build_parser,
    preflight_small_model,
)


def test_trace_has_tool_call_detects_recorded_call(tmp_path: Path) -> None:
    trace = tmp_path / "dev" / "U-TUNE-C1-BASE_rollout00.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps({"event": "llm_turn", "tool_calls": [{"tool_name": "read_file"}]}) + "\n",
        encoding="utf-8",
    )
    assert _trace_has_tool_call(tmp_path, "U-TUNE-C1-BASE") is True


def test_trace_has_tool_call_rejects_text_only_trace(tmp_path: Path) -> None:
    trace = tmp_path / "U-TUNE-C1-BASE_rollout00.jsonl"
    trace.write_text(json.dumps({"event": "llm_turn"}) + "\n", encoding="utf-8")
    assert _trace_has_tool_call(tmp_path, "U-TUNE-C1-BASE") is False


def test_parser_defaults_to_small_coder_model() -> None:
    args = build_parser().parse_args(["--repo", "."])
    assert args.model == DEFAULT_MODEL
    assert DEFAULT_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"
    assert args.rollouts == 10
    assert args.tasks == ""
    assert args.max_model_len == DEFAULT_MAX_MODEL_LEN
    assert DEFAULT_MAX_MODEL_LEN == 8192


def test_parser_supports_final_test_spec_set() -> None:
    args = build_parser().parse_args(["--repo", ".", "--spec-set", "final-test"])
    assert args.spec_set == "final-test"
    assert build_parser().parse_args(["--repo", "."]).spec_set == "tuning"


def test_runtime_install_uses_python313_compatible_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], *, check: bool) -> None:
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr(subprocess, "run", fake_run)
    _install_runtime()
    command = captured["command"]
    assert isinstance(command, list)
    assert DEFAULT_VLLM_VERSION == "0.12.0"
    assert f"vllm=={DEFAULT_VLLM_VERSION}" in command
    assert "openai>=1.99.1" in command
    assert f"https://download.pytorch.org/whl/{DEFAULT_TORCH_BACKEND}" in command
    assert captured["check"] is True


def test_server_command_uses_vllm_012_argument_names() -> None:
    command = _server_command(DEFAULT_MODEL)
    assert "--disable-uvicorn-access-log" in command
    assert "--disable-log-requests" not in command
    assert "--enforce-eager" in command
    assert command[command.index("--tool-call-parser") + 1] == "hermes"
    assert command[command.index("--generation-config") + 1] == DEFAULT_GENERATION_CONFIG
    assert command[command.index("--max-model-len") + 1] == "8192"
    assert DEFAULT_GENERATION_CONFIG == "vllm"


def test_tool_protocol_probe_covers_auto_and_required(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_probe(base_url: str, model: str, tool_choice: str) -> dict[str, object]:
        seen.append(tool_choice)
        return {"tool_choice": tool_choice, "tool_calls": []}

    monkeypatch.setattr(kaggle_small_model_tuning, "_probe_tool_choice", fake_probe)
    result = _run_tool_protocol_probe("http://127.0.0.1:8000/v1", DEFAULT_MODEL)
    assert seen == ["auto", "required"]
    assert set(result) == {"auto", "required"}
    assert _probe_tools()[0]["function"]["name"] == "read_file"


def test_server_env_uses_t4_safe_attention_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASHINFER")
    env = _server_env(DEFAULT_MODEL)
    assert DEFAULT_ATTENTION_BACKEND == "TRITON_ATTN"
    assert env["VLLM_ATTENTION_BACKEND"] == DEFAULT_ATTENTION_BACKEND
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["LLM_MODEL"] == DEFAULT_MODEL
    assert env["NO_PROXY"] == "127.0.0.1,localhost"
    assert env["no_proxy"] == "127.0.0.1,localhost"


def test_main_writes_failure_manifest_when_runtime_install_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "small_model"

    def fail_install() -> None:
        raise RuntimeError("pip install failed")

    monkeypatch.setattr(kaggle_small_model_tuning, "_install_runtime", fail_install)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kaggle_small_model_tuning.py",
            "--repo",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--install-runtime",
        ],
    )
    assert kaggle_small_model_tuning.main() == 1
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert "pip install failed" in manifest["error"]
    assert manifest["vllm_version"] == DEFAULT_VLLM_VERSION


def _small_model_paths(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts/difficulty_tuning.py").write_text("print('ok')\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return repo, model


def test_small_model_parser_has_contract_flags() -> None:
    args = build_parser().parse_args(["--repo", "."])
    assert args.limit == 0
    assert args.preflight_only is False
    assert args.staging is False
    assert args.load_timeout == 180
    assert args.rollout_timeout == 0
    assert args.progress_every == 5
    assert args.provider == "server"
    assert args.system_variant == "s0"
    assert args.readiness_diagnose is False


def test_small_model_staging_defaults_are_bounded() -> None:
    args = build_parser().parse_args(["--repo", ".", "--staging"])
    args = apply_staging_defaults(args, "U-FT01-BASE")
    assert args.limit == 3
    assert args.rollouts == 3
    assert args.tasks == "U-FT01-BASE"
    assert args.rollout_timeout == 120
    assert args.skip_smoke is True


def test_staging_check_requires_zero_errors_and_one_task_success() -> None:
    summary = {
        "errors": 0,
        "summary": [{"task_id": "U-FT01-BASE", "max_rate": 0.4}],
    }
    passed = _staging_check(summary, [{"task_success": True}], rollouts=3)
    assert passed["passed"] is True
    assert passed["task_successes"] == 1
    assert passed["max_rate"] == 0.4
    assert passed["stability_judgement"] == "insufficient"

    degraded = _staging_check(
        {**summary, "errors": 1},
        [{"task_success": True}],
        rollouts=3,
    )
    assert degraded["passed"] is False
    assert degraded["status"] == "degraded"

    no_success = _staging_check(summary, [{"task_success": False}], rollouts=3)
    assert no_success["passed"] is False


def test_small_model_preflight_accepts_local_model(tmp_path: Path) -> None:
    repo, model = _small_model_paths(tmp_path)
    args = build_parser().parse_args(
        [
            "--repo",
            str(repo),
            "--model",
            str(model),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    report = preflight_small_model(args)
    assert report["status"] == "ok"
    assert report["model_kind"] == "local"
    assert report["model_config_bytes"] == 2


def test_small_model_preflight_rejects_unverified_hub_model(tmp_path: Path) -> None:
    repo, _ = _small_model_paths(tmp_path)
    args = build_parser().parse_args(
        [
            "--repo",
            str(repo),
            "--model",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(ContractError) as caught:
        preflight_small_model(args)
    assert caught.value.error_kind == "model_path_unverified"


def test_wait_for_server_timeout_is_categorized() -> None:
    with pytest.raises(ContractError) as caught:
        _wait_for_server("http://127.0.0.1:9/v1", timeout_s=0.0)
    assert caught.value.error_kind == "model_load_timeout"


def test_main_preflight_only_writes_manifest(tmp_path: Path) -> None:
    repo, model = _small_model_paths(tmp_path)
    out_dir = tmp_path / "out"
    code = kaggle_small_model_tuning.main(
        [
            "--repo",
            str(repo),
            "--model",
            str(model),
            "--out-dir",
            str(out_dir),
            "--preflight-only",
        ]
    )
    assert code == 0
    assert (out_dir / "preflight.json").is_file()
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["preflight"]["status"] == "ok"


def test_main_records_error_kind_when_preflight_fails(tmp_path: Path) -> None:
    repo, _ = _small_model_paths(tmp_path)
    out_dir = tmp_path / "out"
    code = kaggle_small_model_tuning.main(
        [
            "--repo",
            str(repo),
            "--model",
            str(tmp_path / "missing-model"),
            "--out-dir",
            str(out_dir),
            "--preflight-only",
        ]
    )
    assert code == 1
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert manifest["error_kind"] == "model_path_missing"
    preflight = json.loads((out_dir / "preflight.json").read_text(encoding="utf-8"))
    assert preflight["error_kind"] == "model_path_missing"
