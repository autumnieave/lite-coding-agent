"""Run the dev-only difficulty ladder against a local Qwen model on Kaggle.

This is the small-base replication path required by the Route B gate.  It starts
a vLLM OpenAI-compatible server, verifies that the model can emit at least one
tool call on a smoke task, and only then runs the same 11-task ladder used for
the API model.  It never imports torch/transformers into the core package and
does not change ``core/loop.py``, compaction or constraint storage.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from training.kaggle.contracts import (
    DEFAULT_LOAD_TIMEOUT_S,
    DEFAULT_PROGRESS_EVERY,
    ContractError,
    classify_exception,
    read_jsonl,
    require_nonempty_file,
    runtime_info,
    validate_summary,
    write_json,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SMOKE_TASK = "U-TUNE-C1-BASE"
DEFAULT_TRAIN_SMOKE_TASK = "U-TR-A-C1-BASE"
DEFAULT_FINAL_SMOKE_TASK = "U-FT01-BASE"
DEFAULT_VLLM_VERSION = "0.12.0"
DEFAULT_TORCH_BACKEND = "cu129"
DEFAULT_ATTENTION_BACKEND = "TRITON_ATTN"
DEFAULT_GENERATION_CONFIG = "vllm"
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_ROLLOUT_TIMEOUT_S = 0.0


def _run_text(command: list[str], *, check: bool = True) -> str:
    completed = subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
    )
    return (completed.stdout or "") + (completed.stderr or "")


def gpu_info() -> dict[str, Any]:
    try:
        output = _run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader",
            ]
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "query": output}


def _wait_for_server(
    base_url: str,
    timeout_s: float = DEFAULT_LOAD_TIMEOUT_S,
    *,
    progress_every_s: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    started = time.monotonic()
    models_url = base_url.rstrip("/") + "/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error = ""
    last_progress = 0.0
    while time.monotonic() < deadline:
        try:
            with opener.open(models_url, timeout=10) as response:
                if response.status == 200:
                    print(
                        f"[vllm] server ready after {time.monotonic() - started:.1f}s",
                        flush=True,
                    )
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started
        if progress_every_s > 0 and elapsed - last_progress >= progress_every_s:
            print(f"[vllm] waiting {elapsed:.0f}s/{timeout_s:.0f}s: {last_error}", flush=True)
            last_progress = elapsed
        time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
    raise ContractError(
        "model_load_timeout",
        f"vLLM 服务在 {timeout_s}s 内未就绪：{last_error}",
    )


def _diagnostic_command(command: list[str], *, timeout_s: float = 15.0) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "duration_s": round(time.monotonic() - started, 2),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.monotonic() - started, 2),
        }


def _socket_probe(host: str, family: int) -> dict[str, Any]:
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(3)
            sock.connect((host, 8000))
        return {"host": host, "connected": True}
    except OSError as exc:
        return {"host": host, "connected": False, "error": f"{type(exc).__name__}: {exc}"}


def diagnose_readiness(
    model: str,
    out_dir: Path,
    *,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    load_timeout: float = DEFAULT_LOAD_TIMEOUT_S,
) -> dict[str, Any]:
    """One bounded server-readiness probe; never runs a ladder."""

    out_dir.mkdir(parents=True, exist_ok=True)
    server_log = out_dir / "vllm_server.log"
    env = _server_env(model)
    started = time.monotonic()
    ready = False
    wait_error = ""
    log_handle = server_log.open("w", encoding="utf-8")
    server = subprocess.Popen(
        _server_command(model, max_model_len),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    try:
        try:
            _wait_for_server(env["LLM_BASE_URL"], timeout_s=load_timeout)
            ready = True
        except ContractError as exc:
            wait_error = f"{exc.error_kind}: {exc}"
        listen = _diagnostic_command(["ss", "-ltnp"])
        if "stdout" in listen:
            listen["stdout"] = "\n".join(
                line for line in listen["stdout"].splitlines() if ":8000" in line
            )
        payload: dict[str, Any] = {
            "server_ready": ready,
            "wait_error": wait_error,
            "server_pid": server.pid,
            "server_returncode": server.poll(),
            "load_timeout_s": load_timeout,
            "host": env["LLM_BASE_URL"],
            "listen_8000": listen,
            "socket_ipv4": _socket_probe("127.0.0.1", socket.AF_INET),
            "socket_ipv6": _socket_probe("::1", socket.AF_INET6),
            "curl_ipv4": _diagnostic_command(
                ["curl", "-v", "--max-time", "10", "http://127.0.0.1:8000/v1/models"]
            ),
            "curl_ipv6": _diagnostic_command(
                ["curl", "-v", "--max-time", "10", "http://[::1]:8000/v1/models"]
            ),
            "curl_localhost": _diagnostic_command(
                ["curl", "-v", "--max-time", "10", "http://localhost:8000/v1/models"]
            ),
            "processes": _diagnostic_command(
                ["bash", "-lc", "ps -ef | grep -E 'vllm|api_server' | grep -v grep"]
            ),
            "duration_s": round(time.monotonic() - started, 1),
        }
        (out_dir / "readiness.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        log_handle.close()


def apply_staging_defaults(args: argparse.Namespace, smoke_task: str) -> argparse.Namespace:
    """Bound staging to three records before a full GPU run."""

    if args.staging:
        if args.limit <= 0:
            args.limit = 3
        if args.rollouts == 10:
            args.rollouts = 3
        if not args.tasks:
            args.tasks = smoke_task
        if args.rollout_timeout <= 0:
            args.rollout_timeout = 120.0
        # The staging run itself is the smoke path; do not add a fourth record.
        args.skip_smoke = True
    return args


def preflight_small_model(args: argparse.Namespace) -> dict[str, Any]:
    """Validate paths, runtime and GPU before starting vLLM."""

    repo = Path(args.repo)
    script = repo / "scripts/difficulty_tuning.py"
    script_bytes = require_nonempty_file(
        script,
        "difficulty_tuning.py",
        missing_kind="repo_missing",
    )
    model = Path(args.model)
    if model.is_dir():
        model_kind = "local"
        config_bytes = require_nonempty_file(
            model / "config.json",
            "model config.json",
            missing_kind="model_path_missing",
            empty_kind="model_config_empty",
        )
        tokenizer_bytes = require_nonempty_file(
            model / "tokenizer_config.json",
            "tokenizer_config.json",
            missing_kind="tokenizer_config_missing",
            empty_kind="tokenizer_config_empty",
        )
    elif "/" in args.model:
        if not args.allow_hub_model:
            raise ContractError(
                "model_path_unverified",
                f"模型是 Hub id 而不是本地挂载目录：{args.model}；"
                "staging 请传本地路径，或显式加 --allow-hub-model",
            )
        model_kind = "hub"
        config_bytes = 0
        tokenizer_bytes = 0
    else:
        raise ContractError("model_path_missing", f"模型路径不存在：{args.model}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "status": "ok",
        "stage": "preflight",
        "model": args.model,
        "model_kind": model_kind,
        "model_config_bytes": config_bytes,
        "tokenizer_config_bytes": tokenizer_bytes,
        "repo": str(repo),
        "script_bytes": script_bytes,
        "spec_set": args.spec_set,
        "tasks": args.tasks,
        "limit": args.limit,
        "staging": bool(args.staging),
        "runtime": runtime_info(),
        "gpu": gpu_info(),
        "load_timeout_s": args.load_timeout,
        "rollout_timeout_s": args.rollout_timeout,
    }


def _trace_has_tool_call(trace_dir: Path, task_id: str) -> bool:
    for path in trace_dir.rglob(f"{task_id}_rollout*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "llm_turn" and event.get("tool_calls"):
                return True
    return False


def _probe_tools() -> list[dict[str, Any]]:
    """Minimal schema used only to distinguish parser failure from model choice."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file from the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]


def _probe_tool_choice(base_url: str, model: str, tool_choice: str) -> dict[str, Any]:
    """Ask vLLM for one tool call with a fixed tool_choice mode."""
    from openai import OpenAI

    client = OpenAI(api_key="local-vllm", base_url=base_url, timeout=120)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a coding agent. Use the provided tool to inspect files; "
                        "never guess file contents."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Call read_file for project/tuning/config_c1.ini, then report "
                        "the first line. You must call the tool before answering."
                    ),
                },
            ],
            tools=_probe_tools(),
            tool_choice=tool_choice,
            temperature=0.0,
            seed=20261019,
            max_tokens=128,
        )
    except Exception as exc:  # noqa: BLE001 - probe failure is diagnostic evidence
        return {"tool_choice": tool_choice, "error": f"{type(exc).__name__}: {exc}"}
    message = response.choices[0].message
    return {
        "tool_choice": tool_choice,
        "response_id": response.id,
        "model": response.model,
        "content": message.content or "",
        "tool_calls": [
            {
                "name": call.function.name,
                "arguments": call.function.arguments,
            }
            for call in (message.tool_calls or ())
        ],
    }


def _run_tool_protocol_probe(base_url: str, model: str) -> dict[str, Any]:
    return {choice: _probe_tool_choice(base_url, model, choice) for choice in ("auto", "required")}


def _load_records(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "runs.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _staging_check(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    rollouts: int,
) -> dict[str, Any]:
    """Judge staging by behaviour, not only by file existence.

    Three rollouts are far below the pre-registered 10-rollout stability
    threshold, so staging reports ``max_rate`` but never claims a stable
    violation. A degraded staging run is evidence of an integration problem,
    not a model-capability result.
    """

    errors = int(summary.get("errors", 0) or 0)
    successes = sum(
        1 for row in rows if row.get("task_success") is True and not row.get("error_kind")
    )
    summary_rows = summary.get("summary", [])
    max_rate = max(
        (float(row.get("max_rate", 0.0) or 0.0) for row in summary_rows if isinstance(row, dict)),
        default=0.0,
    )
    passed = errors == 0 and successes >= 1
    return {
        "passed": passed,
        "status": "ok" if passed else "degraded",
        "errors": errors,
        "task_successes": successes,
        "rollouts": len(rows),
        "rollouts_requested_per_task": rollouts,
        "max_rate": round(max_rate, 4),
        "stability_judgement": "insufficient",
        "note": "staging never declares stable; >=6/10 is required for that",
    }


def _server_command(model: str, max_model_len: int = DEFAULT_MAX_MODEL_LEN) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--served-model-name",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dtype",
        "float16",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        "0.85",
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--generation-config",
        DEFAULT_GENERATION_CONFIG,
        "--disable-uvicorn-access-log",
        "--seed",
        "20261019",
    ]


def _server_env(model: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "LLM_API_KEY": "local-vllm",
            "LLM_BASE_URL": "http://127.0.0.1:8000/v1",
            "LLM_MODEL": model,
            "PYTHONUTF8": "1",
            # FlashInfer JIT needs libcuda stubs that Kaggle's T4 image lacks.
            "VLLM_ATTENTION_BACKEND": DEFAULT_ATTENTION_BACKEND,
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    return env


def _install_runtime() -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--extra-index-url",
        f"https://download.pytorch.org/whl/{DEFAULT_TORCH_BACKEND}",
        f"vllm=={DEFAULT_VLLM_VERSION}",
        "openai>=1.99.1",
    ]
    print("INSTALL", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_ladder(
    repo: Path,
    model: str,
    out_dir: Path,
    trace_dir: Path,
    *,
    rollouts: int,
    smoke_task: str,
    skip_smoke: bool,
    tasks: str,
    max_model_len: int,
    spec_set: str = "tuning",
    limit: int = 0,
    load_timeout: float = DEFAULT_LOAD_TIMEOUT_S,
    rollout_timeout: float = 0.0,
    progress_every: int = 1,
    resume: bool = False,
    staging: bool = False,
    provider: str = "server",
    system_variant: str = "s0",
    adapter_path: str = "",
    train_version: str = "train-v2",
) -> dict[str, Any]:
    experiment_name = {
        "final-test": "final_test",
        "train": "train_v2_eval",
    }.get(spec_set, "difficulty_tuning")
    if staging:
        experiment_name = f"{experiment_name}_staging"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    server_log = out_dir / "vllm_server.log"
    env = dict(os.environ)
    env.update(
        {
            "LLM_API_KEY": "local-vllm",
            "LLM_BASE_URL": "http://127.0.0.1:8000/v1",
            "LLM_MODEL": model,
            "PYTHONUTF8": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    started = time.monotonic()
    server: subprocess.Popen[str] | None = None
    log_handle: Any = None
    protocol_probe: dict[str, Any] | None = None
    try:
        if provider == "server":
            env = _server_env(model)
            os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
            os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
            log_handle = server_log.open("w", encoding="utf-8")
            server = subprocess.Popen(
                _server_command(model, max_model_len),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            _wait_for_server(env["LLM_BASE_URL"], timeout_s=load_timeout)
            protocol_probe = _run_tool_protocol_probe(env["LLM_BASE_URL"], model)
            (out_dir / "tool_protocol_probe.json").write_text(
                json.dumps(protocol_probe, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        provider_args = ["--provider", provider]
        if provider == "offline":
            provider_args += [
                "--offline-model",
                model,
                "--max-model-len",
                str(max_model_len),
            ]
        provider_args += ["--system-variant", system_variant]
        if spec_set == "train":
            provider_args += ["--train-version", train_version]
        if adapter_path:
            provider_args += ["--adapter-path", adapter_path]
        smoke_record: dict[str, Any] = {"skipped": skip_smoke}
        if not skip_smoke:
            smoke_out = out_dir / "smoke"
            smoke_trace = trace_dir / "smoke"
            smoke_cmd = [
                sys.executable,
                str(repo / "scripts/difficulty_tuning.py"),
                "--spec-set",
                spec_set,
                "--tasks",
                smoke_task,
                "--rollouts",
                "1",
                "--limit",
                "1",
                "--concurrency",
                "1",
                "--rollout-timeout",
                str(rollout_timeout),
                "--progress-every",
                "1",
                "--sampling-temperature",
                "0",
                "--sampling-seed",
                "20261019",
                "--out-dir",
                str(smoke_out),
                "--trace-dir",
                str(smoke_trace),
                *provider_args,
            ]
            subprocess.run(smoke_cmd, check=True, env=env)
            records = _load_records(smoke_out)
            smoke_record = {
                "skipped": False,
                "records": records,
                "tool_call_observed": _trace_has_tool_call(smoke_trace, smoke_task),
                "protocol_probe": protocol_probe,
            }
            if not smoke_record["tool_call_observed"]:
                raise RuntimeError(
                    "smoke 未观察到 tool_call；vLLM 工具调用协议未生效，不能继续全量实验"
                )

        full_cmd = [
            sys.executable,
            str(repo / "scripts/difficulty_tuning.py"),
            "--spec-set",
            spec_set,
            "--rollouts",
            str(rollouts),
            "--concurrency",
            "1" if provider == "offline" else "4",
            "--sampling-temperature",
            "0",
            "--sampling-seed",
            "20261019",
            "--out-dir",
            str(out_dir / experiment_name),
            "--trace-dir",
            str(trace_dir / experiment_name),
            *provider_args,
        ]
        if tasks:
            full_cmd += ["--tasks", tasks]
        if limit:
            full_cmd += ["--limit", str(limit)]
        if rollout_timeout:
            full_cmd += ["--rollout-timeout", str(rollout_timeout)]
        if progress_every:
            full_cmd += ["--progress-every", str(progress_every)]
        if resume:
            full_cmd += ["--resume"]
        subprocess.run(full_cmd, check=True, env=env)
        summary_path = out_dir / experiment_name / "summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
        )
        runs_path = out_dir / experiment_name / "runs.jsonl"
        rows = read_jsonl(runs_path)
        if not rows:
            raise ContractError("output_empty", f"运行结束但 runs.jsonl 为空：{runs_path}")
        if summary is None:
            raise ContractError("summary_missing", f"没有生成 summary.json：{summary_path}")
        validate_summary(summary, ("records", "errors", "checks", "stable_checks", "summary"))
        staging_check = _staging_check(summary, rows, rollouts=rollouts) if staging else None
        status = "degraded" if staging_check and not staging_check["passed"] else "ok"
        return {
            "model": model,
            "rollouts_per_task": rollouts,
            "max_model_len": max_model_len,
            "tasks": tasks,
            "spec_set": spec_set,
            "limit": limit,
            "load_timeout_s": load_timeout,
            "rollout_timeout_s": rollout_timeout,
            "progress_every": progress_every,
            "resume": resume,
            "staging": staging,
            "provider": provider,
            "system_variant": system_variant,
            "adapter_path": adapter_path,
            "smoke": smoke_record,
            "summary": summary,
            "status": status,
            "staging_check": staging_check,
            "records_written": len(rows),
            "duration_s": round(time.monotonic() - started, 1),
            "server_log": str(server_log) if server is not None else "",
        }
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
        if log_handle is not None:
            log_handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kaggle small-model difficulty ladder")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/small_model"))
    parser.add_argument("--trace-dir", type=Path, default=Path("/kaggle/working/traces"))
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条 rollout；0 表示全部")
    parser.add_argument("--smoke-task", default="", help="empty picks the spec-set default")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--install-runtime", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--readiness-diagnose", action="store_true")
    parser.add_argument("--preflight-out", default="")
    parser.add_argument("--staging", action="store_true")
    parser.add_argument("--allow-hub-model", action="store_true")
    parser.add_argument("--load-timeout", type=float, default=DEFAULT_LOAD_TIMEOUT_S)
    parser.add_argument("--rollout-timeout", type=float, default=DEFAULT_ROLLOUT_TIMEOUT_S)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--provider", choices=("server", "offline"), default="server")
    parser.add_argument("--system-variant", choices=("s0", "s1", "s1t", "s2"), default="s0")
    parser.add_argument("--adapter-path", default="", help="可选 LoRA adapter")
    parser.add_argument("--tasks", default="", help="comma-separated task ids; empty means all")
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--spec-set", choices=("tuning", "final-test", "train"), default="tuning")
    parser.add_argument("--train-version", choices=("train-v1", "train-v2"), default="train-v2")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    smoke_task = args.smoke_task or (
        DEFAULT_FINAL_SMOKE_TASK
        if args.spec_set == "final-test"
        else DEFAULT_TRAIN_SMOKE_TASK
        if args.spec_set == "train"
        else DEFAULT_SMOKE_TASK
    )
    args = apply_staging_defaults(args, smoke_task)
    manifest: dict[str, Any] = {
        "gpu": gpu_info(),
        "generation_config": DEFAULT_GENERATION_CONFIG,
        "python_version": sys.version.split()[0],
        "vllm_version": DEFAULT_VLLM_VERSION,
        "torch_backend": DEFAULT_TORCH_BACKEND,
        "max_model_len": args.max_model_len,
        "tasks": args.tasks,
        "spec_set": args.spec_set,
        "limit": args.limit,
        "staging": bool(args.staging),
        "load_timeout_s": args.load_timeout,
        "rollout_timeout_s": args.rollout_timeout,
        "progress_every": args.progress_every,
        "resume": bool(args.resume),
        "preflight_only": bool(args.preflight_only),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    stage = "install"
    preflight_path = (
        Path(args.preflight_out) if args.preflight_out else args.out_dir / "preflight.json"
    )
    try:
        if args.install_runtime:
            _install_runtime()
        stage = "preflight"
        preflight = preflight_small_model(args)
        write_json(preflight_path, preflight)
        manifest["preflight"] = preflight
        if args.preflight_only:
            manifest["status"] = "ok"
            return 0
        if args.readiness_diagnose:
            stage = "readiness"
            diagnosis = diagnose_readiness(
                args.model,
                args.out_dir,
                max_model_len=args.max_model_len,
                load_timeout=args.load_timeout,
            )
            manifest["readiness"] = diagnosis
            if diagnosis["server_ready"]:
                manifest["status"] = "ok"
            else:
                manifest["status"] = "degraded"
                manifest["error_kind"] = "readiness_failed"
            return 0 if diagnosis["server_ready"] else 1
        stage = "run"
        manifest["result"] = run_ladder(
            args.repo,
            args.model,
            args.out_dir,
            args.trace_dir,
            rollouts=args.rollouts,
            smoke_task=smoke_task,
            skip_smoke=args.skip_smoke,
            tasks=args.tasks,
            max_model_len=args.max_model_len,
            spec_set=args.spec_set,
            limit=args.limit,
            load_timeout=args.load_timeout,
            rollout_timeout=args.rollout_timeout,
            progress_every=args.progress_every,
            resume=args.resume,
            staging=args.staging,
            provider=args.provider,
            system_variant=args.system_variant,
            adapter_path=args.adapter_path,
            train_version=args.train_version,
        )
        if manifest["result"].get("status") == "degraded":
            manifest["status"] = "degraded"
            manifest["error_kind"] = "staging_degraded"
        else:
            manifest["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - Kaggle must persist the failure manifest
        manifest["status"] = "error"
        manifest["stage"] = stage
        manifest["error_kind"] = classify_exception(exc)
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        if stage == "preflight":
            write_json(
                preflight_path,
                {
                    "status": "error",
                    "stage": "preflight",
                    "error_kind": manifest["error_kind"],
                    "error": manifest["error"],
                    "runtime": runtime_info(),
                    "gpu": gpu_info(),
                },
            )
    finally:
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0 if manifest.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
