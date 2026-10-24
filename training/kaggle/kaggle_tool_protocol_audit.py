"""Probe-only Kaggle audit for Qwen tool-call protocol compatibility.

This entrypoint starts vLLM, checks the model's native chat template, sends one
``tool_choice=auto`` request with the existing tool schemas, and stops.  It
never runs the difficulty ladder and never changes the Agent loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.kaggle.kaggle_small_model_tuning import (  # noqa: E402
    DEFAULT_TORCH_BACKEND,
    DEFAULT_VLLM_VERSION,
    _install_runtime,
    _server_command,
    _server_env,
    _wait_for_server,
    gpu_info,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUT_DIR = Path("/kaggle/working/protocol_audit")


def _tool_specs(repo: Path) -> list[dict[str, Any]]:
    """Load the same built-in tool schemas used by the Agent."""
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from agent.tools import build_default_registry

    with tempfile.TemporaryDirectory(prefix="tool-protocol-audit-") as root:
        return build_default_registry(root).specs()


def _chat_template_flags(template: Any) -> dict[str, Any]:
    if isinstance(template, dict):
        template = next(iter(template.values()), "")
    text = str(template or "")
    return {
        "available": bool(text),
        "length": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None,
        "has_tools_branch": "{%- if tools %}" in text or "{% if tools %}" in text,
        "has_tool_call_tag": "<tool_call>" in text,
        "has_tool_response_tag": "<tool_response>" in text,
        "has_tool_calls_field": "tool_calls" in text,
    }


def _load_chat_template(model: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    return _chat_template_flags(getattr(tokenizer, "chat_template", None))


def _content_has_fenced_json(content: str) -> bool:
    return "```" in content and "{" in content and "}" in content


def _probe(
    base_url: str,
    model: str,
    tools: list[dict[str, Any]],
    tool_choice: str,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key="local-vllm", base_url=base_url, timeout=180)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a coding agent. Use the provided tool to inspect files; "
                        "never guess file contents. Do not answer with a code block."
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
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.0,
            seed=20261019,
            max_tokens=256,
        )
    except Exception as exc:  # noqa: BLE001 - probe failure is the evidence
        return {
            "tool_choice": tool_choice,
            "protocol_error": f"{type(exc).__name__}: {exc}",
            "tool_calls": [],
            "content": "",
            "content_has_fenced_json": False,
        }
    finally:
        client.close()

    message = response.choices[0].message
    content = message.content or ""
    return {
        "tool_choice": tool_choice,
        "response_id": response.id,
        "model": response.model,
        "finish_reason": response.choices[0].finish_reason,
        "content": content,
        "content_has_fenced_json": _content_has_fenced_json(content),
        "tool_calls": [
            {"name": call.function.name, "arguments": call.function.arguments}
            for call in (message.tool_calls or ())
        ],
        "protocol_error": None,
    }


def run_probe(repo: Path, model: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _server_env(model)
    server_log = out_dir / "vllm_server.log"
    log_handle = server_log.open("w", encoding="utf-8")
    server = subprocess.Popen(
        _server_command(model),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    started = time.monotonic()
    try:
        _wait_for_server(env["LLM_BASE_URL"])
        tools = _tool_specs(repo)
        template = _load_chat_template(model)
        auto = _probe(env["LLM_BASE_URL"], model, tools, "auto")
        required = None
        if not auto["tool_calls"]:
            required = _probe(env["LLM_BASE_URL"], model, tools, "required")
        result = {
            "model": model,
            "tools_offered": [item["function"]["name"] for item in tools],
            "chat_template": template,
            "auto": auto,
            "required_control": required,
            "duration_s": round(time.monotonic() - started, 1),
            "vllm_version": DEFAULT_VLLM_VERSION,
            "torch_backend": DEFAULT_TORCH_BACKEND,
            "gpu": gpu_info(),
        }
        (out_dir / "tool_protocol_audit.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        log_handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen tool-call protocol audit")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--install-runtime", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest: dict[str, Any] = {
        "model": args.model,
        "gpu": gpu_info(),
        "python_version": sys.version.split()[0],
        "vllm_version": DEFAULT_VLLM_VERSION,
        "torch_backend": DEFAULT_TORCH_BACKEND,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        if args.install_runtime:
            _install_runtime()
        manifest["result"] = run_probe(args.repo, args.model, args.out_dir)
        manifest["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - write a failure manifest for Kaggle
        manifest["status"] = "error"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
