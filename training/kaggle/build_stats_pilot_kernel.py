"""Build the Kaggle T4 kernel for the stage-1.5 baseline/SFT pilot."""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path
from string import Template

KERNEL_TEMPLATE = Template(
    """from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
import zlib
from pathlib import Path

CODE_B64 = "$code_b64"
REPO = Path("/kaggle/working/repo")
MODEL = Path("/kaggle/temp/qwen2.5-7b-instruct")
OUT = Path("/kaggle/working/stats_pilot")
ADAPTER_SHA256 = "f5b685d4da795a783fbcb21b15880daebfa08c64b1661792f7e2c6caab29fea7"
PILOT_TASKS = "$pilot_tasks"
TEMPERATURE = 0.7
TOP_P = 0.9
BASE_SEED = 20261019


def status(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    Path("/kaggle/working/kernel_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def extract_bundle() -> None:
    REPO.mkdir(parents=True, exist_ok=True)
    blob = Path("/kaggle/working/repo.zip")
    blob.write_bytes(zlib.decompress(base64.b64decode(CODE_B64)))
    with zipfile.ZipFile(blob) as archive:
        archive.extractall(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def run(cmd: list[str], timeout: float, stage: str) -> None:
    print("RUN", stage, json.dumps(cmd, ensure_ascii=False), flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    started = time.perf_counter()
    try:
        subprocess.run(cmd, check=True, timeout=timeout, cwd=str(REPO), env=env)
    except subprocess.TimeoutExpired as exc:
        status(
            {
                "status": "error",
                "error_kind": "run_timeout",
                "stage": stage,
                "timeout_s": timeout,
            }
        )
        raise RuntimeError(f"{stage} exceeded {timeout}s") from exc
    finally:
        print(f"STAGE_SECONDS {stage}={time.perf_counter() - started:.1f}", flush=True)


def install_and_model() -> None:
    status({"status": "running", "stage": "install"})
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-r",
            str(REPO / "training/requirements-kaggle.txt"),
        ],
        2400,
        "install",
    )
    if not (MODEL / "config.json").is_file():
        status({"status": "running", "stage": "download_model"})
        run(
            [
                sys.executable,
                "-c",
                "from huggingface_hub import snapshot_download;"
                "snapshot_download('Qwen/Qwen2.5-7B-Instruct', "
                f"local_dir='{MODEL}', max_workers=4)",
            ],
            3600,
            "download_model",
        )


def find_adapter() -> Path:
    candidates = sorted(Path("/kaggle/input").rglob("adapter_config.json"))
    for config in candidates:
        folder = config.parent
        weights = [
            path
            for path in folder.glob("adapter_model.*")
            if path.is_file() and path.stat().st_size > 0
        ]
        if weights:
            return folder
    raise FileNotFoundError("non-empty train-v5 adapter not found under /kaggle/input")


def run_pilot(adapter: Path, *, label: str, limit: int, rollouts: int, timeout: float) -> None:
    command = [
        sys.executable,
        str(REPO / "training/kaggle/stats_pilot.py"),
        "--model-path",
        str(MODEL),
        "--adapter-path",
        str(adapter),
        "--out-dir",
        str(OUT / label),
        "--trace-dir",
        str(OUT / "traces" / label),
        "--tasks",
        PILOT_TASKS,
        "--rollouts",
        str(rollouts),
        "--max-model-len",
        "8192",
        "--max-tokens",
        "512",
        "--load-timeout",
        "240",
        "--rollout-timeout",
        "300",
        "--temperature",
        str(TEMPERATURE),
        "--top-p",
        str(TOP_P),
        "--seed",
        str(BASE_SEED),
    ]
    if limit:
        command += ["--limit", str(limit)]
    run(command, timeout, label)


def main() -> int:
    extract_bundle()
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = {
        "device": "unknown",
    }
    try:
        import torch

        if torch.cuda.is_available():
            gpu = {
                "cuda": True,
                "device": torch.cuda.get_device_name(0),
                "gpu_memory_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
                ),
            }
    except Exception as exc:
        gpu = {"cuda": False, "error": str(exc)}
    install_and_model()
    adapter = find_adapter()
    adapter_sha = sha256(adapter / "adapter_model.safetensors")
    status(
        {
            "status": "running",
            "stage": "staging",
            "gpu": gpu,
            "adapter": str(adapter),
            "adapter_sha256": adapter_sha,
            "sampling": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "base_seed": BASE_SEED,
                "seed_policy": "base_seed_plus_rollout_index",
            },
        }
    )
    if adapter_sha != ADAPTER_SHA256:
        status(
            {
                "status": "error",
                "error_kind": "adapter_sha_mismatch",
                "expected": ADAPTER_SHA256,
                "actual": adapter_sha,
            }
        )
        return 2

    run_pilot(adapter, label="staging", limit=6, rollouts=1, timeout=2400)
    staging = json.loads((OUT / "staging/pilot_summary.json").read_text(encoding="utf-8"))
    staging_conditions = list(staging["conditions"].values())
    staging_errors = sum(int(item.get("errors") or 0) for item in staging_conditions)
    staging_success = sum(int(item.get("task_success") or 0) for item in staging_conditions)
    staging_ok = (
        all(item.get("status") == "completed" for item in staging_conditions)
        and staging_errors == 0
        and staging_success >= 1
    )
    status(
        {
            "status": "running" if staging_ok else "degraded",
            "stage": "staging_result",
            "staging_ok": staging_ok,
            "staging_errors": staging_errors,
            "staging_success": staging_success,
            "staging": staging,
        }
    )
    if not staging_ok:
        return 2

    status({"status": "running", "stage": "full_pilot"})
    run_pilot(adapter, label="full", limit=0, rollouts=10, timeout=28800)
    full = json.loads((OUT / "full/pilot_summary.json").read_text(encoding="utf-8"))
    (Path("/kaggle/working/stats_pilot_summary.json")).write_text(
        json.dumps(full, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
    )
    status(
        {
            "status": "ok",
            "stage": "done",
            "gpu": gpu,
            "adapter_sha256": adapter_sha,
            "staging": staging,
            "full": full,
        }
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        status(
            {
                "status": "error",
                "error_kind": type(exc).__name__,
                "error": str(exc)[:2000],
                "traceback": traceback.format_exc()[-5000:],
            }
        )
        raise
"""
)


def build_kernel(
    *,
    bundle_zip: Path,
    out_dir: Path,
    kernel_id: str,
    pilot_tasks: str,
) -> dict[str, object]:
    payload = base64.b64encode(zlib.compress(bundle_zip.read_bytes(), 9)).decode("ascii")
    code = KERNEL_TEMPLATE.substitute(code_b64=payload, pilot_tasks=pilot_tasks)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "repo.zip").write_bytes(bundle_zip.read_bytes())
    (out_dir / "run.py").write_text(code, encoding="utf-8", newline="\n")
    metadata = {
        "id": kernel_id,
        "title": "Lite Agent Stage 1.5 Stats Pilot v2 Round 2",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": ["autumnsq/lite-agent-method-assistant-train-v5"],
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "kernel_id": kernel_id,
        "run_py_bytes": (out_dir / "run.py").stat().st_size,
        "bundle_zip_bytes": bundle_zip.stat().st_size,
        "pilot_tasks": pilot_tasks,
    }
    (out_dir / "kernel_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build the stats pilot Kaggle kernel")
    parser.add_argument("--bundle-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--pilot-tasks", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_kernel(
        bundle_zip=args.bundle_zip.resolve(),
        out_dir=args.out_dir.resolve(),
        kernel_id=args.kernel_id,
        pilot_tasks=args.pilot_tasks,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
