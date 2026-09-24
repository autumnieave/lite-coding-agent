"""Build the Kaggle T4 kernel for the layered session-constraint probe.

The kernel mounts the train-v5 assistant-only adapter, verifies its SHA-256,
then runs the real compactor and forced-choice probe on 8 frozen middle-state
tasks. It writes per-condition JSONL plus an aggregate analysis.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import zipfile
import zlib
from collections.abc import Iterable
from pathlib import Path
from string import Template

TASKS_SHA256 = "f0a8067a9483bcfbf6564fa579871210a27104a291a7a266cc60372f97840716"
ADAPTER_SHA256 = "f5b685d4da795a783fbcb21b15880daebfa08c64b1661792f7e2c6caab29fea7"
ADAPTER_SOURCE_KERNEL = "autumnsq/lite-agent-method-assistant-train-v5"
PY_ROOTS = ("src", "scripts", "training")
EXTRA_FILES = (
    "pyproject.toml",
    "training/requirements-kaggle.txt",
    "private/session_constraint/tasks.json",
)

KERNEL_TEMPLATE = Template(
    """from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
import zlib
from pathlib import Path

CODE_B64 = "$code_b64"
EXPECTED_TASKS_SHA256 = "$tasks_sha256"
EXPECTED_ADAPTER_SHA256 = "$adapter_sha256"
REPO = Path("/kaggle/working/repo")
MODEL = Path("/kaggle/temp/qwen2.5-7b-instruct")
OUT = Path("/kaggle/working/session_constraint")
TASKS_FILE = REPO / "private/session_constraint/tasks.json"
PROBE_TIMEOUT = 18000


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def status(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.setdefault("updated_at", now())
    Path("/kaggle/working/kernel_status.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
    )
    print("STATUS", json.dumps(body, ensure_ascii=False), flush=True)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_report() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        return {
            "cuda": True,
            "device": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    except Exception as exc:
        return {"cuda": False, "error": str(exc)}


def extract_bundle() -> None:
    REPO.mkdir(parents=True, exist_ok=True)
    blob = Path("/kaggle/working/repo.zip")
    blob.write_bytes(zlib.decompress(base64.b64decode(CODE_B64)))
    with zipfile.ZipFile(blob) as archive:
        archive.extractall(REPO)
    for path in (REPO, REPO / "src"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def run(cmd: list[str], timeout: float, stage: str) -> None:
    print("RUN", stage, json.dumps(cmd, ensure_ascii=False), flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "src")])
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


def verify_tasks() -> dict:
    actual = sha256(TASKS_FILE)
    if actual != EXPECTED_TASKS_SHA256:
        status(
            {
                "status": "error",
                "error_kind": "tasks_sha_mismatch",
                "expected": EXPECTED_TASKS_SHA256,
                "actual": actual,
            }
        )
        raise ValueError("tasks SHA mismatch")
    payload = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") or []
    if not 5 <= len(tasks) <= 10:
        raise ValueError(f"expected 5-10 tasks, got {len(tasks)}")
    return {"tasks_sha256": actual, "tasks": len(tasks)}


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


def find_adapter() -> tuple[Path, str]:
    candidates = sorted(Path("/kaggle/input").rglob("adapter_config.json"))
    checked: list[dict] = []
    for config in candidates:
        folder = config.parent
        weights = folder / "adapter_model.safetensors"
        if not weights.is_file() or weights.stat().st_size <= 0:
            continue
        digest = sha256(weights)
        checked.append({"path": str(folder), "sha256": digest, "bytes": weights.stat().st_size})
        if digest == EXPECTED_ADAPTER_SHA256:
            return folder, digest
    status(
        {
            "status": "error",
            "error_kind": "adapter_sha_mismatch",
            "expected": EXPECTED_ADAPTER_SHA256,
            "checked": checked,
        }
    )
    raise ValueError("adapter SHA mismatch")


def main() -> int:
    started = time.perf_counter()
    extract_bundle()
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = gpu_report()
    status({"status": "running", "stage": "verify_inputs", "gpu": gpu})
    inputs = verify_tasks()
    install_and_model()
    adapter, adapter_sha = find_adapter()
    status(
        {
            "status": "running",
            "stage": "adapter_ready",
            "adapter": str(adapter),
            "adapter_sha256": adapter_sha,
            "inputs": inputs,
        }
    )
    run(
        [
            sys.executable,
            str(REPO / "training/kaggle/session_constraint_probe.py"),
            "--model-path",
            str(MODEL),
            "--adapter-path",
            str(adapter),
            "--tasks",
            str(TASKS_FILE),
            "--out-dir",
            str(OUT),
            "--condition",
            "both",
            "--max-model-len",
            "8192",
        ],
        PROBE_TIMEOUT,
        "probe",
    )
    run(
        [
            sys.executable,
            str(REPO / "scripts/analyze_session_constraint_probe.py"),
            "--input-dir",
            str(OUT),
            "--out",
            str(OUT / "analysis.json"),
        ],
        600,
        "analysis",
    )
    analysis = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
    summary = {
        "status": "ok",
        "gpu": gpu,
        "tasks_sha256": inputs["tasks_sha256"],
        "tasks": inputs["tasks"],
        "adapter_sha256": adapter_sha,
        "analysis": analysis,
        "duration_s": round(time.perf_counter() - started, 1),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
    )
    status({"status": "ok", "stage": "done", "duration_s": summary["duration_s"]})
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _python_files(root: Path) -> Iterable[Path]:
    for relative in PY_ROOTS:
        for path in sorted((root / relative).rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield path


def build_bundle(root: Path, bundle_path: Path) -> list[Path]:
    files = list(_python_files(root)) + [root / relative for relative in EXTRA_FILES]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"session-constraint bundle inputs missing: {missing}")
    files = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    with zipfile.ZipFile(
        bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    return files


def build_kernel(*, root: Path, out_dir: Path, kernel_id: str) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "repo.zip"
    files = build_bundle(root, bundle_path)
    actual = {
        "tasks_sha256": sha256(root / "private/session_constraint/tasks.json"),
    }
    expected = {"tasks_sha256": TASKS_SHA256}
    if actual != expected:
        raise ValueError(f"input SHA mismatch: actual={actual} expected={expected}")
    payload = base64.b64encode(zlib.compress(bundle_path.read_bytes(), 9)).decode("ascii")
    code = KERNEL_TEMPLATE.substitute(
        code_b64=payload,
        tasks_sha256=TASKS_SHA256,
        adapter_sha256=ADAPTER_SHA256,
    )
    (out_dir / "run.py").write_text(code, encoding="utf-8", newline="\n")
    metadata = {
        "id": kernel_id,
        "title": "Lite Agent Session Constraint Layered",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [ADAPTER_SOURCE_KERNEL],
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "kernel_id": kernel_id,
        "adapter_source": ADAPTER_SOURCE_KERNEL,
        "adapter_sha256": ADAPTER_SHA256,
        "tasks_sha256": TASKS_SHA256,
        "run_py_bytes": (out_dir / "run.py").stat().st_size,
        "bundle_zip_bytes": bundle_path.stat().st_size,
        "bundle_files": len(files),
    }
    (out_dir / "kernel_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build the session-constraint Kaggle kernel")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_kernel(
        root=args.root.resolve(),
        out_dir=args.out_dir.resolve(),
        kernel_id=args.kernel_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
