"""Build the Kaggle T4 kernel for the GRPO group-variance preflight.

This is a full-loop runner, not the old final-turn resampling probe. It mounts
the train-v5 assistant-only adapter from the existing Kaggle kernel source,
verifies its SHA-256, and evaluates a frozen 20-task subset of the 45-task
middle-state pool with 8 rollouts per task.
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

TRAIN_SHA256 = "6ee7e79d87cb7c41f4c1b91964d70abcf5e886c5ac75450318760a805b78f830"
TASKS_SHA256 = "20e20b3beabaf0c48c00dae56d97c1aea356b1ac89c4cdb0fd9decc799c4fa5b"
POOL_TASKS_SHA256 = "2bd5de68a2adb6eec6a73ade188e6d7b72edf436ed88f297284fc9ebcdbebc7d"
ADAPTER_SHA256 = "f5b685d4da795a783fbcb21b15880daebfa08c64b1661792f7e2c6caab29fea7"
ADAPTER_SOURCE_KERNEL = "autumnsq/lite-agent-method-assistant-train-v5"
ADAPTER_WEIGHTS_REL = (
    "private/kaggle_pull_assistant_v5_full/method/assistant_only/adapter/adapter_model.safetensors"
)
PY_ROOTS = ("src", "scripts", "training")
EXTRA_FILES = (
    "pyproject.toml",
    "training/requirements-kaggle.txt",
    "private/sft_data_v5/train.jsonl",
    "private/final_test_v3/tasks.jsonl",
    "private/grpo_middle_pool/preflight_20.jsonl",
    "private/grpo_middle_pool/preflight_20_manifest.json",
)

KERNEL_TEMPLATE = Template(
    """from __future__ import annotations

import base64
import gc
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
EXPECTED_TRAIN_SHA256 = "$train_sha256"
EXPECTED_TASKS_SHA256 = "$tasks_sha256"
EXPECTED_POOL_SHA256 = "$pool_sha256"
EXPECTED_ADAPTER_SHA256 = "$adapter_sha256"
REPO = Path("/kaggle/working/repo")
MODEL = Path("/kaggle/temp/qwen2.5-7b-instruct")
OUT = Path("/kaggle/working/grpo_preflight")
TRACE_ROOT = Path("/kaggle/temp/grpo_preflight_traces")
TRAIN_FILE = REPO / "private/sft_data_v5/train.jsonl"
TASKS_FILE = REPO / "private/final_test_v3/tasks.jsonl"
POOL_FILE = REPO / "private/grpo_middle_pool/preflight_20.jsonl"
ADAPTER: Path | None = None
BASE_SEED = 20261019
TEMPERATURE = 0.7
TOP_P = 0.9
STAGING_ROLLOUTS = 8
FULL_ROLLOUTS = 8
FULL_TIMEOUT = 14400


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


def verify_inputs() -> dict:
    checks = {
        "train_sha256": sha256(TRAIN_FILE),
        "tasks_sha256": sha256(TASKS_FILE),
        "pool_sha256": sha256(POOL_FILE),
    }
    expected = {
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "tasks_sha256": EXPECTED_TASKS_SHA256,
        "pool_sha256": EXPECTED_POOL_SHA256,
    }
    mismatched = [name for name in checks if checks[name] != expected[name]]
    if mismatched:
        status(
            {
                "status": "error",
                "error_kind": "input_sha_mismatch",
                "mismatched": mismatched,
                "actual": checks,
                "expected": expected,
            }
        )
        raise ValueError(f"input SHA mismatch: {mismatched}")
    rows = [
        json.loads(line)
        for line in POOL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 20:
        raise ValueError(f"preflight pool must have 20 rows, got {len(rows)}")
    task_ids = [str(row.get("task_id") or "") for row in rows]
    if len(set(task_ids)) != len(task_ids) or not all(task_ids):
        raise ValueError("preflight pool has duplicate or empty task ids")
    return {"sha256": checks, "tasks": len(rows), "task_ids": task_ids}


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
    raise ValueError("train-v5 assistant-only adapter SHA mismatch")


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


def evaluate(label: str, *, task_limit: int, rollouts: int, timeout: float) -> dict:
    status(
        {
            "status": "running",
            "stage": f"evaluate_{label}",
            "task_limit": task_limit,
            "rollouts": rollouts,
        }
    )
    command = [
        sys.executable,
        str(REPO / "training/kaggle/clean_heldout.py"),
        "--train-file",
        str(TRAIN_FILE),
        "--task-ids-file",
        str(POOL_FILE),
        "--model-path",
        str(MODEL),
        "--adapter-path",
        str(ADAPTER),
        "--out-dir",
        str(OUT / label),
        "--trace-dir",
        str(TRACE_ROOT / label),
        "--rollouts",
        str(rollouts),
        "--task-limit",
        str(task_limit),
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
    run(command, timeout, label)
    return json.loads((OUT / label / "summary.json").read_text(encoding="utf-8"))


def variance_status(task: dict) -> str:
    valid = int(task.get("valid_rollouts") or 0)
    variance = task.get("variance")
    if valid < 2 or variance is None:
        return "not_measured"
    return "measured_positive" if float(variance) > 0.0 else "measured_zero"


def variance_rollup(summary: dict, *, expected_tasks: int, rollouts: int) -> dict:
    condition = summary.get("condition") or {}
    tasks = condition.get("tasks") or []
    rows = []
    for task in tasks:
        rows.append(
            {
                "task_id": task.get("task_id"),
                "family": task.get("family"),
                "difficulty": task.get("difficulty"),
                "rollouts": task.get("rollouts"),
                "valid_rollouts": task.get("valid_rollouts"),
                "errors": task.get("errors"),
                "task_success": task.get("task_success"),
                "pass_rate": task.get("pass_rate"),
                "variance": task.get("variance"),
                "bernoulli_variance": task.get("bernoulli_variance"),
                "variance_status": variance_status(task),
            }
        )
    positive = sum(1 for row in rows if row["variance_status"] == "measured_positive")
    zero = sum(1 for row in rows if row["variance_status"] == "measured_zero")
    not_measured = sum(1 for row in rows if row["variance_status"] == "not_measured")
    complete = len(rows) == expected_tasks and all(
        int(row["valid_rollouts"] or 0) == rollouts
        and int(row["errors"] or 0) == 0
        for row in rows
    )
    ratio = positive / expected_tasks if expected_tasks else 0.0
    if not complete:
        verdict = "preflight_incomplete"
        note = "基础设施未完成，组内方差未验证"
    elif ratio >= 0.30:
        verdict = "pass"
        note = "GRPO 预检通过，组内方差信号存在，可跑全量"
    else:
        verdict = "sparse"
        note = f"实测组内方差稀疏（{positive}/{expected_tasks}），GRPO 不启动"
    return {
        "expected_tasks": expected_tasks,
        "rollouts_per_task": rollouts,
        "tasks_observed": len(rows),
        "complete": complete,
        "measured_positive": positive,
        "measured_zero": zero,
        "not_measured": not_measured,
        "positive_ratio": round(ratio, 4),
        "verdict": verdict,
        "note": note,
        "tasks": rows,
    }


def main() -> int:
    started = time.perf_counter()
    extract_bundle()
    OUT.mkdir(parents=True, exist_ok=True)
    TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    gpu = gpu_report()
    status({"status": "running", "stage": "verify_inputs", "gpu": gpu})
    inputs = verify_inputs()
    install_and_model()
    adapter, adapter_sha = find_adapter()
    globals()["ADAPTER"] = str(adapter)
    status(
        {
            "status": "running",
            "stage": "adapter_ready",
            "adapter": str(adapter),
            "adapter_sha256": adapter_sha,
        }
    )
    staging = evaluate("staging", task_limit=1, rollouts=STAGING_ROLLOUTS, timeout=2400)
    staging_rollup = variance_rollup(staging, expected_tasks=1, rollouts=STAGING_ROLLOUTS)
    staging_ok = (
        staging.get("condition", {}).get("status") == "completed"
        and staging_rollup["complete"]
    )
    status(
        {
            "status": "running" if staging_ok else "degraded",
            "stage": "staging_result",
            "staging_ok": staging_ok,
            "staging_rollup": staging_rollup,
        }
    )
    if not staging_ok:
        summary = {
            "status": "staging_failed",
            "stage": "staging_failed",
            "gpu": gpu,
            "inputs": inputs,
            "adapter": {"path": str(adapter), "sha256": adapter_sha},
            "staging": staging,
            "staging_rollup": staging_rollup,
            "duration_s": round(time.perf_counter() - started, 1),
        }
        (OUT / "grpo_preflight_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
        )
        (Path("/kaggle/working/grpo_preflight_summary.json")).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
        )
        return 2

    full = evaluate("full", task_limit=0, rollouts=FULL_ROLLOUTS, timeout=FULL_TIMEOUT)
    full_rollup = variance_rollup(full, expected_tasks=20, rollouts=FULL_ROLLOUTS)
    duration = round(time.perf_counter() - started, 1)
    summary = {
        "status": "ok" if full_rollup["complete"] else "degraded",
        "stage": "done",
        "gpu": gpu,
        "duration_s": duration,
        "gpu_hours_free_tier": round(duration / 3600, 2),
        "inputs": inputs,
        "adapter": {"path": str(adapter), "sha256": adapter_sha},
        "sampling": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "base_seed": BASE_SEED,
            "seed_policy": "base_seed_plus_rollout_index",
            "rollouts_per_task": FULL_ROLLOUTS,
        },
        "staging": staging,
        "staging_rollup": staging_rollup,
        "full": full,
        "full_rollup": full_rollup,
    }
    (OUT / "grpo_preflight_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
    )
    (Path("/kaggle/working/grpo_preflight_summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
    )
    status({"status": summary["status"], "stage": "done", "summary": summary})
    return 0 if summary["status"] == "ok" else 1


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
        raise FileNotFoundError(f"GRPO preflight bundle inputs missing: {missing}")
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
        "train_sha256": sha256(root / "private/sft_data_v5/train.jsonl"),
        "tasks_sha256": sha256(root / "private/final_test_v3/tasks.jsonl"),
        "pool_sha256": sha256(root / "private/grpo_middle_pool/preflight_20.jsonl"),
        "adapter_sha256": sha256(root / ADAPTER_WEIGHTS_REL),
    }
    expected = {
        "train_sha256": TRAIN_SHA256,
        "tasks_sha256": TASKS_SHA256,
        "pool_sha256": POOL_TASKS_SHA256,
        "adapter_sha256": ADAPTER_SHA256,
    }
    if actual != expected:
        raise ValueError(f"input SHA mismatch: actual={actual} expected={expected}")
    payload = base64.b64encode(zlib.compress(bundle_path.read_bytes(), 9)).decode("ascii")
    code = KERNEL_TEMPLATE.substitute(
        code_b64=payload,
        train_sha256=TRAIN_SHA256,
        tasks_sha256=TASKS_SHA256,
        pool_sha256=POOL_TASKS_SHA256,
        adapter_sha256=ADAPTER_SHA256,
    )
    (out_dir / "run.py").write_text(code, encoding="utf-8", newline="\n")
    metadata = {
        "id": kernel_id,
        "title": "Lite Agent GRPO Preflight V3",
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
        "train_sha256": TRAIN_SHA256,
        "tasks_sha256": TASKS_SHA256,
        "pool_sha256": POOL_TASKS_SHA256,
        "run_py_bytes": (out_dir / "run.py").stat().st_size,
        "bundle_zip_bytes": bundle_path.stat().st_size,
        "bundle_files": len(files),
    }
    (out_dir / "kernel_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build the GRPO preflight Kaggle kernel")
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
