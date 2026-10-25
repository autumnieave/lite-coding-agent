"""Build the Kaggle T4 kernel for train-v6 clean held-out evaluation."""

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

TRAIN_SHA256 = "920a4ee73199b64a81db867deb5bd1e143b40c27420f7d2a5ce4c713a5cdaf70"
VAL_SHA256 = "e49942151205b6231c5dabcd2d55c70010e096c90dbfdb8b9dc31f6d8b7c07e8"
TASKS_SHA256 = "20e20b3beabaf0c48c00dae56d97c1aea356b1ac89c4cdb0fd9decc799c4fa5b"
PY_ROOTS = ("src", "scripts", "training")
EXTRA_FILES = (
    "pyproject.toml",
    "training/requirements-kaggle.txt",
    "training/configs/qlora_7b_t4.json",
    "private/sft_data_v5/train.jsonl",
    "private/sft_data_v6/train.jsonl",
    "private/sft_data_v6/val.jsonl",
    "private/sft_data_v6/MANIFEST.json",
    "private/sft_data_v6/TRAIN_V6_FROZEN.json",
    "private/sft_data_v6/TRAIN_V6_FROZEN.txt",
    "private/final_test_v3/tasks.jsonl",
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
EXPECTED_TRAIN_SHA256 = "$train_sha256"
EXPECTED_VAL_SHA256 = "$val_sha256"
EXPECTED_TASKS_SHA256 = "$tasks_sha256"
REPO = Path("/kaggle/working/repo")
MODEL = Path("/kaggle/temp/qwen2.5-7b-instruct")
OUT = Path("/kaggle/working/clean_heldout")
ADAPTER = OUT / "adapter"
TRACE_ROOT = Path("/kaggle/temp/clean_heldout_traces")
TRAIN_FILE = REPO / "private/sft_data_v6/train.jsonl"
VAL_FILE = REPO / "private/sft_data_v6/val.jsonl"
POOL_FILE = REPO / "private/sft_data_v5/train.jsonl"
TASKS_FILE = REPO / "private/final_test_v3/tasks.jsonl"
BASE_SEED = 20261019
TEMPERATURE = 0.7
TOP_P = 0.9


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


def verify_inputs() -> dict:
    checks = {
        "train_sha256": sha256(TRAIN_FILE),
        "val_sha256": sha256(VAL_FILE),
        "tasks_sha256": sha256(TASKS_FILE),
    }
    expected = {
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "val_sha256": EXPECTED_VAL_SHA256,
        "tasks_sha256": EXPECTED_TASKS_SHA256,
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
    manifest = json.loads((REPO / "private/sft_data_v6/MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["checks"]["heldout_tasks_remaining_in_train"]:
        raise ValueError("train-v6 still contains held-out tasks")
    if manifest["checks"]["heldout_tasks_remaining_in_val"]:
        raise ValueError("train-v6 val contains held-out tasks")
    if manifest["train_examples"] <= 0 or manifest["val_examples"] <= 0:
        raise ValueError("train-v6 split is empty")
    return {"sha256": checks, "manifest": manifest}


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


def tokenizer_gate() -> dict:
    status({"status": "running", "stage": "tokenizer_gate"})
    run(
        [
            sys.executable,
            "-m",
            "training.validate_train_v4",
            "--train-file",
            str(TRAIN_FILE),
            "--val-file",
            str(VAL_FILE),
            "--tokenizer-path",
            str(MODEL),
            "--max-length",
            "3072",
            "--out",
            str(OUT / "tokenizer_gate.json"),
        ],
        1800,
        "tokenizer_gate",
    )
    return json.loads((OUT / "tokenizer_gate.json").read_text(encoding="utf-8"))


def train() -> dict:
    status({"status": "running", "stage": "train_sft"})
    run(
        [
            sys.executable,
            str(REPO / "training/kaggle/method_runner.py"),
            "--repo",
            str(REPO),
            "--mode",
            "sft",
            "--config",
            str(REPO / "training/configs/qlora_7b_t4.json"),
            "--model-path",
            str(MODEL),
            "--train-file",
            str(TRAIN_FILE),
            "--val-file",
            str(VAL_FILE),
            "--output-dir",
            str(ADAPTER),
            "--loss-mode",
            "assistant_only",
            "--seed",
            str(BASE_SEED),
            "--heldout-out-root",
            str(OUT / "unused_heldout"),
            "--heldout-trace-root",
            str(OUT / "unused_traces"),
            "--heldout-tasks",
            "U-TR-A-C1-HARD",
            "--train-version",
            "train-v6",
        ],
        21600,
        "train_sft",
    )
    manifest_path = ADAPTER / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("training did not write run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights = ADAPTER / "adapter_model.safetensors"
    if not weights.is_file() or weights.stat().st_size <= 0:
        raise ValueError("adapter_model.safetensors is missing or empty")
    manifest["adapter_sha256"] = sha256(weights)
    manifest["adapter_bytes"] = weights.stat().st_size
    return manifest


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
        str(POOL_FILE),
        "--task-ids-file",
        str(TASKS_FILE),
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


def summarize_runs(label: str) -> dict:
    path = OUT / label / "sft/runs.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    valid = [row for row in rows if not row.get("error_kind")]
    return {
        "rows": len(rows),
        "valid": len(valid),
        "errors": len(rows) - len(valid),
        "nonempty_replies": sum(1 for row in valid if str(row.get("reply") or "").strip()),
        "task_success": sum(1 for row in valid if row.get("task_success")),
        "within_steps": sum(1 for row in valid if row.get("within_steps")),
        "tool_selection_ok": sum(1 for row in valid if row.get("tool_selection_ok")),
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
    gate = tokenizer_gate()
    train_manifest = train()
    status(
        {
            "status": "running",
            "stage": "train_done",
            "adapter_sha256": train_manifest["adapter_sha256"],
            "adapter_bytes": train_manifest["adapter_bytes"],
            "optimizer_steps": train_manifest.get("optimizer_steps"),
        }
    )
    staging = evaluate("staging", task_limit=3, rollouts=1, timeout=1800)
    staging_runs = summarize_runs("staging")
    staging_ok = (
        staging["condition"].get("status") == "completed"
        and staging_runs["errors"] == 0
        and staging_runs["nonempty_replies"] >= 1
    )
    status(
        {
            "status": "running" if staging_ok else "degraded",
            "stage": "staging_result",
            "staging_ok": staging_ok,
            "staging_runs": staging_runs,
        }
    )
    if not staging_ok:
        summary = {
            "status": "degraded",
            "stage": "staging_failed",
            "gpu": gpu,
            "inputs": inputs,
            "tokenizer_gate": gate,
            "train_manifest": train_manifest,
            "staging": staging,
            "staging_runs": staging_runs,
        }
        (OUT / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
        )
        return 2

    full = evaluate("full", task_limit=0, rollouts=10, timeout=28800)
    full_runs = summarize_runs("full")
    duration = round(time.perf_counter() - started, 1)
    summary = {
        "status": "ok" if full["condition"].get("status") == "completed" else "degraded",
        "stage": "done",
        "gpu": gpu,
        "duration_s": duration,
        "gpu_hours_free_tier": round(duration / 3600, 2),
        "inputs": inputs,
        "tokenizer_gate": gate,
        "train_manifest": train_manifest,
        "staging": staging,
        "staging_runs": staging_runs,
        "full": full,
        "full_runs": full_runs,
        "sampling": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "base_seed": BASE_SEED,
            "seed_policy": "base_seed_plus_rollout_index",
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
    )
    (Path("/kaggle/working/clean_heldout_summary.json")).write_text(
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
        raise FileNotFoundError(f"clean held-out bundle inputs missing: {missing}")
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
    train_sha = sha256(root / "private/sft_data_v6/train.jsonl")
    val_sha = sha256(root / "private/sft_data_v6/val.jsonl")
    tasks_sha = sha256(root / "private/final_test_v3/tasks.jsonl")
    expected = {
        "train_sha256": TRAIN_SHA256,
        "val_sha256": VAL_SHA256,
        "tasks_sha256": TASKS_SHA256,
    }
    actual = {
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "tasks_sha256": tasks_sha,
    }
    if actual != expected:
        raise ValueError(f"input SHA mismatch: actual={actual} expected={expected}")
    payload = base64.b64encode(zlib.compress(bundle_path.read_bytes(), 9)).decode("ascii")
    code = KERNEL_TEMPLATE.substitute(
        code_b64=payload,
        train_sha256=train_sha,
        val_sha256=val_sha,
        tasks_sha256=tasks_sha,
    )
    (out_dir / "run.py").write_text(code, encoding="utf-8", newline="\n")
    metadata = {
        "id": kernel_id,
        "title": "Lite Agent Clean Heldout v6",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "kernel_id": kernel_id,
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "tasks_sha256": tasks_sha,
        "run_py_bytes": (out_dir / "run.py").stat().st_size,
        "bundle_zip_bytes": bundle_path.stat().st_size,
        "bundle_files": len(files),
    }
    (out_dir / "kernel_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build the clean held-out Kaggle kernel")
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
