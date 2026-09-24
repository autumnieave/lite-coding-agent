"""Build the train-v6 GRPO preflight kernel.

The adapter is read from the train-v6 clean-held-out kernel and verified by
SHA-256. A 45-task screen (8 rollouts each) selects tasks in the 2/8–6/8 band;
if at least 10 remain, up to 20 are re-run with a different seed for the
group-variance preflight.
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

TRAIN_V5_SHA256 = "6ee7e79d87cb7c41f4c1b91964d70abcf5e886c5ac75450318760a805b78f830"
TASKS_SHA256 = "20e20b3beabaf0c48c00dae56d97c1aea356b1ac89c4cdb0fd9decc799c4fa5b"
POOL_TASKS_SHA256 = "1f80a63acc59d03262013e4864474f352e7e6c6c53b90abc894957c855dfffdd"
POOL_PROMPTS_SHA256 = "32453d1f860701e334aed9639870e7193998193dcf997e9b37f2068f0eacc3e1"
ADAPTER_SHA256 = "37928a28d67a5a8350f3b49450fc6945c3ffe05c09da7ea0de36136b4a6a4b96"
ADAPTER_SOURCE_KERNEL = "autumnsq/lite-agent-clean-heldout-v6"
PY_ROOTS = ("src", "scripts", "training")
EXTRA_FILES = (
    "pyproject.toml",
    "training/requirements-kaggle.txt",
    "private/sft_data_v5/train.jsonl",
    "private/final_test_v3/tasks.jsonl",
    "private/grpo_middle_pool/prompts.jsonl",
    "private/grpo_middle_pool/task_ids.jsonl",
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
EXPECTED_TRAIN_V5_SHA256 = "$train_v5_sha256"
EXPECTED_TASKS_SHA256 = "$tasks_sha256"
EXPECTED_POOL_SHA256 = "$pool_prompts_sha256"
EXPECTED_ADAPTER_SHA256 = "$adapter_sha256"
REPO = Path("/kaggle/working/repo")
MODEL = Path("/kaggle/temp/qwen2.5-7b-instruct")
OUT = Path("/kaggle/working/grpo_preflight_v6")
TRACE_ROOT = Path("/kaggle/temp/grpo_preflight_v6_traces")
TRAIN_FILE = REPO / "private/sft_data_v5/train.jsonl"
TASKS_FILE = REPO / "private/final_test_v3/tasks.jsonl"
POOL_FILE = REPO / "private/grpo_middle_pool/prompts.jsonl"
BASE_SEED = 20261019
PREFLIGHT_SEED = 20262019
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
    finally:
        print(f"STAGE_SECONDS {stage}={time.perf_counter() - started:.1f}", flush=True)


def verify_inputs() -> dict:
    checks = {
        "train_v5_sha256": sha256(TRAIN_FILE),
        "tasks_sha256": sha256(TASKS_FILE),
        "pool_prompts_sha256": sha256(POOL_FILE),
    }
    expected = {
        "train_v5_sha256": EXPECTED_TRAIN_V5_SHA256,
        "tasks_sha256": EXPECTED_TASKS_SHA256,
        "pool_prompts_sha256": EXPECTED_POOL_SHA256,
    }
    mismatched = [name for name in checks if checks[name] != expected[name]]
    if mismatched:
        raise ValueError(f"input SHA mismatch: {mismatched}; actual={checks}")
    rows = [
        json.loads(line)
        for line in POOL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    task_ids = [str(row.get("task_id") or "") for row in rows]
    if len(rows) != 45 or len(set(task_ids)) != 45 or not all(task_ids):
        raise ValueError("pool must contain 45 unique non-empty task ids")
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
    raise ValueError(f"train-v6 adapter SHA mismatch; checked={checked}")


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


def evaluate(label: str, *, task_ids_file: Path, rollouts: int, seed: int, timeout: float) -> dict:
    status({"status": "running", "stage": f"evaluate_{label}", "rollouts": rollouts, "seed": seed})
    command = [
        sys.executable,
        str(REPO / "training/kaggle/clean_heldout.py"),
        "--train-file",
        str(TRAIN_FILE),
        "--task-ids-file",
        str(task_ids_file),
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
        str(seed),
    ]
    run(command, timeout, label)
    return json.loads((OUT / label / "summary.json").read_text(encoding="utf-8"))


def main() -> int:
    started = time.perf_counter()
    extract_bundle()
    OUT.mkdir(parents=True, exist_ok=True)
    TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    gpu = gpu_report()
    inputs = verify_inputs()
    install_and_model()
    globals()["ADAPTER"] = find_adapter()[0]
    adapter_sha = sha256(Path(ADAPTER) / "adapter_model.safetensors")
    from scripts.grpo_preflight_v6 import (
        decision,
        deterministic_subset,
        middle_tasks,
        variance_rollup,
    )

    staging_ids = OUT / "staging_ids.jsonl"
    staging_ids.write_text(json.dumps({"task_id": inputs["task_ids"][0]}) + "\\n", encoding="utf-8")
    staging = evaluate(
        "staging", task_ids_file=staging_ids, rollouts=8, seed=BASE_SEED, timeout=1800
    )
    staging_tasks = staging.get("condition", {}).get("tasks") or []
    if len(staging_tasks) != 1 or int(staging_tasks[0].get("errors") or 0) != 0:
        summary = {"status": "staging_failed", "gpu": gpu, "inputs": inputs, "staging": staging}
    else:
        screen = evaluate(
            "screen45", task_ids_file=POOL_FILE, rollouts=8, seed=BASE_SEED, timeout=18000
        )
        screen_tasks = screen.get("condition", {}).get("tasks") or []
        middle = middle_tasks(screen_tasks)
        action = decision(len(middle))
        summary = {
            "status": "ok" if len(screen_tasks) == 45 else "degraded",
            "gpu": gpu,
            "inputs": inputs,
            "adapter": {"path": str(ADAPTER), "sha256": adapter_sha},
            "staging": staging,
            "screen": {
                "tasks": len(screen_tasks),
                "middle_count": len(middle),
                "middle_task_ids": [task["task_id"] for task in middle],
                "decision": action,
            },
        }
        if action == "run_preflight":
            selected = deterministic_subset(middle, count=20, seed=BASE_SEED)
            selected_file = OUT / "selected_20.jsonl"
            selected_file.write_text(
                "".join(json.dumps({"task_id": task["task_id"]}) + "\\n" for task in selected),
                encoding="utf-8",
            )
            preflight = evaluate(
                "preflight20",
                task_ids_file=selected_file,
                rollouts=8,
                seed=PREFLIGHT_SEED,
                timeout=9000,
            )
            preflight_tasks = preflight.get("condition", {}).get("tasks") or []
            rollup = variance_rollup(preflight_tasks, expected_tasks=len(selected))
            summary["preflight"] = {
                "selected_tasks": len(selected),
                "selected_task_ids": [task["task_id"] for task in selected],
                "seed": PREFLIGHT_SEED,
                "rollup": rollup,
            }
            summary["decision"] = (
                "grpo_preflight_pass" if rollup["verdict"] == "pass" else "grpo_do_not_start"
            )
        elif action == "insufficient_middle_tasks":
            summary["decision"] = "insufficient_middle_tasks_do_not_start"
        else:
            summary["decision"] = "sft_saturated_grpo_should_not_run"
    summary["duration_s"] = round(time.perf_counter() - started, 1)
    summary["gpu_hours_free_tier"] = round(summary["duration_s"] / 3600, 2)
    (OUT / "grpo_preflight_v6_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
    )
    (Path("/kaggle/working/grpo_preflight_v6_summary.json")).write_text(
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
        raise FileNotFoundError(f"GRPO v6 preflight bundle inputs missing: {missing}")
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
        "train_v5_sha256": sha256(root / "private/sft_data_v5/train.jsonl"),
        "tasks_sha256": sha256(root / "private/final_test_v3/tasks.jsonl"),
        "pool_prompts_sha256": sha256(root / "private/grpo_middle_pool/prompts.jsonl"),
        "pool_tasks_sha256": sha256(root / "private/grpo_middle_pool/task_ids.jsonl"),
    }
    expected = {
        "train_v5_sha256": TRAIN_V5_SHA256,
        "tasks_sha256": TASKS_SHA256,
        "pool_prompts_sha256": POOL_PROMPTS_SHA256,
        "pool_tasks_sha256": POOL_TASKS_SHA256,
    }
    if actual != expected:
        raise ValueError(f"input SHA mismatch: actual={actual} expected={expected}")
    payload = base64.b64encode(zlib.compress(bundle_path.read_bytes(), 9)).decode("ascii")
    code = KERNEL_TEMPLATE.substitute(
        code_b64=payload,
        train_v5_sha256=TRAIN_V5_SHA256,
        tasks_sha256=TASKS_SHA256,
        pool_prompts_sha256=POOL_PROMPTS_SHA256,
        adapter_sha256=ADAPTER_SHA256,
    )
    (out_dir / "run.py").write_text(code, encoding="utf-8", newline="\n")
    metadata = {
        "id": kernel_id,
        "title": "Lite Agent GRPO Preflight V6",
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
        "train_v5_sha256": TRAIN_V5_SHA256,
        "tasks_sha256": TASKS_SHA256,
        "pool_prompts_sha256": POOL_PROMPTS_SHA256,
        "pool_tasks_sha256": POOL_TASKS_SHA256,
        "run_py_bytes": (out_dir / "run.py").stat().st_size,
        "bundle_zip_bytes": bundle_path.stat().st_size,
        "bundle_files": len(files),
    }
    (out_dir / "kernel_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build the train-v6 GRPO preflight kernel")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_kernel(
        root=args.root.resolve(), out_dir=args.out_dir.resolve(), kernel_id=args.kernel_id
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
