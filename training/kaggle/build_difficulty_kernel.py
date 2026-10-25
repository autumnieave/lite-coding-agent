"""Build a Kaggle T4 kernel for one shard of the 409-task difficulty screen."""

from __future__ import annotations

import argparse
import base64
import json
import zipfile
import zlib
from pathlib import Path
from string import Template

KERNEL_TEMPLATE = Template(
    """from __future__ import annotations

import base64
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
SHARD_INDEX = $shard_index
SHARD_COUNT = $shard_count
RUN_STAGING = $run_staging
REPO = Path("/kaggle/working/repo")
MODEL = Path("/kaggle/temp/$model_dir")
MODEL_ID = "$model_id"
OUT = Path("/kaggle/working/difficulty")
MAX_MODEL_LEN = $max_model_len
MAX_TOKENS = $max_tokens
STAGING_TASKS = $staging_tasks
FULL_SCREEN = $full_screen
TEMPERATURE = 0.7
TOP_P = 0.9
BASE_SEED = 20261019
ROLLOUTS = 8
FULL_TIMEOUT = $full_timeout
TASK_IDS = "$task_ids"


def status(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    Path("/kaggle/working/kernel_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


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


def rebuild_partial(label: str) -> None:
    # Reconstruct the summary from appended rollouts after a session timeout.
    status({"status": "degraded", "stage": "rebuild_partial", "label": label})
    out_dir = OUT / label
    run(
        [
            sys.executable,
            "-m",
            "scripts.rebuild_difficulty_summary",
            "--runs",
            str(out_dir / "runs.jsonl"),
            "--out-dir",
            str(out_dir),
            "--train-file",
            str(REPO / "private/sft_data_v5/train.jsonl"),
            "--shard-index",
            str(SHARD_INDEX),
            "--shard-count",
            str(SHARD_COUNT),
            "--rollouts",
            str(ROLLOUTS),
        ],
        1800,
        "rebuild_partial",
    )


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
                f"snapshot_download({MODEL_ID!r}, "
                f"local_dir={str(MODEL)!r}, max_workers=4)",
            ],
            3600,
            "download_model",
        )


def base_command(label: str) -> list[str]:
    out = OUT / label
    traces = OUT / "traces" / label
    return [
        sys.executable,
        "-m",
        "training.kaggle.difficulty_screen",
        "--train-file",
        str(REPO / "private/sft_data_v5/train.jsonl"),
        "--model-path",
        str(MODEL),
        "--out-dir",
        str(out),
        "--trace-dir",
        str(traces),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-tokens",
        str(MAX_TOKENS),
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


def main() -> int:
    extract_bundle()
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = {"device": "unknown"}
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

    if RUN_STAGING:
        status({"status": "running", "stage": "staging", "gpu": gpu})
        staging = base_command("staging")
        staging += ["--staging", "--staging-tasks", str(STAGING_TASKS)]
        run(staging, 2400, "staging")
        staging_summary = json.loads(
            (OUT / "staging/difficulty_summary.json").read_text(encoding="utf-8")
        )
        baseline = staging_summary["baseline"]
        staging_ok = (
            baseline.get("status") == "completed"
            and int(baseline.get("errors") or 0) == 0
            and int(baseline.get("task_success") or 0) >= 1
        )
        status(
            {
                "status": "running" if staging_ok else "degraded",
                "stage": "staging_result",
                "staging_ok": staging_ok,
                "staging": baseline,
            }
        )
        if not staging_ok:
            return 2
        if not FULL_SCREEN:
            status({"status": "ok", "stage": "preflight_done", "gpu": gpu, "staging": baseline})
            return 0

    status(
        {
            "status": "running",
            "stage": "full_shard",
            "gpu": gpu,
            "shard": {"index": SHARD_INDEX, "count": SHARD_COUNT},
            "sampling": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "base_seed": BASE_SEED,
                "seed_policy": "base_seed_plus_rollout_index",
                "rollouts": ROLLOUTS,
            },
        }
    )
    label = f"shard-{SHARD_INDEX:02d}-of-{SHARD_COUNT:02d}"
    full = base_command(label)
    full += [
        "--shard-index",
        str(SHARD_INDEX),
        "--shard-count",
        str(SHARD_COUNT),
        "--rollouts",
        str(ROLLOUTS),
        "--resume",
    ]
    if TASK_IDS:
        ids_path = Path("/kaggle/working/task_ids.txt")
        ids_path.write_text(
            "\\n".join(item.strip() for item in TASK_IDS.split(",") if item.strip()) + "\\n",
            encoding="utf-8",
        )
        full += ["--task-ids-file", str(ids_path)]
    degraded = False
    try:
        run(full, FULL_TIMEOUT, "full_shard")
    except RuntimeError as exc:
        if "full_shard exceeded" not in str(exc):
            raise
        degraded = True
        rebuild_partial(label)
    summary = json.loads(
        (OUT / f"{label}/difficulty_summary.json").read_text(encoding="utf-8")
    )
    destination = Path("/kaggle/working/difficulty_shard_summary.json")
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
    )
    status(
        {
            "status": "degraded" if degraded else "ok",
            "stage": "done_partial" if degraded else "done",
            "gpu": gpu,
            "shard": {"index": SHARD_INDEX, "count": SHARD_COUNT},
            "middle_state": summary["baseline"].get("middle_state"),
        }
    )
    return 3 if degraded else 0


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

PY_ROOTS = ("src", "scripts", "training")
EXTRA_FILES = (
    "pyproject.toml",
    "training/requirements-kaggle.txt",
    "private/sft_data_v5/train.jsonl",
)


def _python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for relative in PY_ROOTS:
        for path in sorted((root / relative).rglob("*.py")):
            if "__pycache__" not in path.parts:
                files.append(path)
    return files


def build_bundle(root: Path, bundle_path: Path) -> list[Path]:
    files = _python_files(root) + [root / relative for relative in EXTRA_FILES]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"difficulty bundle inputs missing: {missing}")
    files = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    with zipfile.ZipFile(
        bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    return files


def build_kernel(
    *,
    root: Path,
    out_dir: Path,
    kernel_id: str,
    shard_index: int,
    shard_count: int,
    title: str,
    model_id: str,
    model_dir: str,
    max_model_len: int,
    max_tokens: int,
    staging_tasks: int,
    full_screen: bool,
    full_timeout: int = 27000,
    task_ids: str = "",
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "repo.zip"
    files = build_bundle(root, bundle_path)
    payload = base64.b64encode(zlib.compress(bundle_path.read_bytes(), 9)).decode("ascii")
    code = KERNEL_TEMPLATE.substitute(
        code_b64=payload,
        shard_index=shard_index,
        shard_count=shard_count,
        run_staging="True" if shard_index == 0 else "False",
        model_id=model_id,
        model_dir=model_dir,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        staging_tasks=staging_tasks,
        full_screen="True" if full_screen else "False",
        full_timeout=full_timeout,
        task_ids=task_ids,
    )
    (out_dir / "run.py").write_text(code, encoding="utf-8", newline="\n")
    metadata = {
        "id": kernel_id,
        "title": title,
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
        "shard_index": shard_index,
        "shard_count": shard_count,
        "run_staging": shard_index == 0,
        "model_id": model_id,
        "model_dir": model_dir,
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        "staging_tasks": staging_tasks,
        "full_screen": full_screen,
        "full_timeout_s": full_timeout,
        "task_ids_count": len([x for x in task_ids.split(",") if x.strip()]),
        "run_py_bytes": (out_dir / "run.py").stat().st_size,
        "bundle_zip_bytes": bundle_path.stat().st_size,
        "bundle_files": len(files),
    }
    (out_dir / "kernel_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build one difficulty-screen Kaggle shard")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=3)
    parser.add_argument("--title", default="Lite Agent Held-out Difficulty Screen")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-dir", default="qwen2.5-7b-instruct")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--staging-tasks", type=int, default=5)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--full-timeout", type=int, default=27000)
    parser.add_argument("--task-ids", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_kernel(
        root=args.root.resolve(),
        out_dir=args.out_dir,
        kernel_id=args.kernel_id,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        title=args.title,
        model_id=args.model_id,
        model_dir=args.model_dir,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        staging_tasks=args.staging_tasks,
        full_screen=not args.preflight_only,
        full_timeout=args.full_timeout,
        task_ids=args.task_ids,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
