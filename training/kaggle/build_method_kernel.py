"""Build a self-contained Kaggle kernel for one method condition."""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path
from string import Template


def build_kernel(
    *,
    root: Path,
    bundle_zip: Path,
    out_dir: Path,
    kernel_id: str,
    experiment: str,
    train_version: str = "train-v4",
) -> dict[str, object]:
    payload = base64.b64encode(zlib.compress(bundle_zip.read_bytes(), 9)).decode("ascii")
    template = Template(
        """from __future__ import annotations

import base64
import gc
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
EXPERIMENT = "$experiment"
KERNEL_RUN = "$kernel_run"
REPO = Path("/kaggle/working/repo")
MODEL = Path("/kaggle/temp/qwen2.5-7b-instruct")
OUT = Path("/kaggle/working/method") / EXPERIMENT
DATA = REPO / "$data_dir"
TRAIN_VERSION = "$train_version"
TASKS = (
    "U-TR-A-C1-HARD,U-TR-B-CHECK-BASE,U-TR-B-C1-BASE,"
    "U-TR-B-CTX-HARD,U-TR-B-TOOL-HARD,U-TR-B-CONFLICT-WEAK"
)
SEED = 20261019
# Held-out task specs stay on the historical train-v2 definitions so that this
# run is comparable with earlier 7B anchors. This is the spec source version,
# not the training-data version recorded in TRAIN_VERSION.
HELDOUT_SPEC_VERSION = "train-v2"


def status(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    Path("/kaggle/working/kernel_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def gpu_report() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        return {
            "cuda": True,
            "device": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        }
    except Exception as exc:
        return {"cuda": False, "error": str(exc)}


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


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


def extract_bundle() -> None:
    REPO.mkdir(parents=True, exist_ok=True)
    blob = Path("/kaggle/working/method_bundle.zip")
    blob.write_bytes(zlib.decompress(base64.b64decode(CODE_B64)))
    with zipfile.ZipFile(blob) as archive:
        archive.extractall(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def install_and_model() -> None:
    status({"status": "running", "stage": "install", "experiment": EXPERIMENT})
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
        status({"status": "running", "stage": "download_model", "experiment": EXPERIMENT})
        run(
            [
                sys.executable,
                "-c",
                "from huggingface_hub import snapshot_download;"
                f"snapshot_download('Qwen/Qwen2.5-7B-Instruct', "
                f"local_dir='{MODEL}', max_workers=4)",
            ],
            3600,
            "download_model",
        )


def tokenizer_gate() -> dict:
    gate_path = OUT / "tokenizer_gate.json"
    run(
        [
            sys.executable,
            "-m",
            "training.validate_train_v4",
            "--train-file",
            str(DATA / "train.jsonl"),
            "--val-file",
            str(DATA / "val.jsonl"),
            "--tokenizer-path",
            str(MODEL),
            "--max-length",
            "3072",
            "--out",
            str(gate_path),
        ],
        1200,
        "tokenizer_gate",
    )
    result = json.loads(gate_path.read_text(encoding="utf-8"))
    if result.get("status") != "ok":
        raise RuntimeError(f"tokenizer gate failed: {result}")
    return result


def train_sft(
    *,
    train_file: Path,
    output_dir: Path,
    loss_mode: str,
    seed: int,
    label: str,
) -> None:
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
            str(train_file),
            "--val-file",
            str(DATA / "val.jsonl"),
            "--output-dir",
            str(output_dir),
            "--loss-mode",
            loss_mode,
            "--seed",
            str(seed),
            "--heldout-out-root",
            str(OUT / "unused_heldout"),
            "--heldout-trace-root",
            str(OUT / "unused_traces"),
            "--heldout-tasks",
            TASKS,
        ],
        14400,
        label,
    )


def train_dpo(*, sft_adapter: Path, output_dir: Path) -> None:
    run(
        [
            sys.executable,
            str(REPO / "training/kaggle/method_runner.py"),
            "--repo",
            str(REPO),
            "--mode",
            "dpo",
            "--config",
            str(REPO / "training/configs/dpo_7b_t4.json"),
            "--model-path",
            str(MODEL),
            "--train-file",
            str(REPO / "private/dpo_v1/dpo_train.jsonl"),
            "--val-file",
            str(REPO / "private/dpo_v1/dpo_val.jsonl"),
            "--init-adapter",
            str(sft_adapter),
            "--output-dir",
            str(output_dir),
            "--heldout-out-root",
            str(OUT / "unused_heldout"),
            "--heldout-trace-root",
            str(OUT / "unused_traces"),
            "--heldout-tasks",
            TASKS,
        ],
        14400,
        "train_dpo",
    )


def evaluate(*, adapter: Path, baseline_adapter: Path | None = None) -> dict:
    command = [
        sys.executable,
        str(REPO / "training/kaggle/heldout_compare.py"),
        "--repo",
        str(REPO),
        "--model",
        str(MODEL),
        "--val-file",
        str(DATA / "val.jsonl"),
        "--adapter-path",
        str(adapter),
        "--out-root",
        str(OUT / "heldout"),
        "--trace-root",
        str(OUT / "heldout_traces"),
        "--rollouts",
        "10",
        "--train-version",
        HELDOUT_SPEC_VERSION,
        "--provider",
        "transformers",
        "--quantization",
        "nf4",
        "--max-model-len",
        "8192",
        "--tasks",
        TASKS,
    ]
    if baseline_adapter is not None:
        command += ["--baseline-adapter", str(baseline_adapter)]
    run(command, 14400, "heldout_compare")
    return json.loads((OUT / "heldout/comparison.json").read_text(encoding="utf-8"))


def build_mix_file() -> Path:
    out = DATA / "train_mix15.jsonl"
    run(
        [
            sys.executable,
            "-m",
            "training.data_mix",
            "--base-train-file",
            str(DATA / "train.jsonl"),
            "--general-file",
            str(DATA / "general_dolly.jsonl"),
            "--out-file",
            str(out),
            "--general-ratio",
            "0.15",
            "--seed",
            str(SEED),
        ],
        300,
        "build_mix",
    )
    return out


def main() -> int:
    extract_bundle()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    gpu = gpu_report()
    install_and_model()
    gate = tokenizer_gate()
    status(
        {
            "status": "running",
            "stage": "tokenizer_gate_done",
            "experiment": EXPERIMENT,
            "gate": gate,
            "gpu": gpu,
        }
    )

    adapter = OUT / "adapter"
    summary_adapter = adapter
    comparison = None
    if EXPERIMENT == "baseline":
        train_sft(
            train_file=DATA / "train.jsonl",
            output_dir=adapter,
            loss_mode="full",
            seed=SEED,
            label="train_sft_baseline",
        )
        comparison = evaluate(adapter=adapter)
    elif EXPERIMENT == "baseline_seed2":
        train_sft(
            train_file=DATA / "train.jsonl",
            output_dir=adapter,
            loss_mode="full",
            seed=20261020,
            label="train_sft_baseline_seed2",
        )
        comparison = evaluate(adapter=adapter)
    elif EXPERIMENT == "assistant_only":
        train_sft(
            train_file=DATA / "train.jsonl",
            output_dir=adapter,
            loss_mode="assistant_only",
            seed=SEED,
            label="train_sft_assistant_only",
        )
        comparison = evaluate(adapter=adapter)
    elif EXPERIMENT == "mix15":
        mix_file = build_mix_file()
        train_sft(
            train_file=mix_file,
            output_dir=adapter,
            loss_mode="full",
            seed=SEED,
            label="train_sft_mix15",
        )
        comparison = evaluate(adapter=adapter)
    elif EXPERIMENT == "dpo":
        sft_adapter = OUT / "sft_adapter"
        dpo_adapter = OUT / "dpo_adapter"
        summary_adapter = dpo_adapter
        train_sft(
            train_file=DATA / "train.jsonl",
            output_dir=sft_adapter,
            loss_mode="full",
            seed=SEED,
            label="train_sft_for_dpo",
        )
        train_dpo(sft_adapter=sft_adapter, output_dir=dpo_adapter)
        comparison = evaluate(adapter=dpo_adapter, baseline_adapter=sft_adapter)
    else:
        raise ValueError(f"unknown experiment: {EXPERIMENT}")

    gc.collect()
    train_manifest_path = summary_adapter / "run_manifest.json"
    train_manifest = (
        json.loads(train_manifest_path.read_text(encoding="utf-8"))
        if train_manifest_path.is_file()
        else {}
    )
    summary = {
        "status": "ok",
        "experiment": EXPERIMENT,
        "kernel_run": KERNEL_RUN,
        "gpu": gpu,
        "duration_s": round(time.perf_counter() - started, 1),
        "train_version": TRAIN_VERSION,
        "train_sha256": sha256(DATA / "train.jsonl"),
        "val_sha256": sha256(DATA / "val.jsonl"),
        "train_manifest": train_manifest,
        "heldout": comparison,
        "adapter_sha256": sha256(summary_adapter / "adapter_model.safetensors"),
        "adapter_bytes": (
            (summary_adapter / "adapter_model.safetensors").stat().st_size
            if (summary_adapter / "adapter_model.safetensors").is_file()
            else 0
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
    )
    status({"status": "ok", "stage": "done", "summary": summary})
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
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_zip = out_dir / "repo.zip"
    repo_zip.write_bytes(bundle_zip.read_bytes())
    data_dir = "private/sft_data_v5" if train_version == "train-v5" else "private/sft_data_v4"
    kernel_run = f"{experiment}-{train_version}"
    code = template.substitute(
        code_b64=payload,
        experiment=experiment,
        kernel_run=kernel_run,
        train_version=train_version,
        data_dir=data_dir,
    )
    (out_dir / "run.py").write_text(code, encoding="utf-8")
    title_experiment = experiment.replace("_only", "")
    metadata = {
        "id": kernel_id,
        "title": f"Lite Agent Method {title_experiment} {train_version}",
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
        "experiment": experiment,
        "train_version": train_version,
        "run_py_bytes": (out_dir / "run.py").stat().st_size,
        "bundle_zip_bytes": repo_zip.stat().st_size,
        "root": str(root),
    }
    (out_dir / "kernel_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build a train-v4 Kaggle method kernel")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--bundle-zip", type=Path, default=Path("private/kaggle_method_bundle_min/repo.zip")
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument(
        "--experiment",
        choices=("baseline", "baseline_seed2", "assistant_only", "mix15", "dpo"),
        required=True,
    )
    parser.add_argument("--train-version", choices=("train-v4", "train-v5"), default="train-v4")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_kernel(
        root=args.root.resolve(),
        bundle_zip=args.bundle_zip.resolve(),
        out_dir=args.out_dir.resolve(),
        kernel_id=args.kernel_id,
        experiment=args.experiment,
        train_version=args.train_version,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
