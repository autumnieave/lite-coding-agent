"""Kaggle T4 Outlines constrained decoding baseline.

Input JSONL: one record per rollout, with `run_id`, `rollout_index`, `prompt`.
The script records every prompt, output, schema verdict, duration and metadata. It does not
claim results when run in `--dry-run` mode.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.constrained_decoding import (  # noqa: E402
    build_summary_schema,
    run_with_outlines,
)
from training.eval_metrics import (  # noqa: E402
    summarize_by_run,
    summarize_generation_records,
)
from training.kaggle.contracts import (  # noqa: E402
    DEFAULT_INFERENCE_TIMEOUT_S,
    DEFAULT_LOAD_TIMEOUT_S,
    DEFAULT_PROGRESS_EVERY,
    ContractError,
    append_jsonl,
    classify_exception,
    completed_keys,
    latest_records,
    read_jsonl,
    require_nonempty_file,
    runtime_info,
    validate_summary,
    write_json,
)

SUMMARY_REQUIRED = (
    "status",
    "backend",
    "model_path",
    "prompts_requested",
    "prompts_completed",
    "records",
    "overall",
    "by_run",
    "metadata",
    "error_kind",
    "staging",
)


def load_prompts(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("run_id", "rollout_index", "prompt"):
            if key not in row:
                raise ValueError(f"prompt 记录缺少字段 {key}: {row}")
        rows.append(row)
    return rows


def resolve_model_path(
    candidate: Path,
    search_root: Path | None = None,
    *,
    preferred_model: str = "Qwen2.5-Coder-7B-Instruct",
) -> Path:
    """Resolve a mounted model directory without treating it as a Hub repo id."""

    if (candidate / "config.json").is_file():
        return candidate
    root = search_root or Path("/kaggle/input")
    if not root.is_dir():
        raise FileNotFoundError(f"模型路径不存在：{candidate}；搜索根目录也不存在：{root}")
    configs = sorted({path.parent for path in root.rglob("config.json")})
    if not configs:
        entries = sorted(path.name for path in root.iterdir())
        raise FileNotFoundError(
            f"模型路径不存在：{candidate}；{root} 下没有 config.json。目录项：{entries}"
        )
    preferred_key = preferred_model.lower().replace(".", "-")
    preferred = [
        path for path in configs if preferred_key in path.as_posix().lower().replace(".", "-")
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(configs) == 1:
        return configs[0]
    raise FileNotFoundError(
        f"找到多个候选模型目录，无法唯一确定：{[str(path) for path in configs]}"
    )


def model_version_from_path(model_path: Path) -> str:
    """Return the mounted model identity without hard-coding the 7B variant."""

    name = model_path.name
    if name.startswith("Qwen"):
        return f"Qwen/{name}"
    return name


def gpu_info() -> dict[str, object]:
    """Best-effort GPU metadata; preflight must still work without torch."""

    try:
        import torch
    except ImportError as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if not torch.cuda.is_available():
        return {"available": False, "error": "CUDA 不可用"}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    }


def apply_staging_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Make staging a bounded three-record run unless the caller overrides it."""

    if args.staging:
        if args.limit <= 0:
            args.limit = 3
        if args.max_new_tokens > 256:
            args.max_new_tokens = 256
    return args


def prepare_run(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object], Path]:
    """Validate input and model contracts without loading model weights."""

    prompts_path = Path(args.prompts_file)
    prompt_bytes = require_nonempty_file(prompts_path, "constrained prompts")
    prompts = load_prompts(prompts_path)
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        raise ContractError("input_empty", "筛选后没有可运行的 prompt")

    try:
        model_path = resolve_model_path(
            Path(args.model_path),
            Path(args.search_root),
            preferred_model=args.preferred_model,
        )
    except FileNotFoundError as exc:
        raise ContractError("model_path_missing", str(exc)) from exc
    config_bytes = require_nonempty_file(
        model_path / "config.json",
        "model config.json",
        missing_kind="model_path_missing",
        empty_kind="model_config_empty",
    )
    tokenizer_bytes = require_nonempty_file(
        model_path / "tokenizer_config.json",
        "tokenizer_config.json",
        missing_kind="tokenizer_config_missing",
        empty_kind="tokenizer_config_empty",
    )
    report: dict[str, object] = {
        "status": "ok",
        "stage": "preflight",
        "model_path": str(model_path),
        "model_version": model_version_from_path(model_path),
        "model_config_bytes": config_bytes,
        "tokenizer_config_bytes": tokenizer_bytes,
        "prompts_file": str(prompts_path),
        "prompts_bytes": prompt_bytes,
        "prompts": len(prompts),
        "runs": sorted({int(row["run_id"]) for row in prompts}),
        "runtime": runtime_info(),
        "gpu": gpu_info(),
        "staging": bool(args.staging),
        "limit": args.limit,
        "load_timeout_s": args.load_timeout,
        "inference_timeout_s": args.inference_timeout,
    }
    return prompts, report, model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kaggle Outlines baseline")
    parser.add_argument(
        "--prompts-file", default="/kaggle/input/lite-agent-training/constrained_prompts.jsonl"
    )
    parser.add_argument("--model-path", default="/kaggle/input/qwen2.5-coder-7b-instruct")
    parser.add_argument("--out", default="/kaggle/working/constrained_decoding_results.jsonl")
    parser.add_argument(
        "--summary-out", default="/kaggle/working/constrained_decoding_summary.json"
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--search-root", default="/kaggle/input")
    parser.add_argument("--preferred-model", default="Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--load-timeout", type=float, default=DEFAULT_LOAD_TIMEOUT_S)
    parser.add_argument("--inference-timeout", type=float, default=DEFAULT_INFERENCE_TIMEOUT_S)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--preflight-out", default="/kaggle/working/constrained_decoding_preflight.json"
    )
    parser.add_argument("--staging", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_summary(
    *,
    status: str,
    error_kind: str | None,
    model_path: Path | None,
    prompts_requested: int,
    records: Sequence[dict[str, object]],
    preflight: dict[str, object],
    started: float,
    args: argparse.Namespace,
    resumed_prompts: int = 0,
) -> dict[str, object]:
    return {
        "status": status,
        "error_kind": error_kind,
        "model_path": str(model_path) if model_path is not None else args.model_path,
        "backend": "outlines",
        "prompts_requested": prompts_requested,
        "prompts_completed": len(records),
        "records": len(records),
        "overall": summarize_generation_records(records),
        "by_run": summarize_by_run(records),
        "staging": bool(args.staging),
        "runtime": runtime_info(),
        "preflight": preflight,
        "metadata": {
            "model_version": model_version_from_path(model_path) if model_path else None,
            "system_fingerprint": "N/A (local outlines; no API)",
            "response_id": None,
            "gpu": preflight.get("gpu"),
            "duration_s": round(time.perf_counter() - started, 2),
            "rollouts": len(records),
            "resumed_prompts": resumed_prompts,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "load_timeout_s": args.load_timeout,
            "inference_timeout_s": args.inference_timeout,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = apply_staging_defaults(build_parser().parse_args(argv))
    started = time.perf_counter()
    if args.dry_run:
        try:
            prompt_bytes = require_nonempty_file(Path(args.prompts_file), "constrained prompts")
            prompts = load_prompts(Path(args.prompts_file))
            if args.limit:
                prompts = prompts[: args.limit]
            if not prompts:
                raise ContractError("input_empty", "筛选后没有可运行的 prompt")
        except ContractError as exc:
            print(
                json.dumps(
                    {"status": "dry_run_error", "error_kind": exc.error_kind, "error": str(exc)},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "model_path": args.model_path,
                    "prompts": len(prompts),
                    "prompt_bytes": prompt_bytes,
                    "runs": sorted({int(row["run_id"]) for row in prompts}),
                    "schema_keys": sorted(build_summary_schema()["properties"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    try:
        prompts, preflight, model_path = prepare_run(args)
    except ContractError as exc:
        report = {
            "status": "error",
            "stage": "preflight",
            "error_kind": exc.error_kind,
            "error": str(exc),
            "runtime": runtime_info(),
            "staging": bool(args.staging),
        }
        write_json(Path(args.preflight_out), report)
        summary = build_summary(
            status="error",
            error_kind=exc.error_kind,
            model_path=None,
            prompts_requested=0,
            records=(),
            preflight=report,
            started=started,
            args=args,
        )
        write_json(Path(args.summary_out), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1

    write_json(Path(args.preflight_out), preflight)
    print(json.dumps({"preflight": preflight}, ensure_ascii=False), flush=True)
    if args.preflight_only:
        return 0

    out = Path(args.out)
    if args.overwrite:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
    existing_records = [] if args.overwrite else read_jsonl(out)
    done = completed_keys(out, ("run_id", "rollout_index"))
    pending = [
        row for row in prompts if (int(row["run_id"]), int(row["rollout_index"])) not in done
    ]
    if not pending:
        summary = build_summary(
            status="completed",
            error_kind=None,
            model_path=model_path,
            prompts_requested=len(prompts),
            records=latest_records(existing_records, ("run_id", "rollout_index")),
            preflight=preflight,
            started=started,
            args=args,
            resumed_prompts=len(existing_records),
        )
        validate_summary(summary, SUMMARY_REQUIRED)
        write_json(Path(args.summary_out), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    try:
        new_records = run_with_outlines(
            pending,
            model_path=str(model_path),
            max_new_tokens=args.max_new_tokens,
            quantize_4bit=args.quantize_4bit,
            on_record=lambda record: append_jsonl(out, record),
            load_timeout_s=args.load_timeout,
            inference_timeout_s=args.inference_timeout,
            progress_every=args.progress_every,
        )
    except BaseException as exc:  # noqa: BLE001 - persist the failure category
        error_kind = classify_exception(exc)
        summary = build_summary(
            status="error",
            error_kind=error_kind,
            model_path=model_path,
            prompts_requested=len(prompts),
            records=existing_records,
            preflight=preflight,
            started=started,
            args=args,
            resumed_prompts=len(existing_records),
        )
        summary["error"] = f"{type(exc).__name__}: {exc}"
        write_json(Path(args.summary_out), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1

    all_records = latest_records([*existing_records, *new_records], ("run_id", "rollout_index"))
    error_kinds = sorted({str(row["error_kind"]) for row in all_records if row.get("error_kind")})
    if not all_records:
        status = "error"
        error_kind: str | None = "output_empty"
    elif error_kinds:
        status = "partial"
        error_kind = error_kinds[0]
    else:
        status = "completed"
        error_kind = None
    summary = build_summary(
        status=status,
        error_kind=error_kind,
        model_path=model_path,
        prompts_requested=len(prompts),
        records=all_records,
        preflight=preflight,
        started=started,
        args=args,
        resumed_prompts=len(existing_records),
    )
    if error_kinds:
        summary["error_kinds"] = error_kinds
    validate_summary(summary, SUMMARY_REQUIRED)
    write_json(Path(args.summary_out), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
