"""Shared preflight and runtime contracts for Kaggle jobs."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import platform
import signal
import subprocess
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_LOAD_TIMEOUT_S = 180
DEFAULT_INFERENCE_TIMEOUT_S = 120
DEFAULT_PROGRESS_EVERY = 5


class ContractError(RuntimeError):
    """A failed contract check with a stable machine-readable category."""

    def __init__(self, error_kind: str, message: str) -> None:
        super().__init__(message)
        self.error_kind = error_kind


def classify_exception(exc: BaseException) -> str:
    """Map an exception to the error kinds used by the submission contract."""

    if isinstance(exc, ContractError):
        return exc.error_kind
    if isinstance(exc, FileNotFoundError):
        return "model_path_missing"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, MemoryError):
        return "oom"
    if isinstance(exc, ImportError):
        return "dependency_missing"
    if isinstance(exc, subprocess.CalledProcessError):
        return "child_process_failed"
    return "unknown"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_versions(
    names: Sequence[str] = (
        "vllm",
        "torch",
        "transformers",
        "outlines",
        "bitsandbytes",
        "openai",
    ),
) -> dict[str, str | None]:
    return {name: package_version(name) for name in names}


def runtime_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": runtime_versions(),
    }


def require_nonempty_file(
    path: Path,
    label: str,
    *,
    missing_kind: str = "input_missing",
    empty_kind: str = "input_empty",
) -> int:
    """Return the byte size, or raise a categorized contract error."""

    if not path.is_file():
        raise ContractError(missing_kind, f"{label} 不存在：{path}")
    size = path.stat().st_size
    if size <= 0:
        raise ContractError(empty_kind, f"{label} 为空：{path}")
    return size


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one record and flush immediately so a timeout leaves evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def completed_keys(path: Path, keys: Sequence[str]) -> set[tuple[Any, ...]]:
    """Load composite keys that completed successfully.

    Rows carrying ``error_kind`` are retried on the next run so an append-only
    result file can recover from a timeout without losing its earlier rows.
    """

    found: set[tuple[Any, ...]] = set()
    for row in read_jsonl(path):
        if all(key in row for key in keys) and not row.get("error_kind"):
            found.add(tuple(row[key] for key in keys))
    return found


def latest_records(rows: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    """Collapse append-only reruns so the last row for a key wins."""

    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if all(key in row for key in keys):
            merged[tuple(row[key] for key in keys)] = dict(row)
    return list(merged.values())


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_summary(summary: Mapping[str, Any], required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(summary))
    if missing:
        raise ContractError(
            "summary_schema_invalid",
            f"summary 缺少必需字段：{missing}",
        )


@contextlib.contextmanager
def time_limit(seconds: float | int | None, error_kind: str) -> Iterator[None]:
    """Raise ``ContractError`` after ``seconds`` on POSIX main threads.

    Kaggle is Linux, so model loading and generation can be interrupted with
    ``SIGALRM``.  Windows and worker threads have no portable preemptive timer;
    there the context manager is a no-op and callers still record elapsed time.
    """

    if not seconds or seconds <= 0:
        yield
        return
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise(signum: int, frame: object) -> None:
        raise ContractError(error_kind, f"操作超过 {float(seconds):.0f}s 未完成")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
