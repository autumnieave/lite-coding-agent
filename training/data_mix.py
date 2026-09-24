"""Build the general-data arm for the catastrophic-forgetting comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

GENERAL_SYSTEM = "你是一个简洁、准确的中文/英文通用助手。只回答用户问题，不要调用工具。"


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_dolly_rows(path: Path, *, limit: int = 200) -> list[dict[str, Any]]:
    """Convert Dolly instruction/response pairs to the project's SFT row shape."""

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        instruction = str(source.get("instruction") or "").strip()
        response = str(source.get("response") or "").strip()
        context = str(source.get("context") or "").strip()
        if not instruction or not response:
            continue
        user = instruction if not context else f"{instruction}\n\n参考资料：\n{context}"
        rows.append(
            {
                "id": f"train-v4:general:dolly:{len(rows):04d}",
                "messages": [
                    {"role": "system", "content": GENERAL_SYSTEM},
                    {"role": "user", "content": user},
                ],
                "completion": response,
                "metadata": {
                    "source": "dolly_general",
                    "source_dataset": "databricks/databricks-dolly-15k",
                    "license": "CC-BY-SA-3.0",
                    "category": "GENERAL",
                    "dimension": "general",
                    "difficulty": "general",
                    "context_tier": "mixed",
                    "split": "general",
                    "source_category": str(source.get("category") or ""),
                },
            }
        )
        if len(rows) >= limit:
            break
    if len(rows) < limit:
        raise ValueError(f"requested {limit} Dolly rows but found only {len(rows)}")
    return rows


def mix_train_file(
    *,
    base_train_file: Path,
    general_file: Path,
    out_file: Path,
    general_ratio: float,
    seed: int = 20261019,
) -> dict[str, Any]:
    """Append a deterministic general-data sample to a constraint-heavy train file."""

    if not 0.0 < general_ratio < 0.5:
        raise ValueError("general_ratio must be in (0, 0.5)")
    base_rows = [
        json.loads(line)
        for line in base_train_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    general_rows = [
        json.loads(line)
        for line in general_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not base_rows or not general_rows:
        raise ValueError("base and general train files must both be non-empty")
    count = min(len(general_rows), round(len(base_rows) * general_ratio / (1 - general_ratio)))
    if count < 1:
        raise ValueError("general mix resolved to zero rows")
    sampled = random.Random(seed).sample(general_rows, count)
    rows = [*base_rows, *sampled]
    _write_jsonl(out_file, rows)
    payload = out_file.read_bytes()
    return {
        "base_rows": len(base_rows),
        "general_available": len(general_rows),
        "general_sampled": len(sampled),
        "total_rows": len(rows),
        "requested_general_ratio": general_ratio,
        "actual_general_ratio": round(len(sampled) / len(rows), 4),
        "seed": seed,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "out_file": str(out_file),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build Dolly data and a mixed train file")
    parser.add_argument(
        "--dolly-file",
        type=Path,
        default=Path("private/public_data_raw/databricks-dolly-15k.jsonl"),
    )
    parser.add_argument(
        "--general-out",
        type=Path,
        default=Path("private/sft_data_v4/general_dolly.jsonl"),
    )
    parser.add_argument("--general-limit", type=int, default=200)
    parser.add_argument(
        "--general-file",
        type=Path,
        default=None,
        help="reuse an already-built general JSONL instead of rebuilding Dolly rows",
    )
    parser.add_argument(
        "--base-train-file",
        type=Path,
        default=Path("private/sft_data_v4/train.jsonl"),
    )
    parser.add_argument(
        "--mixed-out",
        "--out-file",
        dest="mixed_out",
        type=Path,
        default=Path("private/sft_data_v4/train_mix15.jsonl"),
    )
    parser.add_argument("--general-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261019)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.general_file is not None:
        if not args.general_file.is_file():
            raise FileNotFoundError(f"general file not found: {args.general_file}")
        general_path = args.general_file
        general_rows = sum(
            1 for line in general_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    else:
        rows = build_dolly_rows(args.dolly_file, limit=args.general_limit)
        _write_jsonl(args.general_out, rows)
        general_path = args.general_out
        general_rows = len(rows)
    manifest = mix_train_file(
        base_train_file=args.base_train_file,
        general_file=general_path,
        out_file=args.mixed_out,
        general_ratio=args.general_ratio,
        seed=args.seed,
    )
    manifest["general_rows"] = general_rows
    manifest["general_sha256"] = hashlib.sha256(general_path.read_bytes()).hexdigest()
    (args.mixed_out.parent / "mix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
