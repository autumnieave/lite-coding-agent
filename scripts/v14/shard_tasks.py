"""Split the v14 step-0 task file into balanced shards for parallel workers."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    lines = [
        line for line in args.tasks_file.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.shards):
        shard = lines[index :: args.shards]
        path = args.out_dir / f"tasks_shard{index}.jsonl"
        path.write_text("\n".join(shard) + "\n", encoding="utf-8")
        print(path, len(shard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
