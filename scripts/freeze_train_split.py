"""Freeze the disjoint train split without running any rollout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.train_split import TRAIN_VERSION, TRAIN_VERSIONS, freeze_train_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="freeze train split")
    parser.add_argument("--out-dir", type=Path, default=Path("private/unified_dataset"))
    parser.add_argument("--version", choices=TRAIN_VERSIONS, default=TRAIN_VERSION)
    args = parser.parse_args()
    result = freeze_train_split(args.out_dir, args.version)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
