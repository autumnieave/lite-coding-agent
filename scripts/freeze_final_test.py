"""Freeze the final held-out test manifest without running it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.final_test import freeze_final_test  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="freeze final test manifest")
    parser.add_argument("--out-dir", type=Path, default=Path("private/unified_dataset"))
    args = parser.parse_args()
    result = freeze_final_test(args.out_dir, root=Path.cwd())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
