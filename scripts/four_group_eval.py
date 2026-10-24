"""Config-driven entry point for the four pre-registered comparison groups.

Only S0/S1/S2 system-side groups are executable today. Training groups are
pre-registered but intentionally fail closed until a real adapter/checkpoint is
available; this prevents an accidental baseline being reported as training.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from difficulty_tuning import build_parser, main_async  # noqa: E402

from training.system_variants import get_variant  # noqa: E402

GROUP_TO_VARIANT = {
    "S0": "s0",
    "S1": "s1t",
    "S2": "s2",
}
TRAINING_GROUPS = {"SFT", "S1+SFT", "S2+SFT"}


def build_parser_four() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="four-group comparison runner")
    parser.add_argument(
        "--group",
        required=True,
        choices=sorted({*GROUP_TO_VARIANT, *TRAINING_GROUPS}),
    )
    parser.add_argument("--provider", choices=("server", "offline"), default="server")
    parser.add_argument("--spec-set", choices=("tuning", "final-test"), default="final-test")
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("private/four_group"))
    parser.add_argument("--trace-dir", type=Path, default=Path("private/traces/four_group"))
    return parser


def main() -> int:
    args = build_parser_four().parse_args()
    if args.group in TRAINING_GROUPS:
        print(
            f"{args.group}: training checkpoint/adapter is not available; "
            "pre-registered but not runnable",
            file=sys.stderr,
        )
        return 2
    variant = get_variant(GROUP_TO_VARIANT[args.group])
    forwarded = [
        "--spec-set",
        args.spec_set,
        "--rollouts",
        str(args.rollouts),
        "--concurrency",
        str(args.concurrency),
        "--provider",
        args.provider,
        "--system-variant",
        variant.id,
        "--out-dir",
        str(args.out_dir / args.group.lower()),
        "--trace-dir",
        str(args.trace_dir / args.group.lower()),
    ]
    if args.limit:
        forwarded += ["--limit", str(args.limit)]
    parsed = build_parser().parse_args(forwarded)
    return asyncio.run(main_async(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
