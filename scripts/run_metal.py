#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc_bind.cli import build_parser, paths_from_args
from lc_bind.models.metal import run_v1, run_v2


def main():
    parser = build_parser(
        "GEAR metal-binding experiments. Requires --gear-dir.",
        gear=True,
    )
    parser.add_argument(
        "--version",
        choices=("v1", "v2"),
        default="v2",
        help="v1 is GEAR Ultimate metal; v2 is the metal-only screen.",
    )
    args = parser.parse_args()
    runner = run_v1 if args.version == "v1" else run_v2
    runner(paths_from_args(args))


if __name__ == "__main__":
    main()
