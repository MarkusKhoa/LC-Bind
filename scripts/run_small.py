#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc_bind.cli import run_module

if __name__ == "__main__":
    run_module(
        "lc_bind.models.small",
        "Small-molecule pocket pipeline. Requires --gear-dir.",
        gear=True,
    )
