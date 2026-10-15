"""Argument parsing shared by the pipeline scripts."""

from __future__ import annotations

import argparse
from importlib import import_module

from lc_bind.paths import resolve_paths


def build_parser(description: str, gear: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Directory of the project data files. Default: ./data or LC_BIND_DATA_ROOT.",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Directory for outputs. Default: ./results or LC_BIND_RESULTS_ROOT.",
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Directory for intermediate caches. Default: <results>/cache.",
    )
    if gear:
        parser.add_argument(
            "--gear-dir",
            default=None,
            help="Directory that contains the gear_ultimate package. Or set LC_BIND_GEAR_DIR.",
        )
    return parser


def paths_from_args(args):
    gear = getattr(args, "gear_dir", None)
    return resolve_paths(
        data_root=args.data_root,
        results_root=args.results_root,
        cache_root=args.cache_root,
        gear_dir=gear,
    )


def _coerce(raw: str):
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def run_module(module: str, description: str, gear: bool, argv=None) -> None:
    parser = build_parser(description, gear=gear)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a notebook switch. Example: --set RUN_FAMILY_SCREEN=false",
    )
    args = parser.parse_args(argv)
    overrides = {}
    for item in args.set:
        if "=" not in item:
            parser.error(f"--set expects NAME=VALUE, got {item}")
        name, raw = item.split("=", 1)
        overrides[name] = _coerce(raw)
    imported = import_module(module)
    imported.run(paths_from_args(args), **overrides)
