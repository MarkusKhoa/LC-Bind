"""GEAR metal-binding drivers.

``run_v1`` is the original GEAR Ultimate metal notebook.
``run_v2`` is the metal-only ultimate v2 screen.
Both need ``--gear-dir`` pointing at a checkout that contains ``gear_ultimate``.
"""

from __future__ import annotations


def run_v1(paths, **overrides):
    from lc_bind.models.metal_v1 import run

    return run(paths, **overrides)


def run_v2(paths, **overrides):
    from lc_bind.models.metal_v2 import run

    return run(paths, **overrides)
