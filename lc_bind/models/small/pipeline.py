"""Runner for SMALL_ULTIMATE_V1_DATA_DRIVEN_POCKET-3.ipynb."""
from __future__ import annotations

import sys

from lc_bind.paths import bind_session
from lc_bind.session import S


def _apply(overrides):
    for key, value in overrides.items():
        if value is not None:
            setattr(S, key, value)


def run(paths, **overrides):
    bind_session(paths, S)
    gear = paths.gear_dir
    if gear is not None and (gear / 'gear_ultimate').exists():
        text = str(gear.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)
    else:
        paths.require_gear()
    from lc_bind.utils import pdb as pdb_cache
    from lc_bind.models.small import pipeline_stages
    from lc_bind.models.small import features
    from lc_bind.models.small import screen
    from lc_bind.models.small import pocket
    from lc_bind.models.small import lock
    from lc_bind.models.small import v2
    pdb_cache.prepare()
    pipeline_stages.prepare()
    features.prepare()
    screen.prepare()
    pocket.prepare()
    lock.prepare()
    v2.prepare()
    _apply(overrides)
    pipeline_stages.stage_01()
    _apply(overrides)
    pipeline_stages.stage_03()
    _apply(overrides)
    pipeline_stages.stage_04()
    _apply(overrides)
    features.stage_05()
    _apply(overrides)
    features.stage_07()
    _apply(overrides)
    pdb_cache.stage_11()
    _apply(overrides)
    features.stage_12()
    _apply(overrides)
    features.stage_13()
    _apply(overrides)
    screen.stage_15()
    _apply(overrides)
    screen.stage_16()
    _apply(overrides)
    screen.stage_18()
    _apply(overrides)
    screen.stage_19()
    _apply(overrides)
    screen.stage_20()
    _apply(overrides)
    screen.stage_21()
    _apply(overrides)
    pocket.stage_23()
    _apply(overrides)
    pocket.stage_24()
    _apply(overrides)
    pocket.stage_25()
    _apply(overrides)
    pocket.stage_26()
    _apply(overrides)
    lock.stage_28()
    _apply(overrides)
    lock.stage_29()
    _apply(overrides)
    lock.stage_30()
    _apply(overrides)
    lock.stage_32()
    _apply(overrides)
    lock.stage_34()
    _apply(overrides)
    lock.stage_35()
    _apply(overrides)
    v2.stage_37()
    _apply(overrides)
    v2.stage_39()
    _apply(overrides)
    v2.stage_40()
    _apply(overrides)
    v2.stage_41()
    _apply(overrides)
    v2.stage_42()
    _apply(overrides)
    v2.stage_43()
    _apply(overrides)
    v2.stage_44()
    _apply(overrides)
    S.RESULT_KEY = 'small'
    return S

