"""Runner for MASTER_BINDING_BASELINE_REPRODUCTION_300_46_FULL_v10_BINDNODE24_FIXED-2.ipynb."""
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
    from lc_bind.utils import pdb as pdb_cache
    from lc_bind.models.baselines import setup
    from lc_bind.models.baselines import data
    from lc_bind.models.baselines import metrics
    from lc_bind.models.baselines import bindembed21
    from lc_bind.models.baselines import bindnode24
    from lc_bind.models.baselines import evaluate
    from lc_bind.models.baselines import structures
    from lc_bind.models.baselines import esmbind
    from lc_bind.models.baselines import metalbind
    pdb_cache.prepare()
    setup.prepare()
    data.prepare()
    metrics.prepare()
    bindembed21.prepare()
    bindnode24.prepare()
    evaluate.prepare()
    structures.prepare()
    esmbind.prepare()
    metalbind.prepare()
    _apply(overrides)
    setup.stage_02()
    _apply(overrides)
    setup.stage_03()
    _apply(overrides)
    setup.stage_06()
    _apply(overrides)
    setup.stage_08()
    _apply(overrides)
    data.stage_11()
    _apply(overrides)
    data.stage_12()
    _apply(overrides)
    setup.stage_15()
    _apply(overrides)
    data.stage_16()
    _apply(overrides)
    metrics.stage_18()
    _apply(overrides)
    bindembed21.stage_22()
    _apply(overrides)
    bindembed21.stage_26()
    _apply(overrides)
    bindembed21.stage_27()
    _apply(overrides)
    bindembed21.stage_28()
    _apply(overrides)
    bindembed21.stage_30()
    _apply(overrides)
    bindembed21.stage_31()
    _apply(overrides)
    bindembed21.stage_32()
    _apply(overrides)
    bindnode24.stage_34()
    _apply(overrides)
    bindnode24.stage_35()
    _apply(overrides)
    bindnode24.stage_36()
    _apply(overrides)
    bindnode24.stage_38()
    _apply(overrides)
    bindnode24.stage_39()
    _apply(overrides)
    bindnode24.stage_41()
    _apply(overrides)
    bindnode24.stage_42()
    _apply(overrides)
    bindnode24.stage_44()
    _apply(overrides)
    evaluate.stage_46()
    _apply(overrides)
    evaluate.stage_48()
    _apply(overrides)
    evaluate.stage_50()
    _apply(overrides)
    structures.stage_52()
    _apply(overrides)
    esmbind.stage_54()
    _apply(overrides)
    esmbind.stage_55()
    _apply(overrides)
    esmbind.stage_56()
    _apply(overrides)
    esmbind.stage_57()
    _apply(overrides)
    esmbind.stage_58()
    _apply(overrides)
    esmbind.stage_59()
    _apply(overrides)
    metalbind.stage_61()
    _apply(overrides)
    metalbind.stage_62()
    _apply(overrides)
    metalbind.stage_63()
    _apply(overrides)
    metalbind.stage_64()
    _apply(overrides)
    metalbind.stage_65()
    _apply(overrides)
    metalbind.stage_66()
    _apply(overrides)
    metalbind.stage_67()
    _apply(overrides)
    metalbind.stage_68()
    _apply(overrides)
    metalbind.stage_69()
    _apply(overrides)
    metalbind.stage_70()
    _apply(overrides)
    metalbind.stage_71()
    _apply(overrides)
    data.stage_74()
    _apply(overrides)
    evaluate.stage_77()
    _apply(overrides)
    evaluate.stage_79()
    _apply(overrides)
    evaluate.stage_80()
    _apply(overrides)
    evaluate.stage_81()
    _apply(overrides)
    S.RESULT_KEY = 'baselines'
    return S

