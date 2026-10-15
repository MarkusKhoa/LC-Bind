"""Extracted from GEAR_ULTIMATE_METAL_BINDING-1.ipynb. Section: metal_v1."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Path, json, np, os, pd, platform, subprocess, sys, torch
    from pathlib import Path
    import os, sys
    import os
    import torch
    import subprocess
    import numpy as np
    import pandas as pd
    import platform, json

def stage_02():
    S.BASE = Path(str(S.BASE))
    S.ULTIMATE_DIR = S.GEAR_DIR
    if not S.ULTIMATE_DIR.exists():
        S.local = Path(str(S.CACHE_ROOT / 'GEAR_ULTIMATE'))
        if S.local.exists():
            S.ULTIMATE_DIR = S.local
        else:
            raise FileNotFoundError('GEAR_ULTIMATE folder not found. Upload/extract the supplied bundle to ' + str(S.BASE / 'GEAR_ULTIMATE') + ' first.')
    print('BASE:', S.BASE)
    print('ULTIMATE_DIR:', S.ULTIMATE_DIR)

def stage_03():
    S.BASE = Path(str(S.BASE))
    print('=== GPU ===')
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2), 'GB')
    print('\n=== EXISTING FEATURE CACHE ===')
    S.CACHE = S.CACHE_ROOT
    if S.CACHE.exists():
        S.files = [p for p in S.CACHE.rglob('*') if p.is_file()]
        print('Cache files:', len(S.files))
        print('Total cache:', round(sum((p.stat().st_size for p in S.files)) / 1024 ** 3, 2), 'GB')
        S.keywords = ['esm', 'spatial', 'local', 'atom', 'chem', 'access', 'shell', 'multi', 'seqctx']
        for S.key in S.keywords:
            S.hits = [p for p in S.files if S.key.lower() in str(p).lower()]
            S.size = sum((p.stat().st_size for p in S.hits))
            print(f'{S.key:10s}: {len(S.hits):5d} files | {S.size / 1024 ** 3:.3f} GB')
    print('\n=== NEW V2 EXPERIMENT CACHE ===')
    S.versions = ['metal_v2_01_family_mlp', 'metal_v2_02_component_mlp', 'metal_v2_03_combo_mlp', 'metal_v2_04_locked_model_family', 'metal_v2_05_graph_diagnostic', 'metal_v2_06_final_locked_external']
    for S.v in S.versions:
        S.d = S.RESULTS_ROOT / S.v / 'metal_binding'
        if not S.d.exists():
            print(f'{S.v:42s}: NOT STARTED')
            continue
        S.fs = [p for p in S.d.rglob('*') if p.is_file()]
        print(f'{S.v:42s}: {len(S.fs)} files | {sum((p.stat().st_size for p in S.fs)) / 1024 ** 2:.1f} MB')

# Optional: only enable when you are actually generating MSAs.
def stage_07():
    S.INSTALL_EVOLUTION_DEPS = False
    if S.INSTALL_EVOLUTION_DEPS:
        import subprocess

def stage_09():
    sys.path.insert(0, str(S.ULTIMATE_DIR))
    from gear_ultimate import __version__ as GEAR_ULTIMATE_VERSION
    S.GEAR_ULTIMATE_VERSION = GEAR_ULTIMATE_VERSION
    from gear_ultimate import gear_core as core
    S.core = core
    from gear_ultimate import gear_structure as structure
    S.structure = structure
    from gear_ultimate import gear_mechanistic as mech
    S.mech = mech
    from gear_ultimate import gear_legacy as legacy
    S.legacy = legacy
    from gear_ultimate import gear_graph as graph
    S.graph = graph
    from gear_ultimate import gear_evolution as evo
    S.evo = evo
    from gear_ultimate import gear_suite as suite
    S.suite = suite
    from gear_ultimate import gear_reporting as reporting
    S.reporting = reporting
    S.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('GEAR Ultimate:', S.GEAR_ULTIMATE_VERSION)
    print('DEVICE:', S.DEVICE)
    print('Torch:', torch.__version__)
    print('KAN available:', S.core.HAS_KAN)

# Data / feature preparation
def stage_11():
    S.PREPARE_CONTEXT = True
    S.ESM_BATCH_SIZE = 8
    S.RUN_HISTORICAL_KAN = False
    S.RUN_STANDARDIZED_13 = False
    S.STANDARDIZED_MODELS = ('MLP', 'KAN', 'XGBoost')
    S.RUN_MECHANISTIC_GRID = False
    S.MECHANISTIC_MODELS = ('MLP',)
    S.RUN_FINAL_FOCUSED = False
    S.FINAL_MODELS = ('MLP',)
    S.RUN_NO_ESM_GRID = False
    S.NO_ESM_MODELS = ('MLP',)
    S.RUN_MAXIMAL_GRAPH = False
    S.RUN_CLEAN_GRAPH = False
    S.RUN_MSA_DEV = False
    S.MSA_BATCH_SIZE = 25
    S.MSA_MAX_BATCHES = None
    S.ATTACH_EVO_DEV = False
    S.RUN_EVO_DEV_CV = False
    S.EVO_DEV_MODELS = ('MLP',)
    S.RUN_MSA_EXTERNAL = False
    S.ATTACH_EVO_ALL = False
    S.RUN_EVO_FULL = False
    S.EVO_FULL_MODELS = ('MLP',)
    S.RUN_EVO_GRAPH = False
    print('Switches loaded.')

def stage_13():
    S.ctx = None
    if S.PREPARE_CONTEXT:
        S.ctx = S.suite.prepare_context(base=S.BASE, run_version='ultimate_context_v1', models=('MLP', 'KAN', 'XGBoost'), ensure_base=True, add_mech=True, add_multi=True, add_seqctx=True, esm_batch_size=S.ESM_BATCH_SIZE, device=S.DEVICE)
        display(S.suite.audit_context(S.ctx))
        display(S.reporting.dimension_table(S.ctx.stores['train']))

def stage_15():
    if S.ctx is not None:
        S.explicit_blocks = ['AA', 'SPATIAL', 'LOCAL', 'ATOM', 'CHEM', 'ACCESS', 'SHELL2', 'ENV', 'CONF', 'METALSPEC', 'ELEC3D', 'TOPO', 'SOLV', 'SEQMOTIF', 'SEQCTX']
        display(S.reporting.validate_no_direct_label_features(S.ctx.stores, S.explicit_blocks))

def stage_17():
    if S.RUN_HISTORICAL_KAN:
        if not S.core.HAS_KAN:
            raise RuntimeError('KAN is not available. Re-run the dependency cell and verify pykan.')
        S.hist_cfg = S.legacy.HistoricalConfig(base=str(S.BASE))
        S.historical_result = S.legacy.run_historical_baseline(S.hist_cfg, device=S.DEVICE)
        S.historical_result

def stage_19():
    if S.RUN_STANDARDIZED_13:
        S.std_ctx = S.suite.clone_context_for_run(S.ctx, run_version='ultimate_standardized_13_v1', models=S.STANDARDIZED_MODELS)
        S.std_cv, S.std_ext, S._ = S.suite.run_grid(S.std_ctx, S.suite.STANDARDIZED_13, models=S.STANDARDIZED_MODELS)
        display(S.std_cv)
        display(S.std_ext.sort_values(['dataset', 'model', 'AUPRC'], ascending=[True, True, False]))

def stage_21():
    if S.RUN_MECHANISTIC_GRID:
        S.mech_ctx = S.suite.clone_context_for_run(S.ctx, run_version='ultimate_mechanistic_grid_v1', models=S.MECHANISTIC_MODELS)
        S.mech_cv, S.mech_ext, S.mech_paired = S.suite.run_grid(S.mech_ctx, S.suite.MECHANISTIC_GRID, models=S.MECHANISTIC_MODELS, comparisons=S.suite.MECHANISTIC_COMPARISONS)
        display(S.mech_cv)
        display(S.mech_paired.sort_values(['model', 'mean_delta'], ascending=[True, False]))
        display(S.mech_ext.sort_values(['dataset', 'model', 'AUPRC'], ascending=[True, True, False]))

def stage_23():
    if S.RUN_FINAL_FOCUSED:
        S.final_ctx = S.suite.clone_context_for_run(S.ctx, run_version='ultimate_final_focused_v1', models=S.FINAL_MODELS)
        S.final_cv, S.final_ext, S._ = S.suite.run_grid(S.final_ctx, S.suite.FINAL_COMPARISON_REPS, models=S.FINAL_MODELS)
        display(S.final_cv)
        display(S.final_ext.sort_values(['dataset', 'model', 'AUPRC'], ascending=[True, True, False]))

def stage_25():
    if S.RUN_NO_ESM_GRID:
        S.noesm_ctx = S.suite.clone_context_for_run(S.ctx, run_version='ultimate_no_esm_v1', models=S.NO_ESM_MODELS)
        S.noesm_cv, S.noesm_ext, S._ = S.suite.run_grid(S.noesm_ctx, S.suite.NO_ESM_GRID, models=S.NO_ESM_MODELS)
        display(S.noesm_cv)
        display(S.noesm_ext.sort_values(['dataset', 'model', 'AUPRC'], ascending=[True, True, False]))

def stage_27():
    if S.RUN_MAXIMAL_GRAPH:
        S.max_cv, S.max_ext = S.graph.run_graph_grid(S.ctx.stores, S.ctx.fold_ids, S.BASE, S.graph.MAXIMAL_GRAPH_EXPERIMENT, run_version='ultimate_maximal_noesm_graph_v1', cfg=S.graph.maximal_graph_config(), device=S.DEVICE)
        display(S.max_cv)
        display(S.max_ext)

def stage_28():
    if S.RUN_CLEAN_GRAPH:
        S.clean_cv, S.clean_ext = S.graph.run_graph_grid(S.ctx.stores, S.ctx.fold_ids, S.BASE, S.graph.CLEAN_GRAPH_EXPERIMENTS, run_version='ultimate_clean_graph_v2', cfg=S.graph.GraphConfig(), device=S.DEVICE)
        display(S.clean_cv)
        display(S.clean_ext.sort_values(['dataset', 'AUPRC'], ascending=[True, False]))

def stage_30():
    if S.RUN_MSA_DEV:
        S.evo.run_msa_batches(S.ctx.stores, S.BASE, splits=('train',), batch_size=S.MSA_BATCH_SIZE, max_retries=4, sleep_between=8, msa_mode='mmseqs2_uniref_env', max_batches=S.MSA_MAX_BATCHES)

def stage_31():
    if S.ATTACH_EVO_DEV:
        S.dev_evo_coverage = S.evo.attach_evo(S.ctx.stores, S.BASE, splits=('train',), require_msa=True)
        display(S.dev_evo_coverage)

def stage_33():
    if S.RUN_EVO_DEV_CV:
        S.evo_dev_cv, S.evo_dev_folds = S.suite.run_dev_cv_only(S.ctx, S.suite.EVO_GRID, models=S.EVO_DEV_MODELS, run_version='ultimate_evo_dev_only_v1')
        display(S.evo_dev_cv.sort_values(['model', 'CV_AUPRC'], ascending=[True, False]))

def stage_35():
    if S.RUN_MSA_EXTERNAL:
        S.evo.run_msa_batches(S.ctx.stores, S.BASE, splits=('test300', 'test46'), batch_size=S.MSA_BATCH_SIZE, max_retries=4, sleep_between=8, msa_mode='mmseqs2_uniref_env', max_batches=None)
    if S.ATTACH_EVO_ALL:
        S.all_evo_coverage = S.evo.attach_evo(S.ctx.stores, S.BASE, splits=('train', 'test300', 'test46'), require_msa=True)
        display(S.all_evo_coverage)
    if S.RUN_EVO_FULL:
        S.evo_ctx = S.suite.clone_context_for_run(S.ctx, run_version='ultimate_evo_full_v1', models=S.EVO_FULL_MODELS)
        S.evo_cv, S.evo_ext, S._ = S.suite.run_grid(S.evo_ctx, S.suite.EVO_GRID, models=S.EVO_FULL_MODELS)
        display(S.evo_cv)
        display(S.evo_ext.sort_values(['dataset', 'model', 'AUPRC'], ascending=[True, True, False]))

def stage_37():
    S.EVO_GRAPH_EXPERIMENTS = {'PureGraph_AA_Chem_EVO': {'blocks': ['AA', 'CHEM', 'EVO'], 'architecture': 'pure', 'edge_mode': 'rich'}, 'ResidualGraph_AA_Atomic_Chem_EVO': {'blocks': ['AA', 'ATOM', 'CHEM', 'EVO'], 'architecture': 'residual', 'edge_mode': 'rich'}, 'ResidualGraph_DonorFocused_EVO': {'blocks': ['AA', 'ATOM', 'CHEM', 'EVO'], 'architecture': 'residual', 'edge_mode': 'donor'}, 'ResidualGraph_Compact_EVO': {'blocks': ['AA', 'ATOM', 'CHEM', *S.suite.MULTI2, 'EVO'], 'architecture': 'residual', 'edge_mode': 'rich'}}
    if S.RUN_EVO_GRAPH:
        S.evo_graph_cv, S.evo_graph_ext = S.graph.run_graph_grid(S.ctx.stores, S.ctx.fold_ids, S.BASE, S.EVO_GRAPH_EXPERIMENTS, run_version='ultimate_evo_graph_v1', cfg=S.graph.GraphConfig(), device=S.DEVICE)
        display(S.evo_graph_cv)
        display(S.evo_graph_ext.sort_values(['dataset', 'AUPRC'], ascending=[True, False]))

def stage_39():
    display(S.suite.known_result_history())

# Final environment snapshot for the run log.
def stage_45():
    S.snapshot = {'gear_ultimate_version': S.GEAR_ULTIMATE_VERSION, 'python': platform.python_version(), 'torch': torch.__version__, 'device': str(S.DEVICE), 'base': str(S.BASE), 'kan_available': bool(S.core.HAS_KAN)}
    print(json.dumps(S.snapshot, indent=2))

STAGES = [stage_02, stage_03, stage_07, stage_09, stage_11, stage_13, stage_15, stage_17, stage_19, stage_21, stage_23, stage_25, stage_27, stage_28, stage_30, stage_31, stage_33, stage_35, stage_37, stage_39, stage_45]


import sys
from lc_bind.paths import bind_session


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
    prepare()
    _apply(overrides)
    stage_02()
    _apply(overrides)
    stage_03()
    _apply(overrides)
    stage_07()
    _apply(overrides)
    stage_09()
    _apply(overrides)
    stage_11()
    _apply(overrides)
    stage_13()
    _apply(overrides)
    stage_15()
    _apply(overrides)
    stage_17()
    _apply(overrides)
    stage_19()
    _apply(overrides)
    stage_21()
    _apply(overrides)
    stage_23()
    _apply(overrides)
    stage_25()
    _apply(overrides)
    stage_27()
    _apply(overrides)
    stage_28()
    _apply(overrides)
    stage_30()
    _apply(overrides)
    stage_31()
    _apply(overrides)
    stage_33()
    _apply(overrides)
    stage_35()
    _apply(overrides)
    stage_37()
    _apply(overrides)
    stage_39()
    _apply(overrides)
    stage_45()
    _apply(overrides)
    S.RESULT_KEY = 'metal_v1'
    return S

