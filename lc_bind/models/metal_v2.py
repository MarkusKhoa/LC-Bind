"""Extracted from GEAR_METAL_ONLY_ULTIMATE_V2_1-1.ipynb. Section: metal_v2."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Path, ast, inspect, json, math, np, os, pd, platform, subprocess, sys, torch
    from pathlib import Path
    import os, sys, json, ast, inspect, platform, subprocess, math
    import numpy as np
    import pandas as pd
    import torch
    import os

def block_dim(blocks):
    return int(sum((S.BLOCK_DIMS[b] for b in blocks)))

def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def representation_stats_from_folds(fold_df):
    rep_col = _find_col(fold_df, ['representation', 'experiment', 'feature_set', 'rep'])
    score_col = _find_col(fold_df, ['AUPRC', 'val_AUPRC', 'CV_AUPRC', 'auprc'])
    fold_col = _find_col(fold_df, ['fold', 'fold_id', 'split'])
    if rep_col is None or score_col is None:
        return None
    q = fold_df.groupby(rep_col)[score_col].agg(['mean', 'std', 'count']).reset_index().rename(columns={rep_col: 'representation'})
    q['se'] = q['std'] / np.sqrt(q['count'].clip(lower=1))
    return q

# 1. DRIVE / PATHS / DEPENDENCIES
def stage_02():
    S.BASE = Path(str(S.BASE))
    S.ULTIMATE_DIR = S.GEAR_DIR
    if not S.ULTIMATE_DIR.exists():
        raise FileNotFoundError(f'{S.ULTIMATE_DIR} not found. Keep the existing GEAR_ULTIMATE backend on Drive.')
    sys.path.insert(0, str(S.ULTIMATE_DIR))
    from gear_ultimate import __version__ as GEAR_ULTIMATE_VERSION
    S.GEAR_ULTIMATE_VERSION = GEAR_ULTIMATE_VERSION
    from gear_ultimate import gear_core as core
    S.core = core
    from gear_ultimate import gear_structure as structure
    S.structure = structure
    from gear_ultimate import gear_mechanistic as mech
    S.mech = mech
    from gear_ultimate import gear_graph as graph
    S.graph = graph
    from gear_ultimate import gear_suite as suite
    S.suite = suite
    from gear_ultimate import gear_reporting as reporting
    S.reporting = reporting
    S.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    S.RUN_ROOT = S.RESULTS_ROOT / 'GEAR_METAL_ONLY_ULTIMATE_V2'
    S.RUN_ROOT.mkdir(parents=True, exist_ok=True)
    print('BASE:', S.BASE)
    print('ULTIMATE_DIR:', S.ULTIMATE_DIR)
    print('RUN_ROOT:', S.RUN_ROOT)
    print('GEAR version:', S.GEAR_ULTIMATE_VERSION)
    print('Torch:', torch.__version__)
    print('Device:', S.DEVICE)
    print('KAN available:', S.core.HAS_KAN)

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

# 2. MASTER SWITCHES
def stage_05():
    S.PREPARE_CONTEXT = True
    S.ESM_BATCH_SIZE = 8
    S.RUN_FAMILY_SCREEN = True
    S.RUN_COMPONENT_SCREEN = True
    S.RUN_COMBINATION_LADDER = True
    S.RUN_MODEL_FAMILY_SCREEN = True
    S.RUN_GRAPH_DIAGNOSTIC = False
    S.RUN_ESM_GRAPH = False
    S.RUN_FINAL_EXTERNAL = True
    S.SELECTION_POLICY = 'best_mean'
    S.FORCE_RERUN = False
    S.RUN_VERSION_FAMILY = 'metal_v2_01_family_mlp'
    S.RUN_VERSION_COMPONENT = 'metal_v2_02_component_mlp'
    S.RUN_VERSION_COMBO = 'metal_v2_03_combo_mlp'
    S.RUN_VERSION_MODEL = 'metal_v2_04_locked_model_family'
    S.RUN_VERSION_GRAPH = 'metal_v2_05_graph_diagnostic'
    S.RUN_VERSION_FINAL = 'metal_v2_06_final_locked_external'
    print('Selection policy:', S.SELECTION_POLICY)
    print('Graph diagnostic:', S.RUN_GRAPH_DIAGNOSTIC)
    print('Final external:', S.RUN_FINAL_EXTERNAL)

# 3. PREPARE / REUSE CONTEXT
def stage_07():
    S.ctx = S.suite.prepare_context(base=S.BASE, run_version='metal_v2_context', models=('MLP', 'KAN', 'XGBoost'), ensure_base=True, add_mech=True, add_multi=True, add_seqctx=True, esm_batch_size=S.ESM_BATCH_SIZE, device=S.DEVICE)
    display(S.suite.audit_context(S.ctx))
    display(S.reporting.dimension_table(S.ctx.stores['train']))
    S.explicit_blocks = ['AA', 'SPATIAL', 'LOCAL', 'ATOM', 'CHEM', 'ACCESS', 'SHELL2', 'ENV', 'CONF', 'METALSPEC', 'ELEC3D', 'TOPO', 'SOLV', 'SEQMOTIF', 'SEQCTX']
    display(S.reporting.validate_no_direct_label_features(S.ctx.stores, S.explicit_blocks))

# 4. FEATURE CONTRACT
def stage_08():
    S.BLOCK_DIMS = {'AA': 20, 'ESM': 1280, 'SPATIAL': 112, 'LOCAL': 8, 'ATOM': 15, 'CHEM': 24, 'ACCESS': 10, 'SHELL2': 12, 'ENV': 8, 'CONF': 4, 'METALSPEC': 18, 'ELEC3D': 8, 'TOPO': 12, 'SOLV': 10, 'SEQMOTIF': 10, 'SEQCTX': 182}
    S.FAMILIES = {'Structure': ['SPATIAL', 'LOCAL'], 'Atomic': ['ATOM'], 'Chem': ['CHEM'], 'MECH1': ['ACCESS', 'SHELL2', 'ENV', 'CONF'], 'MULTI2': ['METALSPEC', 'ELEC3D', 'TOPO', 'SOLV', 'SEQMOTIF'], 'SeqCtx': ['SEQCTX']}
    S.MECHANISTIC_COMPONENTS = {'ACCESS': ['ACCESS'], 'SHELL2': ['SHELL2'], 'ENV': ['ENV'], 'CONF': ['CONF'], 'METALSPEC': ['METALSPEC'], 'ELEC3D': ['ELEC3D'], 'TOPO': ['TOPO'], 'SOLV': ['SOLV'], 'SEQMOTIF': ['SEQMOTIF']}
    S.feature_inventory = []
    for S.fam, S.blocks in S.FAMILIES.items():
        S.feature_inventory.append({'family': S.fam, 'blocks': '+'.join(S.blocks), 'dim': block_dim(S.blocks)})
    display(pd.DataFrame(S.feature_inventory))
    print('Backend MULTI2:', getattr(S.suite, 'MULTI2', 'not exposed'))
    print('Official fold count:', len(S.ctx.fold_ids))

# 5. STAGE A — DEV-ONLY FAMILY SCREEN, MLP FIXED
def stage_10():
    S.FAMILY_GRID = {'ESM': ['ESM'], 'ESM+Structure': ['ESM', *S.FAMILIES['Structure']], 'ESM+Atomic': ['ESM', *S.FAMILIES['Atomic']], 'ESM+Chem': ['ESM', *S.FAMILIES['Chem']], 'ESM+MECH1': ['ESM', *S.FAMILIES['MECH1']], 'ESM+MULTI2': ['ESM', *S.FAMILIES['MULTI2']], 'ESM+SeqCtx': ['ESM', *S.FAMILIES['SeqCtx']]}
    if S.RUN_FAMILY_SCREEN:
        S.family_cv, S.family_folds = S.suite.run_dev_cv_only(S.ctx, S.FAMILY_GRID, models=('MLP',), run_version=S.RUN_VERSION_FAMILY)
        display(S.family_cv.sort_values('CV_AUPRC', ascending=False))
        S.family_cv.to_csv(S.RUN_ROOT / 'family_screen_cv.csv', index=False)
        S.family_folds.to_csv(S.RUN_ROOT / 'family_screen_folds.csv', index=False)
    else:
        S.family_cv = pd.read_csv(S.RUN_ROOT / 'family_screen_cv.csv')
        S.family_folds = pd.read_csv(S.RUN_ROOT / 'family_screen_folds.csv')
    S.family_rank = S.family_cv.copy()
    S.base_score = float(S.family_rank.loc[S.family_rank['representation'].eq('ESM'), 'CV_AUPRC'].iloc[0])
    S.family_rank['delta_vs_ESM'] = S.family_rank['CV_AUPRC'] - S.base_score
    S.family_rank['feature_dim'] = S.family_rank['representation'].map({k: block_dim(v) for k, v in S.FAMILY_GRID.items()})
    display(S.family_rank.sort_values(['delta_vs_ESM', 'CV_AUPRC'], ascending=False))
    S.family_rank.to_csv(S.RUN_ROOT / 'family_screen_ranked.csv', index=False)

# 6. STAGE B — COMPONENT SCREEN CONDITIONAL ON ESM+ATOM+CHEM
def stage_12():
    S.CORE_BLOCKS = ['ESM', 'ATOM', 'CHEM']
    S.COMPONENT_GRID = {'Core_ESM+Atomic+Chem': S.CORE_BLOCKS}
    for S.name, S.blocks in S.MECHANISTIC_COMPONENTS.items():
        S.COMPONENT_GRID[f'Core+{S.name}'] = [*S.CORE_BLOCKS, *S.blocks]
    S.COMPONENT_GRID['Core+MECH1'] = [*S.CORE_BLOCKS, *S.FAMILIES['MECH1']]
    S.COMPONENT_GRID['Core+MULTI2'] = [*S.CORE_BLOCKS, *S.FAMILIES['MULTI2']]
    if S.RUN_COMPONENT_SCREEN:
        S.component_cv, S.component_folds = S.suite.run_dev_cv_only(S.ctx, S.COMPONENT_GRID, models=('MLP',), run_version=S.RUN_VERSION_COMPONENT)
        display(S.component_cv.sort_values('CV_AUPRC', ascending=False))
        S.component_cv.to_csv(S.RUN_ROOT / 'component_screen_cv.csv', index=False)
        S.component_folds.to_csv(S.RUN_ROOT / 'component_screen_folds.csv', index=False)
    else:
        S.component_cv = pd.read_csv(S.RUN_ROOT / 'component_screen_cv.csv')
        S.component_folds = pd.read_csv(S.RUN_ROOT / 'component_screen_folds.csv')
    S.core_score = float(S.component_cv.loc[S.component_cv['representation'].eq('Core_ESM+Atomic+Chem'), 'CV_AUPRC'].iloc[0])
    S.component_rank = S.component_cv.copy()
    S.component_rank['delta_vs_core'] = S.component_rank['CV_AUPRC'] - S.core_score
    S.component_rank['feature_dim'] = S.component_rank['representation'].map({k: block_dim(v) for k, v in S.COMPONENT_GRID.items()})
    display(S.component_rank.sort_values('delta_vs_core', ascending=False))
    S.component_rank.to_csv(S.RUN_ROOT / 'component_screen_ranked.csv', index=False)

# 7. STAGE C — DEV-ONLY COMBINATION LADDER, MLP FIXED
def stage_14():
    S.COMBO_GRID = {'ESM': ['ESM'], 'ESM+Atomic': ['ESM', 'ATOM'], 'ESM+Chem': ['ESM', 'CHEM'], 'ESM+Atomic+Chem': ['ESM', 'ATOM', 'CHEM'], 'ESM+Atomic+Chem+MECH1': ['ESM', 'ATOM', 'CHEM', *S.FAMILIES['MECH1']], 'ESM+Atomic+Chem+MULTI2': ['ESM', 'ATOM', 'CHEM', *S.FAMILIES['MULTI2']], 'ESM+Atomic+Chem+MECH1+MULTI2': ['ESM', 'ATOM', 'CHEM', *S.FAMILIES['MECH1'], *S.FAMILIES['MULTI2']], 'ESM+Atomic+Chem+MULTI2+Structure': ['ESM', 'ATOM', 'CHEM', *S.FAMILIES['MULTI2'], *S.FAMILIES['Structure']], 'ESM+Atomic+Chem+MULTI2+SeqCtx': ['ESM', 'ATOM', 'CHEM', *S.FAMILIES['MULTI2'], *S.FAMILIES['SeqCtx']], 'ESM+Atomic+Chem+MECH1+MULTI2+Structure': ['ESM', 'ATOM', 'CHEM', *S.FAMILIES['MECH1'], *S.FAMILIES['MULTI2'], *S.FAMILIES['Structure']], 'ESM+Atomic+Chem+MECH1+MULTI2+SeqCtx': ['ESM', 'ATOM', 'CHEM', *S.FAMILIES['MECH1'], *S.FAMILIES['MULTI2'], *S.FAMILIES['SeqCtx']], 'ESM+ALL_EXPLICIT': ['ESM', 'ATOM', 'CHEM', *S.FAMILIES['MECH1'], *S.FAMILIES['MULTI2'], *S.FAMILIES['Structure'], *S.FAMILIES['SeqCtx']]}
    if S.RUN_COMBINATION_LADDER:
        S.combo_cv, S.combo_folds = S.suite.run_dev_cv_only(S.ctx, S.COMBO_GRID, models=('MLP',), run_version=S.RUN_VERSION_COMBO)
        display(S.combo_cv.sort_values('CV_AUPRC', ascending=False))
        S.combo_cv.to_csv(S.RUN_ROOT / 'combo_ladder_cv.csv', index=False)
        S.combo_folds.to_csv(S.RUN_ROOT / 'combo_ladder_folds.csv', index=False)
    else:
        S.combo_cv = pd.read_csv(S.RUN_ROOT / 'combo_ladder_cv.csv')
        S.combo_folds = pd.read_csv(S.RUN_ROOT / 'combo_ladder_folds.csv')

# 8. LOCK REPRESENTATION USING DEV ONLY
def stage_16():
    S.combo_rank = S.combo_cv.copy()
    S.combo_rank['feature_dim'] = S.combo_rank['representation'].map({k: block_dim(v) for k, v in S.COMBO_GRID.items()})
    S.combo_rank = S.combo_rank.sort_values('CV_AUPRC', ascending=False).reset_index(drop=True)
    S.BEST_MEAN_REP = str(S.combo_rank.iloc[0]['representation'])
    S.fold_stats = representation_stats_from_folds(S.combo_folds)
    if S.fold_stats is not None and (not S.fold_stats.empty):
        S.fold_stats['feature_dim'] = S.fold_stats['representation'].map({k: block_dim(v) for k, v in S.COMBO_GRID.items()})
        S.best_row = S.fold_stats.sort_values('mean', ascending=False).iloc[0]
        S.cutoff = float(S.best_row['mean'] - S.best_row['se'])
        S.eligible = S.fold_stats[S.fold_stats['mean'] >= S.cutoff].copy()
        S.eligible = S.eligible.sort_values(['feature_dim', 'mean'], ascending=[True, False])
        S.ONE_SE_REP = str(S.eligible.iloc[0]['representation'])
    else:
        S.ONE_SE_REP = S.BEST_MEAN_REP
    if S.SELECTION_POLICY == 'best_mean':
        S.LOCKED_REP = S.BEST_MEAN_REP
    elif S.SELECTION_POLICY == 'one_se':
        S.LOCKED_REP = S.ONE_SE_REP
    else:
        raise ValueError("SELECTION_POLICY must be 'best_mean' or 'one_se'")
    S.LOCKED_BLOCKS = list(S.COMBO_GRID[S.LOCKED_REP])
    S.decision_table = S.combo_rank[['representation', 'CV_AUPRC', 'feature_dim']].copy()
    display(S.decision_table)
    if S.fold_stats is not None:
        print('\nFold-level representation statistics:')
        display(S.fold_stats.sort_values('mean', ascending=False))
    print('\nBEST_MEAN_REP =', S.BEST_MEAN_REP)
    print('ONE_SE_REP   =', S.ONE_SE_REP)
    print('LOCK POLICY  =', S.SELECTION_POLICY)
    print('LOCKED_REP   =', S.LOCKED_REP)
    print('LOCKED_BLOCKS=', S.LOCKED_BLOCKS)
    print('LOCKED_DIM   =', block_dim(S.LOCKED_BLOCKS))
    S.representation_decision = {'selection_policy': S.SELECTION_POLICY, 'best_mean_representation': S.BEST_MEAN_REP, 'one_se_representation': S.ONE_SE_REP, 'locked_representation': S.LOCKED_REP, 'locked_blocks': S.LOCKED_BLOCKS, 'locked_dimension': block_dim(S.LOCKED_BLOCKS)}
    with open(S.RUN_ROOT / 'locked_representation.json', 'w') as S.f:
        json.dump(S.representation_decision, S.f, indent=2)

# 9. DEV-ONLY MODEL FAMILY SCREEN ON THE LOCKED REPRESENTATION
def stage_18():
    S.MODEL_CANDIDATES = ['MLP', 'XGBoost']
    if S.core.HAS_KAN:
        S.MODEL_CANDIDATES.insert(1, 'KAN')
    S.LOCKED_GRID = {S.LOCKED_REP: S.LOCKED_BLOCKS}
    if S.RUN_MODEL_FAMILY_SCREEN:
        S.model_cv, S.model_folds = S.suite.run_dev_cv_only(S.ctx, S.LOCKED_GRID, models=tuple(S.MODEL_CANDIDATES), run_version=S.RUN_VERSION_MODEL)
        display(S.model_cv.sort_values('CV_AUPRC', ascending=False))
        S.model_cv.to_csv(S.RUN_ROOT / 'model_family_cv.csv', index=False)
        S.model_folds.to_csv(S.RUN_ROOT / 'model_family_folds.csv', index=False)
    else:
        S.model_cv = pd.read_csv(S.RUN_ROOT / 'model_family_cv.csv')
        S.model_folds = pd.read_csv(S.RUN_ROOT / 'model_family_folds.csv')
    S.model_rank = S.model_cv.sort_values('CV_AUPRC', ascending=False).reset_index(drop=True)
    S.LOCKED_MODEL = str(S.model_rank.iloc[0]['model'])
    print('LOCKED MODEL FROM DEV =', S.LOCKED_MODEL)
    display(S.model_rank)
    S.model_decision = {'candidate_models': S.representation_decision, 'locked_model': S.MODEL_CANDIDATES, 'locked_model_cv_auprc': S.LOCKED_MODEL}
    with open(S.RUN_ROOT / 'locked_model.json', 'w') as S.f:
        json.dump(S.model_decision, S.f, indent=2)

# 10. OPTIONAL GRAPH DIAGNOSTIC
def stage_20():
    S.graph_cv = None
    S.graph_ext = None
    if S.RUN_GRAPH_DIAGNOSTIC:
        S.explicit_locked = [b for b in S.LOCKED_BLOCKS if b != 'ESM']
        S.explicit_node_blocks = ['AA'] + [b for b in S.explicit_locked if b != 'AA']
        S.GRAPH_EXPERIMENTS = {'ResidualGraph_ExplicitLocked_Rich': {'blocks': S.explicit_node_blocks, 'architecture': 'residual', 'edge_mode': 'rich'}, 'ResidualGraph_ExplicitLocked_Donor': {'blocks': S.explicit_node_blocks, 'architecture': 'residual', 'edge_mode': 'donor'}}
        if S.RUN_ESM_GRAPH:
            S.GRAPH_EXPERIMENTS['ResidualGraph_ESMLocked_Rich'] = {'blocks': S.LOCKED_BLOCKS, 'architecture': 'residual', 'edge_mode': 'rich'}
            S.GRAPH_EXPERIMENTS['ResidualGraph_ESMLocked_Donor'] = {'blocks': S.LOCKED_BLOCKS, 'architecture': 'residual', 'edge_mode': 'donor'}
        S.graph_cv, S.graph_ext = S.graph.run_graph_grid(S.ctx.stores, S.ctx.fold_ids, S.BASE, S.GRAPH_EXPERIMENTS, run_version=S.RUN_VERSION_GRAPH, cfg=S.graph.GraphConfig(), device=S.DEVICE)
        display(S.graph_cv.sort_values('CV_AUPRC', ascending=False))
        display(S.graph_ext.sort_values(['dataset', 'AUPRC'], ascending=[True, False]))
        S.graph_cv.to_csv(S.RUN_ROOT / 'graph_cv.csv', index=False)
        S.graph_ext.to_csv(S.RUN_ROOT / 'graph_external_exploratory.csv', index=False)
    else:
        print('Graph diagnostic OFF. This is recommended for the first feature-selection run.')

# 11. FINAL LOCKED EXTERNAL RUN
def stage_22():
    S.final_cv = None
    S.final_ext = None
    if S.RUN_FINAL_EXTERNAL:
        S.final_ctx = S.suite.clone_context_for_run(S.ctx, run_version=S.RUN_VERSION_FINAL, models=(S.LOCKED_MODEL,))
        S.final_grid = {S.LOCKED_REP: S.LOCKED_BLOCKS}
        S.final_cv, S.final_ext, S._ = S.suite.run_grid(S.final_ctx, S.final_grid, models=(S.LOCKED_MODEL,))
        display(S.final_cv)
        display(S.final_ext.sort_values(['dataset', 'AUPRC'], ascending=[True, False]))
        S.final_cv.to_csv(S.RUN_ROOT / 'FINAL_locked_cv.csv', index=False)
        S.final_ext.to_csv(S.RUN_ROOT / 'FINAL_locked_external.csv', index=False)
    else:
        print('Final external run OFF.')

# 12. READ LOCKED BASELINE RESULTS — NO BASELINE INFERENCE
def stage_24():
    S.BASELINE_ROOT = S.BASE / 'baseline_reproduction_master_v6' / 'results'
    S.baseline_files = {'direct': S.BASELINE_ROOT / 'FULL_DIRECT_BASELINE_METRICS_300_46.csv', 'ESMBind': S.BASELINE_ROOT / 'ESMBind_TRANSFER_FULL_METRICS_300_46.csv', 'MetalBind': S.BASELINE_ROOT / 'MetalBind_TRANSFER_FULL_METRICS_300_46.csv'}
    S.baseline_frames = []
    for S.group, S.fp in S.baseline_files.items():
        if not S.fp.exists():
            print('Missing:', S.fp)
            continue
        S.z = pd.read_csv(S.fp)
        if 'operating_point' in S.z.columns:
            S.z = S.z[S.z['operating_point'].eq('fixed_0.5')].copy()
        if 'label' in S.z.columns:
            S.z = S.z[S.z['label'].eq('metal_binding')].copy()
        S.z['baseline_group'] = S.group
        S.baseline_frames.append(S.z)
    if S.baseline_frames:
        S.locked_baselines = pd.concat(S.baseline_frames, ignore_index=True, sort=False)
        S.locked_baselines.to_csv(S.RUN_ROOT / 'locked_baseline_metal_rows.csv', index=False)
        S.show_cols = [c for c in ['method', 'dataset', 'label', 'AP_AUPRC', 'AUPRC', 'AUROC', 'global_positive_F1', 'global_MCC', 'protein_mean_F1', 'protein_mean_MCC'] if c in S.locked_baselines.columns]
        print('LOCKED BASELINES — METAL ONLY')
        display(S.locked_baselines[S.show_cols])
    if S.final_ext is not None:
        print('\nOUR FINAL LOCKED EXTERNAL RESULT')
        display(S.final_ext.sort_values(['dataset', 'AUPRC'], ascending=[True, False]))

# 13. FINAL DIGEST / MANIFEST
def stage_26():
    S.digest = {'gear_ultimate_version': S.GEAR_ULTIMATE_VERSION, 'python': platform.python_version(), 'torch': torch.__version__, 'device': str(S.DEVICE), 'primary_metric': 'AUPRC', 'development_set': 'DevSet1010 official 5-fold protein split', 'external_sets': ['TestSet300', 'TestSetNew46'], 'selection_policy': S.SELECTION_POLICY, 'locked_representation': S.LOCKED_REP, 'locked_blocks': S.LOCKED_BLOCKS, 'locked_dimension': block_dim(S.LOCKED_BLOCKS), 'locked_model': S.LOCKED_MODEL, 'graph_diagnostic_run': bool(S.RUN_GRAPH_DIAGNOSTIC), 'final_external_run': bool(S.RUN_FINAL_EXTERNAL)}
    with open(S.RUN_ROOT / 'FINAL_DECISION_MANIFEST.json', 'w') as S.f:
        json.dump(S.digest, S.f, indent=2)
    print(json.dumps(S.digest, indent=2))
    print('\nArtifacts:')
    for S.fp in sorted(S.RUN_ROOT.glob('*')):
        print(' -', S.fp.name)
    print('\n' + '=' * 90)
    print('GEAR METAL-ONLY ULTIMATE V2: PIPELINE COMPLETE')
    print('Feature selection source: DevSet1010 only')
    print('Locked representation:', S.LOCKED_REP)
    print('Locked model:', S.LOCKED_MODEL)
    if S.RUN_FINAL_EXTERNAL:
        print('External confirmation: TestSet300 + TestSetNew46 complete')
    print('=' * 90)

STAGES = [stage_02, stage_03, stage_05, stage_07, stage_08, stage_10, stage_12, stage_14, stage_16, stage_18, stage_20, stage_22, stage_24, stage_26]


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
    stage_05()
    _apply(overrides)
    stage_07()
    _apply(overrides)
    stage_08()
    _apply(overrides)
    stage_10()
    _apply(overrides)
    stage_12()
    _apply(overrides)
    stage_14()
    _apply(overrides)
    stage_16()
    _apply(overrides)
    stage_18()
    _apply(overrides)
    stage_20()
    _apply(overrides)
    stage_22()
    _apply(overrides)
    stage_24()
    _apply(overrides)
    stage_26()
    _apply(overrides)
    S.RESULT_KEY = 'metal_v2'
    return S

