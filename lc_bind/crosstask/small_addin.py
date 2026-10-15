"""Extracted from SMALL_STANDALONE_XTASK_ADDIN-4.ipynb. Section: small_addin."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Path, ast, average_precision_score, get_block, get_record, gzip, json, np, os, parse_list, pd, set_block, subprocess, sys, time, torch
    from pathlib import Path
    import os, sys, json, ast, time, subprocess
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import average_precision_score
    import os
    import gzip, time, json
    from lc_bind.io.datasets import parse_list
    from lc_bind.utils.cache import get_block, get_record, set_block

def load_df(path):
    df = pd.read_csv(path)
    df['prot_id'] = df['prot_id'].astype(str)
    df['sequence'] = df['sequence'].astype(str)
    for col in ['metal_binding', 'nuclear_binding', 'small_binding', 'any_ligand_binding_sites']:
        if col in df.columns:
            df[col] = df[col].apply(parse_list)
    return df

def reconstruct_oof_from_fold_predictions(pred_dir, label_col):
    """
    Saved foldX_val.npy was produced on:
      va_df = train_df[train_df.prot_id.isin(val_set)]
    preserving train_df order.
    """
    offsets = {}
    cursor = 0
    for r in S.train_df.itertuples():
        L = len(r.sequence)
        offsets[str(r.prot_id)] = (cursor, cursor + L)
        cursor += L
    oof = np.full(cursor, np.nan, dtype=np.float32)
    for fold, val_ids in S.OFFICIAL_FOLDS.items():
        val_set = set(val_ids)
        va_df = S.train_df[S.train_df.prot_id.isin(val_set)].copy()
        fp = Path(pred_dir) / f'fold{fold}_val.npy'
        if not fp.exists():
            raise FileNotFoundError(fp)
        pred = np.load(fp).astype(np.float32)
        expected = int(va_df.sequence.str.len().sum())
        if pred.shape != (expected,):
            raise RuntimeError(f'{fp}: shape {pred.shape}, expected {(expected,)}')
        pos = 0
        for r in va_df.itertuples():
            L = len(r.sequence)
            a, b = offsets[str(r.prot_id)]
            oof[a:b] = pred[pos:pos + L]
            pos += L
    if not np.all(np.isfinite(oof)):
        raise RuntimeError('Incomplete reconstructed OOF')
    y = np.concatenate(S.train_df[label_col].apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
    return (oof, y)

def official_fold_indices():
    offsets = {}
    cursor = 0
    for r in S.train_df.itertuples():
        L = len(r.sequence)
        offsets[str(r.prot_id)] = (cursor, cursor + L)
        cursor += L
    out = {}
    for fold, ids in S.OFFICIAL_FOLDS.items():
        z = []
        for pid in ids:
            a, b = offsets[str(pid)]
            z.extend(range(a, b))
        out[fold] = np.asarray(z, dtype=np.int64)
    return out

def weight_search(y, p1, p2):
    rows = []
    for w in np.linspace(0, 1, 21):
        p = w * p1 + (1 - w) * p2
        aps = [average_precision_score(y[idx], p[idx]) for _, idx in sorted(S.FOLD_IDX.items())]
        rows.append({'w_V1': float(w), 'w_MS': float(1 - w), 'CV_AUPRC_mean': float(np.mean(aps)), 'CV_AUPRC_std': float(np.std(aps, ddof=1)), 'OOF_AUPRC': float(average_precision_score(y, p))})
    return pd.DataFrame(rows).sort_values(['CV_AUPRC_mean', 'w_V1'], ascending=[False, False]).reset_index(drop=True)

def _read_maybe_gzip_csv(fp):
    with open(fp, 'rb') as f:
        magic = f.read(2)
    if magic == b'\x1f\x8b':
        return pd.read_csv(fp)
    print('WARNING: counterpart has .gz suffix but is plain CSV; reading safely as plain CSV.')
    return pd.read_csv(fp, compression=None)

# STANDALONE BOOTSTRAP — fresh Colab runtime
def stage_01():
    S.BASE = Path(str(S.BASE))
    S.ULTIMATE_CANDIDATES = [S.GEAR_DIR, S.BASE / 'GEAR_ULTIMATE_OTHER_BINDINGS', S.BASE / 'GEAR_ULTIMATE']
    S.ULTIMATE_DIR = next((p for p in S.ULTIMATE_CANDIDATES if p is not None and (Path(p) / 'gear_ultimate').exists()), None)
    if S.ULTIMATE_DIR is None:
        raise FileNotFoundError('GEAR backend not found under GEAR_ULTIMATE_OTHER_BINDINGS or GEAR_ULTIMATE')
    S.req = S.ULTIMATE_DIR / 'requirements.txt'
    if S.req.exists():
        pass
    sys.path.insert(0, str(S.ULTIMATE_DIR))
    from gear_ultimate import gear_evolution as evo
    S.evo = evo
    from gear_ultimate import gear_suite as suite
    S.suite = suite
    from gear_ultimate import gear_reporting as reporting
    S.reporting = reporting
    S.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if S.DEVICE.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    print('BASE:', S.BASE)
    print('GEAR backend:', S.ULTIMATE_DIR)
    print('DEVICE:', S.DEVICE)

# Standalone settings
def stage_02():
    print('Standalone add-in runtime started.')

# DATA / OFFICIAL FOLDS / STORE HELPERS
def stage_03():
    S.TRAIN_CSV = S.DATA_ROOT / 'development_set/full_grouped_train_binding_sites_df.csv'
    S.TEST300_CSV = S.DATA_ROOT / 'development_set/full_grouped_test_binding_sites_df.csv'
    S.TEST46_CSV = S.DATA_ROOT / 'independent_set/grouped_test_46_new_binding_sites.csv'
    S.raw_train = load_df(S.TRAIN_CSV)
    S.test300_df = load_df(S.TEST300_CSV)
    S.test46_df = load_df(S.TEST46_CSV)
    S.BASELINE_WORK = S.BASE / 'baseline_reproduction_master_v6'
    S.BN24_CANDIDATES = [S.BASELINE_WORK / 'repos' / 'bindnode24', S.BASE / 'repos' / 'bindnode24']
    S.BN24 = next((p for p in S.BN24_CANDIDATES if (p / '.git').exists()), None)
    if S.BN24 is None:
        S.BN24 = Path(str(S.CACHE_ROOT / 'bindnode24_standalone_addin'))
        if not S.BN24.exists():
            subprocess.run(['git', 'clone', '-q', 'https://github.com/erckert/bindNode24.git', str(S.BN24)], check=True)
    S.fold_root = S.BN24 / 'dataset' / 'cv_splits' / 'predicted_structures'
    S.OFFICIAL_FOLDS = {}
    S.official_order = []
    S.seen = set()
    for S.k in range(1, 6):
        S.ids = [x.strip() for x in (S.fold_root / f'ids_split{S.k}.txt').read_text().splitlines() if x.strip()]
        if S.seen.intersection(S.ids):
            raise RuntimeError(f'Fold overlap at fold {S.k}')
        S.OFFICIAL_FOLDS[S.k] = S.ids
        S.official_order.extend(S.ids)
        S.seen.update(S.ids)
    S.train_lookup = S.raw_train.set_index('prot_id', drop=False)
    S.missing = set(S.official_order) - set(S.raw_train.prot_id)
    if S.missing:
        raise RuntimeError(f'Missing official Dev IDs: {sorted(S.missing)[:20]}')
    S.train_df = S.train_lookup.loc[S.official_order].reset_index(drop=True)
    assert len(S.train_df) == 1010
    assert len(S.test300_df) == 300
    assert len(S.test46_df) == 46
    S.SPLIT_DF = {'DevSet1010': S.train_df, 'TestSet300': S.test300_df, 'TestSetNew46': S.test46_df}
    S.STORE_KEY = {'DevSet1010': 'train', 'TestSet300': 'test300', 'TestSetNew46': 'test46'}
    S.FOLD_IDX = official_fold_indices()
    print('Dataset/fold bootstrap: PASS')

# SMALL STANDALONE CONTEXT + CACHED EXPLICIT BLOCKS + EVO CTRL
def stage_04():
    S.ADDIN_ROOT = S.RESULTS_ROOT / 'ADDIN_STANDALONE_SMALL_XTASK'
    S.ADDIN_ROOT.mkdir(parents=True, exist_ok=True)
    S.XTASK_ROOT = S.RESULTS_ROOT / 'CROSSTASK_EXPORTS'
    S.XTASK_ROOT.mkdir(parents=True, exist_ok=True)
    S.ctx = S.suite.prepare_context(base=S.BASE, run_version='standalone_small_addin_context', label_column='small_binding', models=('MLP',), ensure_base=True, add_mech=True, add_multi=True, add_seqctx=True, esm_batch_size=8, device=S.DEVICE)
    for S.split, S.df in S.SPLIT_DF.items():
        S.store_ids = set(map(str, S.ctx.stores[S.STORE_KEY[S.split]].keys()))
        S.csv_ids = set(S.df.prot_id.astype(str))
        if S.store_ids != S.csv_ids:
            raise RuntimeError(f'{S.split}: store/CSV mismatch')
    S.FEATURE_CACHE = S.CACHE_ROOT / 'small_ultimate_v1' / 'small_explicit_v1'
    S.NEW_BLOCKS = {'SM_SEQCHEM32': 32, 'SM_POCKET3D44': 44, 'SM_LIGCHEM20': 20, 'SM_TOPO16': 16, 'SM_AROMSTACK12': 12, 'SM_PROTCTX8': 8}
    for S.split, S.df in S.SPLIT_DF.items():
        S.store = S.ctx.stores[S.STORE_KEY[S.split]]
        S.outdir = S.FEATURE_CACHE / S.split
        for S.r in S.df.itertuples():
            S.fp = S.outdir / f'{S.r.prot_id}.npz'
            if not S.fp.exists():
                raise FileNotFoundError(f'Missing cached small feature file: {S.fp}')
            S.z = np.load(S.fp)
            for S.k, S.d in S.NEW_BLOCKS.items():
                S.a = S.z[S.k].astype(np.float32, copy=False)
                if S.a.shape != (len(S.r.sequence), S.d):
                    raise RuntimeError(f'{S.split}/{S.r.prot_id}/{S.k}: {S.a.shape}')
                set_block(S.store, S.r.prot_id, S.k, S.a)
    print('Cached small explicit blocks attached.')
    S.EVO_AVAILABLE = False
    try:
        S.coverage = S.evo.attach_evo(S.ctx.stores, S.BASE, splits=('train', 'test300', 'test46'), require_msa=True)
        display(S.coverage)
        S.EVO_AVAILABLE = True
        print('EVO attached as control.')
    except Exception as e:
        print('EVO unavailable -> skip control.')
        print(repr(e))
    display(S.reporting.dimension_table(S.ctx.stores['train']))

# RECONSTRUCT SMALL V1/MS OOF FROM SAVED FOLD PREDICTIONS
def stage_05():
    S.V1_PRED_DIR = S.RESULTS_ROOT / 'SMALL_ULTIMATE_V1_DATA_DRIVEN_POCKET' / 'pocket_states' / 'SM_PocketHG_Attention'
    S.MS_PRED_DIR = S.RESULTS_ROOT / 'SMALL_V1_ATTENTION_MULTISCALE_POCKETS' / 'states' / 'SM_PocketHG_Attention_MS'
    S.V1_OOF, S.Y_DEV = reconstruct_oof_from_fold_predictions(S.V1_PRED_DIR, 'small_binding')
    S.MS_OOF, S.Y2 = reconstruct_oof_from_fold_predictions(S.MS_PRED_DIR, 'small_binding')
    assert np.array_equal(S.Y_DEV, S.Y2)
    S.ws = weight_search(S.Y_DEV, S.V1_OOF, S.MS_OOF)
    display(S.ws.head(10))
    S.ws.to_csv(S.ADDIN_ROOT / 'DEV_WEIGHT_SEARCH.csv', index=False)
    S.best = S.ws.iloc[0]
    S.W_V1 = float(S.best.w_V1)
    S.W_MS = float(S.best.w_MS)
    S.FINAL_OOF = S.W_V1 * S.V1_OOF + S.W_MS * S.MS_OOF
    print('Locked weights:', S.W_V1, S.W_MS)
    print('Dev mean AUPRC:', float(S.best.CV_AUPRC_mean))
    S.rows = []
    S.pos = 0
    for S.r in S.train_df.itertuples():
        S.L = len(S.r.sequence)
        S.y = np.asarray(S.r.small_binding, dtype=np.int8)
        for S.i in range(S.L):
            S.rows.append({'dataset': 'DevSet1010', 'prot_id': str(S.r.prot_id), 'residue_index': S.i, 'y_small': int(S.y[S.i]), 'prob_FINAL': float(S.FINAL_OOF[S.pos + S.i]), 'prob_V1': float(S.V1_OOF[S.pos + S.i]), 'prob_MS': float(S.MS_OOF[S.pos + S.i])})
        S.pos += S.L
    S.export = pd.DataFrame(S.rows)
    S.export_fp = S.XTASK_ROOT / 'SMALL_DEV_OOF.csv.gz'
    S.tmp_fp = S.XTASK_ROOT / 'SMALL_DEV_OOF.csv.gz.tmp'
    S.export.to_csv(S.tmp_fp, index=False)
    os.replace(S.tmp_fp, S.export_fp)
    print('Small OOF export:', S.export_fp)

# FIX SMALL OOF EXPORT — WRITE REAL GZIP
def stage_06():
    S.XTASK_ROOT = S.RESULTS_ROOT / 'CROSSTASK_EXPORTS'
    S.export_fp = S.XTASK_ROOT / 'SMALL_DEV_OOF.csv.gz'
    S.tmp_fp = S.XTASK_ROOT / 'SMALL_DEV_OOF.tmp.csv.gz'
    assert 'export' in globals(), 'Need the previous OOF export cell state.'
    S.export.to_csv(S.tmp_fp, index=False, compression='gzip')
    os.replace(S.tmp_fp, S.export_fp)
    S._check = pd.read_csv(S.export_fp)
    print('FIXED:', S.export_fp)
    print('rows:', len(S._check))
    print('columns:', S._check.columns.tolist())
    print(S._check.head())

# WAIT FOR NUCLEAR EXPORT -> ATTACH X-TASK -> SCREEN
def stage_07():
    S.COUNTERPART = S.XTASK_ROOT / 'NUCLEAR_DEV_OOF.csv.gz'
    S.WAIT_SECONDS = 900
    S.POLL_SECONDS = 10
    S.t0 = time.time()
    while not S.COUNTERPART.exists() and time.time() - S.t0 < S.WAIT_SECONDS:
        print(f'Waiting for Nuclear export... {int(time.time() - S.t0)}s / {S.WAIT_SECONDS}s', end='\r')
        time.sleep(S.POLL_SECONDS)
    S.XTASK_NUCLEAR_AVAILABLE = S.COUNTERPART.exists()
    print('\nNuclear cross-task export available:', S.XTASK_NUCLEAR_AVAILABLE)
    if S.XTASK_NUCLEAR_AVAILABLE:
        S.xdf = pd.read_csv(S.COUNTERPART)
        S.xdf['prot_id'] = S.xdf['prot_id'].astype(str)
        S.store = S.ctx.stores['train']
        for S.r in S.train_df.itertuples():
            S.pid = str(S.r.prot_id)
            S.sub = S.xdf[S.xdf.prot_id.eq(S.pid)].sort_values('residue_index')
            if len(S.sub) != len(S.r.sequence):
                raise RuntimeError(f'{S.pid}: cross-task length mismatch')
            S.arr = S.sub[['prob_FINAL', 'prob_V1', 'prob_MS']].to_numpy(np.float32)
            set_block(S.store, S.pid, 'XTASK_NUCLEAR3', S.arr)
    S.BASE_BLOCKS = ['ESM', 'ACCESS', 'SM_POCKET3D44', 'SM_TOPO16']
    S.GRID = {'S_BASE_POCKET_GEOMETRY': S.BASE_BLOCKS}
    if S.XTASK_NUCLEAR_AVAILABLE:
        S.GRID['S_BASE_PLUS_XTASK_NUCLEAR'] = S.BASE_BLOCKS + ['XTASK_NUCLEAR3']
    if S.EVO_AVAILABLE:
        S.GRID['S_BASE_PLUS_EVO_CONTROL'] = S.BASE_BLOCKS + ['EVO']
        if S.XTASK_NUCLEAR_AVAILABLE:
            S.GRID['S_BASE_PLUS_EVO_XTASK_NUCLEAR'] = S.BASE_BLOCKS + ['EVO', 'XTASK_NUCLEAR3']
    print('Screening:')
    for S.k, S.v in S.GRID.items():
        print(' ', S.k, '=', S.v)
    S.cv, S.folds = S.suite.run_dev_cv_only(S.ctx, S.GRID, models=('MLP',), run_version='standalone_small_xtask_screen')
    S.base = float(S.cv.loc[S.cv.representation.eq('S_BASE_POCKET_GEOMETRY'), 'CV_AUPRC'].iloc[0])
    S.cv = S.cv.copy()
    S.cv['delta_vs_base'] = S.cv['CV_AUPRC'] - S.base
    S.cv = S.cv.sort_values('CV_AUPRC', ascending=False).reset_index(drop=True)
    display(S.cv)
    display(S.folds)
    S.cv.to_csv(S.ADDIN_ROOT / 'REPRESENTATION_SCREEN.csv', index=False)
    S.folds.to_csv(S.ADDIN_ROOT / 'REPRESENTATION_SCREEN_FOLDS.csv', index=False)
    S.best = S.cv.iloc[0]
    S.decision = {'base_dev_auprc': S.base, 'best_representation': str(S.best.representation), 'best_dev_auprc': float(S.best.CV_AUPRC), 'gain_vs_base': float(S.best.delta_vs_base), 'promote_if_gain_ge_0.002': bool(S.best.delta_vs_base >= 0.002), 'EVO_control_available': bool(S.EVO_AVAILABLE), 'XTASK_nuclear_available': bool(S.XTASK_NUCLEAR_AVAILABLE)}
    (S.ADDIN_ROOT / 'DECISION.json').write_text(json.dumps(S.decision, indent=2))
    print('\nFINAL DECISION')
    print(json.dumps(S.decision, indent=2))
    print('Saved:', S.ADDIN_ROOT)

# Attach ONLY clean OOF component predictions:
def stage_08():
    print('=' * 100)
    print('SMALL CLEAN V3 — TRUE NUCLEAR -> SMALL CROSS-TASK SCREEN')
    print('=' * 100)
    S.COUNTERPART = S.XTASK_ROOT / 'NUCLEAR_DEV_OOF.csv.gz'
    S.WAIT_SECONDS = 300
    S.POLL_SECONDS = 5
    S.t0 = time.time()
    while not S.COUNTERPART.exists() and time.time() - S.t0 < S.WAIT_SECONDS:
        print(f'Waiting for Nuclear export... {int(time.time() - S.t0)}s', end='\r')
        time.sleep(S.POLL_SECONDS)
    if not S.COUNTERPART.exists():
        raise FileNotFoundError(S.COUNTERPART)
    S.xdf = _read_maybe_gzip_csv(S.COUNTERPART)
    S.xdf['prot_id'] = S.xdf['prot_id'].astype(str)
    S.xdf['residue_index'] = S.xdf['residue_index'].astype(int)
    S.required = {'prot_id', 'residue_index', 'y_nuclear', 'prob_V1', 'prob_MS'}
    S.missing = S.required - set(S.xdf.columns)
    if S.missing:
        raise RuntimeError(f'Wrong/old Nuclear export. Missing columns: {sorted(S.missing)}; got {S.xdf.columns.tolist()}')
    if 'y_small' in S.xdf.columns:
        raise RuntimeError('WRONG FILE: Nuclear export contains y_small. Repair NUCLEAR_DEV_OOF first.')
    if S.xdf.duplicated(['prot_id', 'residue_index']).any():
        raise RuntimeError('Duplicate (prot_id,residue_index) in Nuclear OOF export.')
    S.expected_rows = int(S.train_df.sequence.str.len().sum())
    if len(S.xdf) != S.expected_rows:
        raise RuntimeError(f'Nuclear OOF row count mismatch: {len(S.xdf)} vs {S.expected_rows}')
    if set(S.xdf.prot_id) != set(S.train_df.prot_id.astype(str)):
        raise RuntimeError('Nuclear OOF protein IDs do not match DevSet1010.')
    print('Nuclear export integrity: PASS')
    print('rows:', len(S.xdf))
    print('columns:', S.xdf.columns.tolist())
    S.xmap = {pid: g.sort_values('residue_index') for pid, g in S.xdf.groupby('prot_id', sort=False)}
    S.train_store = S.ctx.stores[S.STORE_KEY['DevSet1010']]
    for S.r in S.train_df.itertuples():
        S.pid = str(S.r.prot_id)
        S.sub = S.xmap[S.pid]
        S.L = len(S.r.sequence)
        if len(S.sub) != S.L:
            raise RuntimeError(f'{S.pid}: Nuclear OOF length {len(S.sub)} != sequence length {S.L}')
        if not np.array_equal(S.sub.residue_index.to_numpy(), np.arange(S.L)):
            raise RuntimeError(f'{S.pid}: residue_index is not exactly 0..L-1')
        S.arr = S.sub[['prob_V1', 'prob_MS']].to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(S.arr)):
            raise RuntimeError(f'{S.pid}: non-finite Nuclear OOF probability')
        set_block(S.train_store, S.pid, 'XTASK_NUCLEAR2_CLEAN', S.arr)
    print('Attached XTASK_NUCLEAR2_CLEAN: PASS')
    S.BASE_BLOCKS = ['ESM', 'ACCESS', 'SM_POCKET3D44', 'SM_TOPO16']
    S.GRID = {'S_BASE_POCKET_GEOMETRY_CLEANV3': S.BASE_BLOCKS, 'S_BASE_PLUS_XTASK_NUCLEAR2_CLEANV3': S.BASE_BLOCKS + ['XTASK_NUCLEAR2_CLEAN']}
    S.EVO_OK = bool(globals().get('EVO_AVAILABLE', False))
    if S.EVO_OK:
        S.GRID['S_BASE_PLUS_EVO_CONTROL_CLEANV3'] = S.BASE_BLOCKS + ['EVO']
        S.GRID['S_BASE_PLUS_EVO_XTASK_NUCLEAR2_CLEANV3'] = S.BASE_BLOCKS + ['EVO', 'XTASK_NUCLEAR2_CLEAN']
    print('\nFresh representations:')
    for S.k, S.v in S.GRID.items():
        print(' -', S.k, '=', S.v)
    S.RUN_VERSION = 'standalone_small_xtask_screen_CLEAN_V3_20260914'
    S.cv, S.folds = S.suite.run_dev_cv_only(S.ctx, S.GRID, models=('MLP',), run_version=S.RUN_VERSION)
    S.base_name = 'S_BASE_POCKET_GEOMETRY_CLEANV3'
    S.base = float(S.cv.loc[S.cv.representation.eq(S.base_name), 'CV_AUPRC'].iloc[0])
    S.cv = S.cv.copy()
    S.cv['delta_vs_base'] = S.cv['CV_AUPRC'] - S.base
    S.cv = S.cv.sort_values('CV_AUPRC', ascending=False).reset_index(drop=True)
    print('\nCLEAN V3 SMALL RESULT')
    display(S.cv)
    display(S.folds)
    S.cv.to_csv(S.ADDIN_ROOT / 'REPRESENTATION_SCREEN_CLEAN_V3.csv', index=False)
    S.folds.to_csv(S.ADDIN_ROOT / 'REPRESENTATION_SCREEN_FOLDS_CLEAN_V3.csv', index=False)
    S.best = S.cv.iloc[0]
    S.xt = S.cv[S.cv.representation.eq('S_BASE_PLUS_XTASK_NUCLEAR2_CLEANV3')]
    S.xt_score = float(S.xt.CV_AUPRC.iloc[0])
    S.decision = {'run_version': S.RUN_VERSION, 'base_dev_auprc': S.base, 'xtask_nuclear2_dev_auprc': S.xt_score, 'xtask_nuclear2_gain_vs_base': S.xt_score - S.base, 'best_representation': str(S.best.representation), 'best_dev_auprc': float(S.best.CV_AUPRC), 'best_gain_vs_base': float(S.best.delta_vs_base), 'promote_best_if_gain_ge_0.002': bool(S.best.delta_vs_base >= 0.002), 'EVO_control_available': S.EVO_OK, 'cross_task_source_columns': ['prob_V1', 'prob_MS']}
    (S.ADDIN_ROOT / 'DECISION_CLEAN_V3.json').write_text(json.dumps(S.decision, indent=2))
    print('\nDECISION')
    print(json.dumps(S.decision, indent=2))
    print('Saved:', S.ADDIN_ROOT)

STAGES = [stage_01, stage_02, stage_03, stage_04, stage_05, stage_06, stage_07, stage_08]


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
    stage_01()
    _apply(overrides)
    stage_02()
    _apply(overrides)
    stage_03()
    _apply(overrides)
    stage_04()
    _apply(overrides)
    stage_05()
    _apply(overrides)
    stage_06()
    _apply(overrides)
    stage_07()
    _apply(overrides)
    stage_08()
    _apply(overrides)
    S.RESULT_KEY = 'small_xtask'
    return S

