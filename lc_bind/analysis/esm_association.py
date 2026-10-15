"""Extracted from ESM_FEATURE_ASSOCIATION_TESTS.ipynb. Section: esm_association."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global ConditionalLogit, GroupKFold, PCA, Path, StandardScaler, ast, get_block, get_record, np, parse_list, pd, set_block, stats, subprocess, sys, torch, warnings
    from pathlib import Path
    import sys, subprocess, warnings
    import numpy as np
    import torch
    import pandas as pd
    from scipy import stats
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from statsmodels.discrete.conditional_models import ConditionalLogit
    import ast
    from lc_bind.io.datasets import parse_list
    from lc_bind.utils.cache import get_block, get_record, set_block

def list_store_blocks(store, prot_id):
    record = get_record(store, prot_id)
    keys = []
    if isinstance(record, dict):
        features = record.get('features')
        keys = list(features) if isinstance(features, dict) else list(record)
    else:
        for attr in ('features', 'feature_dict', 'blocks'):
            mapping = getattr(record, attr, None)
            if isinstance(mapping, dict):
                keys = list(mapping)
                break
    return [key for key in keys if key != 'ESM' and (not str(key).startswith('_'))]

def attach_npz_blocks(store, frame, cache_dir, expected):
    """Attach cached ligand blocks when every DevSet protein file is present."""
    cache_dir = Path(cache_dir)
    paths = [cache_dir / f'{prot_id}.npz' for prot_id in frame['prot_id']]
    if not all((path.exists() for path in paths)):
        missing = sum((not path.exists() for path in paths))
        print(f'Skip cache {cache_dir}: missing {missing}/{len(paths)} files')
        return []
    attached = []
    for row, path in zip(frame.itertuples(), paths):
        archive = np.load(path)
        for name in expected:
            values = np.asarray(archive[name], dtype=np.float32)
            if values.shape[0] != len(row.sequence):
                raise RuntimeError(f'{row.prot_id}/{name}: {values.shape} vs L={len(row.sequence)}')
            set_block(store, row.prot_id, name, values)
        attached = list(expected)
    print('Attached', attached, 'from', cache_dir)
    return attached

def complete_blocks(store, frame, block_names):
    """Keep blocks present at the sequence length for every protein."""
    kept = []
    for name in block_names:
        ok = True
        for row in frame.itertuples():
            try:
                values = get_block(store, row.prot_id, name)
            except KeyError:
                ok = False
                break
            values = np.asarray(values)
            if values.ndim != 2 or values.shape[0] != len(row.sequence):
                ok = False
                break
        if ok:
            kept.append(name)
        else:
            print(f'Skip incomplete block {name}')
    return kept

def stack_residue_table(store, frame, label_column, block_names):
    """Return ESM, labels, protein ids, and a dict of feature matrices."""
    block_names = complete_blocks(store, frame, block_names)
    esm_parts = []
    y_parts = []
    id_parts = []
    block_parts = {name: [] for name in block_names}
    for row in frame.itertuples():
        length = len(row.sequence)
        labels = np.asarray(getattr(row, label_column), dtype=np.float64)
        embedding = np.asarray(get_block(store, row.prot_id, 'ESM'), dtype=np.float32)
        if embedding.shape[0] != length or len(labels) != length:
            raise RuntimeError(f'{row.prot_id}: ESM/label length mismatch')
        esm_parts.append(embedding)
        y_parts.append(labels)
        id_parts.append(np.full(length, row.prot_id, dtype=object))
        for name in block_names:
            block_parts[name].append(np.asarray(get_block(store, row.prot_id, name), dtype=np.float32))
    return {'esm': np.concatenate(esm_parts, axis=0), 'y': np.concatenate(y_parts, axis=0), 'prot_id': np.concatenate(id_parts, axis=0), 'blocks': {name: np.concatenate(parts, axis=0) for name, parts in block_parts.items()}, 'n_proteins': int(len(frame))}

def _ridge_weights(x_train, y_train, alpha):
    gram = x_train.T @ x_train
    gram = gram.astype(np.float64, copy=False)
    gram.flat[::gram.shape[0] + 1] += float(alpha)
    rhs = x_train.T @ y_train
    return np.linalg.solve(gram, rhs.astype(np.float64, copy=False))

def probe_r2_block(esm, features, groups):
    """Out-of-fold R² of each feature column predicted from raw ESM."""
    n_cols = features.shape[1]
    oof = np.full((features.shape[0], n_cols), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=min(S.PROBE_SPLITS, len(np.unique(groups))))
    for train_idx, test_idx in splitter.split(esm, groups=groups):
        inner = GroupKFold(n_splits=min(5, len(np.unique(groups[train_idx]))))
        inner_train, inner_val = next(inner.split(esm[train_idx], groups=groups[train_idx]))
        fit_idx = train_idx[inner_train]
        val_idx = train_idx[inner_val]
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(esm[fit_idx]).astype(np.float32, copy=False)
        x_val = scaler.transform(esm[val_idx]).astype(np.float32, copy=False)
        y_mu = features[fit_idx].mean(axis=0)
        y_fit = features[fit_idx] - y_mu
        mse = []
        for alpha in S.PROBE_ALPHAS:
            weight = _ridge_weights(x_fit, y_fit, alpha)
            pred = x_val @ weight.astype(np.float32) + y_mu
            mse.append(np.mean((features[val_idx] - pred) ** 2, axis=0))
        mse = np.vstack(mse)
        chosen = S.PROBE_ALPHAS[np.argmin(mse, axis=0)]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(esm[train_idx]).astype(np.float32, copy=False)
        x_test = scaler.transform(esm[test_idx]).astype(np.float32, copy=False)
        y_mu = features[train_idx].mean(axis=0)
        y_train = features[train_idx] - y_mu
        prediction = np.empty((len(test_idx), n_cols), dtype=np.float64)
        for alpha in np.unique(chosen):
            cols = np.flatnonzero(chosen == alpha)
            weight = _ridge_weights(x_train, y_train[:, cols], alpha)
            prediction[:, cols] = x_test @ weight.astype(np.float32) + y_mu[cols]
        oof[test_idx] = prediction
    y_mu = features.mean(axis=0)
    ss_tot = np.sum((features - y_mu) ** 2, axis=0)
    ss_res = np.sum((features - oof) ** 2, axis=0)
    r2 = np.full(n_cols, np.nan, dtype=np.float64)
    usable = ss_tot > 1e-08
    r2[usable] = 1.0 - ss_res[usable] / ss_tot[usable]
    return r2

def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    valid = np.isfinite(p_values)
    raw = p_values[valid]
    m = len(raw)
    if m == 0:
        return adjusted
    order = np.argsort(raw)
    ranked = raw[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = ranked
    adjusted[valid] = out
    return adjusted

def _varying_mask(y, groups):
    keep = np.zeros(len(y), dtype=bool)
    for group in pd.unique(groups):
        idx = np.flatnonzero(groups == group)
        labels = y[idx]
        if len(labels) >= 2 and labels.min() != labels.max():
            keep[idx] = True
    return keep

def _design_pcs(values, n_components, unsupervised_columns):
    """Standardize. Use raw columns when dim <= cap, else PCA."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    dim = scaled.shape[1]
    if dim <= unsupervised_columns:
        keep = np.nanstd(scaled, axis=0) > 1e-08
        if not np.any(keep):
            raise RuntimeError('Block has no variance')
        return scaled[:, keep]
    n_components = min(n_components, dim, scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=S.SEED)
    return pca.fit_transform(scaled)

def nested_lrt(y, groups, esm, block):
    """Protein-stratified conditional logit: ESM PCs vs ESM PCs + block."""
    mask = _varying_mask(y, groups)
    y = y[mask]
    groups = groups[mask]
    esm = esm[mask]
    block = block[mask]
    esm_design = _design_pcs(esm, S.N_ESM_PCS, S.N_ESM_PCS)
    block_design = _design_pcs(block, S.BLOCK_PC_CAP, S.BLOCK_PC_CAP)
    order = np.argsort(groups, kind='mergesort')
    y = y[order]
    groups = groups[order]
    esm_design = esm_design[order]
    block_design = block_design[order]
    codes = pd.factorize(groups, sort=False)[0]
    null = ConditionalLogit(y, esm_design, groups=codes).fit(disp=0, method='newton', maxiter=100)
    alt_design = np.concatenate([esm_design, block_design], axis=1)
    alt = ConditionalLogit(y, alt_design, groups=codes).fit(disp=0, method='newton', maxiter=100)
    df = int(block_design.shape[1])
    lr_stat = float(2.0 * (alt.llf - null.llf))
    if lr_stat < 0:
        lr_stat = np.nan
        p_value = np.nan
    else:
        p_value = float(stats.chi2.sf(lr_stat, df))
    return {'n_residues': int(len(y)), 'n_proteins': int(len(np.unique(codes))), 'n_esm_pcs': int(esm_design.shape[1]), 'n_block_pcs': df, 'll_m0': float(null.llf), 'll_m1': float(alt.llf), 'lr_stat': lr_stat, 'df': df, 'p_value': p_value}

def verdict_row(mean_r2, fdr):
    high = np.isfinite(mean_r2) and mean_r2 >= S.HIGH_R2
    sig = np.isfinite(fdr) and fdr < S.FDR_ALPHA
    if high and sig:
        return 'overlaps_esm_residual_binding_signal'
    if high and (not sig):
        return 'redundant_with_esm'
    if not high and sig:
        return 'complementary_and_associated'
    return 'not_redundant_not_associated'

# 0. DRIVE / INSTALL / IMPORT
def stage_01():
    warnings.filterwarnings('ignore', category=FutureWarning)
    S.BASE = Path(str(S.BASE))
    S.ULTIMATE_CANDIDATES = [S.GEAR_DIR, S.BASE / 'GEAR_ULTIMATE_OTHER_BINDINGS', S.BASE / 'GEAR_ULTIMATE']
    S.ULTIMATE_DIR = next((p for p in S.ULTIMATE_CANDIDATES if p is not None and (Path(p) / 'gear_ultimate').exists()), None)
    if S.ULTIMATE_DIR is None:
        raise FileNotFoundError('GEAR backend not found under GEAR_ULTIMATE_OTHER_BINDINGS or GEAR_ULTIMATE.')
    S.req = S.ULTIMATE_DIR / 'requirements.txt'
    if S.req.exists():
        pass
    sys.path.insert(0, str(S.ULTIMATE_DIR))
    from gear_ultimate import gear_suite as suite
    S.suite = suite
    S.OUT_DIR = S.RESULTS_ROOT / 'esm_feature_association'
    S.OUT_DIR.mkdir(parents=True, exist_ok=True)
    S.ESM_BATCH_SIZE = 8
    S.SEED = 2026
    S.N_ESM_PCS = 32
    S.BLOCK_PC_CAP = 6
    S.PROBE_SPLITS = 5
    S.PROBE_ALPHAS = np.logspace(-1, 5, 8)
    S.HIGH_R2 = 0.5
    S.FDR_ALPHA = 0.05
    print('BASE:', S.BASE)
    print('GEAR backend:', S.ULTIMATE_DIR)
    print('OUT_DIR:', S.OUT_DIR)

# 1. DEVSET LABELS
def stage_02():
    S.TRAIN_CSV = S.DATA_ROOT / 'development_set' / 'full_grouped_train_binding_sites_df.csv'
    S.LABEL_COLUMNS = ['metal_binding', 'nuclear_binding', 'small_binding', 'any_ligand_binding_sites']
    S.dev_df = pd.read_csv(S.TRAIN_CSV)
    S.dev_df['prot_id'] = S.dev_df['prot_id'].astype(str)
    S.dev_df['sequence'] = S.dev_df['sequence'].astype(str)
    for S.column in S.LABEL_COLUMNS:
        if S.column in S.dev_df.columns:
            S.dev_df[S.column] = S.dev_df[S.column].apply(parse_list)
    S.fold_root_candidates = [S.BASE / 'baseline_reproduction_master_v6' / 'repos' / 'bindnode24' / 'dataset' / 'cv_splits' / 'predicted_structures', S.BASE / 'repos' / 'bindnode24' / 'dataset' / 'cv_splits' / 'predicted_structures']
    S.fold_root = next((p for p in S.fold_root_candidates if p.exists()), None)
    if S.fold_root is None:
        raise FileNotFoundError('Official cv split files were not found on Drive.')
    S.official_order = []
    S.seen = set()
    for S.fold_id in range(1, 6):
        S.ids = [line.strip() for line in (S.fold_root / f'ids_split{S.fold_id}.txt').read_text().splitlines() if line.strip()]
        if S.seen.intersection(S.ids):
            raise RuntimeError(f'Fold overlap at fold {S.fold_id}')
        S.official_order.extend(S.ids)
        S.seen.update(S.ids)
    S.dev_df = S.dev_df.set_index('prot_id', drop=False).loc[S.official_order].reset_index(drop=True)
    assert len(S.dev_df) == 1010
    print('DevSet1010:', len(S.dev_df), 'proteins')

# 2. STORE HELPERS + STATISTICS
def stage_03():
    S.SHARED_BLOCKS = ['AA', 'SPATIAL', 'LOCAL', 'ATOM', 'CHEM', 'ACCESS', 'SHELL2', 'ENV', 'CONF', 'METALSPEC', 'ELEC3D', 'TOPO', 'SOLV', 'SEQMOTIF', 'SEQCTX']
    print('Helpers ready')

# 3. LIGAND CONFIGS
def stage_04():
    S.LIGANDS = {'small': {'label_column': 'small_binding', 'run_version': 'small_ultimate_v1_context', 'models': ('MLP',), 'cache_dir': S.CACHE_ROOT / 'small_ultimate_v1' / 'small_explicit_v1' / 'DevSet1010', 'extra_blocks': ['SM_SEQCHEM32', 'SM_POCKET3D44', 'SM_LIGCHEM20', 'SM_TOPO16', 'SM_AROMSTACK12', 'SM_PROTCTX8']}, 'nuclear': {'label_column': 'nuclear_binding', 'run_version': 'nuclear_ultimate_v1_context', 'models': ('MLP',), 'cache_dir': S.CACHE_ROOT / 'nuclear_ultimate_v1' / 'nuclear_explicit_v1' / 'DevSet1010', 'extra_blocks': ['NA_SEQELECTRO32', 'NA_PATCH3D40', 'NA_HBOND16', 'NA_TOPO12', 'NA_PROTCTX8', 'NA_AROMCTRL8']}, 'metal': {'label_column': 'metal_binding', 'run_version': 'metal_v2_context', 'models': ('MLP', 'KAN', 'XGBoost'), 'cache_dir': None, 'extra_blocks': ['METALSPEC', 'CHEM', 'TOPO', 'ACCESS', 'ATOM', 'SOLV', 'SPATIAL', 'LOCAL', 'SEQMOTIF']}}
    S.RUN_LIGANDS = ['small', 'nuclear', 'metal']
    print('Will run:', S.RUN_LIGANDS)

# 4. RUN PROBE R² + NESTED LRT
def stage_05():
    S.summaries = []
    for S.ligand in S.RUN_LIGANDS:
        S.cfg = S.LIGANDS[S.ligand]
        print('=' * 72)
        print('LIGAND', S.ligand)
        S.ctx = S.suite.prepare_context(base=S.BASE, run_version=S.cfg['run_version'], label_column=S.cfg['label_column'], models=S.cfg['models'], ensure_base=True, add_mech=True, add_multi=True, add_seqctx=True, esm_batch_size=S.ESM_BATCH_SIZE, device=torch.device('cpu'))
        S.store = S.ctx.stores['train']
        if len(S.store) != 1010:
            raise RuntimeError(f'{S.ligand}: expected 1010 train proteins, got {len(S.store)}')
        if S.cfg['cache_dir'] is not None:
            attach_npz_blocks(S.store, S.dev_df, S.cfg['cache_dir'], S.cfg['extra_blocks'])
        S.discovered = list_store_blocks(S.store, S.dev_df.iloc[0]['prot_id'])
        S.block_names = []
        for S.name in list(S.cfg['extra_blocks']) + S.SHARED_BLOCKS + S.discovered:
            if S.name not in S.block_names and S.name != 'ESM':
                S.block_names.append(S.name)
        S.table = stack_residue_table(S.store, S.dev_df, S.cfg['label_column'], S.block_names)
        S.present = [name for name in S.block_names if name in S.table['blocks']]
        print(f'stacked proteins={S.table['n_proteins']} residues={len(S.table['y'])} positives={int(S.table['y'].sum())} blocks={S.present}')
        S.probe_rows = []
        S.lrt_rows = []
        for S.name in S.present:
            S.features = S.table['blocks'][S.name].astype(np.float32, copy=False)
            print(f'  probe {S.name} dim={S.features.shape[1]}')
            S.r2 = probe_r2_block(S.table['esm'], S.features, S.table['prot_id'])
            for S.index, S.value in enumerate(S.r2):
                S.probe_rows.append({'ligand': S.ligand, 'block': S.name, 'feature_index': S.index, 'oof_r2': S.value, 'n_residues': int(len(S.table['y'])), 'n_proteins': int(S.table['n_proteins'])})
            print(f'  lrt {S.name}')
            try:
                S.stats_row = nested_lrt(S.table['y'], S.table['prot_id'], S.table['esm'], S.features)
            except Exception as exc:
                print('  LRT failed', S.name, repr(exc))
                S.stats_row = {'n_residues': np.nan, 'n_proteins': np.nan, 'n_esm_pcs': np.nan, 'n_block_pcs': np.nan, 'll_m0': np.nan, 'll_m1': np.nan, 'lr_stat': np.nan, 'df': np.nan, 'p_value': np.nan}
            S.stats_row.update({'ligand': S.ligand, 'block': S.name, 'block_dim': int(S.features.shape[1])})
            S.lrt_rows.append(S.stats_row)
        S.probe_df = pd.DataFrame(S.probe_rows)
        S.lrt_df = pd.DataFrame(S.lrt_rows)
        S.lrt_df['fdr'] = benjamini_hochberg(S.lrt_df['p_value'].to_numpy())
        S.probe_path = S.OUT_DIR / f'esm_probe_r2_{S.ligand}.csv'
        S.lrt_path = S.OUT_DIR / f'esm_nested_lrt_{S.ligand}.csv'
        S.probe_df.to_csv(S.probe_path, index=False)
        S.lrt_df.to_csv(S.lrt_path, index=False)
        S.summary = S.probe_df.groupby('block', sort=False)['oof_r2'].agg(probe_r2_mean='mean', probe_r2_median='median').reset_index()
        S.summary = S.summary.merge(S.lrt_df, on='block', how='outer')
        S.summary['verdict'] = [verdict_row(mean_r2, fdr) for mean_r2, fdr in zip(S.summary['probe_r2_mean'], S.summary['fdr'])]
        S.verdict_path = S.OUT_DIR / f'esm_feature_verdicts_{S.ligand}.csv'
        S.summary.to_csv(S.verdict_path, index=False)
        S.summaries.append(S.summary)
        print('Wrote', S.probe_path)
        print('Wrote', S.lrt_path)
        print('Wrote', S.verdict_path)
        display(S.summary)
    print('Finished ligands:', S.RUN_LIGANDS)

# 5. MARKDOWN SUMMARY
def stage_06():
    S.lines = ['# ESM feature association (DevSet1010)', '', f'High Probe R² means mean out-of-fold R² ≥ {S.HIGH_R2:.2f}.', f'Significant LRT means BH FDR < {S.FDR_ALPHA:.2f}.', '']
    for S.summary in S.summaries:
        S.ligand = S.summary['ligand'].iloc[0]
        S.lines.append(f'## {S.ligand}')
        S.lines.append('')
        S.lines.append('| block | probe R² mean | probe R² median | LRT | df | FDR | verdict |')
        S.lines.append('|---|---:|---:|---:|---:|---:|---|')
        for S.row in S.summary.itertuples(index=False):
            S.lines.append('| {block} | {mean:.3f} | {med:.3f} | {lr} | {df} | {fdr} | {verdict} |'.format(block=S.row.block, mean=S.row.probe_r2_mean, med=S.row.probe_r2_median, lr='nan' if not np.isfinite(S.row.lr_stat) else f'{S.row.lr_stat:.2f}', df='nan' if not np.isfinite(S.row.df) else str(int(S.row.df)), fdr='nan' if not np.isfinite(S.row.fdr) else f'{S.row.fdr:.3g}', verdict=S.row.verdict))
        S.lines.append('')
    S.report = '\n'.join(S.lines)
    print(S.report)
    S.report_path = S.OUT_DIR / 'esm_feature_summary.md'
    S.report_path.write_text(S.report)
    print('Wrote', S.report_path)

STAGES = [stage_01, stage_02, stage_03, stage_04, stage_05, stage_06]


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
    S.RESULT_KEY = 'esm_association'
    return S

