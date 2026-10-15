"""Extracted from ALL_BINDINGS_DATASET_ANATOMY_DEV1010_300_46-1.ipynb. Section: anatomy."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Counter, Path, ast, fisher_exact, get_block, jensenshannon, json, math, np, os, parse_list, pd, subprocess, sys
    from pathlib import Path
    import ast, json, math, os, subprocess, sys
    from collections import Counter
    import numpy as np
    import pandas as pd
    from scipy.stats import fisher_exact
    from scipy.spatial.distance import jensenshannon
    from lc_bind.io.datasets import parse_list
    from lc_bind.utils.cache import get_block

def load_df(path):
    df = pd.read_csv(path)
    df['prot_id'] = df['prot_id'].astype(str)
    df['sequence'] = df['sequence'].astype(str)
    for c in S.ALL_LABELS:
        if c in df.columns:
            df[c] = df[c].apply(parse_list)
    if not df.prot_id.is_unique:
        raise RuntimeError(f'{path}: duplicate prot_id')
    for r in df.itertuples():
        for c in S.ALL_LABELS:
            if hasattr(r, c):
                y = getattr(r, c)
                if len(y) != len(r.sequence):
                    raise RuntimeError(f'{path}/{r.prot_id}/{c}: {len(y)} != {len(r.sequence)}')
    return df

def frac(seq, i, chars, r):
    lo = max(0, i - r)
    hi = min(len(seq), i + r + 1)
    w = seq[lo:hi]
    return sum((x in chars for x in w)) / max(len(w), 1)

def clusters(pos, max_gap):
    pos = np.asarray(pos, dtype=int)
    if len(pos) == 0:
        return []
    out = [[int(pos[0])]]
    for x in pos[1:]:
        if x - out[-1][-1] <= max_gap:
            out[-1].append(int(x))
        else:
            out.append([int(x)])
    return out

def aa_dist(df, task=None, positive_only=False):
    c = Counter()
    for rec in df.itertuples():
        if not positive_only:
            c.update(rec.sequence)
        else:
            y = np.asarray(getattr(rec, task), dtype=np.int8)
            c.update([aa for aa, yy in zip(rec.sequence, y) if yy])
    v = np.asarray([c[a] for a in S.AA20], dtype=float)
    return v / v.sum() if v.sum() else np.zeros_like(v)

def stability_flag(row):
    corr = row['AA_enrichment_correlation_vs_Dev']
    js = row['JS_positive_AA_vs_Dev']
    pr = row['prevalence_ratio_vs_Dev']
    if row['dataset'] == 'DevSet1010':
        return 'REFERENCE'
    if corr >= 0.75 and js <= 0.25 and (0.5 <= pr <= 2.0):
        return 'STABLE'
    if corr >= 0.5 and js <= 0.4:
        return 'MODERATE_SHIFT'
    return 'STRONG_SHIFT'

# 0. DRIVE / DEPENDENCIES / PATH AUTO-DETECTION
def stage_01():
    S.CANDIDATE_BASES = [Path(str(S.BASE)), Path(str(S.ALT_BASE))]
    S.BASE = None
    for S.b in S.CANDIDATE_BASES:
        if (S.b / 'data/development_set/full_grouped_train_binding_sites_df.csv').exists() and (S.b / 'data/development_set/full_grouped_test_binding_sites_df.csv').exists() and (S.b / 'data/independent_set/grouped_test_46_new_binding_sites.csv').exists():
            S.BASE = S.b
            break
    if S.BASE is None:
        raise FileNotFoundError('Project CSVs not found.')
    S.OUT = S.RESULTS_ROOT / 'ALL_BINDINGS_DATASET_ANATOMY_V1'
    S.OUT.mkdir(parents=True, exist_ok=True)
    S.TRAIN_CSV = S.DATA_ROOT / 'development_set/full_grouped_train_binding_sites_df.csv'
    S.TEST300_CSV = S.DATA_ROOT / 'development_set/full_grouped_test_binding_sites_df.csv'
    S.TEST46_CSV = S.DATA_ROOT / 'independent_set/grouped_test_46_new_binding_sites.csv'
    S.TASKS = ['metal_binding', 'nuclear_binding', 'small_binding']
    S.ALL_LABELS = S.TASKS + ['any_ligand_binding_sites']
    print('BASE =', S.BASE)
    print('OUT  =', S.OUT)

# 1. LOAD DATA + EXACT OFFICIAL DevSet1010
def stage_02():
    S.raw_train = load_df(S.TRAIN_CSV)
    S.test300 = load_df(S.TEST300_CSV)
    S.test46 = load_df(S.TEST46_CSV)
    S.other = Path(str(S.ALT_BASE)) if S.BASE.name == 'Protein-binding-Phi' else Path(str(S.BASE))
    S.BN24_CANDIDATES = [S.BASE / 'baseline_reproduction_master_v6' / 'repos' / 'bindnode24', S.BASE / 'repos' / 'bindnode24', S.other / 'baseline_reproduction_master_v6' / 'repos' / 'bindnode24', S.other / 'repos' / 'bindnode24']
    S.BN24 = next((p for p in S.BN24_CANDIDATES if (p / '.git').exists()), None)
    if S.BN24 is None:
        S.BN24 = Path(str(S.CACHE_ROOT / 'bindnode24_dataset_anatomy'))
        if not S.BN24.exists():
            subprocess.run(['git', 'clone', '-q', 'https://github.com/erckert/bindNode24.git', str(S.BN24)], check=True)
    S.fold_root = S.BN24 / 'dataset' / 'cv_splits' / 'predicted_structures'
    S.OFFICIAL_FOLDS = {}
    S.official_order = []
    S.seen = set()
    for S.k in range(1, 6):
        S.fp = S.fold_root / f'ids_split{S.k}.txt'
        S.ids = [x.strip() for x in S.fp.read_text().splitlines() if x.strip()]
        S.overlap = S.seen.intersection(S.ids)
        if S.overlap:
            raise RuntimeError(f'Fold overlap: {sorted(S.overlap)[:10]}')
        S.OFFICIAL_FOLDS[S.k] = S.ids
        S.official_order.extend(S.ids)
        S.seen.update(S.ids)
    S.official = set(S.official_order)
    S.raw_ids = set(S.raw_train.prot_id)
    S.missing = S.official - S.raw_ids
    if S.missing:
        raise RuntimeError(f'Official Dev IDs missing from project CSV: {sorted(S.missing)[:20]}')
    S.train_lookup = S.raw_train.set_index('prot_id', drop=False)
    S.dev = S.train_lookup.loc[S.official_order].reset_index(drop=True)
    assert len(S.dev) == 1010
    assert len(S.test300) == 300
    assert len(S.test46) == 46
    S.SETS = {'DevSet1010': S.dev, 'TestSet300': S.test300, 'TestSetNew46': S.test46}
    print('Raw dev lineage:', len(S.raw_train))
    print('Exact DevSet1010:', len(S.dev))
    print('Excluded IDs:', sorted(S.raw_ids - S.official))
    print('Fold sizes:', {k: len(v) for k, v in S.OFFICIAL_FOLDS.items()})

# 2. CORE PREVALENCE / PROTEIN BURDEN
def stage_04():
    S.rows = []
    for S.dataset, S.df in S.SETS.items():
        for S.task in S.TASKS:
            S.y_all = np.concatenate([np.asarray(x, dtype=np.int8) for x in S.df[S.task]])
            S.per_protein = np.asarray([int(np.sum(x)) for x in S.df[S.task]])
            S.pos_protein = S.per_protein > 0
            S.rows.append({'dataset': S.dataset, 'task': S.task, 'n_proteins': len(S.df), 'n_residues': len(S.y_all), 'n_positive_residues': int(S.y_all.sum()), 'residue_prevalence': float(S.y_all.mean()), 'n_positive_proteins': int(S.pos_protein.sum()), 'positive_protein_fraction': float(S.pos_protein.mean()), 'median_positive_residues_per_positive_protein': float(np.median(S.per_protein[S.pos_protein])) if S.pos_protein.any() else 0, 'mean_positive_residues_per_positive_protein': float(np.mean(S.per_protein[S.pos_protein])) if S.pos_protein.any() else 0, 'max_positive_residues_one_protein': int(S.per_protein.max()), 'median_protein_length': float(np.median(S.df.sequence.str.len()))})
    S.core = pd.DataFrame(S.rows)
    display(S.core)
    S.pivot_prev = S.core.pivot(index='task', columns='dataset', values='residue_prevalence')
    print('Residue prevalence by task/split:')
    display(S.pivot_prev)
    S.core.to_csv(S.OUT / 'core_task_anatomy.csv', index=False)

# 3. PAIRWISE LABEL OVERLAP
def stage_06():
    S.pairs = [('metal_binding', 'nuclear_binding'), ('metal_binding', 'small_binding'), ('nuclear_binding', 'small_binding')]
    S.rows = []
    for S.dataset, S.df in S.SETS.items():
        for S.a, S.b in S.pairs:
            S.ya = np.concatenate([np.asarray(x, dtype=bool) for x in S.df[S.a]])
            S.yb = np.concatenate([np.asarray(x, dtype=bool) for x in S.df[S.b]])
            S.pa = np.array([np.sum(x) > 0 for x in S.df[S.a]], dtype=bool)
            S.pb = np.array([np.sum(x) > 0 for x in S.df[S.b]], dtype=bool)
            S.inter = int((S.ya & S.yb).sum())
            S.union = int((S.ya | S.yb).sum())
            S.rows.append({'dataset': S.dataset, 'task_a': S.a, 'task_b': S.b, 'residue_intersection': S.inter, 'residue_jaccard': S.inter / max(S.union, 1), 'fraction_a_positive_residues_also_b': S.inter / max(int(S.ya.sum()), 1), 'fraction_b_positive_residues_also_a': S.inter / max(int(S.yb.sum()), 1), 'protein_intersection': int((S.pa & S.pb).sum()), 'protein_jaccard': int((S.pa & S.pb).sum()) / max(int((S.pa | S.pb).sum()), 1)})
    S.overlap = pd.DataFrame(S.rows)
    display(S.overlap)
    S.overlap.to_csv(S.OUT / 'pairwise_label_overlap.csv', index=False)

# 4. AA ENRICHMENT FINGERPRINT
def stage_08():
    S.AA20 = list('ACDEFGHIKLMNPQRSTVWY')
    S.rows = []
    for S.dataset, S.df in S.SETS.items():
        for S.task in S.TASKS:
            S.pos = []
            S.neg = []
            S.allseq = []
            for S.r in S.df.itertuples():
                S.seq = S.r.sequence
                S.y = np.asarray(getattr(S.r, S.task), dtype=np.int8)
                S.allseq.extend(S.seq)
                S.pos.extend([aa for aa, yy in zip(S.seq, S.y) if yy])
                S.neg.extend([aa for aa, yy in zip(S.seq, S.y) if not yy])
            S.ac = Counter(S.allseq)
            S.pc = Counter(S.pos)
            S.nc = Counter(S.neg)
            S.nall = sum(S.ac.values())
            S.npos = sum(S.pc.values())
            S.nneg = sum(S.nc.values())
            for S.aa in S.AA20:
                S.a = S.pc[S.aa]
                S.b = S.npos - S.a
                S.c = S.nc[S.aa]
                S.d = S.nneg - S.c
                S.odds, S.p = fisher_exact([[S.a, S.b], [S.c, S.d]])
                S.fa = S.ac[S.aa] / max(S.nall, 1)
                S.fp = S.a / max(S.npos, 1)
                S.rows.append({'dataset': S.dataset, 'task': S.task, 'aa': S.aa, 'global_frequency': S.fa, 'positive_frequency': S.fp, 'log2_enrichment': float(np.log2((S.fp + 1e-08) / (S.fa + 1e-08))), 'odds_ratio': float(S.odds), 'fisher_p': float(S.p), 'positive_count': int(S.a)})
    S.aa = pd.DataFrame(S.rows)
    S.aa.to_csv(S.OUT / 'aa_enrichment_all_tasks.csv', index=False)
    for S.task in S.TASKS:
        print('\n###', S.task)
        S.p = S.aa[S.aa.task.eq(S.task)].pivot(index='aa', columns='dataset', values='log2_enrichment')
        S.p['mean'] = S.p.mean(axis=1)
        display(S.p.sort_values('mean', ascending=False))

# 5. TASK-SIGNATURE SIMILARITY
def stage_10():
    S.task_signature_rows = []
    for S.dataset in S.SETS:
        S.p = S.aa[S.aa.dataset.eq(S.dataset)].pivot(index='aa', columns='task', values='log2_enrichment')
        S.corr = S.p.corr()
        print('\n', S.dataset, 'AA-enrichment signature correlation')
        display(S.corr)
        for S.a, S.b in S.pairs:
            S.task_signature_rows.append({'dataset': S.dataset, 'task_a': S.a, 'task_b': S.b, 'AA_enrichment_correlation': float(S.corr.loc[S.a, S.b])})
    S.task_signature_similarity = pd.DataFrame(S.task_signature_rows)
    S.task_signature_similarity.to_csv(S.OUT / 'task_signature_similarity.csv', index=False)

# 6. LOCAL SEQUENCE-CONTEXT FINGERPRINTS
def stage_12():
    S.GROUPS = {'basic_KRH': set('KRH'), 'strong_basic_KR': set('KR'), 'acidic_DE': set('DE'), 'polar_NQSTY': set('NQSTY'), 'aromatic_FWYH': set('FWYH'), 'hydrophobic_AVILMFWY': set('AVILMFWY'), 'sulfur_CM': set('CM'), 'gly_pro_GP': set('GP')}
    S.RADII = [2, 4, 7, 10]
    S.rows = []
    for S.dataset, S.df in S.SETS.items():
        for S.task in S.TASKS:
            S.posvals = {(g, r): [] for g in S.GROUPS for r in S.RADII}
            S.negvals = {(g, r): [] for g in S.GROUPS for r in S.RADII}
            for S.rec in S.df.itertuples():
                S.y = np.asarray(getattr(S.rec, S.task), dtype=np.int8)
                for S.i, S.yy in enumerate(S.y):
                    S.target = S.posvals if S.yy else S.negvals
                    for S.g, S.chars in S.GROUPS.items():
                        for S.rad in S.RADII:
                            S.target[S.g, S.rad].append(frac(S.rec.sequence, S.i, S.chars, S.rad))
            for S.g in S.GROUPS:
                for S.rad in S.RADII:
                    S.p = np.asarray(S.posvals[S.g, S.rad], dtype=float)
                    S.n = np.asarray(S.negvals[S.g, S.rad], dtype=float)
                    S.sp = np.sqrt((S.p.var() + S.n.var()) / 2 + 1e-12)
                    S.smd = (S.p.mean() - S.n.mean()) / S.sp
                    S.rows.append({'dataset': S.dataset, 'task': S.task, 'feature': S.g, 'radius': S.rad, 'positive_mean': float(S.p.mean()), 'negative_mean': float(S.n.mean()), 'standardized_mean_difference': float(S.smd)})
    S.seqctx = pd.DataFrame(S.rows)
    S.seqctx.to_csv(S.OUT / 'sequence_context_all_tasks.csv', index=False)
    for S.dataset in S.SETS:
        print('\n###', S.dataset, 'top task/context effects')
        display(S.seqctx[S.seqctx.dataset.eq(S.dataset)].assign(abs_smd=lambda x: x.standardized_mean_difference.abs()).sort_values('abs_smd', ascending=False).head(30))

# 7. POSITIVE CLUSTERING
def stage_14():
    S.rows = []
    for S.dataset, S.df in S.SETS.items():
        for S.task in S.TASKS:
            for S.rec in S.df.itertuples():
                S.y = np.asarray(getattr(S.rec, S.task), dtype=np.int8)
                S.pos = np.flatnonzero(S.y)
                for S.gap in [1, 2, 3, 5]:
                    S.cc = clusters(S.pos, S.gap)
                    S.rows.append({'dataset': S.dataset, 'task': S.task, 'prot_id': str(S.rec.prot_id), 'max_gap': S.gap, 'n_positive': len(S.pos), 'n_clusters': len(S.cc), 'largest_cluster': max((len(c) for c in S.cc), default=0), 'fraction_positive_in_cluster_ge2': sum((len(c) for c in S.cc if len(c) >= 2)) / max(len(S.pos), 1)})
    S.cluster = pd.DataFrame(S.rows)
    S.cluster_summary = S.cluster[S.cluster.n_positive.gt(0)].groupby(['dataset', 'task', 'max_gap']).agg(positive_proteins=('prot_id', 'nunique'), mean_largest_cluster=('largest_cluster', 'mean'), mean_fraction_positive_in_cluster_ge2=('fraction_positive_in_cluster_ge2', 'mean')).reset_index()
    display(S.cluster_summary)
    S.cluster.to_csv(S.OUT / 'positive_cluster_per_protein_all_tasks.csv', index=False)
    S.cluster_summary.to_csv(S.OUT / 'positive_cluster_summary_all_tasks.csv', index=False)

# 8. CROSS-SPLIT STABILITY
def stage_16():
    S.dev_global = aa_dist(S.dev)
    S.rows = []
    for S.task in S.TASKS:
        S.dev_pos = aa_dist(S.dev, S.task, True)
        S.dev_e = S.aa[S.aa.dataset.eq('DevSet1010') & S.aa.task.eq(S.task)].set_index('aa').loc[S.AA20, 'log2_enrichment'].to_numpy()
        S.dev_prev = float(S.core[S.core.dataset.eq('DevSet1010') & S.core.task.eq(S.task)].residue_prevalence.iloc[0])
        for S.dataset, S.df in S.SETS.items():
            S.test_global = aa_dist(S.df)
            S.test_pos = aa_dist(S.df, S.task, True)
            S.test_e = S.aa[S.aa.dataset.eq(S.dataset) & S.aa.task.eq(S.task)].set_index('aa').loc[S.AA20, 'log2_enrichment'].to_numpy()
            S.prev = float(S.core[S.core.dataset.eq(S.dataset) & S.core.task.eq(S.task)].residue_prevalence.iloc[0])
            S.rows.append({'task': S.task, 'dataset': S.dataset, 'JS_global_AA_vs_Dev': float(jensenshannon(S.dev_global, S.test_global, base=2)), 'JS_positive_AA_vs_Dev': float(jensenshannon(S.dev_pos, S.test_pos, base=2)), 'AA_enrichment_correlation_vs_Dev': float(np.corrcoef(S.dev_e, S.test_e)[0, 1]), 'residue_prevalence': S.prev, 'prevalence_ratio_vs_Dev': S.prev / max(S.dev_prev, 1e-12)})
    S.stability = pd.DataFrame(S.rows)
    display(S.stability)
    S.stability.to_csv(S.OUT / 'cross_split_stability_all_tasks.csv', index=False)

# 9. OPTIONAL CACHED-FEATURE MECHANISM FINGERPRINT
def stage_18():
    S.GEAR_DIR_CANDIDATES = [S.GEAR_DIR, S.BASE / 'GEAR_ULTIMATE', S.other / 'GEAR_ULTIMATE']
    S.GEAR_DIR = next((p for p in S.GEAR_DIR_CANDIDATES if p is not None and (Path(p) / 'gear_ultimate').exists()), None)
    S.feature_fingerprint = pd.DataFrame()
    if S.GEAR_DIR is None:
        print('GEAR_ULTIMATE backend not found -> optional feature fingerprint skipped.')
    else:
        sys.path.insert(0, str(S.GEAR_DIR))
        from gear_ultimate import gear_suite as suite
        S.suite = suite
        from gear_ultimate import gear_core as core_mod
        S.core_mod = core_mod
        import torch
        S.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        S.ctx = S.suite.prepare_context(base=S.BASE, run_version='all_bindings_anatomy_context', models=('MLP',), ensure_base=True, add_mech=True, add_multi=True, add_seqctx=True, esm_batch_size=8, device=S.DEVICE)
        S.STORE_KEY = {'DevSet1010': 'train', 'TestSet300': 'test300', 'TestSetNew46': 'test46'}

        def _store_mapping(store):
            if isinstance(store, dict):
                return store
            for attr in ('records', 'proteins', 'data', 'items_by_id'):
                obj = getattr(store, attr, None)
                if isinstance(obj, dict):
                    return obj
            raise TypeError(type(store))

        def _container(rec):
            if isinstance(rec, dict):
                if isinstance(rec.get('features'), dict):
                    return rec['features']
                return rec
            for attr in ('features', 'feature_dict', 'blocks'):
                obj = getattr(rec, attr, None)
                if isinstance(obj, dict):
                    return obj
            return None

        def get_block(store, pid, b):
            mp = _store_mapping(store)
            rec = mp[str(pid)] if str(pid) in mp else mp[pid]
            cont = _container(rec)
            if isinstance(cont, dict) and b in cont:
                return np.asarray(cont[b])
            if isinstance(rec, dict) and b in rec:
                return np.asarray(rec[b])
            if hasattr(rec, b):
                return np.asarray(getattr(rec, b))
            raise KeyError(b)
        S.BLOCKS = ['SPATIAL', 'LOCAL', 'ATOM', 'CHEM', 'ACCESS', 'SHELL2', 'ENV', 'CONF', 'METALSPEC', 'ELEC3D', 'TOPO', 'SOLV', 'SEQMOTIF', 'SEQCTX']
        S.rows = []
        for S.dataset, S.df in S.SETS.items():
            S.store = S.ctx.stores[S.STORE_KEY[S.dataset]]
            for S.b in S.BLOCKS:
                S.Xparts = []
                S.Yparts = {t: [] for t in S.TASKS}
                S.ok = True
                for S.rec in S.df.itertuples():
                    try:
                        S.x = get_block(S.store, str(S.rec.prot_id), S.b)
                    except Exception:
                        S.ok = False
                        break
                    if S.x.shape[0] != len(S.rec.sequence):
                        S.ok = False
                        break
                    S.Xparts.append(S.x.astype(np.float32, copy=False))
                    for S.t in S.TASKS:
                        S.Yparts[S.t].append(np.asarray(getattr(S.rec, S.t), dtype=np.int8))
                if not S.ok:
                    print('Skip unavailable block:', S.dataset, S.b)
                    continue
                S.X = np.concatenate(S.Xparts, axis=0)
                S.mu = S.X.mean(axis=0, keepdims=True)
                S.sd = S.X.std(axis=0, keepdims=True)
                S.Z = (S.X - S.mu) / (S.sd + 1e-06)
                for S.t in S.TASKS:
                    S.y = np.concatenate(S.Yparts[S.t]).astype(bool)
                    if S.y.sum() == 0 or (~S.y).sum() == 0:
                        continue
                    S.effect = S.Z[S.y].mean(axis=0) - S.Z[~S.y].mean(axis=0)
                    S.rows.append({'dataset': S.dataset, 'task': S.t, 'block': S.b, 'dim': S.X.shape[1], 'effect_rms': float(np.sqrt(np.mean(S.effect ** 2))), 'effect_abs_mean': float(np.mean(np.abs(S.effect))), 'effect_signed_mean': float(np.mean(S.effect))})
        S.feature_fingerprint = pd.DataFrame(S.rows)
        if len(S.feature_fingerprint):
            display(S.feature_fingerprint.sort_values(['dataset', 'task', 'effect_rms'], ascending=[True, True, False]))
            S.feature_fingerprint.to_csv(S.OUT / 'cached_feature_mechanism_fingerprint.csv', index=False)

# 10. AUTOMATIC DECISION TABLES
def stage_20():
    S.task_corr_by_split = []
    for S.dataset in S.SETS:
        S.p = S.aa[S.aa.dataset.eq(S.dataset)].pivot(index='aa', columns='task', values='log2_enrichment')
        S.corr = S.p.corr()
        S.vals = []
        for S.i, S.a in enumerate(S.TASKS):
            for S.b in S.TASKS[S.i + 1:]:
                S.vals.append(S.corr.loc[S.a, S.b])
        S.task_corr_by_split.append({'dataset': S.dataset, 'mean_pairwise_task_AA_signature_correlation': float(np.mean(S.vals)), 'max_pairwise_task_AA_signature_correlation': float(np.max(S.vals))})
    S.task_distinctiveness = pd.DataFrame(S.task_corr_by_split)
    display(S.task_distinctiveness)
    S.stability_view = S.stability.copy()
    S.stability_view['stability_flag'] = S.stability_view.apply(stability_flag, axis=1)
    display(S.stability_view)
    S.hints = []
    for S.task in S.TASKS:
        S.aa_p = S.aa[S.aa.task.eq(S.task)].pivot(index='aa', columns='dataset', values='log2_enrichment')
        S.aa_p['min'] = S.aa_p.min(axis=1)
        S.consistent = S.aa_p[S.aa_p['min'] > 0].sort_values('min', ascending=False)
        S.ctx = S.seqctx[S.seqctx.task.eq(S.task)].groupby(['feature', 'radius']).standardized_mean_difference.agg(['mean', 'min', 'max']).reset_index()
        S.hints.append({'task': S.task, 'consistently_enriched_AA': ','.join(S.consistent.index.tolist()[:8]), 'strongest_context_feature': str(S.ctx.iloc[S.ctx['mean'].abs().argmax()]['feature']) if len(S.ctx) else '', 'strongest_context_mean_SMD': float(S.ctx.iloc[S.ctx['mean'].abs().argmax()]['mean']) if len(S.ctx) else np.nan})
    S.mechanism_hints = pd.DataFrame(S.hints)
    display(S.mechanism_hints)
    S.task_distinctiveness.to_csv(S.OUT / 'task_distinctiveness.csv', index=False)
    S.stability_view.to_csv(S.OUT / 'stability_verdicts.csv', index=False)
    S.mechanism_hints.to_csv(S.OUT / 'mechanism_hints.csv', index=False)
    print('\nSaved to:', S.OUT)

STAGES = [stage_01, stage_02, stage_04, stage_06, stage_08, stage_10, stage_12, stage_14, stage_16, stage_18, stage_20]


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
    prepare()
    _apply(overrides)
    stage_01()
    _apply(overrides)
    stage_02()
    _apply(overrides)
    stage_04()
    _apply(overrides)
    stage_06()
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
    S.RESULT_KEY = 'anatomy'
    return S

