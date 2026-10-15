"""Extracted from NUCLEAR_DATASET_AUDIT_DEV1010_300_46.ipynb. Section: nuclear_audit."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Counter, Path, ast, fisher_exact, jensenshannon, json, mannwhitneyu, math, np, os, parse_list, pd, roc_auc_score, subprocess, sys
    from pathlib import Path
    import ast, json, math, os, subprocess, sys
    from collections import Counter
    import numpy as np
    import pandas as pd
    from scipy.stats import fisher_exact, mannwhitneyu
    from scipy.spatial.distance import jensenshannon
    from sklearn.metrics import roc_auc_score
    from lc_bind.io.datasets import parse_list

def load_df(path):
    df = pd.read_csv(path)
    df['prot_id'] = df['prot_id'].astype(str)
    df['sequence'] = df['sequence'].astype(str)
    for c in S.LABELS:
        if c in df.columns:
            df[c] = df[c].apply(parse_list)
    if not df.prot_id.is_unique:
        raise RuntimeError(f'{path}: duplicate prot_id')
    for r in df.itertuples():
        for c in S.LABELS:
            if hasattr(r, c):
                y = getattr(r, c)
                if len(y) != len(r.sequence):
                    raise RuntimeError(f'{path}/{r.prot_id}/{c}: {len(y)} labels != {len(r.sequence)} residues')
    return df

def local_fraction(seq, i, chars, radius):
    lo = max(0, i - radius)
    hi = min(len(seq), i + radius + 1)
    w = seq[lo:hi]
    return sum((aa in chars for aa in w)) / max(len(w), 1)

def clusters_from_positions(pos, max_gap=2):
    pos = np.asarray(pos, dtype=int)
    if len(pos) == 0:
        return []
    groups = [[int(pos[0])]]
    for x in pos[1:]:
        if x - groups[-1][-1] <= max_gap:
            groups[-1].append(int(x))
        else:
            groups.append([int(x)])
    return groups

def aa_distribution(df, positive_only=False):
    counts = Counter()
    for r in df.itertuples():
        if not positive_only:
            counts.update(r.sequence)
        else:
            y = np.asarray(r.nuclear_binding, dtype=np.int8)
            counts.update((aa for aa, yy in zip(r.sequence, y) if yy == 1))
    v = np.array([counts[a] for a in S.AA20], dtype=float)
    return v / v.sum() if v.sum() else np.zeros_like(v)

def add(name, rationale, priority_level):
    S.priority.append({'feature_family': name, 'priority': priority_level, 'rationale_from_dataset_audit': rationale})

# 0. DRIVE / PATH AUTO-DETECTION
def stage_01():
    S.CANDIDATE_BASES = [Path(str(S.BASE)), Path(str(S.ALT_BASE))]
    S.BASE = None
    for S.b in S.CANDIDATE_BASES:
        if (S.b / 'data/development_set/full_grouped_train_binding_sites_df.csv').exists() and (S.b / 'data/development_set/full_grouped_test_binding_sites_df.csv').exists() and (S.b / 'data/independent_set/grouped_test_46_new_binding_sites.csv').exists():
            S.BASE = S.b
            break
    if S.BASE is None:
        raise FileNotFoundError('Could not find the three project CSVs under Protein-binding-Phi or Protein-binding.')
    S.OUT = S.RESULTS_ROOT / 'NUCLEAR_DATASET_AUDIT_V1'
    S.OUT.mkdir(parents=True, exist_ok=True)
    S.TRAIN_CSV = S.DATA_ROOT / 'development_set/full_grouped_train_binding_sites_df.csv'
    S.TEST300_CSV = S.DATA_ROOT / 'development_set/full_grouped_test_binding_sites_df.csv'
    S.TEST46_CSV = S.DATA_ROOT / 'independent_set/grouped_test_46_new_binding_sites.csv'
    print('BASE =', S.BASE)
    print('OUT  =', S.OUT)

# 1. LOAD DATA + EXACT OFFICIAL DevSet1010
def stage_02():
    S.LABELS = ['metal_binding', 'nuclear_binding', 'small_binding', 'any_ligand_binding_sites']
    S.raw_train = load_df(S.TRAIN_CSV)
    S.test300 = load_df(S.TEST300_CSV)
    S.test46 = load_df(S.TEST46_CSV)
    S.BN24_CANDIDATES = [S.BASE / 'baseline_reproduction_master_v6' / 'repos' / 'bindnode24', S.BASE / 'repos' / 'bindnode24']
    S.other = Path(str(S.ALT_BASE)) if S.BASE.name == 'Protein-binding-Phi' else Path(str(S.BASE))
    S.BN24_CANDIDATES += [S.other / 'baseline_reproduction_master_v6' / 'repos' / 'bindnode24', S.other / 'repos' / 'bindnode24']
    S.BN24 = next((p for p in S.BN24_CANDIDATES if (p / '.git').exists()), None)
    if S.BN24 is None:
        S.BN24 = Path(str(S.CACHE_ROOT / 'bindnode24_dataset_audit'))
        if not S.BN24.exists():
            subprocess.run(['git', 'clone', '-q', 'https://github.com/erckert/bindNode24.git', str(S.BN24)], check=True)
    S.fold_root = S.BN24 / 'dataset' / 'cv_splits' / 'predicted_structures'
    S.fold_ids = {}
    S.official_order = []
    S.seen = set()
    for S.k in range(1, 6):
        S.ids = [x.strip() for x in (S.fold_root / f'ids_split{S.k}.txt').read_text().splitlines() if x.strip()]
        if S.seen.intersection(S.ids):
            raise RuntimeError(f'Fold {S.k}: overlap with earlier folds')
        S.fold_ids[S.k] = S.ids
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
    S.excluded = sorted(S.raw_ids - S.official)
    S.SETS = {'DevSet1010': S.dev, 'TestSet300': S.test300, 'TestSetNew46': S.test46}
    print('Raw development lineage:', len(S.raw_train))
    print('DevSet1010:', len(S.dev))
    print('Excluded from DevSet1010:', len(S.excluded), S.excluded)
    print('Fold sizes:', {k: len(v) for k, v in S.fold_ids.items()})

# 2. PREVALENCE / POSITIVE-PROTEIN / BURDEN AUDIT
def stage_04():
    S.summary_rows = []
    S.burden_rows = []
    for S.name, S.df in S.SETS.items():
        S.total_res = int(S.df.sequence.str.len().sum())
        S.all_y = np.concatenate([np.asarray(x, dtype=np.int8) for x in S.df.nuclear_binding])
        S.protein_pos = np.array([int(np.sum(x)) for x in S.df.nuclear_binding])
        S.n_pos_proteins = int(np.sum(S.protein_pos > 0))
        S.summary_rows.append({'dataset': S.name, 'n_proteins': len(S.df), 'n_residues': S.total_res, 'n_nuclear_positive_residues': int(S.all_y.sum()), 'residue_positive_rate': float(S.all_y.mean()), 'n_positive_proteins': S.n_pos_proteins, 'positive_protein_fraction': S.n_pos_proteins / len(S.df), 'median_positive_residues_positive_protein': float(np.median(S.protein_pos[S.protein_pos > 0])) if S.n_pos_proteins else 0, 'mean_positive_residues_positive_protein': float(np.mean(S.protein_pos[S.protein_pos > 0])) if S.n_pos_proteins else 0, 'max_positive_residues_one_protein': int(S.protein_pos.max()), 'median_protein_length': float(np.median(S.df.sequence.str.len())), 'mean_protein_length': float(np.mean(S.df.sequence.str.len()))})
        for S.pid, S.L, S.npos in zip(S.df.prot_id, S.df.sequence.str.len(), S.protein_pos):
            S.burden_rows.append({'dataset': S.name, 'prot_id': S.pid, 'length': int(S.L), 'n_nuclear_positive': int(S.npos), 'protein_positive': bool(S.npos > 0), 'within_protein_positive_fraction': S.npos / max(S.L, 1)})
    S.dataset_summary = pd.DataFrame(S.summary_rows)
    S.protein_burden = pd.DataFrame(S.burden_rows)
    display(S.dataset_summary)
    S.dataset_summary.to_csv(S.OUT / 'dataset_summary.csv', index=False)
    S.protein_burden.to_csv(S.OUT / 'protein_burden.csv', index=False)

# 3. POSITIVE-RESIDUE AMINO-ACID ENRICHMENT
def stage_06():
    S.AA20 = list('ACDEFGHIKLMNPQRSTVWY')
    S.aa_rows = []
    for S.name, S.df in S.SETS.items():
        S.all_seq = ''.join(S.df.sequence.tolist())
        S.pos_aa = []
        S.neg_aa = []
        for S.r in S.df.itertuples():
            S.y = np.asarray(S.r.nuclear_binding, dtype=np.int8)
            S.seq = S.r.sequence
            S.pos_aa.extend([aa for aa, yy in zip(S.seq, S.y) if yy == 1])
            S.neg_aa.extend([aa for aa, yy in zip(S.seq, S.y) if yy == 0])
        S.all_counts = Counter(S.all_seq)
        S.pos_counts = Counter(S.pos_aa)
        S.neg_counts = Counter(S.neg_aa)
        S.n_all = sum(S.all_counts.values())
        S.n_pos = sum(S.pos_counts.values())
        S.n_neg = sum(S.neg_counts.values())
        for S.aa in S.AA20:
            S.a = S.pos_counts[S.aa]
            S.b = S.n_pos - S.a
            S.c = S.neg_counts[S.aa]
            S.d = S.n_neg - S.c
            S.odds, S.p = fisher_exact([[S.a, S.b], [S.c, S.d]])
            S.f_all = S.all_counts[S.aa] / max(S.n_all, 1)
            S.f_pos = S.a / max(S.n_pos, 1)
            S.log2_enr = np.log2((S.f_pos + 1e-08) / (S.f_all + 1e-08))
            S.aa_rows.append({'dataset': S.name, 'aa': S.aa, 'all_frequency': S.f_all, 'positive_frequency': S.f_pos, 'log2_enrichment_positive_vs_all': float(S.log2_enr), 'odds_ratio_positive_vs_negative': float(S.odds), 'fisher_p': float(S.p), 'positive_count': int(S.a)})
    S.aa_enrichment = pd.DataFrame(S.aa_rows)
    for S.name in S.SETS:
        print('\n', S.name)
        display(S.aa_enrichment[S.aa_enrichment.dataset.eq(S.name)].sort_values('log2_enrichment_positive_vs_all', ascending=False))
    S.aa_enrichment.to_csv(S.OUT / 'aa_enrichment.csv', index=False)

# 4. POSITIVE VS NEGATIVE SEQUENCE-NEIGHBORHOOD CONTEXT
def stage_08():
    S.GROUPS = {'basic_KRH': set('KRH'), 'strong_basic_KR': set('KR'), 'acidic_DE': set('DE'), 'polar_NQSTY': set('NQSTY'), 'aromatic_FWYH': set('FWYH'), 'gly_arg_GR': set('GR')}
    S.WINDOWS = [2, 4, 7, 10]
    S.context_rows = []
    for S.name, S.df in S.SETS.items():
        S.pos_vals = {}
        S.neg_vals = {}
        for S.g in S.GROUPS:
            for S.rad in S.WINDOWS:
                S.pos_vals[S.g, S.rad] = []
                S.neg_vals[S.g, S.rad] = []
        for S.r in S.df.itertuples():
            S.seq = S.r.sequence
            S.y = np.asarray(S.r.nuclear_binding, dtype=np.int8)
            for S.i, S.yy in enumerate(S.y):
                S.target = S.pos_vals if S.yy else S.neg_vals
                for S.g, S.chars in S.GROUPS.items():
                    for S.rad in S.WINDOWS:
                        S.target[S.g, S.rad].append(local_fraction(S.seq, S.i, S.chars, S.rad))
        for S.g in S.GROUPS:
            for S.rad in S.WINDOWS:
                S.p = np.asarray(S.pos_vals[S.g, S.rad], dtype=float)
                S.n = np.asarray(S.neg_vals[S.g, S.rad], dtype=float)
                S.sp = np.sqrt((S.p.var() + S.n.var()) / 2 + 1e-12)
                S.smd = (S.p.mean() - S.n.mean()) / S.sp
                S.context_rows.append({'dataset': S.name, 'feature': S.g, 'radius_residues': S.rad, 'positive_mean': float(S.p.mean()), 'negative_mean': float(S.n.mean()), 'standardized_mean_difference': float(S.smd)})
    S.context_df = pd.DataFrame(S.context_rows)
    for S.name in S.SETS:
        print('\n', S.name, 'top sequence-context effects')
        display(S.context_df[S.context_df.dataset.eq(S.name)].assign(abs_smd=lambda x: x.standardized_mean_difference.abs()).sort_values('abs_smd', ascending=False).head(20))
    S.context_df.to_csv(S.OUT / 'sequence_context_effects.csv', index=False)

# 5. POSITIVE CLUSTERING / RUN STATISTICS
def stage_10():
    S.cluster_rows = []
    for S.name, S.df in S.SETS.items():
        for S.r in S.df.itertuples():
            S.y = np.asarray(S.r.nuclear_binding, dtype=np.int8)
            S.pos = np.flatnonzero(S.y)
            for S.gap in [1, 2, 3, 5]:
                S.cs = clusters_from_positions(S.pos, max_gap=S.gap)
                S.cluster_rows.append({'dataset': S.name, 'prot_id': str(S.r.prot_id), 'max_gap': S.gap, 'n_positive': int(len(S.pos)), 'n_clusters': int(len(S.cs)), 'largest_cluster_size': int(max((len(c) for c in S.cs), default=0)), 'mean_cluster_size': float(np.mean([len(c) for c in S.cs])) if S.cs else 0.0, 'fraction_positives_in_multiresidue_cluster': float(sum((len(c) for c in S.cs if len(c) >= 2)) / max(len(S.pos), 1))})
    S.cluster_df = pd.DataFrame(S.cluster_rows)
    S.cluster_summary = S.cluster_df[S.cluster_df.n_positive.gt(0)].groupby(['dataset', 'max_gap']).agg(positive_proteins=('prot_id', 'nunique'), mean_clusters=('n_clusters', 'mean'), mean_largest_cluster=('largest_cluster_size', 'mean'), mean_fraction_in_multiresidue_cluster=('fraction_positives_in_multiresidue_cluster', 'mean')).reset_index()
    display(S.cluster_summary)
    S.cluster_df.to_csv(S.OUT / 'positive_cluster_per_protein.csv', index=False)
    S.cluster_summary.to_csv(S.OUT / 'positive_cluster_summary.csv', index=False)

# 6. CROSS-LABEL OVERLAP
def stage_12():
    S.overlap_rows = []
    for S.name, S.df in S.SETS.items():
        S.nuc_all, S.met_all, S.sma_all = ([], [], [])
        S.protein_flags = []
        for S.r in S.df.itertuples():
            S.nuc = np.asarray(S.r.nuclear_binding, dtype=np.int8)
            S.met = np.asarray(S.r.metal_binding, dtype=np.int8)
            S.sma = np.asarray(S.r.small_binding, dtype=np.int8)
            S.nuc_all.append(S.nuc)
            S.met_all.append(S.met)
            S.sma_all.append(S.sma)
            S.protein_flags.append({'nuc': S.nuc.sum() > 0, 'met': S.met.sum() > 0, 'small': S.sma.sum() > 0})
        S.nuc = np.concatenate(S.nuc_all).astype(bool)
        S.met = np.concatenate(S.met_all).astype(bool)
        S.sma = np.concatenate(S.sma_all).astype(bool)
        S.p = pd.DataFrame(S.protein_flags)
        S.overlap_rows.append({'dataset': S.name, 'nuclear_positive_residues': int(S.nuc.sum()), 'residue_nuclear_and_metal': int((S.nuc & S.met).sum()), 'residue_nuclear_and_small': int((S.nuc & S.sma).sum()), 'fraction_nuclear_residues_also_metal': float((S.nuc & S.met).sum() / max(S.nuc.sum(), 1)), 'fraction_nuclear_residues_also_small': float((S.nuc & S.sma).sum() / max(S.nuc.sum(), 1)), 'positive_nuclear_proteins': int(S.p.nuc.sum()), 'protein_nuclear_and_metal': int((S.p.nuc & S.p.met).sum()), 'protein_nuclear_and_small': int((S.p.nuc & S.p.small).sum()), 'fraction_nuclear_proteins_also_metal': float((S.p.nuc & S.p.met).sum() / max(S.p.nuc.sum(), 1)), 'fraction_nuclear_proteins_also_small': float((S.p.nuc & S.p.small).sum() / max(S.p.nuc.sum(), 1))})
    S.overlap_df = pd.DataFrame(S.overlap_rows)
    display(S.overlap_df)
    S.overlap_df.to_csv(S.OUT / 'cross_label_overlap.csv', index=False)

# 7. SPLIT-SHIFT ANALYSIS
def stage_14():
    S.shift_rows = []
    S.dev_all = aa_distribution(S.dev, False)
    S.dev_pos = aa_distribution(S.dev, True)
    for S.name, S.df in S.SETS.items():
        S.all_dist = aa_distribution(S.df, False)
        S.pos_dist = aa_distribution(S.df, True)
        S.shift_rows.append({'dataset': S.name, 'JS_global_AA_vs_Dev': float(jensenshannon(S.dev_all, S.all_dist, base=2)), 'JS_positive_AA_vs_Dev': float(jensenshannon(S.dev_pos, S.pos_dist, base=2)), 'median_length': float(np.median(S.df.sequence.str.len())), 'mean_length': float(np.mean(S.df.sequence.str.len()))})
    S.shift_df = pd.DataFrame(S.shift_rows)
    display(S.shift_df)
    S.shift_df.to_csv(S.OUT / 'split_shift_summary.csv', index=False)
    S.pos_pivot = S.aa_enrichment.pivot(index='aa', columns='dataset', values='log2_enrichment_positive_vs_all')
    print('Correlation of nuclear-positive AA enrichment patterns:')
    display(S.pos_pivot.corr())
    S.pos_pivot.to_csv(S.OUT / 'positive_AA_enrichment_by_split.csv')

# 8. READ EXISTING NUCLEAR BASELINE RESULTS
def stage_16():
    S.result_roots = [S.BASE / 'baseline_reproduction_master_v6' / 'results', S.other / 'baseline_reproduction_master_v6' / 'results']
    S.baseline_root = next((p for p in S.result_roots if (p / 'FULL_DIRECT_BASELINE_METRICS_300_46.csv').exists()), None)
    S.nuclear_baselines = pd.DataFrame()
    if S.baseline_root is not None:
        S.z = pd.read_csv(S.baseline_root / 'FULL_DIRECT_BASELINE_METRICS_300_46.csv')
        if 'label' in S.z.columns:
            S.z = S.z[S.z.label.eq('nuclear_binding')].copy()
        if 'operating_point' in S.z.columns:
            S.z = S.z[S.z.operating_point.eq('fixed_0.5')].copy()
        S.keep = [c for c in ['method', 'dataset', 'label', 'AP_AUPRC', 'AUPRC', 'AUROC', 'global_positive_F1', 'global_MCC', 'protein_mean_F1', 'protein_mean_MCC'] if c in S.z.columns]
        S.nuclear_baselines = S.z[S.keep].copy()
        display(S.nuclear_baselines)
        S.nuclear_baselines.to_csv(S.OUT / 'existing_nuclear_baseline_metrics.csv', index=False)
    else:
        print('Frozen baseline metric table not found; skipping this optional section.')

# 9. AUTOMATIC INTERPRETATION
def stage_18():
    S.aa_mat = S.aa_enrichment.pivot(index='aa', columns='dataset', values='log2_enrichment_positive_vs_all')
    S.aa_mat['mean_enrichment'] = S.aa_mat.mean(axis=1)
    S.aa_mat['min_enrichment'] = S.aa_mat[['DevSet1010', 'TestSet300', 'TestSetNew46']].min(axis=1)
    S.aa_mat['max_enrichment'] = S.aa_mat[['DevSet1010', 'TestSet300', 'TestSetNew46']].max(axis=1)
    S.consistent_enriched = S.aa_mat[S.aa_mat.min_enrichment > 0].sort_values('mean_enrichment', ascending=False)
    S.consistent_depleted = S.aa_mat[S.aa_mat.max_enrichment < 0].sort_values('mean_enrichment')
    print('=' * 88)
    print('NUCLEAR DATASET MECHANISM AUDIT')
    print('=' * 88)
    print('\nConsistently enriched nuclear-positive amino acids:')
    display(S.consistent_enriched)
    print('\nConsistently depleted nuclear-positive amino acids:')
    display(S.consistent_depleted)
    S.ctx_mean = S.context_df.groupby(['feature', 'radius_residues']).standardized_mean_difference.agg(['mean', 'min', 'max']).reset_index()
    S.ctx_mean['abs_mean'] = S.ctx_mean['mean'].abs()
    print('\nMost consistent sequence-context effects:')
    display(S.ctx_mean.sort_values('abs_mean', ascending=False).head(20))
    print('\nDataset shift:')
    display(S.shift_df)
    print('\nCross-label overlap:')
    display(S.overlap_df)
    S.priority = []
    S.basic_rows = S.ctx_mean[S.ctx_mean.feature.isin(['basic_KRH', 'strong_basic_KR'])]
    if len(S.basic_rows) and S.basic_rows['mean'].max() > 0.1:
        add('NA_ELECTRO / POSITIVE_PATCH', 'Nuclear positives show enrichment in local basic-residue context across splits.', 'HIGH')
    if set(['K', 'R']).intersection(S.consistent_enriched.index):
        add('NA_HBOND_CATIONIC', 'K/R enrichment suggests explicit cationic/H-bond chemistry may add interpretable signal beyond ESM.', 'HIGH')
    if set(['F', 'W', 'Y', 'H']).intersection(S.consistent_enriched.index):
        add('NA_AROMATIC', 'Aromatic residue enrichment is consistent enough to justify a stacking/aromatic-patch block.', 'MEDIUM-HIGH')
    S.cs = S.cluster_summary[S.cluster_summary.max_gap.eq(2)]
    if len(S.cs) and S.cs.mean_fraction_in_multiresidue_cluster.mean() > 0.4:
        add('NA_PATCH / PATCH_GRAPH', 'A large fraction of positive residues occur in near-contiguous clusters, supporting patch-aware reasoning.', 'HIGH')
    else:
        add('NA_PATCH', 'Patch features remain biologically plausible, but sequence clustering alone is not strong enough to make graph reasoning mandatory.', 'MEDIUM')
    add('NA_SURFACE', 'Nucleic-acid interaction requires accessible interfaces; prior GEAR runs also showed ACCESS transfer signal. Structural confirmation should be tested directly.', 'HIGH')
    add('EVO48', 'Previous nuclear experiments showed complementary evolutionary signal; retain as an orthogonal biological-context branch.', 'MEDIUM-HIGH')
    S.priority_df = pd.DataFrame(S.priority).drop_duplicates('feature_family')
    display(S.priority_df)
    S.priority_df.to_csv(S.OUT / 'recommended_feature_priorities.csv', index=False)
    S.manifest = {'base': str(S.BASE), 'output': str(S.OUT), 'devset_definition': 'union/order of official bindNode24 five folds', 'label': 'nuclear_binding', 'n_sets': 3}
    (S.OUT / 'audit_manifest.json').write_text(json.dumps(S.manifest, indent=2))
    print('\nSaved audit artifacts to:')
    print(S.OUT)
    print('=' * 88)

STAGES = [stage_01, stage_02, stage_04, stage_06, stage_08, stage_10, stage_12, stage_14, stage_16, stage_18]


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
    S.RESULT_KEY = 'nuclear_audit'
    return S

