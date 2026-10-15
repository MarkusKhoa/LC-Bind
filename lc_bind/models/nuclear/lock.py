"""Extracted from NUCLEAR_ULTIMATE_V1_DATA_DRIVEN_PATCH-4.ipynb. Section: lock."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Chain, Counter, DataLoader, Dataset, F, Model, MultiArrayDataset, PDBIO, PDBParser, PatchProteinDataset, Path, Structure, Subset, accuracy_score, ast, average_precision_score, balanced_accuracy_score, best_mcc_threshold, binary_metrics, cKDTree, deepcopy, defaultdict, f1_score, hashlib, json, math, matthews_corrcoef, nn, np, os, pd, platform, precision_score, predict_flat, predict_patch, recall_score, roc_auc_score, seq1, shutil, subprocess, sys, tarfile, torch
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import os, sys, json, ast, math, shutil, subprocess, platform
    from collections import Counter, defaultdict
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, Subset
    from scipy.spatial import cKDTree
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, matthews_corrcoef, precision_score, recall_score
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.SeqUtils import seq1
    from copy import deepcopy
    import shutil
    import tarfile
    import hashlib
    import os
    from collections import defaultdict, Counter
    import json, math, numpy as np, pandas as pd
    import json, numpy as np, pandas as pd, torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
    from lc_bind.models.shared import MultiArrayDataset, predict_flat
    from lc_bind.utils.metrics import best_mcc_threshold, binary_metrics
    from lc_bind.models.nuclear.patch import PatchProteinDataset, predict_patch

def predict_local_external(split):
    pack = S.PACKS[split]
    ds = MultiArrayDataset(pack['mods'], pack['y'])
    preds = []
    for fold, state in S.ARCH_STATES[S.LOCKED_ARCH].items():
        model = S.ARCH_FACTORY[S.LOCKED_ARCH](S.DIMS).to(S.DEVICE)
        model.load_state_dict(state)
        preds.append(predict_flat(model, ds))
        del model
        torch.cuda.empty_cache()
    return np.mean(np.stack(preds), axis=0)

def predict_patch_external(split):
    ds = PatchProteinDataset(split, S.SPLIT_DF[split])
    preds = []
    for fold, state in S.PATCH_STATES[S.BEST_PATCH].items():
        model = S.PATCH_FACTORY[S.BEST_PATCH](S.DIMS).to(S.DEVICE)
        model.load_state_dict(state)
        p, _ = predict_patch(model, ds)
        preds.append(p)
        del model
        torch.cuda.empty_cache()
    return np.mean(np.stack(preds), axis=0)

def score_rep(df, name):
    x = df.loc[df.representation.eq(name), 'CV_AUPRC']
    return float(x.iloc[0]) if len(x) else np.nan

def verdict(gain):
    if gain >= 0.01:
        return 'STRONG'
    if gain >= 0.005:
        return 'CLEAR'
    if gain >= S.MIN_MEAN_AUPRC_GAIN:
        return 'MODEST_BUT_REAL'
    if gain > 0:
        return 'WEAK'
    return 'NO_GAIN'

# 18. GLOBAL DEV-ONLY LOCK
def stage_28():
    S.candidates = pd.DataFrame([{'family': 'LOCAL_MULTIMODAL', 'model': S.LOCKED_ARCH, 'Dev_CV_AUPRC': S.LOCKED_ARCH_SCORE, 'complexity_rank': 1}, {'family': 'PATCH_HYPERGRAPH', 'model': S.BEST_PATCH, 'Dev_CV_AUPRC': S.BEST_PATCH_SCORE, 'complexity_rank': 2}])
    S.winner = S.candidates.sort_values('complexity_rank').iloc[0].to_dict()
    for S._, S.r in S.candidates.sort_values('complexity_rank').iloc[1:].iterrows():
        if float(S.r.Dev_CV_AUPRC) - float(S.winner['Dev_CV_AUPRC']) >= S.GLOBAL_COMPLEXITY_MARGIN:
            S.winner = S.r.to_dict()
    S.GLOBAL_FAMILY = str(S.winner['family'])
    S.GLOBAL_MODEL = str(S.winner['model'])
    S.GLOBAL_SCORE = float(S.winner['Dev_CV_AUPRC'])
    display(S.candidates)
    print('GLOBAL LOCK:', S.GLOBAL_FAMILY, S.GLOBAL_MODEL, S.GLOBAL_SCORE)
    (S.RUN_ROOT / 'GLOBAL_LOCK.json').write_text(json.dumps({'family': S.GLOBAL_FAMILY, 'model': S.GLOBAL_MODEL, 'dev_cv_auprc': S.GLOBAL_SCORE, 'complexity_margin': S.GLOBAL_COMPLEXITY_MARGIN}, indent=2))

# 19. METRIC HELPERS + DEV OOF THRESHOLDS
def stage_29():
    S.dev_y = S.PACKS['DevSet1010']['y'].astype(int)
    S.LOCAL_OOF = S.ARCH_OOF[S.LOCKED_ARCH]
    S.LOCAL_THR, S.LOCAL_OOF_MCC = best_mcc_threshold(S.dev_y, S.LOCAL_OOF)
    S.PATCH_OOF_LOCK = S.PATCH_OOF[S.BEST_PATCH]
    S.PATCH_THR, S.PATCH_OOF_MCC = best_mcc_threshold(S.dev_y, S.PATCH_OOF_LOCK)
    print('Local threshold:', S.LOCAL_THR, 'OOF MCC:', S.LOCAL_OOF_MCC)
    print('Patch threshold:', S.PATCH_THR, 'OOF MCC:', S.PATCH_OOF_MCC)

# 20. FINAL EXTERNAL EVALUATION — LOCKED FAMILY WINNERS
def stage_30():
    S.rows = []
    if S.RUN_FINAL_EXTERNAL:
        for S.split in ['TestSet300', 'TestSetNew46']:
            S.y = S.PACKS[S.split]['y'].astype(int)
            S.lp = predict_local_external(S.split)
            S.pp = predict_patch_external(S.split)
            for S.family, S.model, S.p, S.thr, S.devscore in [('LOCAL_MULTIMODAL', S.LOCKED_ARCH, S.lp, S.LOCAL_THR, S.LOCKED_ARCH_SCORE), ('PATCH_HYPERGRAPH', S.BEST_PATCH, S.pp, S.PATCH_THR, S.BEST_PATCH_SCORE)]:
                S.m05 = binary_metrics(S.y, S.p, 0.5)
                S.moo = binary_metrics(S.y, S.p, S.thr)
                S.row = {'family': S.family, 'model': S.model, 'dataset': S.split, 'Dev_CV_AUPRC': S.devscore, 'Dev_OOF_MCC_threshold': S.thr}
                for S.k, S.v in S.m05.items():
                    S.row[f'{S.k}_fixed0.5'] = S.v
                for S.k, S.v in S.moo.items():
                    S.row[f'{S.k}_OOFthr'] = S.v
                S.rows.append(S.row)
            pd.DataFrame({'prot_id': S.PACKS[S.split]['pids'], 'residue_index': S.PACKS[S.split]['ridx'], 'y_true': S.y, 'local_probability': S.lp, 'patch_probability': S.pp}).to_csv(S.RUN_ROOT / f'predictions_{S.split}.csv.gz', index=False)
    S.external_df = pd.DataFrame(S.rows)
    S.external_df.to_csv(S.RUN_ROOT / 'external_family_winners.csv', index=False)
    display(S.external_df)
    S.headline = S.external_df[S.external_df.family.eq(S.GLOBAL_FAMILY) & S.external_df.model.eq(S.GLOBAL_MODEL)].copy()
    S.headline.to_csv(S.RUN_ROOT / 'FINAL_HEADLINE_EXTERNAL.csv', index=False)
    print('HEADLINE:')
    display(S.headline)

# 21. BASELINE / HISTORICAL CONTROL TABLES
def stage_32():
    S.baseline_frames = []
    for S.fp in [S.BASELINE_WORK / 'results' / 'FULL_DIRECT_BASELINE_METRICS_300_46.csv']:
        if S.fp.exists():
            S.z = pd.read_csv(S.fp)
            if 'label' in S.z.columns:
                S.z = S.z[S.z.label.eq('nuclear_binding')].copy()
            if 'operating_point' in S.z.columns:
                S.z = S.z[S.z.operating_point.eq('fixed_0.5')].copy()
            S.baseline_frames.append(S.z)
    S.baselines = pd.concat(S.baseline_frames, ignore_index=True, sort=False) if S.baseline_frames else pd.DataFrame()
    if len(S.baselines):
        S.cols = [c for c in ['method', 'dataset', 'label', 'AP_AUPRC', 'AUPRC', 'AUROC', 'global_positive_F1', 'global_MCC', 'protein_mean_F1', 'protein_mean_MCC'] if c in S.baselines.columns]
        print('LOCKED PUBLISHED-REPRODUCTION BASELINES')
        display(S.baselines[S.cols])
        S.baselines.to_csv(S.RUN_ROOT / 'locked_nuclear_baselines.csv', index=False)
    S.historic_runs = ['ultimate_other_generic_core_v1', 'ultimate_other_standardized13_transfer_v1', 'ultimate_other_metal_mechanistic_transfer_v1', 'ultimate_other_evo_full_v1', 'ultimate_other_generic_graph_v1', 'ultimate_other_evo_graph_v1']
    S.historic = []
    for S.run in S.historic_runs:
        S.root = S.RESULTS_ROOT / S.run / 'nuclear_binding'
        S.fp = S.root / 'cv_summary.csv'
        if S.fp.exists():
            S.z = pd.read_csv(S.fp)
            S.z.insert(0, 'run_version', S.run)
            S.historic.append(S.z)
    S.historic_cv = pd.concat(S.historic, ignore_index=True, sort=False) if S.historic else pd.DataFrame()
    if len(S.historic_cv):
        print('PRIOR NUCLEAR DEV CONTROLS')
        display(S.historic_cv.sort_values('CV_AUPRC', ascending=False).head(30))
        S.historic_cv.to_csv(S.RUN_ROOT / 'prior_nuclear_cv_controls.csv', index=False)

# 22. GAIN DECOMPOSITION / HYPOTHESIS VERDICTS
def stage_34():
    S.mechanism_rows = []
    for S.name in S.MECHANISM_GRID:
        S.s = score_rep(S.combo_cv, S.name)
        S.mechanism_rows.append({'representation': S.name, 'Dev_CV_AUPRC': S.s, 'delta_vs_ESM': S.s - S.base_score})
    S.mechanism_gain = pd.DataFrame(S.mechanism_rows).sort_values('Dev_CV_AUPRC', ascending=False)
    display(S.mechanism_gain)
    S.mechanism_gain.to_csv(S.RUN_ROOT / 'INSIGHT_mechanism_gain.csv', index=False)
    S.arch_gain = S.LOCKED_ARCH_SCORE - S.concat_score
    S.patch_gain = S.BEST_PATCH_SCORE - S.LOCKED_ARCH_SCORE
    S.summary = {'ESM_baseline': S.base_score, 'best_representation': S.BEST_REP_NAME, 'best_representation_dev_auprc': S.BEST_REP_SCORE, 'best_representation_gain_vs_ESM': S.BEST_REP_SCORE - S.base_score, 'locked_multimodal_architecture': S.LOCKED_ARCH, 'locked_multimodal_dev_auprc': S.LOCKED_ARCH_SCORE, 'architecture_gain_vs_concat': S.arch_gain, 'architecture_verdict': verdict(S.arch_gain), 'best_patch_hypergraph': S.BEST_PATCH, 'best_patch_dev_auprc': S.BEST_PATCH_SCORE, 'patch_gain_vs_locked_local': S.patch_gain, 'patch_verdict': verdict(S.patch_gain), 'patch_alpha_mean': float(np.mean(S.alphas)) if S.alphas else None, 'global_family': S.GLOBAL_FAMILY, 'global_model': S.GLOBAL_MODEL, 'global_dev_auprc': S.GLOBAL_SCORE, 'EVO_available': S.EVO_AVAILABLE, 'ESMIF_available': S.ESMIF_AVAILABLE}
    (S.RUN_ROOT / 'INSIGHT_summary.json').write_text(json.dumps(S.summary, indent=2))
    print(json.dumps(S.summary, indent=2))
    print('\n' + '=' * 96)
    print('NUCLEAR ULTIMATE V1 — INTERPRETATION')
    print('=' * 96)
    for S.key in ['ESM+ACCESS+ELEC3D', 'ESM+BACKBONE_ATTRACTION', 'ESM+PATCH_ORGANIZATION', 'ESM+ELECTRO_PATCH', 'ESM+ELECTRO_PATCH_HBOND', 'ESM+ELECTRO_PATCH_HBOND+AROMCTRL', 'ESM+ELECTRO_PATCH_HBOND+EVO']:
        if S.key in set(S.combo_cv.representation):
            S.s = score_rep(S.combo_cv, S.key)
            print(f'{S.key:42s} {S.s:.5f} ({S.s - S.base_score:+.5f} vs ESM)')
    print('\nArchitecture:')
    print(S.LOCKED_ARCH, f'{S.LOCKED_ARCH_SCORE:.5f}', 'gain vs concat =', f'{S.arch_gain:+.5f}', verdict(S.arch_gain))
    print('\nPatch hypergraph:')
    print(S.BEST_PATCH, f'{S.BEST_PATCH_SCORE:.5f}', 'gain vs local =', f'{S.patch_gain:+.5f}', verdict(S.patch_gain))
    if S.alphas:
        print('Learned patch residual alpha:', [round(x, 4) for x in S.alphas], 'mean=', round(float(np.mean(S.alphas)), 4))
    print('\nGLOBAL DEV-LOCKED WINNER:')
    print(S.GLOBAL_FAMILY, '/', S.GLOBAL_MODEL, '/', f'{S.GLOBAL_SCORE:.5f}')
    print('\nExternal headline:')
    display(S.headline)
    print('\nArtifacts:')
    print(S.RUN_ROOT)
    print('=' * 96)

# 23. FINAL MANIFEST
def stage_35():
    S.manifest = {'gear_version': S.GEAR_VERSION, 'python': platform.python_version(), 'torch': torch.__version__, 'device': str(S.DEVICE), 'task': 'nuclear_binding', 'development_set': 'DevSet1010 official bindNode24 5-fold split', 'external_sets': ['TestSet300', 'TestSetNew46'], 'selection_metric': 'residue-level Dev CV AUPRC', 'audit_driven_design': {'core_existing': ['ACCESS', 'ELEC3D'], 'new_blocks': S.NEW_BLOCKS, 'aromatic_role': 'control, not core prior', 'patch_reasoning': 'soft residual due Test46 interface-size shift'}, 'best_representation': S.BEST_REP_NAME, 'best_representation_blocks': S.BEST_REP_BLOCKS, 'locked_architecture': S.LOCKED_ARCH, 'best_patch_model': S.BEST_PATCH, 'global_family': S.GLOBAL_FAMILY, 'global_model': S.GLOBAL_MODEL, 'global_dev_auprc': S.GLOBAL_SCORE, 'evo_available': S.EVO_AVAILABLE, 'esmif_available': S.ESMIF_AVAILABLE, 'summary': S.summary}
    (S.RUN_ROOT / 'FINAL_MANIFEST.json').write_text(json.dumps(S.manifest, indent=2))
    print(json.dumps(S.manifest, indent=2))
    print('\n' + '=' * 96)
    print('NUCLEAR ULTIMATE V1 COMPLETE')
    print('=' * 96)
    print('Headline:', S.GLOBAL_FAMILY, '/', S.GLOBAL_MODEL)
    print('Dev CV AUPRC:', S.GLOBAL_SCORE)
    print('Saved to:', S.RUN_ROOT)
    print('=' * 96)

# 24. FINAL CONSOLIDATED PERFORMANCE COMPARISON
def stage_36():
    print('=' * 110)
    print('NUCLEAR ULTIMATE V1 — FINAL CONSOLIDATED COMPARISON')
    print('=' * 110)
    S.dev_rep_rows = []
    S.dev_rep_rows.append({'stage': 'Representation', 'method': 'ESM baseline', 'framework': 'GEAR MLP', 'Dev_CV_AUPRC': float(S.base_score), 'delta_vs_ESM': 0.0})
    for S.rep_name in ['ESM+ACCESS+ELEC3D', 'ESM+BACKBONE_ATTRACTION', 'ESM+PATCH_ORGANIZATION', 'ESM+ELECTRO_PATCH', 'ESM+ELECTRO_PATCH_HBOND', 'ESM+ELECTRO_PATCH_HBOND+AROMCTRL', 'ESM+ELECTRO_PATCH_HBOND+EVO']:
        if S.rep_name in set(S.combo_cv['representation']):
            S.score = float(S.combo_cv.loc[S.combo_cv['representation'].eq(S.rep_name), 'CV_AUPRC'].iloc[0])
            S.dev_rep_rows.append({'stage': 'Representation', 'method': S.rep_name, 'framework': 'GEAR MLP', 'Dev_CV_AUPRC': S.score, 'delta_vs_ESM': S.score - float(S.base_score)})
    S.dev_rep = pd.DataFrame(S.dev_rep_rows).sort_values('Dev_CV_AUPRC', ascending=False).reset_index(drop=True)
    print('\n[A] DEV REPRESENTATION / FEATURE ABLATION')
    display(S.dev_rep)
    S.dev_rep.to_csv(S.RUN_ROOT / 'FINAL_COMPARISON_DEV_REPRESENTATION.csv', index=False)
    S.dev_arch_rows = []
    for S._, S.r in S.arch_summary.iterrows():
        S.dev_arch_rows.append({'family': 'LOCAL_MULTIMODAL', 'method': str(S.r['architecture']), 'Dev_CV_AUPRC_mean': float(S.r['CV_AUPRC_mean']), 'Dev_CV_AUPRC_SD': float(S.r['CV_AUPRC_std']), 'OOF_AUPRC': float(S.r['OOF_AUPRC']), 'delta_vs_ConcatMLP': float(S.r['CV_AUPRC_mean']) - float(S.concat_score)})
    for S._, S.r in S.patch_summary.iterrows():
        S.dev_arch_rows.append({'family': 'PATCH_HYPERGRAPH', 'method': str(S.r['architecture']), 'Dev_CV_AUPRC_mean': float(S.r['CV_AUPRC_mean']), 'Dev_CV_AUPRC_SD': float(S.r['CV_AUPRC_std']), 'OOF_AUPRC': float(S.r['OOF_AUPRC']), 'delta_vs_ConcatMLP': float(S.r['CV_AUPRC_mean']) - float(S.concat_score)})
    S.dev_arch = pd.DataFrame(S.dev_arch_rows).sort_values('Dev_CV_AUPRC_mean', ascending=False).reset_index(drop=True)
    print('\n[B] DEV MATCHED ARCHITECTURE COMPARISON')
    display(S.dev_arch)
    S.dev_arch.to_csv(S.RUN_ROOT / 'FINAL_COMPARISON_DEV_ARCHITECTURE.csv', index=False)
    S.baseline_fp = S.BASELINE_WORK / 'results' / 'FULL_DIRECT_BASELINE_METRICS_300_46.csv'
    if not S.baseline_fp.exists():
        raise FileNotFoundError(f'Baseline file not found: {S.baseline_fp}')
    S.b = pd.read_csv(S.baseline_fp)
    if 'label' in S.b.columns:
        S.b = S.b[S.b['label'].eq('nuclear_binding')].copy()
    if 'operating_point' in S.b.columns:
        S.b = S.b[S.b['operating_point'].eq('fixed_0.5')].copy()
    S.dataset_map = {'TestSet300': 'TestSet300', 'TestSetNew46': 'TestSetNew46', 'Test300': 'TestSet300', 'Test46': 'TestSetNew46'}
    if 'dataset' in S.b.columns:
        S.b['dataset'] = S.b['dataset'].replace(S.dataset_map)
    S.benchmark_rows = []
    for S._, S.r in S.headline.iterrows():
        S.benchmark_rows.append({'method': f'Our {S.GLOBAL_MODEL}', 'family': S.GLOBAL_FAMILY, 'dataset': str(S.r['dataset']), 'AUPRC': float(S.r['AUPRC_fixed0.5']), 'AUROC': float(S.r['AUROC_fixed0.5']), 'F1': float(S.r['F1_fixed0.5']), 'MCC': float(S.r['MCC_fixed0.5']), 'precision': float(S.r['precision_fixed0.5']), 'recall': float(S.r['recall_fixed0.5']), 'threshold': 0.5, 'source': 'Current nuclear V1'})
    for S._, S.r in S.b.iterrows():
        S.method = str(S.r['method'])
        if S.method not in {'bindEmbed21DL', 'bindNode24'}:
            continue
        S.auprc_col = 'AP_AUPRC' if 'AP_AUPRC' in S.r.index else 'AUPRC'
        S.benchmark_rows.append({'method': S.method, 'family': 'PUBLISHED_BASELINE', 'dataset': str(S.r['dataset']), 'AUPRC': float(S.r[S.auprc_col]), 'AUROC': float(S.r['AUROC']), 'F1': float(S.r['global_positive_F1']), 'MCC': float(S.r['global_MCC']), 'precision': np.nan, 'recall': np.nan, 'threshold': 0.5, 'source': 'Locked direct reproduction'})
    S.benchmark = pd.DataFrame(S.benchmark_rows)
    S.dataset_order = {'TestSet300': 0, 'TestSetNew46': 1}
    S.method_order = {f'Our {S.GLOBAL_MODEL}': 0, 'bindNode24': 1, 'bindEmbed21DL': 2}
    S.benchmark['_ds_order'] = S.benchmark['dataset'].map(S.dataset_order).fillna(99)
    S.benchmark['_method_order'] = S.benchmark['method'].map(S.method_order).fillna(99)
    S.benchmark = S.benchmark.sort_values(['_ds_order', '_method_order']).drop(columns=['_ds_order', '_method_order']).reset_index(drop=True)
    print('\n[C] EXTERNAL BENCHMARK — FIXED THRESHOLD 0.5')
    display(S.benchmark[['dataset', 'method', 'AUPRC', 'AUROC', 'F1', 'MCC']])
    S.benchmark.to_csv(S.RUN_ROOT / 'FINAL_BENCHMARK_COMPARISON.csv', index=False)
    S.our_name = f'Our {S.GLOBAL_MODEL}'
    S.delta_rows = []
    for S.dataset in ['TestSet300', 'TestSetNew46']:
        S.ours = S.benchmark[S.benchmark['dataset'].eq(S.dataset) & S.benchmark['method'].eq(S.our_name)]
        if len(S.ours) != 1:
            continue
        S.ours = S.ours.iloc[0]
        for S.baseline_name in ['bindNode24', 'bindEmbed21DL']:
            S.ref = S.benchmark[S.benchmark['dataset'].eq(S.dataset) & S.benchmark['method'].eq(S.baseline_name)]
            if len(S.ref) != 1:
                continue
            S.ref = S.ref.iloc[0]
            S.delta_rows.append({'dataset': S.dataset, 'comparison': f'{S.our_name} - {S.baseline_name}', 'delta_AUPRC': float(S.ours['AUPRC'] - S.ref['AUPRC']), 'delta_AUROC': float(S.ours['AUROC'] - S.ref['AUROC']), 'delta_F1': float(S.ours['F1'] - S.ref['F1']), 'delta_MCC': float(S.ours['MCC'] - S.ref['MCC'])})
    S.delta_df = pd.DataFrame(S.delta_rows)
    print('\n[D] DELTA OF OUR LOCKED MODEL VS BASELINES')
    display(S.delta_df)
    S.delta_df.to_csv(S.RUN_ROOT / 'FINAL_DELTA_VS_BASELINES.csv', index=False)
    S.best_rep_row = S.dev_rep.iloc[0]
    S.best_local_row = S.dev_arch[S.dev_arch['family'].eq('LOCAL_MULTIMODAL')].sort_values('Dev_CV_AUPRC_mean', ascending=False).iloc[0]
    S.best_patch_row = S.dev_arch[S.dev_arch['family'].eq('PATCH_HYPERGRAPH')].sort_values('Dev_CV_AUPRC_mean', ascending=False).iloc[0]
    S.ladder = pd.DataFrame([{'step': 1, 'stage': 'Sequence baseline', 'model': 'ESM', 'Dev_AUPRC': float(S.base_score), 'comparison_scope': 'GEAR MLP representation'}, {'step': 2, 'stage': 'Best explicit representation', 'model': str(S.best_rep_row['method']), 'Dev_AUPRC': float(S.best_rep_row['Dev_CV_AUPRC']), 'comparison_scope': 'GEAR MLP representation'}, {'step': 3, 'stage': 'Matched local custom baseline', 'model': 'ConcatMLP', 'Dev_AUPRC': float(S.concat_score), 'comparison_scope': 'Custom architecture'}, {'step': 4, 'stage': 'Best local custom architecture', 'model': str(S.best_local_row['method']), 'Dev_AUPRC': float(S.best_local_row['Dev_CV_AUPRC_mean']), 'comparison_scope': 'Custom architecture'}, {'step': 5, 'stage': 'Best patch hypergraph', 'model': str(S.best_patch_row['method']), 'Dev_AUPRC': float(S.best_patch_row['Dev_CV_AUPRC_mean']), 'comparison_scope': 'Custom architecture'}])
    print('\n[E] SCIENTIFIC PERFORMANCE LADDER')
    display(S.ladder)
    S.ladder.to_csv(S.RUN_ROOT / 'FINAL_SCIENTIFIC_LADDER.csv', index=False)
    S.rep_gain = float(S.BEST_REP_SCORE) - float(S.base_score)
    S.patch_gain_matched = float(S.BEST_PATCH_SCORE) - float(S.concat_score)
    S.test300_delta_bn24 = S.delta_df.loc[S.delta_df['dataset'].eq('TestSet300') & S.delta_df['comparison'].str.endswith('bindNode24'), 'delta_AUPRC']
    S.test46_delta_bn24 = S.delta_df.loc[S.delta_df['dataset'].eq('TestSetNew46') & S.delta_df['comparison'].str.endswith('bindNode24'), 'delta_AUPRC']
    S.test300_delta_bn24 = float(S.test300_delta_bn24.iloc[0]) if len(S.test300_delta_bn24) else np.nan
    S.test46_delta_bn24 = float(S.test46_delta_bn24.iloc[0]) if len(S.test46_delta_bn24) else np.nan
    S.comparison_summary = {'task': 'nuclear_binding', 'ESM_Dev_CV_AUPRC': float(S.base_score), 'best_explicit_representation': S.BEST_REP_NAME, 'best_explicit_representation_Dev_CV_AUPRC': float(S.BEST_REP_SCORE), 'explicit_representation_gain_vs_ESM': S.rep_gain, 'matched_custom_baseline': 'ConcatMLP', 'ConcatMLP_Dev_CV_AUPRC': float(S.concat_score), 'best_patch_hypergraph': S.BEST_PATCH, 'best_patch_hypergraph_Dev_CV_AUPRC': float(S.BEST_PATCH_SCORE), 'patch_gain_vs_ConcatMLP': S.patch_gain_matched, 'Test300_delta_AUPRC_vs_bindNode24': S.test300_delta_bn24, 'Test46_delta_AUPRC_vs_bindNode24': S.test46_delta_bn24, 'global_locked_family': S.GLOBAL_FAMILY, 'global_locked_model': S.GLOBAL_MODEL}
    (S.RUN_ROOT / 'FINAL_COMPARISON_SUMMARY.json').write_text(json.dumps(S.comparison_summary, indent=2))
    print('\n[F] KEY VERDICT')
    print('-' * 110)
    print(f'ESM -> best explicit representation: {S.base_score:.5f} -> {S.BEST_REP_SCORE:.5f} ({S.rep_gain:+.5f})')
    print(f'ConcatMLP -> best patch hypergraph: {S.concat_score:.5f} -> {S.BEST_PATCH_SCORE:.5f} ({S.patch_gain_matched:+.5f})')
    print(f'TestSet300 ΔAUPRC vs bindNode24: {S.test300_delta_bn24:+.5f}')
    print(f'TestSetNew46 ΔAUPRC vs bindNode24: {S.test46_delta_bn24:+.5f}')
    if S.test300_delta_bn24 > 0 and S.test46_delta_bn24 >= 0:
        S.ext_verdict = 'OUTPERFORMS bindNode24 on both external sets.'
    elif S.test300_delta_bn24 > 0 and S.test46_delta_bn24 > -0.005:
        S.ext_verdict = 'CLEAR Test300 improvement; essentially competitive with bindNode24 on shifted Test46.'
    elif S.test300_delta_bn24 > 0:
        S.ext_verdict = 'CLEAR Test300 improvement, but the gain does not fully transfer to shifted Test46.'
    else:
        S.ext_verdict = 'No consistent external advantage over bindNode24.'
    print('External verdict:', S.ext_verdict)
    print('-' * 110)
    print('\nIMPORTANT INTERPRETATION NOTE:')
    print('The GEAR-MLP representation ladder and the custom architecture ladder use different trainers/objectives. Therefore, do NOT interpret BEST_REP_SCORE -> ConcatMLP as a pure architecture effect.')
    print('The clean architecture comparison is ConcatMLP vs the other custom fusion / patch-hypergraph models.')
    print('\nSaved consolidated outputs to:')
    print(S.RUN_ROOT)
    print('=' * 110)

STAGES = [stage_28, stage_29, stage_30, stage_32, stage_34, stage_35, stage_36]

