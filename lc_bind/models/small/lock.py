"""Extracted from SMALL_ULTIMATE_V1_DATA_DRIVEN_POCKET-3.ipynb. Section: lock."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Chain, Counter, DataLoader, Dataset, F, Model, MultiArrayDataset, PDBIO, PDBParser, Path, PocketProteinDataset, Structure, Subset, accuracy_score, ast, average_precision_score, balanced_accuracy_score, best_mcc_threshold, binary_metrics, cKDTree, deepcopy, defaultdict, f1_score, hashlib, json, math, matthews_corrcoef, nn, np, os, pd, platform, precision_score, predict_flat, predict_pocket, recall_score, roc_auc_score, seq1, shutil, subprocess, sys, tarfile, torch
    from pathlib import Path
    import os, sys, json, ast, math, shutil, subprocess, platform
    from collections import Counter, defaultdict
    import numpy as np
    import pandas as pd
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
    import json
    import hashlib
    import os
    import math
    from collections import Counter
    import json, numpy as np, pandas as pd, torch
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
    from lc_bind.models.shared import MultiArrayDataset, predict_flat
    from lc_bind.utils.metrics import best_mcc_threshold, binary_metrics
    from lc_bind.models.small.pocket import PocketProteinDataset, predict_pocket

def predict_local_external(split):
    pack = S.PACKS[split]
    ds = MultiArrayDataset(pack['mods'], pack['y'])
    preds = []
    for fold, state in S.ARCH_STATES[S.LOCKED_ARCH].items():
        model = S.ARCH_FACTORY[S.LOCKED_ARCH](S.DIMS).to(S.DEVICE)
        model.load_state_dict(state)
        preds.append(predict_flat(model, ds))
        del model
        if S.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    return np.mean(np.stack(preds), axis=0)

def predict_pocket_external(split):
    ds = PocketProteinDataset(split, S.SPLIT_DF[split])
    preds = []
    for fold, state in S.POCKET_STATES[S.BEST_POCKET].items():
        model = S.POCKET_FACTORY[S.BEST_POCKET](S.DIMS).to(S.DEVICE)
        model.load_state_dict(state)
        p, _ = predict_pocket(model, ds)
        preds.append(p)
        del model
        if S.DEVICE.type == 'cuda':
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
    S.candidates = pd.DataFrame([{'family': 'LOCAL_MULTIMODAL', 'model': S.LOCKED_ARCH, 'Dev_CV_AUPRC': S.LOCKED_ARCH_SCORE, 'complexity_rank': 1}, {'family': 'POCKET_HYPERGRAPH', 'model': S.BEST_POCKET, 'Dev_CV_AUPRC': S.BEST_POCKET_SCORE, 'complexity_rank': 2}])
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
    S.POCKET_OOF_LOCK = S.POCKET_OOF[S.BEST_POCKET]
    S.POCKET_THR, S.POCKET_OOF_MCC = best_mcc_threshold(S.dev_y, S.POCKET_OOF_LOCK)
    print('Local threshold:', S.LOCAL_THR, 'OOF MCC:', S.LOCAL_OOF_MCC)
    print('Pocket threshold:', S.POCKET_THR, 'OOF MCC:', S.POCKET_OOF_MCC)

# 20. FINAL EXTERNAL EVALUATION — LOCKED FAMILY WINNERS
def stage_30():
    S.rows = []
    if S.RUN_FINAL_EXTERNAL:
        for S.split in ['TestSet300', 'TestSetNew46']:
            S.y = S.PACKS[S.split]['y'].astype(int)
            S.lp = predict_local_external(S.split)
            S.pp = predict_pocket_external(S.split)
            for S.family, S.model, S.p, S.thr, S.devscore in [('LOCAL_MULTIMODAL', S.LOCKED_ARCH, S.lp, S.LOCAL_THR, S.LOCKED_ARCH_SCORE), ('POCKET_HYPERGRAPH', S.BEST_POCKET, S.pp, S.POCKET_THR, S.BEST_POCKET_SCORE)]:
                S.m05 = binary_metrics(S.y, S.p, 0.5)
                S.moo = binary_metrics(S.y, S.p, S.thr)
                S.row = {'family': S.family, 'model': S.model, 'dataset': S.split, 'Dev_CV_AUPRC': S.devscore, 'Dev_OOF_MCC_threshold': S.thr}
                for S.k, S.v in S.m05.items():
                    S.row[f'{S.k}_fixed0.5'] = S.v
                for S.k, S.v in S.moo.items():
                    S.row[f'{S.k}_OOFthr'] = S.v
                S.rows.append(S.row)
            pd.DataFrame({'prot_id': S.PACKS[S.split]['pids'], 'residue_index': S.PACKS[S.split]['ridx'], 'y_true': S.y, 'local_probability': S.lp, 'pocket_probability': S.pp}).to_csv(S.RUN_ROOT / f'predictions_{S.split}.csv.gz', index=False)
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
    S.fp = S.BASELINE_WORK / 'results' / 'FULL_DIRECT_BASELINE_METRICS_300_46.csv'
    if S.fp.exists():
        S.z = pd.read_csv(S.fp)
        if 'label' in S.z.columns:
            S.z = S.z[S.z.label.eq('small_binding')].copy()
        if 'operating_point' in S.z.columns:
            S.z = S.z[S.z.operating_point.eq('fixed_0.5')].copy()
        S.baseline_frames.append(S.z)
    S.baselines = pd.concat(S.baseline_frames, ignore_index=True, sort=False) if S.baseline_frames else pd.DataFrame()
    if len(S.baselines):
        S.cols = [c for c in ['method', 'dataset', 'label', 'AP_AUPRC', 'AUPRC', 'AUROC', 'global_positive_F1', 'global_MCC', 'protein_mean_F1', 'protein_mean_MCC'] if c in S.baselines.columns]
        print('LOCKED REPRODUCTION BASELINES')
        display(S.baselines[S.cols])
        S.baselines.to_csv(S.RUN_ROOT / 'locked_small_baselines.csv', index=False)
    S.historic_runs = ['ultimate_other_generic_core_v1', 'ultimate_other_standardized13_transfer_v1', 'ultimate_other_metal_transfer_v1', 'ultimate_other_metal_mechanistic_transfer_v1', 'ultimate_other_evo_full_v1', 'ultimate_other_generic_graph_v1', 'ultimate_other_evo_graph_v1']
    S.historic = []
    for S.run in S.historic_runs:
        S.root = S.RESULTS_ROOT / S.run / 'small_binding'
        S.fp = S.root / 'cv_summary.csv'
        if S.fp.exists():
            S.z = pd.read_csv(S.fp)
            S.z.insert(0, 'run_version', S.run)
            S.historic.append(S.z)
    S.historic_cv = pd.concat(S.historic, ignore_index=True, sort=False) if S.historic else pd.DataFrame()
    if len(S.historic_cv):
        print('PRIOR SMALL DEV CONTROLS')
        display(S.historic_cv.sort_values('CV_AUPRC', ascending=False).head(40))
        S.historic_cv.to_csv(S.RUN_ROOT / 'prior_small_cv_controls.csv', index=False)

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
    S.pocket_gain = S.BEST_POCKET_SCORE - S.LOCKED_ARCH_SCORE
    S.summary = {'ESM_baseline': S.base_score, 'best_representation': S.BEST_REP_NAME, 'best_representation_dev_auprc': S.BEST_REP_SCORE, 'best_representation_gain_vs_ESM': S.BEST_REP_SCORE - S.base_score, 'locked_multimodal_architecture': S.LOCKED_ARCH, 'locked_multimodal_dev_auprc': S.LOCKED_ARCH_SCORE, 'architecture_gain_vs_concat': S.arch_gain, 'architecture_verdict': verdict(S.arch_gain), 'best_pocket_hypergraph': S.BEST_POCKET, 'best_pocket_dev_auprc': S.BEST_POCKET_SCORE, 'pocket_gain_vs_locked_local': S.pocket_gain, 'pocket_verdict': verdict(S.pocket_gain), 'pocket_alpha_mean': float(np.mean(S.alphas)) if S.alphas else None, 'global_family': S.GLOBAL_FAMILY, 'global_model': S.GLOBAL_MODEL, 'global_dev_auprc': S.GLOBAL_SCORE, 'EVO_available': S.EVO_AVAILABLE, 'ESMIF_available': S.ESMIF_AVAILABLE}
    (S.RUN_ROOT / 'INSIGHT_summary.json').write_text(json.dumps(S.summary, indent=2))
    print(json.dumps(S.summary, indent=2))
    print('\n' + '=' * 96)
    print('SMALL ULTIMATE V1 — INTERPRETATION')
    print('=' * 96)
    for S.key in ['ESM+ACCESS+TOPO', 'ESM+POCKET_GEOMETRY', 'ESM+AROM_HYDRO', 'ESM+LIGAND_CHEMISTRY', 'ESM+POCKET_CHEM', 'ESM+POCKET_CHEM+SOLV', 'ESM+POCKET_CHEM+ATOM_CHEM_CONTROL', 'ESM+POCKET_CHEM+EVO_CONTROL']:
        if S.key in set(S.combo_cv.representation):
            S.s = score_rep(S.combo_cv, S.key)
            print(f'{S.key:44s} {S.s:.5f} ({S.s - S.base_score:+.5f} vs ESM)')
    print('\nArchitecture:')
    print(S.LOCKED_ARCH, f'{S.LOCKED_ARCH_SCORE:.5f}', 'gain vs concat =', f'{S.arch_gain:+.5f}', verdict(S.arch_gain))
    print('\nPocket hypergraph:')
    print(S.BEST_POCKET, f'{S.BEST_POCKET_SCORE:.5f}', 'gain vs local =', f'{S.pocket_gain:+.5f}', verdict(S.pocket_gain))
    if S.alphas:
        print('Learned pocket residual alpha:', [round(x, 4) for x in S.alphas], 'mean=', round(float(np.mean(S.alphas)), 4))
    print('\nGLOBAL DEV-LOCKED WINNER:')
    print(S.GLOBAL_FAMILY, '/', S.GLOBAL_MODEL, '/', f'{S.GLOBAL_SCORE:.5f}')
    print('\nExternal headline:')
    display(S.headline)
    print('\nPocket coverage:')
    display(S.pocket_audit.groupby('dataset').agg(mean_candidate_pockets=('n_candidate_pockets', 'mean'), median_candidate_pockets=('n_candidate_pockets', 'median'), mean_covered_fraction=('covered_residue_fraction', 'mean')).reset_index())
    print('\nArtifacts:')
    print(S.RUN_ROOT)
    print('=' * 96)

# 23. FINAL MANIFEST
def stage_35():
    S.manifest = {'gear_version': S.GEAR_VERSION, 'python': platform.python_version(), 'torch': torch.__version__, 'device': str(S.DEVICE), 'task': 'small_binding', 'development_set': 'DevSet1010 official bindNode24 5-fold split', 'external_sets': ['TestSet300', 'TestSetNew46'], 'selection_metric': 'residue-level Dev CV AUPRC', 'audit_driven_design': {'core_existing': ['ACCESS', 'TOPO', 'CHEM', 'SOLV'], 'new_blocks': S.NEW_BLOCKS, 'core_mechanism': 'pocket accessibility + geometry/topology + aromatic/hydrophobic/sulfur chemistry', 'electrostatics_role': 'secondary/contextual, not core', 'EVO_role': 'optional control because prior small-binding EVO was not consistently beneficial', 'pocket_reasoning': 'soft residual pocket hypergraph'}, 'best_representation': S.BEST_REP_NAME, 'best_representation_blocks': S.BEST_REP_BLOCKS, 'locked_architecture': S.LOCKED_ARCH, 'best_pocket_model': S.BEST_POCKET, 'global_family': S.GLOBAL_FAMILY, 'global_model': S.GLOBAL_MODEL, 'global_dev_auprc': S.GLOBAL_SCORE, 'evo_available': S.EVO_AVAILABLE, 'esmif_available': S.ESMIF_AVAILABLE, 'summary': S.summary}
    (S.RUN_ROOT / 'FINAL_MANIFEST.json').write_text(json.dumps(S.manifest, indent=2))
    print(json.dumps(S.manifest, indent=2))
    print('\n' + '=' * 96)
    print('SMALL ULTIMATE V1 COMPLETE')
    print('=' * 96)
    print('Headline:', S.GLOBAL_FAMILY, '/', S.GLOBAL_MODEL)
    print('Dev CV AUPRC:', S.GLOBAL_SCORE)
    print('Saved to:', S.RUN_ROOT)
    print('=' * 96)

STAGES = [stage_28, stage_29, stage_30, stage_32, stage_34, stage_35]

