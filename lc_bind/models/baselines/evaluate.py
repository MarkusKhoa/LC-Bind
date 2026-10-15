"""Extracted from MASTER_BINDING_BASELINE_REPRODUCTION_300_46_FULL_v10_BINDNODE24_FIXED-2.ipynb. Section: evaluate."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Bio, Chain, Model, PDBIO, PDBParser, Path, SeqIO, Structure, T5EncoderModel, T5Tokenizer, accuracy_score, ast, auc, average_precision_score, balanced_accuracy_score, brier_score_loss, choose_f1_threshold, choose_mcc_threshold, deepcopy, f1_score, fbeta_score, global_threshold_metrics, h5py, hashlib, importlib, jaccard_score, json, matthews_corrcoef, np, os, pd, pickle, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve, seq1, shutil, student_t, subprocess, sys, time, torch, tqdm
    import hashlib
    from pathlib import Path
    import os, sys, json, ast, shutil, subprocess, pickle, hashlib, time
    import numpy as np
    import pandas as pd
    import importlib.util
    import subprocess
    import sys
    from Bio import SeqIO
    import Bio
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score, fbeta_score, matthews_corrcoef, precision_score, recall_score, accuracy_score, balanced_accuracy_score, jaccard_score, brier_score_loss, auc, roc_curve
    from scipy.stats import t as student_t
    import h5py, torch
    from transformers import T5EncoderModel, T5Tokenizer
    from tqdm.auto import tqdm
    from copy import deepcopy
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.SeqUtils import seq1
    import shutil
    import json
    from lc_bind.models.baselines.metrics import choose_f1_threshold, choose_mcc_threshold, global_threshold_metrics

def sha256_file(path, chunk=1024 * 1024):
    path = Path(path)
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

# 13A. PUBLISHED-SCALE REPRODUCTION CHECK
def stage_46():
    S.published = pd.DataFrame([['TestSet300', 'any_ligand_binding_sites', 'bindEmbed21DL', 0.43, 0.41], ['TestSet300', 'metal_binding', 'bindEmbed21DL', 0.22, 0.22], ['TestSet300', 'nuclear_binding', 'bindEmbed21DL', 0.24, 0.22], ['TestSet300', 'small_binding', 'bindEmbed21DL', 0.33, 0.31], ['TestSetNew46', 'any_ligand_binding_sites', 'bindEmbed21DL', 0.37, np.nan], ['TestSetNew46', 'metal_binding', 'bindEmbed21DL', 0.26, np.nan], ['TestSetNew46', 'nuclear_binding', 'bindEmbed21DL', 0.19, np.nan], ['TestSetNew46', 'small_binding', 'bindEmbed21DL', 0.29, np.nan], ['TestSet300', 'any_ligand_binding_sites', 'bindNode24', 0.44, 0.42], ['TestSet300', 'metal_binding', 'bindNode24', 0.26, 0.26], ['TestSet300', 'nuclear_binding', 'bindNode24', 0.26, 0.25], ['TestSet300', 'small_binding', 'bindNode24', 0.35, 0.34]], columns=['dataset', 'label', 'method', 'published_protein_F1', 'published_protein_MCC'])
    S.fixed = S.full_direct[S.full_direct.operating_point.eq('fixed_0.5')].copy()
    S.chk = S.fixed.merge(S.published, on=['dataset', 'label', 'method'], how='inner')
    S.chk['delta_F1'] = S.chk['protein_mean_F1'] - S.chk['published_protein_F1']
    S.chk['delta_MCC'] = S.chk['protein_mean_MCC'] - S.chk['published_protein_MCC']
    S.chk['F1_within_tolerance'] = S.chk.delta_F1.abs() <= S.REPRO_TOLERANCE_ABS
    S.chk['MCC_within_tolerance'] = S.chk.apply(lambda r: True if pd.isna(r.published_protein_MCC) else abs(r.delta_MCC) <= S.REPRO_TOLERANCE_ABS, axis=1)
    display(S.chk[['dataset', 'label', 'method', 'protein_mean_F1', 'published_protein_F1', 'delta_F1', 'F1_within_tolerance', 'protein_mean_MCC', 'published_protein_MCC', 'delta_MCC', 'MCC_within_tolerance']])
    S.chk.to_csv(S.RESULTS / 'REPRODUCTION_GUARD.csv', index=False)
    S.bad = S.chk[~(S.chk.F1_within_tolerance & S.chk.MCC_within_tolerance)]
    if len(S.bad):
        raise RuntimeError('Reproduction guard failed. DO NOT REPORT these numbers yet. Inspect ProtT5 generation, sequence mapping, checkpoint mapping, and label convention.\n' + S.bad[['dataset', 'label', 'method', 'delta_F1', 'delta_MCC']].to_string(index=False))
    print('Published-scale reproduction guard PASSED.')

# 14A. COMPACT TABLES
def stage_48():
    S.main_cols = ['dataset', 'label', 'method', 'comparison_tier', 'AP_AUPRC', 'AUROC', 'Brier', 'global_positive_F1', 'global_MCC', 'global_precision_PPV', 'global_recall_sensitivity', 'global_specificity', 'negative_class_F1', 'class_macro_F1', 'class_weighted_F1', 'class_micro_F1', 'protein_mean_F1', 'protein_F1_CI95_legacy_halfwidth', 'protein_mean_MCC', 'protein_MCC_CI95_legacy_halfwidth', 'protein_macro_positive_n', 'protein_false_alarm_only_n', 'protein_TN_only_excluded_n']
    S.main = S.fixed[S.main_cols].copy()
    S.main.to_csv(S.RESULTS / 'PAPER_READY_FIXED_0_5_300_46.csv', index=False)
    display(S.main.sort_values(['dataset', 'label', 'method']))
    S.supp = S.full_direct[S.full_direct.operating_point.ne('fixed_0.5')].copy()
    S.supp.to_csv(S.RESULTS / 'SUPPLEMENTARY_OOF_OPERATING_POINTS_300_46.csv', index=False)

# 15A. TEST-ORACLE DIAGNOSTICS
def stage_50():
    S.oracle_rows = []
    for S.method, S.probs, S.tier in [('bindEmbed21DL', globals().get('bind21_probs', {}), 'diagnostic'), ('bindNode24', globals().get('bind24_probs', {}), 'diagnostic')]:
        for S.dataset, S.df in S.DATASETS.items():
            if S.dataset not in S.probs:
                continue
            for S.label in S.LABELS:
                S.y = np.concatenate([np.asarray(v, np.int8) for v in S.df[S.label]])
                S.p = np.concatenate([np.asarray(S.probs[S.dataset][S.label][pid], float) for pid in S.df.prot_id])
                S.tf1 = choose_f1_threshold(S.y, S.p)
                S.tmcc = choose_mcc_threshold(S.y, S.p)
                for S.op, S.t in [('TEST_ORACLE_F1', S.tf1), ('TEST_ORACLE_MCC', S.tmcc)]:
                    S.z = global_threshold_metrics(S.y, S.p, S.t)
                    S.z.update({'method': S.method, 'dataset': S.dataset, 'label': S.label, 'operating_point': S.op})
                    S.oracle_rows.append(S.z)
    S.oracle = pd.DataFrame(S.oracle_rows)
    S.oracle.to_csv(S.RESULTS / 'DIAGNOSTIC_TEST_ORACLE_NOT_FOR_CLAIMS.csv', index=False)

# 19B. MASTER LOCKED-MODEL TABLE
def stage_77():
    S.frames = []
    if 'full_direct' in globals() and len(S.full_direct):
        S.frames.append(S.full_direct.copy())
    if 'esmb_df' in globals() and len(S.esmb_df):
        S.frames.append(S.esmb_df.copy())
    if 'metalb_df' in globals() and len(S.metalb_df):
        S.frames.append(S.metalb_df.copy())
    S.master_locked = pd.concat(S.frames, ignore_index=True) if S.frames else pd.DataFrame()
    if len(S.master_locked):
        S.master_locked.to_csv(S.RESULTS / 'MASTER_LOCKED_MODEL_FULL_METRICS_300_46.csv', index=False)
        display(S.master_locked.sort_values(['dataset', 'label', 'method', 'operating_point']))
    else:
        print('No completed model branches yet.')

# PUBLIC MODEL / BENCHMARK ARTIFACT SHA256 MANIFEST
def stage_79():
    S.artifact_rows = []
    for S.name, S.p in [('project_TestSet300', S.TEST300_CSV), ('project_TestSetNew46', S.TEST46_CSV)]:
        if Path(S.p).exists():
            S.artifact_rows.append(['benchmark', S.name, str(S.p), Path(S.p).stat().st_size, sha256_file(S.p)])
    for S.i in range(1, 6):
        S.p = S.REPOS / 'bind21' / 'trained_models' / f'checkpoint{S.i}.pt'
        if S.p.exists():
            S.artifact_rows.append(['bindEmbed21DL', f'checkpoint{S.i}', str(S.p), S.p.stat().st_size, sha256_file(S.p)])
    for S.i in range(5):
        S.p = S.REPOS / 'bindnode24' / 'trained_model_dicts' / 'bindNode24' / f'trained_model_{S.i}.state_dict'
        if S.p.exists():
            S.artifact_rows.append(['bindNode24', f'trained_model_{S.i}', str(S.p), S.p.stat().st_size, sha256_file(S.p)])
    for S.i in range(1, 6):
        S.p = S.REPOS / 'esmbind' / 'multi_modal_binding' / 'model' / 'trained_weights' / f'fold_{S.i}.pt'
        if S.p.exists():
            S.artifact_rows.append(['ESMBind', f'fold_{S.i}', str(S.p), S.p.stat().st_size, sha256_file(S.p)])
    for S.ion in S.METALB_IONS:
        S.p = S.REPOS / 'metalbind' / 'Pretrained_model' / S.ion / 'best.pth'
        if S.p.exists():
            S.artifact_rows.append(['MetalBind', S.ion, str(S.p), S.p.stat().st_size, sha256_file(S.p)])
    S.artifact_manifest = pd.DataFrame(S.artifact_rows, columns=['method', 'artifact', 'path', 'bytes', 'sha256'])
    S.artifact_manifest.to_csv(S.RESULTS / 'PUBLIC_MODEL_ARTIFACT_SHA256_MANIFEST.csv', index=False)
    display(S.artifact_manifest)
    if S.RUN_BIND21:
        assert (S.artifact_manifest.method == 'bindEmbed21DL').sum() == 5
    if S.RUN_BINDNODE24:
        assert (S.artifact_manifest.method == 'bindNode24').sum() == 5
    if S.RUN_ESMBIND_TRANSFER:
        assert (S.artifact_manifest.method == 'ESMBind').sum() == 5
    if S.RUN_METALBIND_TRANSFER:
        assert (S.artifact_manifest.method == 'MetalBind').sum() == len(S.METALB_IONS)
    print('Public checkpoint provenance audit passed.')

# HARD COMPLETION AUDIT
def stage_80():
    S.expected = []
    if S.RUN_BIND21:
        S.expected += [('bindEmbed21DL', ds, lab) for ds in S.DATASETS for lab in S.LABELS]
    if S.RUN_BINDNODE24:
        S.expected += [('bindNode24', ds, lab) for ds in S.DATASETS for lab in S.LABELS]
    if S.RUN_ESMBIND_TRANSFER:
        S.expected += [('ESMBind_7ion_max_TRANSFER', ds, 'metal_binding') for ds in S.DATASETS]
    if S.RUN_METALBIND_TRANSFER:
        S.expected += [('MetalBind_13ion_max_TRANSFER', ds, 'metal_binding') for ds in S.DATASETS]
    S.fixed_master = S.master_locked[S.master_locked.operating_point.eq('fixed_0.5')].copy() if len(S.master_locked) else pd.DataFrame()
    S.actual = set(zip(S.fixed_master.method, S.fixed_master.dataset, S.fixed_master.label)) if len(S.fixed_master) else set()
    S.missing = [x for x in S.expected if x not in S.actual]
    S.status = pd.DataFrame(S.expected, columns=['method', 'dataset', 'label'])
    S.status['complete'] = S.status.apply(lambda r: (r.method, r.dataset, r.label) in S.actual, axis=1)
    S.status.to_csv(S.RESULTS / 'FINAL_BRANCH_COMPLETION_AUDIT.csv', index=False)
    display(S.status)
    if S.missing:
        raise RuntimeError('FINAL COMPLETION GUARD FAILED. Do not use the master comparison yet. Missing rows:\n' + '\n'.join(map(str, S.missing[:30])))
    print(f'FINAL COMPLETION GUARD PASSED: {len(S.expected)} / {len(S.expected)} required fixed-0.5 rows present.')

# 19A. FINAL REPRODUCIBILITY MANIFEST
def stage_81():
    S.manifest = {'timestamp_utc': pd.Timestamp.utcnow().isoformat(), 'repo_pins': {k: {'url': v[0], 'sha': v[1]} for k, v in S.REPO_SPECS.items()}, 'datasets': {k: {'proteins': len(v), 'csv': str(S.TEST300_CSV if k == 'TestSet300' else S.TEST46_CSV)} for k, v in S.DATASETS.items()}, 'fixed_threshold': S.FIXED_THRESHOLD, 'OOF_thresholds_enabled': S.RUN_OOF_THRESHOLDS, 'AP_definition': 'sklearn.average_precision_score', 'PR_AUC_trapezoid_saved_separately': True, 'global_metrics_definition': 'all residues concatenated first', 'per_protein_definition': 'calculate per protein; include iff TP+FP+FN>0; average included proteins', 'any_probability': 'max(metal,nuclear,small)', 'ESMBind_transfer_probability': 'max(MG,FE,CU,CO,CA,MN,ZN)', 'MetalBind_transfer_probability': 'max over 13 official ion-model probabilities after author surface preprocessing and full-sequence remapping', 'bootstrap_reps': S.BOOTSTRAP_REPS, 'notes': ['Main paper comparison should use fixed 0.5 + per-protein means for literature consistency.', 'OOF-F1 and OOF-MCC operating points are reported separately.', 'Test-oracle thresholds are diagnostic only and never reportable.', 'bindNode24 TestSetNew46 is an external application using project structures, not a public native reproduction.', 'ESMBind/MetalBind run all external proteins and are scored only against metal_binding; no positive-only filtering.']}
    (S.MANIFEST / 'run_manifest.json').write_text(json.dumps(S.manifest, indent=2))
    print(S.MANIFEST / 'run_manifest.json')

STAGES = [stage_46, stage_48, stage_50, stage_77, stage_79, stage_80, stage_81]

