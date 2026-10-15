"""Extracted from MASTER_BINDING_BASELINE_REPRODUCTION_300_46_FULL_v10_BINDNODE24_FIXED-2.ipynb. Section: metrics."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Bio, Chain, Model, PDBIO, PDBParser, Path, SeqIO, Structure, T5EncoderModel, T5Tokenizer, accuracy_score, ast, auc, average_precision_score, balanced_accuracy_score, brier_score_loss, deepcopy, f1_score, fbeta_score, h5py, hashlib, importlib, jaccard_score, json, matthews_corrcoef, np, os, pd, pickle, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve, seq1, shutil, student_t, subprocess, sys, time, torch, tqdm
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score, fbeta_score, matthews_corrcoef, precision_score, recall_score, accuracy_score, balanced_accuracy_score, jaccard_score, brier_score_loss, auc, roc_curve
    from scipy.stats import t as student_t
    from pathlib import Path
    import os, sys, json, ast, shutil, subprocess, pickle, hashlib, time
    import numpy as np
    import pandas as pd
    import importlib.util
    import subprocess
    import sys
    from Bio import SeqIO
    import Bio
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
    import hashlib

def confusion(y, pred):
    y = np.asarray(y, np.int8)
    pred = np.asarray(pred, np.int8)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return (tp, fp, tn, fn)

def safe_div(a, b):
    return float(a / b) if b else 0.0

def ci_legacy(x):
    x = np.asarray(x, float)
    return np.nan if len(x) < 2 else float(np.std(x, ddof=0) / np.sqrt(len(x)) * student_t.ppf(0.975, len(x)))

def ci_standard(x):
    x = np.asarray(x, float)
    return np.nan if len(x) < 2 else float(np.std(x, ddof=1) / np.sqrt(len(x)) * student_t.ppf(0.975, len(x) - 1))

def bootstrap_mean_ci(x, reps=2000, seed=42):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, float)
    for i in range(reps):
        vals[i] = np.mean(rng.choice(x, size=len(x), replace=True))
    return tuple(np.percentile(vals, [2.5, 97.5]).astype(float))

def threshold_free_metrics(y, p):
    y = np.asarray(y, np.int8)
    p = np.asarray(p, float)
    pr, rc, _ = precision_recall_curve(y, p)
    pr_auc = float(auc(rc[::-1], pr[::-1]))
    return {'n_residues': int(len(y)), 'n_positive': int(y.sum()), 'n_negative': int(len(y) - y.sum()), 'prevalence': float(y.mean()), 'AP_AUPRC': float(average_precision_score(y, p)), 'PR_AUC_trapezoid': pr_auc, 'AUROC': float(roc_auc_score(y, p)), 'Brier': float(brier_score_loss(y, p))}

def global_threshold_metrics(y, p, t):
    y = np.asarray(y, np.int8)
    p = np.asarray(p, float)
    pred = (p >= t).astype(np.int8)
    tp, fp, tn, fn = confusion(y, pred)
    spec = safe_div(tn, tn + fp)
    npv = safe_div(tn, tn + fn)
    return {'threshold': float(t), 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'global_accuracy': float(accuracy_score(y, pred)), 'global_balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'global_precision_PPV': float(precision_score(y, pred, zero_division=0)), 'global_recall_sensitivity': float(recall_score(y, pred, zero_division=0)), 'global_specificity': spec, 'global_NPV': npv, 'global_positive_F1': float(f1_score(y, pred, average='binary', pos_label=1, zero_division=0)), 'global_F1': float(f1_score(y, pred, average='binary', pos_label=1, zero_division=0)), 'negative_class_F1': float(f1_score(y, pred, average='binary', pos_label=0, zero_division=0)), 'class_macro_F1': float(f1_score(y, pred, average='macro', zero_division=0)), 'class_weighted_F1': float(f1_score(y, pred, average='weighted', zero_division=0)), 'class_micro_F1': float(f1_score(y, pred, average='micro', zero_division=0)), 'global_F2': float(fbeta_score(y, pred, beta=2, zero_division=0)), 'global_Jaccard': float(jaccard_score(y, pred, zero_division=0)), 'global_MCC': float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0, 'predicted_positive_rate': float(pred.mean())}

def per_protein_table(df, probs, label, t):
    rows = []
    for r in df.itertuples():
        pid = str(r.prot_id)
        y = np.asarray(getattr(r, label), np.int8)
        p = np.asarray(probs[pid], float)
        assert len(y) == len(p), (pid, len(y), len(p))
        pred = (p >= t).astype(np.int8)
        tp, fp, tn, fn = confusion(y, pred)
        include = tp + fp + fn > 0
        false_alarm_only = tp + fn == 0 and fp > 0
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        f1 = safe_div(2 * tp, 2 * tp + fp + fn)
        den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        mcc = float((tp * tn - fp * fn) / np.sqrt(den)) if den > 0 else 0.0
        accuracy = safe_div(tp + tn, tp + fp + tn + fn)
        rows.append([pid, include, false_alarm_only, tp, fp, tn, fn, precision, recall, specificity, f1, mcc, accuracy])
    return pd.DataFrame(rows, columns=['prot_id', 'included_macro_positive', 'false_alarm_only', 'TP', 'FP', 'TN', 'FN', 'precision', 'recall', 'specificity', 'F1', 'MCC', 'accuracy'])

def per_protein_summary(pt, reps=2000, seed=42):
    inc = pt[pt.included_macro_positive].copy()
    out = {'protein_macro_positive_n': int(len(inc)), 'protein_TN_only_excluded_n': int((~pt.included_macro_positive).sum()), 'protein_false_alarm_only_n': int(pt.false_alarm_only.sum())}
    for metric in ['precision', 'recall', 'specificity', 'F1', 'MCC', 'accuracy']:
        x = inc[metric].to_numpy(float)
        out[f'protein_mean_{metric}'] = float(np.mean(x)) if len(x) else np.nan
        out[f'protein_{metric}_CI95_legacy_halfwidth'] = ci_legacy(x)
        out[f'protein_{metric}_CI95_standard_halfwidth'] = ci_standard(x)
        if metric in ['F1', 'MCC']:
            lo, hi = bootstrap_mean_ci(x, reps, seed)
            out[f'protein_{metric}_bootstrap95_low'] = lo
            out[f'protein_{metric}_bootstrap95_high'] = hi
    return out

def evaluate_probability_dict(df, probs, label, method, dataset, operating_point, t, comparison_tier):
    ids = [str(x) for x in df.prot_id]
    missing = [pid for pid in ids if pid not in probs]
    assert not missing, f'{method}/{dataset}/{label}: missing {missing[:10]}'
    y = np.concatenate([np.asarray(v, np.int8) for v in df[label]])
    p = np.concatenate([np.asarray(probs[str(pid)], float) for pid in df.prot_id])
    tf = threshold_free_metrics(y, p)
    gt = global_threshold_metrics(y, p, t)
    pt = per_protein_table(df, probs, label, t)
    ps = per_protein_summary(pt, S.BOOTSTRAP_REPS, S.RANDOM_SEED)
    out = {'method': method, 'dataset': dataset, 'label': label, 'comparison_tier': comparison_tier, 'operating_point': operating_point}
    return (out, pt)

def save_probs(probs, path):
    rows = []
    for pid, p in probs.items():
        for i, v in enumerate(np.asarray(p, float)):
            rows.append((pid, i, float(v)))
    pd.DataFrame(rows, columns=['prot_id', 'residue_index', 'probability']).to_csv(path, index=False, compression='gzip')

def choose_f1_threshold(y, p):
    y = np.asarray(y, np.int8)
    p = np.asarray(p, float)
    pr, rc, th = precision_recall_curve(y, p)
    if len(th) == 0:
        return 0.5
    f = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-15)
    return float(th[int(np.nanargmax(f))])

def choose_mcc_threshold(y, p):
    y = np.asarray(y, np.int8)
    p = np.asarray(p, float)
    fpr, tpr, th = roc_curve(y, p, drop_intermediate=False)
    P = float(y.sum())
    N = float(len(y) - y.sum())
    tp = tpr * P
    fp = fpr * N
    fn = P - tp
    tn = N - fp
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = np.zeros_like(th, dtype=float)
    ok = den > 0
    mcc[ok] = (tp[ok] * tn[ok] - fp[ok] * fn[ok]) / np.sqrt(den[ok])
    idx = int(np.nanargmax(mcc))
    t = float(th[idx])
    if not np.isfinite(t):
        t = float(np.nextafter(np.max(p), -np.inf))
    return t

def thresholds_from_oof(y_by_label, p_by_label):
    rows = []
    out = {}
    for label in S.LABELS:
        y = np.asarray(y_by_label[label], np.int8)
        p = np.asarray(p_by_label[label], float)
        tf1 = choose_f1_threshold(y, p)
        tmcc = choose_mcc_threshold(y, p)
        out[label] = {'OOF_F1': tf1, 'OOF_MCC': tmcc}
        mf1 = global_threshold_metrics(y, p, tf1)
        mmcc = global_threshold_metrics(y, p, tmcc)
        rows.append([label, tf1, mf1['global_F1'], mf1['global_MCC'], tmcc, mmcc['global_F1'], mmcc['global_MCC']])
    return (out, pd.DataFrame(rows, columns=['label', 'OOF_F1_threshold', 'OOF_F1_at_threshold', 'OOF_MCC_at_F1_threshold', 'OOF_MCC_threshold', 'OOF_F1_at_MCC_threshold', 'OOF_MCC_at_threshold']))

# 5A. FULL METRIC IMPLEMENTATION
def stage_18():
    S.EPS = 1e-15

STAGES = [stage_18]

