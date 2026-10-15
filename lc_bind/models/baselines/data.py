"""Extracted from MASTER_BINDING_BASELINE_REPRODUCTION_300_46_FULL_v10_BINDNODE24_FIXED-2.ipynb. Section: data."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Bio, Chain, Model, PDBIO, PDBParser, Path, SeqIO, Structure, T5EncoderModel, T5Tokenizer, accuracy_score, ast, auc, average_precision_score, balanced_accuracy_score, brier_score_loss, deepcopy, f1_score, fbeta_score, h5py, hashlib, importlib, jaccard_score, json, matthews_corrcoef, np, os, parse_list, pd, pickle, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve, seq1, shutil, student_t, subprocess, sys, time, torch, tqdm
    from Bio import SeqIO
    from pathlib import Path
    import os, sys, json, ast, shutil, subprocess, pickle, hashlib, time
    import numpy as np
    import pandas as pd
    import importlib.util
    import subprocess
    import sys
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
    import hashlib
    from lc_bind.io.datasets import parse_list

def load_frame(path):
    df = pd.read_csv(path)
    df['prot_id'] = df['prot_id'].astype(str)
    for c in S.LABELS:
        if c not in df.columns:
            raise KeyError(f'{path}: missing {c}')
        df[c] = df[c].apply(parse_list)
    return df

def write_fasta(df, path, style='plain'):
    with open(path, 'w') as f:
        for r in df.itertuples():
            if style == 'plain':
                header = r.prot_id
            elif style == 'esmbind':
                header = f'sp|{r.prot_id}|{r.prot_id}'
            else:
                raise ValueError(style)
            f.write(f'>{header}\n{r.sequence}\n')

def write_ids(df, path):
    with open(path, 'w') as f:
        for pid in df.prot_id:
            f.write(str(pid) + '\n')

def collect_author_fastas(root):
    seqs = {}
    for fp in Path(root).rglob('*'):
        if fp.suffix.lower() not in ['.fa', '.faa', '.fasta', '.fas']:
            continue
        try:
            for rec in SeqIO.parse(str(fp), 'fasta'):
                seqs.setdefault(str(rec.id), []).append((str(fp), str(rec.seq)))
        except Exception:
            pass
    return seqs

def audit_author_sequences(df, author_root, method, dataset, require_all=False):
    idx = collect_author_fastas(author_root)
    rows = []
    mismatches = []
    missing = []
    for r in df.itertuples():
        candidates = idx.get(str(r.prot_id), [])
        if not candidates:
            missing.append(str(r.prot_id))
            rows.append([method, dataset, r.prot_id, 'missing', False, ''])
            continue
        equal = [(fp, seq) for fp, seq in candidates if seq == r.sequence]
        ok = bool(equal)
        rows.append([method, dataset, r.prot_id, 'found', ok, equal[0][0] if equal else candidates[0][0]])
        if not ok:
            mismatches.append(str(r.prot_id))
    if mismatches:
        raise RuntimeError(f'{method}/{dataset}: same-ID author sequence mismatch for {mismatches[:10]}')
    if require_all and missing:
        raise RuntimeError(f'{method}/{dataset}: author sequence unavailable for {missing[:10]}')
    return pd.DataFrame(rows, columns=['method', 'dataset', 'prot_id', 'author_status', 'sequence_equal', 'author_file'])

def scan_fastas(root, method):
    rows = []
    for fp in Path(root).rglob('*'):
        if fp.suffix.lower() not in ['.fa', '.faa', '.fasta', '.fas']:
            continue
        try:
            for rec in SeqIO.parse(str(fp), 'fasta'):
                seq = str(rec.seq)
                if seq in S.external_seq:
                    for dataset, pid in S.external_seq[seq]:
                        rows.append([method, dataset, pid, str(fp), rec.id, len(seq)])
        except Exception:
            pass
    return rows

# 4A. LOAD / AUDIT PROJECT CSVs
def stage_11():
    S.LABELS = ['any_ligand_binding_sites', 'metal_binding', 'small_binding', 'nuclear_binding']
    S.train_df = load_frame(S.TRAIN_CSV)
    S.test300_df = load_frame(S.TEST300_CSV)
    S.test46_df = load_frame(S.TEST46_CSV)
    S.DATASETS = {'TestSet300': S.test300_df, 'TestSetNew46': S.test46_df}
    S.rows = []
    for S.name, S.df in [('DevSet1010_project', S.train_df), *S.DATASETS.items()]:
        assert S.df.prot_id.is_unique, S.name
        for S.label in S.LABELS:
            for S.r in S.df.itertuples():
                assert len(S.r.sequence) == len(getattr(S.r, S.label)), (S.name, S.label, S.r.prot_id)
            S.nres = int(sum((len(x) for x in S.df[S.label])))
            S.pos = int(sum((sum(x) for x in S.df[S.label])))
            S.rows.append([S.name, S.label, len(S.df), S.nres, S.pos, S.nres - S.pos, S.pos / max(S.nres, 1)])
    S.audit_df = pd.DataFrame(S.rows, columns=['dataset', 'label', 'proteins', 'residues', 'positives', 'negatives', 'prevalence'])
    display(S.audit_df)
    S.audit_df.to_csv(S.RESULTS / 'dataset_audit.csv', index=False)

# 4B. CANONICAL FASTA / ID FILES FOR BOTH EXTERNAL SETS
def stage_12():
    for S.name, S.df in S.DATASETS.items():
        write_fasta(S.df, S.CANON / f'{S.name}.fasta', 'plain')
        write_fasta(S.df, S.CANON / f'{S.name}_esmbind.fasta', 'esmbind')
        write_ids(S.df, S.CANON / f'{S.name}_ids.txt')
    print(sorted((p.name for p in S.CANON.iterdir())))

# 4C. AUTHOR SEQUENCE PROVENANCE AUDIT
def stage_16():
    S.seq_audits = []
    if (S.REPOS / 'bind21').exists():
        S.seq_audits.append(audit_author_sequences(S.test300_df, S.REPOS / 'bind21', 'bindEmbed21DL', 'TestSet300', False))
        S.seq_audits.append(audit_author_sequences(S.test46_df, S.REPOS / 'bind21', 'bindEmbed21DL', 'TestSetNew46', False))
    if (S.REPOS / 'bindnode24').exists():
        S.seq_audits.append(audit_author_sequences(S.test300_df, S.REPOS / 'bindnode24', 'bindNode24', 'TestSet300', False))
    S.sequence_audit = pd.concat(S.seq_audits, ignore_index=True) if S.seq_audits else pd.DataFrame()
    display(S.sequence_audit.groupby(['method', 'dataset', 'author_status', 'sequence_equal'], dropna=False).size().reset_index(name='n'))
    S.sequence_audit.to_csv(S.RESULTS / 'author_sequence_provenance_audit.csv', index=False)

# 18A. EXACT SEQUENCE OVERLAP AUDIT
def stage_74():
    S.external_seq = {}
    for S.dataset, S.df in S.DATASETS.items():
        for S.r in S.df.itertuples():
            S.external_seq.setdefault(S.r.sequence, []).append((S.dataset, S.r.prot_id))
    S.hits = scan_fastas(S.REPOS / 'esmbind', 'ESMBind') + scan_fastas(S.REPOS / 'metalbind', 'MetalBind')
    S.overlap = pd.DataFrame(S.hits, columns=['method', 'dataset', 'external_id', 'author_file', 'author_record', 'length'])
    display(S.overlap.head(100))
    S.overlap.to_csv(S.RESULTS / 'exact_sequence_overlap_audit.csv', index=False)

STAGES = [stage_11, stage_12, stage_16, stage_74]

