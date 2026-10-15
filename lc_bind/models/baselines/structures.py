"""Extracted from MASTER_BINDING_BASELINE_REPRODUCTION_300_46_FULL_v10_BINDNODE24_FIXED-2.ipynb. Section: structures."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Bio, Chain, Model, PDBIO, PDBParser, Path, SeqIO, Structure, T5EncoderModel, T5Tokenizer, accuracy_score, ast, auc, average_precision_score, balanced_accuracy_score, brier_score_loss, deepcopy, f1_score, fbeta_score, h5py, hashlib, importlib, jaccard_score, json, matthews_corrcoef, np, os, pd, pickle, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve, seq1, shutil, student_t, subprocess, sys, time, torch, tqdm
    from copy import deepcopy
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.SeqUtils import seq1
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
    import shutil
    import json
    import hashlib

def _standard_chain_residues(chain):
    return [r for r in chain if r.id[0] == ' ' and r.resname.strip().upper() in S.AA3]

def _residue_seq(residues):
    return ''.join((seq1(r.resname.strip().upper()) for r in residues))

def resolve_external_structure(dataset, pid):
    p = S.PROJECT_PDB.get(dataset, {}).get(str(pid))
    if p is not None and Path(p).exists():
        return Path(p)
    if dataset == 'TestSet300':
        q = S.REPOS / 'bindnode24' / 'dataset' / 'AlphaFold_predictions' / f'{pid}.pdb'
        if q.exists():
            return q
    return None

def normalize_pdb_exact_chain_A(src, target_sequence, dst):
    """
    Select an exact-sequence chain and write a one-chain canonical PDB.
    Homomers are valid: prefer chain A, else deterministic lexical chain id.
    """
    src = Path(src)
    dst = Path(dst)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('src', str(src))
    model = next(structure.get_models())
    exact = []
    diagnostic = []
    for chain in model:
        residues = _standard_chain_residues(chain)
        seq = _residue_seq(residues)
        diagnostic.append((str(chain.id), len(seq), seq[:20]))
        if seq == target_sequence:
            exact.append((chain, residues))
    if len(exact) == 0:
        raise RuntimeError(f'{src}: no chain exactly matches canonical sequence length {len(target_sequence)}. Chains={diagnostic}')
    exact = sorted(exact, key=lambda x: (0 if str(x[0].id) == 'A' else 1, str(x[0].id)))
    chain, residues = exact[0]
    source_chain = str(chain.id)
    n_exact_matches = len(exact)
    out_structure = Structure('normalized')
    out_model = Model(0)
    out_chain = Chain('A')
    out_structure.add(out_model)
    out_model.add(out_chain)
    for i, res in enumerate(residues, start=1):
        rr = deepcopy(res)
        rr.detach_parent()
        rr.id = (' ', i, ' ')
        out_chain.add(rr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(out_structure)
    io.save(str(dst))
    chk = PDBParser(QUIET=True).get_structure('chk', str(dst))
    cm = next(chk.get_models())
    if 'A' not in cm:
        raise RuntimeError(f'{dst}: normalized chain A missing after write')
    residues2 = _standard_chain_residues(cm['A'])
    seq2 = _residue_seq(residues2)
    if seq2 != target_sequence:
        raise RuntimeError(f'{dst}: read-back sequence mismatch after normalization')
    if len(residues2) != len(target_sequence):
        raise RuntimeError(f'{dst}: residue count mismatch after normalization')
    return (dst, source_chain, n_exact_matches)

# 16A. NORMALIZE ALL EXTERNAL STRUCTURES TO EXACT CANONICAL CHAIN A
def stage_52():
    S.AA3 = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
    S.METAL_EXTERNAL_STRUCT_ROOT = S.CACHE / 'metal_external_normalized_structures'
    S.METAL_EXTERNAL_STRUCT_ROOT.mkdir(parents=True, exist_ok=True)
    S.NORMALIZED_EXTERNAL_PDB = {}
    S.structure_audit = []
    for S.dataset, S.df in S.DATASETS.items():
        S.NORMALIZED_EXTERNAL_PDB[S.dataset] = {}
        S.droot = S.METAL_EXTERNAL_STRUCT_ROOT / S.dataset
        S.droot.mkdir(parents=True, exist_ok=True)
        for S.r in S.df.itertuples():
            S.pid = str(S.r.prot_id)
            S.src = resolve_external_structure(S.dataset, S.pid)
            if S.src is None:
                raise FileNotFoundError(f'{S.dataset}/{S.pid}: no external PDB found')
            S.dst = S.droot / f'{S.pid}.pdb'
            S.dst, S.source_chain, S.n_exact_matches = normalize_pdb_exact_chain_A(S.src, S.r.sequence, S.dst)
            S.NORMALIZED_EXTERNAL_PDB[S.dataset][S.pid] = S.dst
            S.structure_audit.append([S.dataset, S.pid, str(S.src), S.source_chain, S.n_exact_matches, str(S.dst), len(S.r.sequence), 'PASS'])
    S.structure_audit_df = pd.DataFrame(S.structure_audit, columns=['dataset', 'prot_id', 'source_pdb', 'selected_source_chain', 'n_exact_sequence_matching_chains', 'normalized_pdb', 'length', 'status'])
    S.structure_audit_df.to_csv(S.RESULTS / 'metal_specific_structure_normalization_audit.csv', index=False)
    display(S.structure_audit_df.groupby(['dataset', 'status']).size().reset_index(name='n'))
    for S.dataset, S.df in S.DATASETS.items():
        assert len(S.NORMALIZED_EXTERNAL_PDB[S.dataset]) == len(S.df)
        for S.r in S.df.itertuples():
            S.p = S.NORMALIZED_EXTERNAL_PDB[S.dataset][str(S.r.prot_id)]
            if not Path(S.p).exists():
                raise FileNotFoundError(S.p)
    print('External structure normalization complete for all 300 + all 46 proteins.')

STAGES = [stage_52]

