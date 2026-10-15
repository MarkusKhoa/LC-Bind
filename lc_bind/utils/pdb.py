"""Extracted from SMALL_ULTIMATE_V1_DATA_DRIVEN_POCKET-3.ipynb. Section: pdb."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global PDBParser, PDBIO, Structure, Model, Chain, seq1
    from Bio.PDB import PDBIO, PDBParser
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure
    from Bio.SeqUtils import seq1
    global Chain, Counter, DataLoader, Dataset, F, Model, PDBIO, PDBParser, Path, Structure, Subset, accuracy_score, ast, average_precision_score, balanced_accuracy_score, cKDTree, deepcopy, defaultdict, f1_score, hashlib, json, math, matthews_corrcoef, nn, np, os, pd, platform, precision_score, recall_score, roc_auc_score, seq1, shutil, subprocess, sys, tarfile, torch
    from pathlib import Path
    import shutil
    import tarfile
    import json
    import hashlib
    import os
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
    import math
    from collections import Counter
    import json, numpy as np, pandas as pd, torch
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score

def standard_residues(chain):
    return [r for r in chain if r.id[0] == ' ' and r.resname.strip().upper() in S.AA3]

def residue_sequence(rr):
    return ''.join((seq1(r.resname.strip().upper()) for r in rr))

def pdb_exact_ok(path, target_seq):
    """
    Check whether PDB is already exact chain-A normalized.

    Important:
    use this mainly on /content or on a candidate file,
    NOT recursively over Drive.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        st = S._parser.get_structure('check', str(path))
        model = next(st.get_models())
        if 'A' not in model:
            return False
        rr = standard_residues(model['A'])
        return residue_sequence(rr) == target_seq
    except Exception:
        return False

def normalize_exact_chain_A(src, target_seq, dst):
    """
    Normalize ONLY if we truly do not already have
    an exact-normalized copy.
    """
    st = S._parser.get_structure('src', str(src))
    model = next(st.get_models())
    exact = []
    diag = []
    for ch in model:
        rr = standard_residues(ch)
        seq = residue_sequence(rr)
        diag.append((str(ch.id), len(seq)))
        if seq == target_seq:
            exact.append((ch, rr))
    if not exact:
        raise RuntimeError(f'{src}: no exact chain match; target={len(target_seq)}, chains={diag}')
    exact.sort(key=lambda x: (0 if str(x[0].id) == 'A' else 1, str(x[0].id)))
    _, rr = exact[0]
    out_st = Structure('normalized')
    out_model = Model(0)
    out_chain = Chain('A')
    out_st.add(out_model)
    out_model.add(out_chain)
    for i, res in enumerate(rr, start=1):
        r2 = deepcopy(res)
        r2.detach_parent()
        r2.id = (' ', i, ' ')
        out_chain.add(r2)
    dst.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(out_st)
    io.save(str(dst))

def normalized_candidates(split, pid):
    pid = str(pid)
    aliases = {'DevSet1010': ['DevSet1010', 'train'], 'TestSet300': ['TestSet300', 'test300'], 'TestSetNew46': ['TestSetNew46', 'test46']}
    out = []
    for root in S.OLD_NORMALIZED_ROOTS:
        out.append(root / f'{pid}.pdb')
        for alias in aliases.get(split, []):
            out.append(root / alias / f'{pid}.pdb')
    return out

def raw_candidates(pid):
    pid = str(pid)
    return [S.BASE / 'esmFold_pdb_files' / f'{pid}.pdb', S.BASE / 'experimental_pdb' / f'{pid}.pdb', S.BN24 / 'dataset' / 'AlphaFold_predictions' / f'{pid}.pdb']

def get_raw_fallback_index():
    if S.RAW_FALLBACK_INDEX is not None:
        return S.RAW_FALLBACK_INDEX
    print('\nDirect raw paths missed at least one protein.')
    print('Building ONE fallback source index...')
    idx = {}
    for root in S.RAW_ROOTS:
        if not root.exists():
            continue
        print(' scanning:', root)
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith('.pdb'):
                    continue
                p = Path(dirpath) / fn
                idx.setdefault(p.stem, p)
    S.RAW_FALLBACK_INDEX = idx
    print('fallback PDBs indexed:', len(idx))
    return idx

# 5. SHARED EXACT PDB CACHE — NUCLEAR + SMALL
def stage_11():
    S.AA3 = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
    S.LOCAL_SHARED_ROOT = Path(str(S.CACHE_ROOT / 'shared_exact_pdb_v1'))
    S.DRIVE_ARCHIVE = S.CACHE_ROOT / 'shared_exact_pdb_v1.tar.gz'
    S.DRIVE_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    S.PDB_NORM_ROOT = S.LOCAL_SHARED_ROOT
    S.OLD_NORMALIZED_ROOTS = [S.CACHE_ROOT / 'nuclear_ultimate_v1' / 'normalized_pdb', S.CACHE_ROOT / 'small_ultimate_v1' / 'normalized_pdb', S.BASELINE_WORK / 'cache' / 'metal_external_normalized_structures']
    S.RAW_ROOTS = [S.BASE / 'esmFold_pdb_files', S.BASE / 'experimental_pdb', S.BN24 / 'dataset' / 'AlphaFold_predictions']
    S._parser = PDBParser(QUIET=True)
    if S.DRIVE_ARCHIVE.exists() and (not S.LOCAL_SHARED_ROOT.exists()):
        print('Found shared PDB archive.')
        print('Copy Drive -> /content ...')
        S.local_tar = Path('/content') / S.DRIVE_ARCHIVE.name
        shutil.copy2(S.DRIVE_ARCHIVE, S.local_tar)
        print('Extracting locally ...')
        with tarfile.open(S.local_tar, 'r:gz') as S.tf:
            S.tf.extractall('/content')
        S.local_tar.unlink(missing_ok=True)
    S.LOCAL_SHARED_ROOT.mkdir(parents=True, exist_ok=True)
    S.RAW_FALLBACK_INDEX = None
    S.stats = {'already_local': 0, 'reused_old_normalized': 0, 'normalized_raw': 0}
    S.missing = []
    S.TOTAL = sum((len(df) for df in S.SPLIT_DF.values()))
    S.done = 0
    for S.split, S.df in S.SPLIT_DF.items():
        S.outdir = S.LOCAL_SHARED_ROOT / S.split
        S.outdir.mkdir(parents=True, exist_ok=True)
        print('\n', S.split, len(S.df), 'proteins')
        for S.r in S.df.itertuples():
            S.pid = str(S.r.prot_id)
            S.dst = S.outdir / f'{S.pid}.pdb'
            if pdb_exact_ok(S.dst, S.r.sequence):
                S.stats['already_local'] += 1
                S.done += 1
                continue
            S.reused = False
            for S.cand in normalized_candidates(S.split, S.pid):
                if not S.cand.exists():
                    continue
                if pdb_exact_ok(S.cand, S.r.sequence):
                    shutil.copy2(S.cand, S.dst)
                    S.stats['reused_old_normalized'] += 1
                    S.reused = True
                    break
            if S.reused:
                S.done += 1
                if S.done % 100 == 0:
                    print(f'  {S.done}/{S.TOTAL}', S.stats)
                continue
            S.src = None
            for S.cand in raw_candidates(S.pid):
                if S.cand.exists():
                    S.src = S.cand
                    break
            if S.src is None:
                S.src = get_raw_fallback_index().get(S.pid)
            if S.src is None:
                S.missing.append((S.split, S.pid))
                continue
            normalize_exact_chain_A(S.src, S.r.sequence, S.dst)
            if not pdb_exact_ok(S.dst, S.r.sequence):
                raise RuntimeError(f'{S.split}/{S.pid}: normalized PDB failed validation')
            S.stats['normalized_raw'] += 1
            S.done += 1
            if S.done % 100 == 0:
                print(f'  {S.done}/{S.TOTAL}', S.stats)
    if S.missing:
        raise FileNotFoundError(f'Missing source PDB for {len(S.missing)} proteins: {S.missing[:20]}')
    print('\nCACHE BUILD COMPLETE')
    print(S.stats)
    print('\nPacking shared local cache...')
    S.local_tar = Path('/content') / 'shared_exact_pdb_v1.tar.gz'
    with tarfile.open(S.local_tar, 'w:gz') as S.tf:
        S.tf.add(S.LOCAL_SHARED_ROOT, arcname='shared_exact_pdb_v1')
    print('Copying ONE archive to Drive...')
    shutil.copy2(S.local_tar, S.DRIVE_ARCHIVE)
    S.local_tar.unlink(missing_ok=True)
    print('\nExact-sequence shared structures: PASS')
    print('PDB_NORM_ROOT =', S.PDB_NORM_ROOT)
    print('Persistent archive =', S.DRIVE_ARCHIVE)

STAGES = [stage_11]

