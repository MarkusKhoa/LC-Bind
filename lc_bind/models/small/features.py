"""Extracted from SMALL_ULTIMATE_V1_DATA_DRIVEN_POCKET-3.ipynb. Section: features."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Chain, Counter, DataLoader, Dataset, F, Model, PDBIO, PDBParser, Path, Structure, Subset, accuracy_score, ast, average_precision_score, balanced_accuracy_score, cKDTree, deepcopy, defaultdict, f1_score, get_block, get_record, hashlib, json, math, matthews_corrcoef, nn, np, os, parse_list, pd, platform, precision_score, recall_score, residue_sequence, roc_auc_score, seq1, set_block, shutil, standard_residues, subprocess, sys, tarfile, torch
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
    from lc_bind.io.datasets import parse_list
    from lc_bind.utils.cache import get_block, get_record, set_block
    from lc_bind.utils.pdb import residue_sequence, standard_residues

def load_df(path):
    df = pd.read_csv(path)
    df['prot_id'] = df['prot_id'].astype(str)
    df['sequence'] = df['sequence'].astype(str)
    for c in ['metal_binding', 'nuclear_binding', 'small_binding', 'any_ligand_binding_sites']:
        if c in df.columns:
            df[c] = df[c].apply(parse_list)
    for r in df.itertuples():
        if len(r.sequence) != len(r.small_binding):
            raise RuntimeError(f'{r.prot_id}: small label/sequence mismatch')
    return df

def representative_coord(res):
    side = [np.asarray(a.coord, dtype=np.float32) for a in res if a.element != 'H' and a.name not in S.BACKBONE_ATOMS]
    if side:
        return np.mean(np.stack(side), axis=0)
    if 'CA' in res:
        return np.asarray(res['CA'].coord, dtype=np.float32)
    atoms = [np.asarray(a.coord, dtype=np.float32) for a in res]
    if not atoms:
        raise RuntimeError('Residue has no atoms')
    return np.mean(np.stack(atoms), axis=0)

def ring_geometry(res):
    rn = res.resname.strip().upper()
    atoms = S.RING_ATOMS.get(rn)
    if atoms is None:
        return (None, None)
    cc = [np.asarray(res[a].coord, dtype=np.float32) for a in atoms if a in res]
    if len(cc) < 3:
        return (None, None)
    x = np.stack(cc)
    centroid = x.mean(axis=0)
    centered = x - centroid[None, :]
    cov = centered.T @ centered / max(len(x) - 1, 1)
    eigval, eigvec = np.linalg.eigh(cov)
    normal = eigvec[:, np.argmin(eigval)]
    normal = normal / (np.linalg.norm(normal) + 1e-08)
    return (centroid.astype(np.float32), normal.astype(np.float32))

def connected_component_sizes(indices, coords, threshold):
    if len(indices) == 0:
        return {}
    idx = np.asarray(indices, dtype=int)
    cc = np.asarray(coords, dtype=np.float32)
    parent = list(range(len(idx)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = (find(a), find(b))
        if ra != rb:
            parent[rb] = ra
    if len(idx) > 1:
        tree = cKDTree(cc)
        for a, b in tree.query_pairs(threshold):
            union(a, b)
    comp = defaultdict(list)
    for j, ridx in enumerate(idx):
        comp[find(j)].append(int(ridx))
    out = {}
    for members in comp.values():
        n = len(members)
        for r in members:
            out[r] = n
    return out

def sequence_chem_block(seq):
    L = len(seq)
    out = np.zeros((L, 32), dtype=np.float32)
    hydro = np.asarray([S.HYDROPATHY.get(a, 0.0) / 4.5 for a in seq], dtype=np.float32)
    for i, a in enumerate(seq):
        out[i, 0] = hydro[i]
        out[i, 1] = float(a in S.GROUPS['AROM'])
        out[i, 2] = float(a in S.GROUPS['SULFUR'])
        out[i, 3] = float(a in S.GROUPS['HYDRO'])
        out[i, 4] = float(a in S.GROUPS['ACIDIC'])
        out[i, 5] = float(a in S.GROUPS['POLAR'])
        out[i, 6] = float(a in S.GROUPS['GP'])
        out[i, 7] = float(a in S.GROUPS['BASIC'])
        col = 8
        for rad in [2, 4, 7, 10]:
            lo = max(0, i - rad)
            hi = min(L, i + rad + 1)
            w = seq[lo:hi]
            h = hydro[lo:hi]
            vals = [sum((x in S.GROUPS['AROM'] for x in w)) / len(w), sum((x in S.GROUPS['HYDRO'] for x in w)) / len(w), sum((x in S.GROUPS['SULFUR'] for x in w)) / len(w), sum((x in S.GROUPS['ACIDIC'] for x in w)) / len(w), sum((x in S.GROUPS['GP'] for x in w)) / len(w), float(np.mean(h))]
            out[i, col:col + 6] = vals
            col += 6
    return out

def build_small_3d_blocks(pdb_path, seq):
    st = PDBParser(QUIET=True).get_structure('p', str(pdb_path))
    m = next(st.get_models())
    residues = standard_residues(m['A'])
    if residue_sequence(residues) != seq:
        raise RuntimeError(f'{pdb_path}: sequence mismatch')
    coords = np.stack([representative_coord(r) for r in residues], axis=0).astype(np.float32)
    L = len(seq)
    tree = cKDTree(coords)
    hydro = np.asarray([S.HYDROPATHY.get(a, 0.0) / 4.5 for a in seq], dtype=np.float32)
    q = np.asarray([S.CHARGE.get(a, 0.0) for a in seq], dtype=np.float32)
    arom_idx = [i for i, a in enumerate(seq) if a in S.GROUPS['AROM']]
    hydro_idx = [i for i, a in enumerate(seq) if a in S.GROUPS['HYDRO']]
    arom_comp = connected_component_sizes(arom_idx, coords[arom_idx] if arom_idx else np.zeros((0, 3), dtype=np.float32), threshold=8.0)
    hydro_comp = connected_component_sizes(hydro_idx, coords[hydro_idx] if hydro_idx else np.zeros((0, 3), dtype=np.float32), threshold=8.0)
    ring_centroid = {}
    ring_normal = {}
    for i, res in enumerate(residues):
        c, n = ring_geometry(res)
        if c is not None:
            ring_centroid[i] = c
            ring_normal[i] = n
    pocket = np.zeros((L, 44), dtype=np.float32)
    ligchem = np.zeros((L, 20), dtype=np.float32)
    topo = np.zeros((L, 16), dtype=np.float32)
    arom = np.zeros((L, 12), dtype=np.float32)
    neigh6 = [tree.query_ball_point(coords[i], 6.0) for i in range(L)]
    neigh8 = [tree.query_ball_point(coords[i], 8.0) for i in range(L)]
    neigh10 = [tree.query_ball_point(coords[i], 10.0) for i in range(L)]
    for i, a in enumerate(seq):
        ligchem[i, 0] = float(a in S.GROUPS['HBD'])
        ligchem[i, 1] = float(a in S.GROUPS['HBA'])
        ligchem[i, 2] = float(a in S.GROUPS['AROM'])
        ligchem[i, 3] = float(a in S.GROUPS['SULFUR'])
        ligchem[i, 4] = float(a in S.GROUPS['HYDRO'])
        ligchem[i, 5] = float(a in S.GROUPS['POLAR'])
        ligchem[i, 6] = float(a in S.GROUPS['ACIDIC'])
        ligchem[i, 7] = float(a in S.GROUPS['BASIC'])
        ligchem[i, 8] = float(a in S.GROUPS['AMIDE'])
        ligchem[i, 9] = float(a in S.GROUPS['HYDROXYL'])
        c = 10
        for radius in [6.0, 10.0]:
            ids = np.asarray(tree.query_ball_point(coords[i], radius), dtype=int)
            ids = ids[ids != i]
            aa = [seq[j] for j in ids]
            den = max(len(aa), 1)
            ligchem[i, c:c + 5] = [sum((x in S.GROUPS['HBD'] for x in aa)) / den, sum((x in S.GROUPS['HBA'] for x in aa)) / den, sum((x in S.GROUPS['AROM'] for x in aa)) / den, sum((x in S.GROUPS['SULFUR'] for x in aa)) / den, sum((x in S.GROUPS['POLAR'] for x in aa)) / den]
            c += 5
        c = 0
        for radius in [5.0, 7.0, 9.0, 11.0]:
            ids = np.asarray(tree.query_ball_point(coords[i], radius), dtype=int)
            ids = ids[ids != i]
            if len(ids):
                aa = [seq[j] for j in ids]
                hh = hydro[ids]
                dd = np.linalg.norm(coords[ids] - coords[i][None, :], axis=1)
                vals = [min(len(ids), 40) / 40.0, sum((x in S.GROUPS['AROM'] for x in aa)) / len(aa), sum((x in S.GROUPS['HYDRO'] for x in aa)) / len(aa), sum((x in S.GROUPS['SULFUR'] for x in aa)) / len(aa), sum((x in S.GROUPS['POLAR'] for x in aa)) / len(aa), sum((x in S.GROUPS['ACIDIC'] for x in aa)) / len(aa), float(np.mean(hh)), min(float(np.mean(dd)) / 12.0, 1.0)]
            else:
                vals = [0.0] * 8
            pocket[i, c:c + 8] = vals
            c += 8

        def nearest(chars):
            ids = [j for j, x in enumerate(seq) if x in chars and j != i]
            if not ids:
                return 1.0
            d = np.linalg.norm(coords[ids] - coords[i][None, :], axis=1)
            return min(float(d.min()) / 20.0, 1.0)
        pocket[i, 32] = nearest(S.GROUPS['AROM'])
        pocket[i, 33] = nearest(S.GROUPS['SULFUR'])
        pocket[i, 34] = nearest(S.GROUPS['POLAR'])
        pocket[i, 35] = nearest(S.GROUPS['HYDRO'])
        d_all = np.linalg.norm(coords - coords[i][None, :], axis=1)
        shell_counts = [np.sum((d_all > 0.1) & (d_all <= 4.0)), np.sum((d_all > 4.0) & (d_all <= 6.0)), np.sum((d_all > 6.0) & (d_all <= 8.0)), np.sum((d_all > 8.0) & (d_all <= 10.0))]
        pocket[i, 36:40] = np.clip(np.asarray(shell_counts, dtype=np.float32) / 20.0, 0, 1)
        ids = np.asarray(neigh8[i], dtype=int)
        ids = ids[ids != i]
        if len(ids) >= 3:
            cc = coords[ids] - coords[ids].mean(axis=0, keepdims=True)
            cov = cc.T @ cc / max(len(ids) - 1, 1)
            eig = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
            total = float(eig.sum() + 1e-08)
            linearity = (eig[0] - eig[1]) / (eig[0] + 1e-08)
            planarity = (eig[1] - eig[2]) / (eig[0] + 1e-08)
            isotropy = eig[2] / total
            pocket[i, 40] = np.clip(linearity, 0, 1)
            pocket[i, 41] = np.clip(planarity, 0, 1)
            pocket[i, 42] = np.clip(isotropy, 0, 1)
            seqsep = np.abs(ids - i)
            pocket[i, 43] = float(np.mean(seqsep > 10))
        n6 = [j for j in neigh6[i] if j != i]
        n8 = [j for j in neigh8[i] if j != i]
        n10 = [j for j in neigh10[i] if j != i]
        topo[i, 0] = min(len(n6), 25) / 25.0
        topo[i, 1] = min(len(n8), 35) / 35.0
        topo[i, 2] = min(len(n10), 45) / 45.0
        a8 = [j for j in n8 if seq[j] in S.GROUPS['AROM']]
        h8 = [j for j in n8 if seq[j] in S.GROUPS['HYDRO']]
        s8 = [j for j in n8 if seq[j] in S.GROUPS['SULFUR']]
        p8 = [j for j in n8 if seq[j] in S.GROUPS['POLAR']]
        d8 = [j for j in n8 if seq[j] in S.GROUPS['ACIDIC']]
        topo[i, 3] = min(len(a8), 12) / 12.0
        topo[i, 4] = min(len(h8), 20) / 20.0
        topo[i, 5] = min(len(s8), 8) / 8.0
        topo[i, 6] = min(len(p8), 20) / 20.0
        topo[i, 7] = min(len(d8), 15) / 15.0
        topo[i, 8] = min(arom_comp.get(i, 0), 15) / 15.0
        topo[i, 9] = min(hydro_comp.get(i, 0), 25) / 25.0
        if len(n8) >= 2:
            cc = coords[n8]
            t2 = cKDTree(cc)
            edges = len(t2.query_pairs(8.0))
            possible = len(n8) * (len(n8) - 1) / 2
            topo[i, 10] = edges / max(possible, 1)
            seqsep = np.abs(np.asarray(n8) - i)
            topo[i, 11] = float(np.mean(seqsep > 10))
            topo[i, 12] = min(float(np.mean(seqsep)) / 50.0, 1.0)
            comp = Counter((seq[j] for j in n8))
            probs = np.asarray([x / len(n8) for x in comp.values()], dtype=float)
            topo[i, 13] = float(-np.sum(probs * np.log(probs + 1e-12)) / math.log(20))
        topo[i, 14] = np.tanh((len(n6) - (len(n10) - len(n6))) / 10.0)
        topo[i, 15] = np.tanh((len(h8) + len(a8) - len(d8)) / 10.0)
        arom[i, 0] = float(a in S.GROUPS['AROM'])
        arom[i, 1] = float(a == 'W')
        arom[i, 2] = float(a == 'F')
        arom[i, 3] = float(a == 'Y')
        arom[i, 4] = float(a == 'H')
        for col2, radius in zip([5, 6, 7], [6.0, 8.0, 10.0]):
            ids = [j for j in tree.query_ball_point(coords[i], radius) if j != i]
            arom[i, col2] = sum((seq[j] in S.GROUPS['AROM'] for j in ids)) / max(len(ids), 1)
        arom[i, 8] = nearest(S.GROUPS['AROM'])
        arom[i, 9] = min(arom_comp.get(i, 0), 15) / 15.0
        ar_nei = [j for j in neigh10[i] if j in ring_normal and j != i]
        dots = []
        for a1 in range(len(ar_nei)):
            for a2 in range(a1 + 1, len(ar_nei)):
                dots.append(abs(float(np.dot(ring_normal[ar_nei[a1]], ring_normal[ar_nei[a2]]))))
        arom[i, 10] = float(np.mean(dots)) if dots else 0.0
        if ar_nei:
            cent = np.mean(np.stack([ring_centroid[j] for j in ar_nei]), axis=0)
            arom[i, 11] = min(float(np.linalg.norm(coords[i] - cent)) / 20.0, 1.0)
    return (pocket, ligchem, topo, arom)

def protein_context(seq):
    L = len(seq)
    p = np.array([math.log1p(L) / 8.0, sum((a in S.GROUPS['AROM'] for a in seq)) / L, sum((a in S.GROUPS['HYDRO'] for a in seq)) / L, sum((a in S.GROUPS['SULFUR'] for a in seq)) / L, sum((a in S.GROUPS['POLAR'] for a in seq)) / L, sum((a in S.GROUPS['ACIDIC'] for a in seq)) / L, sum((a in S.GROUPS['GP'] for a in seq)) / L, float(np.mean([S.HYDROPATHY.get(a, 0.0) / 4.5 for a in seq]))], dtype=np.float32)
    return np.repeat(p[None, :], L, axis=0)

# 3. DATAFRAME / OFFICIAL-FOLD AUDIT
def stage_05():
    S.TRAIN_CSV = S.DATA_ROOT / 'development_set/full_grouped_train_binding_sites_df.csv'
    S.TEST300_CSV = S.DATA_ROOT / 'development_set/full_grouped_test_binding_sites_df.csv'
    S.TEST46_CSV = S.DATA_ROOT / 'independent_set/grouped_test_46_new_binding_sites.csv'
    S.raw_train = load_df(S.TRAIN_CSV)
    S.test300_df = load_df(S.TEST300_CSV)
    S.test46_df = load_df(S.TEST46_CSV)
    S.BASELINE_WORK = S.BASE / 'baseline_reproduction_master_v6'
    S.BN24_CANDIDATES = [S.BASELINE_WORK / 'repos' / 'bindnode24', S.BASE / 'repos' / 'bindnode24']
    S.BN24 = next((p for p in S.BN24_CANDIDATES if (p / '.git').exists()), None)
    if S.BN24 is None:
        S.BN24 = Path(str(S.CACHE_ROOT / 'bindnode24_small_v1'))
        if not S.BN24.exists():
            subprocess.run(['git', 'clone', '-q', 'https://github.com/erckert/bindNode24.git', str(S.BN24)], check=True)
    S.fold_root = S.BN24 / 'dataset' / 'cv_splits' / 'predicted_structures'
    S.OFFICIAL_FOLDS = {}
    S.official_order = []
    S.seen = set()
    for S.k in range(1, 6):
        S.ids = [x.strip() for x in (S.fold_root / f'ids_split{S.k}.txt').read_text().splitlines() if x.strip()]
        if S.seen.intersection(S.ids):
            raise RuntimeError(f'Fold overlap at fold {S.k}')
        S.OFFICIAL_FOLDS[S.k] = S.ids
        S.official_order.extend(S.ids)
        S.seen.update(S.ids)
    S.train_lookup = S.raw_train.set_index('prot_id', drop=False)
    S.missing = set(S.official_order) - set(S.raw_train.prot_id)
    if S.missing:
        raise RuntimeError(f'Official Dev IDs missing from project CSV: {sorted(S.missing)[:20]}')
    S.train_df = S.train_lookup.loc[S.official_order].reset_index(drop=True)
    assert len(S.train_df) == 1010
    assert len(S.test300_df) == 300
    assert len(S.test46_df) == 46
    S.SPLIT_DF = {'DevSet1010': S.train_df, 'TestSet300': S.test300_df, 'TestSetNew46': S.test46_df}
    S.STORE_KEY = {'DevSet1010': 'train', 'TestSet300': 'test300', 'TestSetNew46': 'test46'}
    for S.split, S.df in S.SPLIT_DF.items():
        S.store_ids = set(map(str, S.ctx.stores[S.STORE_KEY[S.split]].keys()))
        S.csv_ids = set(S.df.prot_id.astype(str))
        if S.store_ids != S.csv_ids:
            raise RuntimeError(f'{S.split}: store/CSV mismatch. missing_in_store={sorted(S.csv_ids - S.store_ids)[:10]} extra_in_store={sorted(S.store_ids - S.csv_ids)[:10]}')
    print('Exact dataset/fold alignment: PASS')
    print('Fold sizes:', {k: len(v) for k, v in S.OFFICIAL_FOLDS.items()})

# 4. STORE HELPERS + OPTIONAL EVO / ESMIF
def stage_07():
    S.EVO_AVAILABLE = False
    if S.ATTACH_EVO_IF_CACHED:
        try:
            S.coverage = S.evo.attach_evo(S.ctx.stores, S.BASE, splits=('train', 'test300', 'test46'), require_msa=True)
            display(S.coverage)
            S.EVO_AVAILABLE = True
            print('EVO48 attached as optional control.')
        except Exception as e:
            print('EVO unavailable/incomplete -> skip:', repr(e))
    S.ESMIF_AVAILABLE = False
    S.ESMIF_CANDIDATE_ROOTS = [S.CACHE_ROOT / 'gear_metal_v3' / 'esmif_v1', S.CACHE_ROOT / 'nuclear_ultimate_v1' / 'esmif_v1']
    if S.ATTACH_ESMIF_IF_CACHED:
        for S.root in S.ESMIF_CANDIDATE_ROOTS:
            S.split_dirs = {'DevSet1010': S.root / 'DevSet1010' / 'esm_if', 'TestSet300': S.root / 'TestSet300' / 'esm_if', 'TestSetNew46': S.root / 'TestSetNew46' / 'esm_if'}
            if not all((d.exists() for d in S.split_dirs.values())):
                continue
            S.good = True
            for S.split, S.df in S.SPLIT_DF.items():
                S.d = S.split_dirs[S.split]
                for S.r in S.df.itertuples():
                    S.fp = S.d / f'{S.r.prot_id}.npy'
                    if not S.fp.exists():
                        S.good = False
                        break
                    S.a = np.load(S.fp, mmap_mode='r')
                    if S.a.shape != (len(S.r.sequence), 512):
                        S.good = False
                        break
                if not S.good:
                    break
            if S.good:
                for S.split, S.df in S.SPLIT_DF.items():
                    S.store = S.ctx.stores[S.STORE_KEY[S.split]]
                    S.d = S.split_dirs[S.split]
                    for S.r in S.df.itertuples():
                        set_block(S.store, S.r.prot_id, 'ESMIF', np.load(S.d / f'{S.r.prot_id}.npy').astype(np.float32))
                S.ESMIF_AVAILABLE = True
                print('Reused complete ESMIF cache:', S.root)
                break
    print('EVO_AVAILABLE  =', S.EVO_AVAILABLE)
    print('ESMIF_AVAILABLE=', S.ESMIF_AVAILABLE)

# 6. SMALL-BINDING-SPECIFIC FEATURE DEFINITIONS
def stage_12():
    S.FEATURE_CACHE = S.CACHE_ROOT / 'small_explicit_v1'
    S.FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    S.HYDROPATHY = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}
    S.GROUPS = {'AROM': set('FWYH'), 'STRONG_AROM': set('FWY'), 'SULFUR': set('CM'), 'HYDRO': set('AVILMFWY'), 'ALIPHATIC': set('AVILM'), 'POLAR': set('NQSTY'), 'ACIDIC': set('DE'), 'BASIC': set('KRH'), 'GP': set('GP'), 'HBD': set('KRHNQSTYW'), 'HBA': set('DEHNQSTYC'), 'AMIDE': set('NQ'), 'HYDROXYL': set('STY')}
    S.CHARGE = {'K': 1.0, 'R': 1.0, 'H': 0.1, 'D': -1.0, 'E': -1.0}
    S.BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'OXT'}
    S.RING_ATOMS = {'PHE': ['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2'], 'TYR': ['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2'], 'HIS': ['CG', 'ND1', 'CE1', 'NE2', 'CD2'], 'TRP': ['CG', 'CD1', 'NE1', 'CE2', 'CD2', 'CE3', 'CZ3', 'CH2', 'CZ2']}

# 7. GENERATE / CACHE / ATTACH SMALL-SPECIFIC BLOCKS
def stage_13():
    S.NEW_BLOCKS = {'SM_SEQCHEM32': 32, 'SM_POCKET3D44': 44, 'SM_LIGCHEM20': 20, 'SM_TOPO16': 16, 'SM_AROMSTACK12': 12, 'SM_PROTCTX8': 8}
    if S.RUN_FEATURE_GENERATION:
        for S.split, S.df in S.SPLIT_DF.items():
            S.outdir = S.FEATURE_CACHE / S.split
            S.outdir.mkdir(parents=True, exist_ok=True)
            for S.n, S.r in enumerate(S.df.itertuples(), start=1):
                S.fp = S.outdir / f'{S.r.prot_id}.npz'
                S.valid = False
                if S.fp.exists() and (not S.FORCE_REBUILD_SM_FEATURES):
                    try:
                        S.z = np.load(S.fp)
                        S.valid = all((S.z[k].shape == (len(S.r.sequence), d) for k, d in S.NEW_BLOCKS.items()))
                    except Exception:
                        S.valid = False
                if not S.valid:
                    S.seqb = sequence_chem_block(S.r.sequence)
                    S.pocket, S.ligchem, S.topo, S.arom = build_small_3d_blocks(S.PDB_NORM_ROOT / S.split / f'{S.r.prot_id}.pdb', S.r.sequence)
                    S.prot = protein_context(S.r.sequence)
                    np.savez_compressed(S.fp, SM_SEQCHEM32=S.seqb, SM_POCKET3D44=S.pocket, SM_LIGCHEM20=S.ligchem, SM_TOPO16=S.topo, SM_AROMSTACK12=S.arom, SM_PROTCTX8=S.prot)
                if S.n % 250 == 0:
                    print(S.split, S.n, '/', len(S.df))
    for S.split, S.df in S.SPLIT_DF.items():
        S.store = S.ctx.stores[S.STORE_KEY[S.split]]
        S.outdir = S.FEATURE_CACHE / S.split
        for S.r in S.df.itertuples():
            S.z = np.load(S.outdir / f'{S.r.prot_id}.npz')
            for S.k, S.d in S.NEW_BLOCKS.items():
                S.a = S.z[S.k].astype(np.float32, copy=False)
                if S.a.shape != (len(S.r.sequence), S.d):
                    raise RuntimeError(f'{S.split}/{S.r.prot_id}/{S.k}: {S.a.shape}')
                set_block(S.store, S.r.prot_id, S.k, S.a)
    print('New small-binding blocks attached.')
    display(S.reporting.dimension_table(S.ctx.stores['train']))
    S.audit_blocks = ['ESM', 'ACCESS', 'ATOM', 'CHEM', 'TOPO', 'SOLV', 'SPATIAL', 'LOCAL', 'SEQMOTIF', *S.NEW_BLOCKS.keys()]
    if S.EVO_AVAILABLE:
        S.audit_blocks.append('EVO')
    if S.ESMIF_AVAILABLE:
        S.audit_blocks.append('ESMIF')
    display(S.reporting.validate_no_direct_label_features(S.ctx.stores, S.audit_blocks))

STAGES = [stage_05, stage_07, stage_12, stage_13]

