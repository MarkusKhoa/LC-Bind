"""Extracted from NUCLEAR_ULTIMATE_V1_DATA_DRIVEN_PATCH-4.ipynb. Section: v2."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Chain, Counter, DataLoader, Dataset, F, Model, NuclearPatchHypergraph, NuclearPatchHypergraphMS, NuclearPatchV2, PDBIO, PDBParser, PatchProteinDataset, Path, Structure, Subset, V2PatchDataset, V2ResidueEncoder, accuracy_score, ast, average_precision_score, balanced_accuracy_score, best_mcc_threshold, binary_metrics, cKDTree, deepcopy, defaultdict, f1_score, get_block, hashlib, json, math, matthews_corrcoef, nn, np, os, pd, platform, precision_score, predict_patch, predict_patch_external, recall_score, residue_sequence, roc_auc_score, run_patch_cv, seq1, set_block, shutil, standard_residues, subprocess, sys, tarfile, torch
    from pathlib import Path
    from collections import defaultdict, Counter
    import json, math, numpy as np, pandas as pd
    from scipy.spatial import cKDTree
    import json, numpy as np, pandas as pd, torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, matthews_corrcoef, precision_score, recall_score
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
    import os, sys, json, ast, math, shutil, subprocess, platform
    from collections import Counter, defaultdict
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, Subset
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
    from lc_bind.utils.cache import get_block, set_block
    from lc_bind.utils.metrics import best_mcc_threshold, binary_metrics
    from lc_bind.utils.pdb import residue_sequence, standard_residues
    from lc_bind.models.nuclear.lock import predict_patch_external
    from lc_bind.models.nuclear.patch import NuclearPatchHypergraph, PatchProteinDataset, predict_patch, run_patch_cv

    class V2PatchDataset(Dataset):
    
        def __init__(self, split, df):
            self.split = split
            self.df = df.reset_index(drop=True)
            self.store = S.ctx.stores[S.STORE_KEY[split]]
    
        def __len__(self):
            return len(self.df)
    
        def __getitem__(self, i):
            r = self.df.iloc[i]
            pid = str(r.prot_id)
            mods = {m: v2_concat_blocks(self.store, pid, blocks).astype(np.float32, copy=False) for m, blocks in S.V2_MODALITY_SPEC.items()}
            er, ep, pf = S.V2_PATCH_RAM[self.split, pid]
            return {'pid': pid, 'mods': mods, 'y': np.asarray(r.nuclear_binding, dtype=np.float32), 'edge_res': er, 'edge_patch': ep, 'patch_feat': pf}

    class V2ResidueEncoder(nn.Module):
    
        def __init__(self, dims, latent=192, heads=6, drop=0.2):
            super().__init__()
            assert latent % heads == 0
            self.keys = sorted(dims)
            self.enc = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, latent), nn.GELU(), nn.Dropout(drop)) for k, d in dims.items()})
            self.attn = nn.MultiheadAttention(latent, heads, dropout=drop, batch_first=True)
    
        def forward(self, x):
            z = torch.stack([self.enc[k](x[k]) for k in self.keys], dim=1)
            a, _ = self.attn(z, z, z, need_weights=False)
            return (z + a).mean(dim=1)

    class NuclearPatchV2(nn.Module):
    
        def __init__(self, dims, latent=192, heads=6, drop=0.2, use_patch=True):
            super().__init__()
            self.use_patch = use_patch
            self.latent = latent
            self.residue_encoder = V2ResidueEncoder(dims, latent=latent, heads=heads, drop=drop)
            self.patch_static = nn.Sequential(nn.LayerNorm(24), nn.Linear(24, 64), nn.GELU())
            self.patch_encoder = nn.Sequential(nn.Linear(latent * 2 + 64, latent), nn.GELU(), nn.Dropout(drop), nn.Linear(latent, latent), nn.GELU())
            self.q = nn.Linear(latent, latent, bias=False)
            self.k = nn.Linear(latent, latent, bias=False)
            self.v = nn.Linear(latent, latent, bias=False)
            self.delta = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, latent), nn.GELU(), nn.Dropout(drop), nn.Linear(latent, latent))
            self.gate = nn.Sequential(nn.LayerNorm(latent * 2), nn.Linear(latent * 2, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, -1.5)
            self.head = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))
    
        def index_mean(self, x, index, n):
            out = x.new_zeros((n, x.shape[-1]))
            cnt = x.new_zeros((n, 1))
            if index.numel():
                out = out.index_add(0, index, x)
                cnt = cnt.index_add(0, index, x.new_ones((len(index), 1)))
            return out / cnt.clamp_min(1.0)
    
        def index_max(self, x, index, n):
            if n == 0:
                return x.new_zeros((0, x.shape[-1]))
            base = x.new_full((n, x.shape[-1]), -1000000000.0)
            if index.numel():
                idx = index[:, None].expand(-1, x.shape[-1])
                out = base.scatter_reduce(0, idx, x, reduce='amax', include_self=True)
            else:
                out = base
            return torch.where(out < -100000000.0, torch.zeros_like(out), out)
    
        def forward(self, mods, edge_res, edge_patch, patch_feat, return_aux=False):
            h = self.residue_encoder(mods)
            gate_mean = h.new_tensor(0.0)
            if self.use_patch and patch_feat.shape[0] > 0 and (edge_res.numel() > 0):
                npatch = patch_feat.shape[0]
                nres = h.shape[0]
                mean = self.index_mean(h[edge_res], edge_patch, npatch)
                mx = self.index_max(h[edge_res], edge_patch, npatch)
                stat = self.patch_static(patch_feat)
                hp = self.patch_encoder(torch.cat([mean, mx, stat], dim=-1))
                q = self.q(h[edge_res])
                k = self.k(hp[edge_patch])
                edge_logits = (q * k).sum(dim=-1) / math.sqrt(self.latent)
                base = edge_logits.new_full((nres,), -1000000000.0)
                max_r = base.scatter_reduce(0, edge_res, edge_logits, reduce='amax', include_self=True)
                max_r = torch.where(max_r < -100000000.0, torch.zeros_like(max_r), max_r)
                ew = torch.exp(edge_logits - max_r[edge_res])
                denom = ew.new_zeros((nres, 1)).index_add(0, edge_res, ew[:, None])
                weighted = self.v(hp[edge_patch]) * ew[:, None]
                msg = h.new_zeros((nres, self.latent)).index_add(0, edge_res, weighted)
                msg = msg / denom.clamp_min(1e-08)
                g = torch.sigmoid(self.gate(torch.cat([h, msg], dim=-1)))
                incident = h.new_zeros((nres, 1)).index_add(0, edge_res, h.new_ones((len(edge_res), 1)))
                g = g * (incident > 0).float()
                h = h + g * self.delta(msg)
                active = incident.squeeze(-1) > 0
                gate_mean = g[active].mean() if active.any() else h.new_tensor(0.0)
            logits = self.head(h).squeeze(-1)
            if return_aux:
                return (logits, {'gate_mean': gate_mean})
            return logits

    class NuclearPatchHypergraphMS(NuclearPatchHypergraph):
    
        def __init__(self, dims, mode='attention', latent=192, drop=0.2):
            super().__init__(dims, mode=mode, latent=latent, moe=False, drop=drop)
            self.patch_static = nn.Sequential(nn.LayerNorm(24), nn.Linear(24, 64), nn.GELU())

def v2_rep_coord(res):
    side = [np.asarray(a.coord, dtype=np.float32) for a in res if a.element != 'H' and a.name not in S.V2_BACKBONE]
    if side:
        return np.mean(np.stack(side), axis=0)
    if 'CA' in res:
        return np.asarray(res['CA'].coord, dtype=np.float32)
    atoms = [np.asarray(a.coord, dtype=np.float32) for a in res]
    return np.mean(np.stack(atoms), axis=0)

def v2_coords(split, pid, seq):
    p = S.PDB_NORM_ROOT / split / f'{pid}.pdb'
    st = PDBParser(QUIET=True).get_structure('v2', str(p))
    m = next(st.get_models())
    rr = standard_residues(m['A'])
    if residue_sequence(rr) != seq:
        raise RuntimeError(f'{split}/{pid}: PDB/sequence mismatch')
    return np.stack([v2_rep_coord(r) for r in rr], axis=0).astype(np.float32)

def v2_exposure_proxy(coords):
    """
    Label-free geometric exposure proxy.

    Fewer residues within 10 Å = more surface exposed.
    Returns [0,1], ranked within each protein.
    """
    tree = cKDTree(coords)
    n10 = np.array([max(len(tree.query_ball_point(coords[i], 10.0)) - 1, 0) for i in range(len(coords))], dtype=np.float32)
    if len(coords) <= 1:
        return (np.ones(len(coords), dtype=np.float32), n10)
    order = np.argsort(np.argsort(n10))
    dense_rank = order.astype(np.float32) / (len(coords) - 1)
    exposure = 1.0 - dense_rank
    return (exposure.astype(np.float32), n10)

def v2_component_sizes(indices, coords, threshold=8.0):
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
        for a, b in cKDTree(cc).query_pairs(threshold):
            union(a, b)
    groups = defaultdict(list)
    for j, rid in enumerate(idx):
        groups[find(j)].append(int(rid))
    out = {}
    for g in groups.values():
        for rid in g:
            out[rid] = len(g)
    return out

def build_na_surftopo16(split, pid, seq):
    coords = v2_coords(split, pid, seq)
    tree = cKDTree(coords)
    exposure, n10 = v2_exposure_proxy(coords)
    L = len(seq)
    q = np.array([S.V2_CHARGE.get(a, 0.0) for a in seq], dtype=np.float32)
    surface_mask = exposure >= 0.5
    surf_basic_idx = [i for i, a in enumerate(seq) if surface_mask[i] and a in S.V2_KRH]
    comp = v2_component_sizes(surf_basic_idx, coords[surf_basic_idx] if surf_basic_idx else np.zeros((0, 3), np.float32), threshold=8.0)
    out = np.zeros((L, 16), dtype=np.float32)
    for i in range(L):
        ids8 = np.asarray(tree.query_ball_point(coords[i], 8.0), dtype=int)
        ids8 = ids8[ids8 != i]
        surf8 = ids8[surface_mask[ids8]] if len(ids8) else np.array([], dtype=int)
        out[i, 0] = exposure[i]
        out[i, 1] = min(float(n10[i]) / 40.0, 1.0)
        out[i, 2] = len(surf8) / max(len(ids8), 1)
        if len(surf8):
            aa = [seq[j] for j in surf8]
            qq = q[surf8]
            dd = np.linalg.norm(coords[surf8] - coords[i][None, :], axis=1)
            out[i, 3] = sum((a in S.V2_KRH for a in aa)) / len(aa)
            out[i, 4] = sum((a in S.V2_KR for a in aa)) / len(aa)
            out[i, 5] = sum((a in S.V2_POL for a in aa)) / len(aa)
            out[i, 6] = sum((a in S.V2_DE for a in aa)) / len(aa)
            out[i, 7] = float(np.mean(qq))
            out[i, 8] = float(np.tanh(np.sum(qq / (dd + 1.0))))
            out[i, 12] = float(np.std(qq))
            out[i, 13] = float(np.tanh((sum((a in S.V2_KRH for a in aa)) - sum((a in S.V2_DE for a in aa))) / 5.0))
            out[i, 14] = sum((a in S.V2_HBD for a in aa)) / len(aa)
            out[i, 15] = float(np.mean(np.abs(surf8 - i) > 10))
        basic = [j for j in surf_basic_idx if j != i]
        if basic:
            d = np.linalg.norm(coords[basic] - coords[i][None, :], axis=1)
            out[i, 9] = min(float(d.min()) / 20.0, 1.0)
        else:
            out[i, 9] = 1.0
        out[i, 10] = min(comp.get(i, 0), 20) / 20.0
        ids12 = np.asarray(tree.query_ball_point(coords[i], 12.0), dtype=int)
        pos = [j for j in ids12 if surface_mask[j] and seq[j] in S.V2_KRH]
        if pos:
            cent = coords[pos].mean(axis=0)
            out[i, 11] = min(float(np.linalg.norm(coords[i] - cent)) / 20.0, 1.0)
        else:
            out[i, 11] = 1.0
    return out

def v2_modality_spec(blocks):
    spec = {}
    if 'ESM' in blocks:
        spec['ESM'] = ['ESM']
    surface = [b for b in ['ACCESS', 'NA_SURFTOPO16'] if b in blocks]
    if surface:
        spec['SURFACE'] = surface
    patch = [b for b in ['NA_PATCH3D40', 'NA_TOPO12'] if b in blocks]
    if patch:
        spec['PATCH_LOCAL'] = patch
    electro = [b for b in ['ELEC3D', 'NA_SEQELECTRO32'] if b in blocks]
    if electro:
        spec['ELECTRO'] = electro
    chem = [b for b in ['NA_HBOND16'] if b in blocks]
    if chem:
        spec['CHEM'] = chem
    assigned = {x for v in spec.values() for x in v}
    other = [b for b in blocks if b not in assigned]
    if other:
        spec['OTHER'] = other
    return spec

def v2_concat_blocks(store, pid, blocks):
    xs = [get_block(store, pid, b).astype(np.float32, copy=False) for b in blocks]
    return np.concatenate(xs, axis=1)

def v2_patch_feature(seq, coords, exposure, members, radius):
    members = np.asarray(members, dtype=int)
    aa = [seq[j] for j in members]
    cc = coords[members]
    qq = np.asarray([S.V2_CHARGE.get(a, 0.0) for a in aa], dtype=np.float32)
    n = len(members)
    cent = cc.mean(axis=0)
    radial = np.linalg.norm(cc - cent[None, :], axis=1)
    pair = []
    seqsep = []
    for x in range(n):
        for y in range(x + 1, n):
            pair.append(float(np.linalg.norm(cc[x] - cc[y])))
            seqsep.append(abs(int(members[x]) - int(members[y])))
    pair = np.asarray(pair if pair else [0.0], dtype=np.float32)
    seqsep = np.asarray(seqsep if seqsep else [0.0], dtype=np.float32)
    comp = Counter(aa)
    probs = np.asarray([c / n for c in comp.values()], dtype=np.float32)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / math.log(20))
    f = np.zeros(24, dtype=np.float32)
    f[0] = radius / 12.0
    f[1] = min(n, 40) / 40.0
    f[2] = float(np.mean(exposure[members]))
    f[3] = float(np.min(exposure[members]))
    f[4] = sum((a in S.V2_KR for a in aa)) / n
    f[5] = sum((a in S.V2_KRH for a in aa)) / n
    f[6] = sum((a in S.V2_POL for a in aa)) / n
    f[7] = sum((a in S.V2_DE for a in aa)) / n
    f[8] = sum((a in S.V2_HBD for a in aa)) / n
    f[9] = float(np.mean(qq))
    f[10] = float(np.std(qq))
    f[11] = float(np.tanh((sum((a in S.V2_KRH for a in aa)) - sum((a in S.V2_DE for a in aa))) / 5.0))
    f[12] = min(float(radial.mean()) / 12.0, 1.0)
    f[13] = min(float(radial.std()) / 8.0, 1.0)
    f[14] = min(float(radial.max()) / 15.0, 1.0)
    f[15] = min(float(pair.mean()) / 15.0, 1.0)
    f[16] = min(float(pair.std()) / 10.0, 1.0)
    f[17] = float(np.mean(seqsep > 10))
    f[18] = min(float(seqsep.mean()) / 50.0, 1.0)
    f[19] = entropy
    if n >= 3:
        z = cc - cc.mean(axis=0, keepdims=True)
        cov = z.T @ z / max(n - 1, 1)
        eig = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
        total = float(eig.sum() + 1e-08)
        f[20] = np.clip((eig[0] - eig[1]) / (eig[0] + 1e-08), 0, 1)
        f[21] = np.clip((eig[1] - eig[2]) / (eig[0] + 1e-08), 0, 1)
        f[22] = np.clip(eig[2] / total, 0, 1)
    f[23] = float(np.mean(np.abs(members - int(np.median(members))) > 10))
    return f

def v2_jaccard(a, b):
    a = set(a)
    b = set(b)
    return len(a & b) / max(len(a | b), 1)

def build_v2_patch_graph(split, pid, seq):
    coords = v2_coords(split, pid, seq)
    tree = cKDTree(coords)
    exposure, _ = v2_exposure_proxy(coords)
    candidates = []
    for seed, a in enumerate(seq):
        if a in S.V2_KRH:
            if exposure[seed] < 0.35:
                continue
        elif a in S.V2_POL:
            if exposure[seed] < 0.65:
                continue
        else:
            continue
        for radius in S.V2_RADII:
            raw = tree.query_ball_point(coords[seed], radius)
            members = tuple(sorted((j for j in raw if exposure[j] >= 0.25)))
            if len(members) < 4:
                continue
            if not any((seq[j] in S.V2_KRH for j in members)):
                continue
            feat = v2_patch_feature(seq, coords, exposure, members, radius)
            score = 2.0 * feat[5] + 0.7 * feat[6] - 0.7 * feat[7] + 1.2 * feat[2] + 0.25 * feat[17] - 0.1 * feat[0]
            candidates.append((float(score), members, feat))
    unique = {}
    for score, members, feat in candidates:
        if members not in unique or score > unique[members][0]:
            unique[members] = (score, feat)
    ranked = sorted([(s, m, f) for m, (s, f) in unique.items()], key=lambda x: x[0], reverse=True)
    selected = []
    for score, members, feat in ranked:
        if any((v2_jaccard(members, m2) >= S.V2_JACCARD_NMS for _, m2, _ in selected)):
            continue
        selected.append((score, members, feat))
        if len(selected) >= S.V2_PATCH_MAX:
            break
    edge_res = []
    edge_patch = []
    patch_feat = []
    for pidx, (_, members, feat) in enumerate(selected):
        patch_feat.append(feat)
        for rid in members:
            edge_res.append(int(rid))
            edge_patch.append(int(pidx))
    return (np.asarray(edge_res, dtype=np.int64), np.asarray(edge_patch, dtype=np.int64), np.asarray(patch_feat, dtype=np.float32).reshape(-1, 24))

def v2_collate(batch):
    keys = sorted(batch[0]['mods'])
    mods = {k: torch.from_numpy(np.concatenate([b['mods'][k] for b in batch], axis=0)).float() for k in keys}
    y = torch.from_numpy(np.concatenate([b['y'] for b in batch])).float()
    er_all = []
    ep_all = []
    pf_all = []
    protein_index = []
    roff = 0
    poff = 0
    for pidx, b in enumerate(batch):
        L = len(b['y'])
        P = len(b['patch_feat'])
        protein_index.append(torch.full((L,), pidx, dtype=torch.long))
        if P:
            pf_all.append(torch.from_numpy(b['patch_feat']).float())
        if len(b['edge_res']):
            er_all.append(torch.from_numpy(b['edge_res']).long() + roff)
            ep_all.append(torch.from_numpy(b['edge_patch']).long() + poff)
        roff += L
        poff += P
    patch_feat = torch.cat(pf_all, 0) if pf_all else torch.zeros((0, 24), dtype=torch.float32)
    edge_res = torch.cat(er_all, 0) if er_all else torch.zeros((0,), dtype=torch.long)
    edge_patch = torch.cat(ep_all, 0) if ep_all else torch.zeros((0,), dtype=torch.long)
    return {'mods': mods, 'y': y, 'protein_index': torch.cat(protein_index, 0), 'edge_res': edge_res, 'edge_patch': edge_patch, 'patch_feat': patch_feat}

def v2_move(b):
    return {'mods': {k: v.to(S.DEVICE, non_blocking=True) for k, v in b['mods'].items()}, 'y': b['y'].to(S.DEVICE, non_blocking=True), 'protein_index': b['protein_index'].to(S.DEVICE, non_blocking=True), 'edge_res': b['edge_res'].to(S.DEVICE, non_blocking=True), 'edge_patch': b['edge_patch'].to(S.DEVICE, non_blocking=True), 'patch_feat': b['patch_feat'].to(S.DEVICE, non_blocking=True)}

def v2_loss(logits, y, protein_index, pos_weight, use_hardneg):
    raw = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction='none')
    nprot = int(protein_index.max().item()) + 1
    sums = raw.new_zeros((nprot,)).index_add(0, protein_index, raw)
    counts = raw.new_zeros((nprot,)).index_add(0, protein_index, raw.new_ones(len(raw)))
    base = (sums / counts.clamp_min(1.0)).mean()
    if not use_hardneg:
        return (base, raw.new_tensor(0.0))
    rank_terms = []
    for p in range(nprot):
        mask = protein_index == p
        lp = logits[mask]
        yp = y[mask] > 0.5
        if not yp.any() or (~yp).sum() == 0:
            continue
        pos_logits = lp[yp]
        neg_logits = lp[~yp]
        k = min(max(4, int(2 * len(pos_logits))), 32, len(neg_logits))
        hard_neg = torch.topk(neg_logits, k=k).values
        pos_anchor = pos_logits.mean()
        rank_terms.append(F.softplus(hard_neg - pos_anchor + S.V2_HARDNEG_MARGIN).mean())
    rank = torch.stack(rank_terms).mean() if rank_terms else base.new_tensor(0.0)
    return (base + S.V2_HARDNEG_LAMBDA * rank, rank)

def v2_predict(model, ds):
    loader = DataLoader(ds, batch_size=S.V2_BATCH, shuffle=False, num_workers=0, pin_memory=S.DEVICE.type == 'cuda', collate_fn=v2_collate)
    pp = []
    yy = []
    gates = []
    model.eval()
    with torch.no_grad():
        for b in loader:
            b = v2_move(b)
            logits, aux = model(b['mods'], b['edge_res'], b['edge_patch'], b['patch_feat'], return_aux=True)
            pp.append(torch.sigmoid(logits).cpu().numpy())
            yy.append(b['y'].cpu().numpy())
            gates.append(float(aux['gate_mean'].detach().cpu()))
    return (np.concatenate(pp), np.concatenate(yy), float(np.mean(gates)))

def v2_train_fold(variant, tr_ds, va_ds, fold):
    cfg = S.V2_VARIANTS[variant]
    torch.manual_seed(S.SEED + 700 + fold)
    np.random.seed(S.SEED + 700 + fold)
    model = NuclearPatchV2(S.V2_DIMS, latent=192, heads=6, drop=0.2, use_patch=cfg['use_patch']).to(S.DEVICE)
    ytr = np.concatenate(tr_ds.df.nuclear_binding.apply(lambda x: np.asarray(x, dtype=np.float32)).tolist())
    npos = max(float(ytr.sum()), 1.0)
    nneg = max(float(len(ytr) - ytr.sum()), 1.0)
    pos_weight = torch.tensor(np.sqrt(nneg / npos), dtype=torch.float32, device=S.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=S.V2_LR, weight_decay=S.V2_WD)
    loader = DataLoader(tr_ds, batch_size=S.V2_BATCH, shuffle=True, num_workers=0, pin_memory=S.DEVICE.type == 'cuda', collate_fn=v2_collate)
    best_ap = -1.0
    best_state = None
    bad = 0
    for epoch in range(1, S.V2_EPOCHS + 1):
        model.train()
        ls = []
        rs = []
        for b in loader:
            b = v2_move(b)
            opt.zero_grad(set_to_none=True)
            logits = model(b['mods'], b['edge_res'], b['edge_patch'], b['patch_feat'])
            loss, rank = v2_loss(logits, b['y'], b['protein_index'], pos_weight, cfg['hardneg'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ls.append(float(loss.detach().cpu()))
            rs.append(float(rank.detach().cpu()))
        vp, vy, vg = v2_predict(model, va_ds)
        vap = float(average_precision_score(vy, vp))
        if vap > best_ap + 1e-05:
            best_ap = vap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch == 1 or epoch % 5 == 0:
            print(variant, 'fold', fold, 'epoch', epoch, 'loss', round(float(np.mean(ls)), 4), 'rank', round(float(np.mean(rs)), 4), 'val_AP', round(vap, 5), 'gate', round(vg, 4))
        if bad >= S.V2_PATIENCE:
            break
    model.load_state_dict(best_state)
    return (model, best_state)

def v2_run_cv(variant):
    offsets = {}
    cursor = 0
    for r in S.train_df.itertuples():
        L = len(r.sequence)
        offsets[str(r.prot_id)] = (cursor, cursor + L)
        cursor += L
    oof = np.full(cursor, np.nan, dtype=np.float32)
    rows = []
    states = {}
    adir = S.V2_STATE_ROOT / variant
    adir.mkdir(parents=True, exist_ok=True)
    for fold, val_ids in S.OFFICIAL_FOLDS.items():
        val_set = set(val_ids)
        tr_df = S.train_df[~S.train_df.prot_id.isin(val_set)].copy()
        va_df = S.train_df[S.train_df.prot_id.isin(val_set)].copy()
        tr_ds = V2PatchDataset('DevSet1010', tr_df)
        va_ds = V2PatchDataset('DevSet1010', va_df)
        state_fp = adir / f'fold{fold}.pt'
        pred_fp = adir / f'fold{fold}_val.npy'
        meta_fp = adir / f'fold{fold}.json'
        expected = int(va_df.sequence.str.len().sum())
        reuse = False
        if state_fp.exists() and pred_fp.exists() and meta_fp.exists():
            try:
                meta = json.loads(meta_fp.read_text())
                pred = np.load(pred_fp)
                reuse = meta.get('code_version') == S.V2_CODE_VERSION and meta.get('representation') == S.V2_REP_NAME and (meta.get('n_val') == expected) and (pred.shape == (expected,)) and np.all(np.isfinite(pred))
            except Exception:
                reuse = False
        if reuse:
            print(variant, 'fold', fold, 'REUSE')
            state = torch.load(state_fp, map_location='cpu')
            pred = np.load(pred_fp).astype(np.float32)
            model = NuclearPatchV2(S.V2_DIMS, use_patch=S.V2_VARIANTS[variant]['use_patch']).to(S.DEVICE)
            model.load_state_dict(state)
            _, _, gate = v2_predict(model, va_ds)
            del model
        else:
            print(variant, 'fold', fold, 'TRAIN')
            model, state = v2_train_fold(variant, tr_ds, va_ds, fold)
            pred, _, gate = v2_predict(model, va_ds)
            torch.save(state, state_fp)
            np.save(pred_fp, pred.astype(np.float32))
            meta_fp.write_text(json.dumps({'code_version': S.V2_CODE_VERSION, 'representation': S.V2_REP_NAME, 'blocks': S.V2_BLOCKS, 'n_val': expected}, indent=2))
            del model
            if S.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        states[fold] = state
        pos = 0
        for r in va_df.itertuples():
            L = len(r.sequence)
            a, b = offsets[str(r.prot_id)]
            oof[a:b] = pred[pos:pos + L]
            pos += L
        yv = np.concatenate(va_df.nuclear_binding.apply(lambda x: np.asarray(x, dtype=np.float32)).tolist())
        rows.append({'variant': variant, 'fold': fold, 'AUPRC': float(average_precision_score(yv, pred)), 'n_positive': int(yv.sum()), 'gate_mean': gate})
    f = pd.DataFrame(rows)
    yall = np.concatenate(S.train_df.nuclear_binding.apply(lambda x: np.asarray(x, dtype=np.float32)).tolist())
    return ({'variant': variant, 'CV_AUPRC_mean': float(f.AUPRC.mean()), 'CV_AUPRC_std': float(f.AUPRC.std(ddof=1)), 'OOF_AUPRC': float(average_precision_score(yall, oof)), 'gate_mean': float(f.gate_mean.mean())}, f, oof, states)

def v2score(name):
    return float(S.v2_summary.loc[S.v2_summary.variant.eq(name), 'CV_AUPRC_mean'].iloc[0])

def v2_best_mcc_threshold(y, p):
    best = (-999, None)
    for t in np.linspace(0.01, 0.99, 199):
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > best[0]:
            best = (m, float(t))
    return best[1]

def v2_predict_external(split, variant):
    ds = V2PatchDataset(split, S.SPLIT_DF[split])
    preds = []
    for fold, state in S.V2_STATES[variant].items():
        model = NuclearPatchV2(S.V2_DIMS, latent=192, heads=6, drop=0.2, use_patch=S.V2_VARIANTS[variant]['use_patch']).to(S.DEVICE)
        model.load_state_dict(state)
        p, _, _ = v2_predict(model, ds)
        preds.append(p)
        del model
        if S.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    return np.mean(np.stack(preds), axis=0)

def v2_metrics(y, p, t=0.5):
    pred = (p >= t).astype(int)
    return {'AUPRC': float(average_precision_score(y, p)), 'AUROC': float(roc_auc_score(y, p)), 'F1': float(f1_score(y, pred, zero_division=0)), 'MCC': float(matthews_corrcoef(y, pred)), 'precision': float(precision_score(y, pred, zero_division=0)), 'recall': float(recall_score(y, pred, zero_division=0))}

def collate_patch_ms(batch):
    keys = sorted(batch[0]['mods'])
    mods = {k: torch.from_numpy(np.concatenate([b['mods'][k] for b in batch], axis=0)).float() for k in keys}
    y = torch.from_numpy(np.concatenate([b['y'] for b in batch])).float()
    patch_feat = []
    edge_res = []
    edge_patch = []
    roff = 0
    poff = 0
    for b in batch:
        L = len(b['y'])
        P = len(b['patch_feat'])
        if P:
            patch_feat.append(torch.from_numpy(b['patch_feat']).float())
        if len(b['edge_res']):
            edge_res.append(torch.from_numpy(b['edge_res']).long() + roff)
            edge_patch.append(torch.from_numpy(b['edge_patch']).long() + poff)
        roff += L
        poff += P
    patch_feat = torch.cat(patch_feat, dim=0) if patch_feat else torch.zeros((0, 24), dtype=torch.float32)
    edge_res = torch.cat(edge_res, dim=0) if edge_res else torch.zeros((0,), dtype=torch.long)
    edge_patch = torch.cat(edge_patch, dim=0) if edge_patch else torch.zeros((0,), dtype=torch.long)
    return {'mods': mods, 'y': y, 'edge_res': edge_res, 'edge_patch': edge_patch, 'patch_feat': patch_feat}

def _metric(y, p, t):
    pred = (p >= t).astype(int)
    return {'AUPRC': float(average_precision_score(y, p)), 'AUROC': float(roc_auc_score(y, p)), 'precision': float(precision_score(y, pred, zero_division=0)), 'recall': float(recall_score(y, pred, zero_division=0)), 'F1': float(f1_score(y, pred, zero_division=0)), 'MCC': float(matthews_corrcoef(y, pred))}

def predict_ms_again(split):
    old_ram = S.PATCH_RAM
    old_factory = S.PATCH_FACTORY
    old_state_root = S.PATCH_STATE_ROOT
    old_collate = S.collate_patch
    S.PATCH_RAM = S.V2_PATCH_RAM
    S.PATCH_FACTORY = {'NA_PatchHG_Attention_MS': lambda dims: NuclearPatchHypergraphMS(dims, mode='attention', latent=S.PATCH_LATENT, drop=0.2)}
    S.collate_patch = collate_patch_ms
    try:
        ds = PatchProteinDataset(split, S.SPLIT_DF[split])
        preds = []
        for fold, state in S.ms_states.items():
            model = S.PATCH_FACTORY['NA_PatchHG_Attention_MS'](S.DIMS).to(S.DEVICE)
            model.load_state_dict(state)
            p, _ = predict_patch(model, ds)
            preds.append(p)
            del model
            if S.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        return np.mean(np.stack(preds), axis=0)
    finally:
        S.PATCH_RAM = old_ram
        S.PATCH_FACTORY = old_factory
        S.PATCH_STATE_ROOT = old_state_root
        S.collate_patch = old_collate

def best_mcc_thr(y, p):
    best = (-999, 0.5)
    for t in np.linspace(0.01, 0.99, 199):
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > best[0]:
            best = (m, float(t))
    return (best[1], best[0])

def predict_v1(split):
    ds = PatchProteinDataset(split, S.SPLIT_DF[split])
    pred = []
    for fold, state in S.PATCH_STATES[S.V1_NAME].items():
        model = S.PATCH_FACTORY[S.V1_NAME](S.DIMS).to(S.DEVICE)
        model.load_state_dict(state)
        p, _ = predict_patch(model, ds)
        pred.append(p)
        del model
        if S.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    return np.mean(np.stack(pred), axis=0)

def predict_ms(split):
    old_ram, old_fac, old_col = (S.PATCH_RAM, S.PATCH_FACTORY, S.collate_patch)
    try:
        S.PATCH_RAM = S.V2_PATCH_RAM
        S.PATCH_FACTORY = {'NA_PatchHG_Attention_MS': lambda dims: NuclearPatchHypergraphMS(dims, mode='attention', latent=S.PATCH_LATENT, drop=0.2)}
        S.collate_patch = collate_patch_ms
        ds = PatchProteinDataset(split, S.SPLIT_DF[split])
        pred = []
        for fold, state in S.ms_states.items():
            model = S.PATCH_FACTORY['NA_PatchHG_Attention_MS'](S.DIMS).to(S.DEVICE)
            model.load_state_dict(state)
            p, _ = predict_patch(model, ds)
            pred.append(p)
            del model
            if S.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        return np.mean(np.stack(pred), axis=0)
    finally:
        S.PATCH_RAM, S.PATCH_FACTORY, S.collate_patch = (old_ram, old_fac, old_col)

def global_metrics(y, p, t=0.5):
    pr = (p >= t).astype(int)
    return {'AUPRC': float(average_precision_score(y, p)), 'AUROC': float(roc_auc_score(y, p)), 'precision': float(precision_score(y, pr, zero_division=0)), 'recall': float(recall_score(y, pr, zero_division=0)), 'F1': float(f1_score(y, pr, zero_division=0)), 'MCC': float(matthews_corrcoef(y, pr))}

def protein_metrics(df, p, t=0.5):
    f1s, mccs = ([], [])
    pos = 0
    for r in df.itertuples():
        L = len(r.sequence)
        y = np.asarray(r.nuclear_binding, dtype=int)
        pr = (p[pos:pos + L] >= t).astype(int)
        f1s.append(f1_score(y, pr, zero_division=0))
        mccs.append(matthews_corrcoef(y, pr))
        pos += L
    return {'protein_mean_F1': float(np.mean(f1s)), 'protein_mean_MCC': float(np.mean(mccs))}

def _counts(y, pred):
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return (tp, tn, fp, fn)

def _mcc_counts(tp, tn, fp, fn):
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return 0.0 if den == 0 else float((tp * tn - fp * fn) / den)

def _safe_div(a, b):
    return 0.0 if b == 0 else float(a / b)

def full_global_metrics(y, p, threshold):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= float(threshold)).astype(np.int8)
    tp, tn, fp, fn = _counts(y, pred)
    precision_pos = _safe_div(tp, tp + fp)
    recall_pos = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    npv = _safe_div(tn, tn + fn)
    f1_pos = _safe_div(2 * tp, 2 * tp + fp + fn)
    micro_f1 = float(f1_score(y, pred, average='micro', zero_division=0))
    macro_f1 = float(f1_score(y, pred, average='macro', zero_division=0))
    return {'n_residues': int(len(y)), 'n_positive': int(np.sum(y == 1)), 'prevalence': float(np.mean(y)), 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'AUPRC_global_residue': float(average_precision_score(y, p)), 'AUROC_global_residue': float(roc_auc_score(y, p)), 'Accuracy_global_residue': float(accuracy_score(y, pred)), 'Balanced_accuracy_global_residue': float(balanced_accuracy_score(y, pred)), 'Precision_positive_global_residue': precision_pos, 'Recall_positive_global_residue': recall_pos, 'Sensitivity_global_residue': recall_pos, 'Specificity_global_residue': specificity, 'NPV_global_residue': npv, 'F1_positive_global_residue': f1_pos, 'MCC_global_residue': _mcc_counts(tp, tn, fp, fn), 'F1_micro_all_classes': micro_f1, 'F1_macro_two_classes': macro_f1}

def protein_mean_metrics_baseline_convention(df, p, threshold):
    """
    Same aggregation convention used by the reproduced direct baseline:
    calculate each protein separately and INCLUDE it iff TP+FP+FN > 0.
    Therefore completely TN-only proteins are excluded.
    """
    p = np.asarray(p, dtype=np.float64)
    pos = 0
    f1s, mccs = ([], [])
    n_total = 0
    n_included = 0
    n_excluded_tn_only = 0
    for r in df.itertuples():
        L = len(r.sequence)
        y = np.asarray(getattr(r, S.LABEL_COL), dtype=np.int8)
        pred = (p[pos:pos + L] >= float(threshold)).astype(np.int8)
        pos += L
        n_total += 1
        tp, tn, fp, fn = _counts(y, pred)
        if tp + fp + fn == 0:
            n_excluded_tn_only += 1
            continue
        n_included += 1
        f1s.append(_safe_div(2 * tp, 2 * tp + fp + fn))
        mccs.append(_mcc_counts(tp, tn, fp, fn))
    assert pos == len(p)
    return {'protein_mean_F1_baseline_convention': float(np.mean(f1s)) if f1s else np.nan, 'protein_mean_MCC_baseline_convention': float(np.mean(mccs)) if mccs else np.nan, 'protein_n_total': int(n_total), 'protein_n_included_baseline_convention': int(n_included), 'protein_n_excluded_TN_only': int(n_excluded_tn_only)}

def optimize_thresholds_exact(y, p):
    """
    Exact search at unique OOF probability cutoffs for pred = (p >= threshold).
    Returns thresholds optimized ONLY on Dev OOF predictions.
    """
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(-p, kind='mergesort')
    ys = y[order]
    ps = p[order]
    tp_c = np.cumsum(ys == 1)
    fp_c = np.cumsum(ys == 0)
    ends = np.r_[np.where(ps[:-1] != ps[1:])[0], len(ps) - 1]
    tp = tp_c[ends].astype(np.float64)
    fp = fp_c[ends].astype(np.float64)
    P = float(np.sum(y == 1))
    N = float(np.sum(y == 0))
    fn = P - tp
    tn = N - fp
    thr = ps[ends]
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros_like(tp), where=den > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=2 * tp + fp + fn > 0)
    tpr = np.divide(tp, P, out=np.zeros_like(tp), where=P > 0)
    tnr = np.divide(tn, N, out=np.zeros_like(tn), where=N > 0)
    youden = tpr + tnr - 1.0
    i_mcc = int(np.nanargmax(mcc))
    i_f1 = int(np.nanargmax(f1))
    i_y = int(np.nanargmax(youden))
    return {'fixed_0.5': 0.5, 'dev_oof_mcc_opt': float(thr[i_mcc]), 'dev_oof_positive_f1_opt': float(thr[i_f1]), 'dev_oof_youden_j_opt': float(thr[i_y])}

def _load_cached(method, split, expected_len):
    fp = S.PRED_CACHE / f'{method}__{split}.npy'
    if fp.exists():
        try:
            p = np.load(fp)
            if p.shape == (expected_len,) and np.all(np.isfinite(p)):
                return p.astype(np.float64)
        except Exception:
            pass
    return None

def _save_cached(method, split, p):
    np.save(S.PRED_CACHE / f'{method}__{split}.npy', np.asarray(p, dtype=np.float32))

def predict_v1_external(split):
    expected = int(sum(S.SPLIT_DF[split].sequence.str.len()))
    cached = _load_cached('V1_Attention', split, expected)
    if cached is not None:
        return cached
    ds = PatchProteinDataset(split, S.SPLIT_DF[split])
    preds = []
    for fold, state in S.PATCH_STATES[S.V1_NAME].items():
        model = S.PATCH_FACTORY[S.V1_NAME](S.DIMS).to(S.DEVICE)
        model.load_state_dict(state)
        p, _ = predict_patch(model, ds)
        preds.append(p)
        del model
        if S.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    p = np.mean(np.stack(preds), axis=0)
    _save_cached('V1_Attention', split, p)
    return p.astype(np.float64)

def predict_ms_external(split):
    expected = int(sum(S.SPLIT_DF[split].sequence.str.len()))
    cached = _load_cached('MS_Attention', split, expected)
    if cached is not None:
        return cached
    old_ram = S.PATCH_RAM
    old_factory = S.PATCH_FACTORY
    old_collate = S.collate_patch
    try:
        S.PATCH_RAM = S.V2_PATCH_RAM
        S.PATCH_FACTORY = {S.MS_NAME: lambda dims: NuclearPatchHypergraphMS(dims, mode='attention', latent=S.PATCH_LATENT, drop=0.2)}
        S.collate_patch = collate_patch_ms
        ds = PatchProteinDataset(split, S.SPLIT_DF[split])
        preds = []
        for fold, state in S.ms_states.items():
            model = S.PATCH_FACTORY[S.MS_NAME](S.DIMS).to(S.DEVICE)
            model.load_state_dict(state)
            p, _ = predict_patch(model, ds)
            preds.append(p)
            del model
            if S.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        p = np.mean(np.stack(preds), axis=0)
        _save_cached('MS_Attention', split, p)
        return p.astype(np.float64)
    finally:
        S.PATCH_RAM = old_ram
        S.PATCH_FACTORY = old_factory
        S.collate_patch = old_collate

def _first_existing(row, candidates, default=np.nan):
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return float(row[c])
    return default

# NUCLEAR V2-A. Surface-topology feature block (Dev-selected)
def stage_38():
    S.V2_RUN_ROOT = S.RESULTS_ROOT / 'NUCLEAR_V2_SELECTIVE_MULTISCALE'
    S.V2_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    S.V2_CACHE_ROOT = S.CACHE_ROOT / 'nuclear_v2_selective_multiscale'
    S.V2_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    S.V2_FORCE_FEATURES = False
    S.V2_CHARGE = {'K': 1.0, 'R': 1.0, 'H': 0.1, 'D': -1.0, 'E': -1.0}
    S.V2_KR = set('KR')
    S.V2_KRH = set('KRH')
    S.V2_DE = set('DE')
    S.V2_POL = set('NQSTY')
    S.V2_HBD = set('KRHNQSTYW')
    S.V2_BACKBONE = {'N', 'CA', 'C', 'O', 'OXT'}
    S.V2_SURF_CACHE = S.V2_CACHE_ROOT / 'surftopo16'
    for S.split, S.df in S.SPLIT_DF.items():
        S.od = S.V2_SURF_CACHE / S.split
        S.od.mkdir(parents=True, exist_ok=True)
        S.store = S.ctx.stores[S.STORE_KEY[S.split]]
        for S.n, S.r in enumerate(S.df.itertuples(), start=1):
            S.fp = S.od / f'{S.r.prot_id}.npy'
            S.good = False
            if S.fp.exists() and (not S.V2_FORCE_FEATURES):
                try:
                    S.a = np.load(S.fp, mmap_mode='r')
                    S.good = S.a.shape == (len(S.r.sequence), 16)
                except Exception:
                    S.good = False
            if not S.good:
                S.a = build_na_surftopo16(S.split, str(S.r.prot_id), S.r.sequence)
                np.save(S.fp, S.a.astype(np.float32))
            else:
                S.a = np.load(S.fp).astype(np.float32)
            set_block(S.store, S.r.prot_id, 'NA_SURFTOPO16', S.a)
            if S.n % 250 == 0:
                print(S.split, S.n, '/', len(S.df))
    print('NA_SURFTOPO16 attached: PASS')

# NUCLEAR V2-B. Dev-only representation re-lock
def stage_39():
    S.V2_REP_GRID = {'V2_ESM': ['ESM'], 'V2_V1_PATCHORG': ['ESM', 'ACCESS', 'NA_PATCH3D40', 'NA_TOPO12'], 'V2_ESM_SURFTOPO': ['ESM', 'NA_SURFTOPO16'], 'V2_PATCHORG_SURFTOPO': ['ESM', 'ACCESS', 'NA_PATCH3D40', 'NA_TOPO12', 'NA_SURFTOPO16'], 'V2_PATCHORG_SURFTOPO_ELEC': ['ESM', 'ACCESS', 'ELEC3D', 'NA_PATCH3D40', 'NA_TOPO12', 'NA_SURFTOPO16']}
    S.v2_rep_cv, S.v2_rep_folds = S.suite.run_dev_cv_only(S.ctx, S.V2_REP_GRID, models=('MLP',), run_version='nuclear_v2_01_surftopo_rep')
    S.v2_rep_cv.to_csv(S.V2_RUN_ROOT / 'representation_cv.csv', index=False)
    S.v2_rep_folds.to_csv(S.V2_RUN_ROOT / 'representation_folds.csv', index=False)
    display(S.v2_rep_cv.sort_values('CV_AUPRC', ascending=False))
    S.V2_REP_NAME = str(S.v2_rep_cv.sort_values('CV_AUPRC', ascending=False).iloc[0].representation)
    S.V2_BLOCKS = list(S.V2_REP_GRID[S.V2_REP_NAME])
    S.V2_REP_SCORE = float(S.v2_rep_cv.loc[S.v2_rep_cv.representation.eq(S.V2_REP_NAME), 'CV_AUPRC'].iloc[0])
    print('V2 LOCKED REPRESENTATION:', S.V2_REP_NAME)
    print('V2 BLOCKS:', S.V2_BLOCKS)
    print('V2 Dev CV AUPRC:', S.V2_REP_SCORE)
    S.V2_MODALITY_SPEC = v2_modality_spec(S.V2_BLOCKS)
    S.V2_DIMS = {}
    for S.m, S.blocks in S.V2_MODALITY_SPEC.items():
        S.sample_pid = str(S.train_df.iloc[0].prot_id)
        S.V2_DIMS[S.m] = v2_concat_blocks(S.ctx.stores['train'], S.sample_pid, S.blocks).shape[1]
    print('V2 modalities:', S.V2_MODALITY_SPEC)
    print('V2 dims:', S.V2_DIMS)
    (S.V2_RUN_ROOT / 'locked_representation.json').write_text(json.dumps({'representation': S.V2_REP_NAME, 'blocks': S.V2_BLOCKS, 'modalities': S.V2_MODALITY_SPEC, 'dev_cv_auprc': S.V2_REP_SCORE}, indent=2))

# NUCLEAR V2-C. Selective multi-scale surface patches
def stage_40():
    S.V2_RADII = (6.0, 8.0, 10.0, 12.0)
    S.V2_PATCH_MAX = 48
    S.V2_JACCARD_NMS = 0.65
    S.V2_FORCE_PATCH = False
    S.V2_PATCH_CACHE = S.V2_CACHE_ROOT / 'selective_multiscale_patches'
    S.V2_PATCH_CACHE.mkdir(parents=True, exist_ok=True)
    S.V2_PATCH_RAM = {}
    S.v2_audit = []
    for S.split, S.df in S.SPLIT_DF.items():
        S.od = S.V2_PATCH_CACHE / S.split
        S.od.mkdir(parents=True, exist_ok=True)
        for S.n, S.r in enumerate(S.df.itertuples(), start=1):
            S.fp = S.od / f'{S.r.prot_id}.npz'
            S.good = False
            if S.fp.exists() and (not S.V2_FORCE_PATCH):
                try:
                    S.z = np.load(S.fp)
                    S.good = S.z['patch_feat'].ndim == 2 and S.z['patch_feat'].shape[1] == 24 and (len(S.z['edge_res']) == len(S.z['edge_patch']))
                except Exception:
                    S.good = False
            if not S.good:
                S.er, S.ep, S.pf = build_v2_patch_graph(S.split, str(S.r.prot_id), S.r.sequence)
                np.savez_compressed(S.fp, edge_res=S.er, edge_patch=S.ep, patch_feat=S.pf)
            S.z = np.load(S.fp)
            S.er = S.z['edge_res'].astype(np.int64)
            S.ep = S.z['edge_patch'].astype(np.int64)
            S.pf = S.z['patch_feat'].astype(np.float32)
            S.V2_PATCH_RAM[S.split, str(S.r.prot_id)] = (S.er, S.ep, S.pf)
            S.covered = len(np.unique(S.er)) if len(S.er) else 0
            S.v2_audit.append({'dataset': S.split, 'prot_id': str(S.r.prot_id), 'n_residues': len(S.r.sequence), 'n_patches': len(S.pf), 'n_memberships': len(S.er), 'coverage': S.covered / max(len(S.r.sequence), 1)})
            if S.n % 250 == 0:
                print(S.split, S.n, '/', len(S.df))
    S.v2_patch_audit = pd.DataFrame(S.v2_audit)
    S.v2_patch_audit.to_csv(S.V2_RUN_ROOT / 'patch_audit.csv', index=False)
    display(S.v2_patch_audit.groupby('dataset').agg(mean_patches=('n_patches', 'mean'), median_patches=('n_patches', 'median'), mean_coverage=('coverage', 'mean'), median_coverage=('coverage', 'median'), mean_memberships=('n_memberships', 'mean')).reset_index())

# NUCLEAR V2-E.
def stage_42():
    S.V2_BATCH = 4
    S.V2_EPOCHS = 36
    S.V2_PATIENCE = 6
    S.V2_LR = 0.0003
    S.V2_WD = 0.0001
    S.V2_HARDNEG_LAMBDA = 0.2
    S.V2_HARDNEG_MARGIN = 0.5
    S.V2_CODE_VERSION = 'nuclear_v2_selective_multiscale_r1'
    S.V2_STATE_ROOT = S.V2_RUN_ROOT / 'states'
    S.V2_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    S.V2_VARIANTS = {'NoPatch_BCE': {'use_patch': False, 'hardneg': False}, 'NoPatch_HardNeg': {'use_patch': False, 'hardneg': True}, 'Patch_BCE': {'use_patch': True, 'hardneg': False}, 'Patch_HardNeg': {'use_patch': True, 'hardneg': True}}
    S.V2_RESULTS = {}
    S.V2_FOLDS = []
    S.V2_OOF = {}
    S.V2_STATES = {}
    for S.variant in S.V2_VARIANTS:
        print('\n' + '=' * 88)
        print('V2:', S.variant)
        print('=' * 88)
        S.s, S.f, S.o, S.st = v2_run_cv(S.variant)
        S.V2_RESULTS[S.variant] = S.s
        S.V2_FOLDS.append(S.f)
        S.V2_OOF[S.variant] = S.o
        S.V2_STATES[S.variant] = S.st
    S.v2_summary = pd.DataFrame(S.V2_RESULTS.values()).sort_values('CV_AUPRC_mean', ascending=False).reset_index(drop=True)
    S.v2_folds = pd.concat(S.V2_FOLDS, ignore_index=True)
    S.v2_summary.to_csv(S.V2_RUN_ROOT / 'v2_cv_summary.csv', index=False)
    S.v2_folds.to_csv(S.V2_RUN_ROOT / 'v2_cv_folds.csv', index=False)
    display(S.v2_summary)
    S.v2_ablation = pd.DataFrame([{'effect': 'Hard-negative loss without patches', 'delta_AUPRC': v2score('NoPatch_HardNeg') - v2score('NoPatch_BCE')}, {'effect': 'Patch reasoning under BCE', 'delta_AUPRC': v2score('Patch_BCE') - v2score('NoPatch_BCE')}, {'effect': 'Patch reasoning under HardNeg', 'delta_AUPRC': v2score('Patch_HardNeg') - v2score('NoPatch_HardNeg')}, {'effect': 'Hard-negative loss with patches', 'delta_AUPRC': v2score('Patch_HardNeg') - v2score('Patch_BCE')}])
    display(S.v2_ablation)
    S.v2_ablation.to_csv(S.V2_RUN_ROOT / 'matched_ablation.csv', index=False)
    S.complexity = {'NoPatch_BCE': 0, 'NoPatch_HardNeg': 1, 'Patch_BCE': 2, 'Patch_HardNeg': 3}
    S.ordered = sorted(S.V2_VARIANTS, key=lambda x: S.complexity[x])
    S.winner = S.ordered[0]
    for S.name in S.ordered[1:]:
        if v2score(S.name) - v2score(S.winner) >= 0.002:
            S.winner = S.name
    S.V2_LOCKED = S.winner
    S.V2_LOCKED_SCORE = v2score(S.winner)
    print('V2 LOCKED:', S.V2_LOCKED, S.V2_LOCKED_SCORE)

# NUCLEAR V2-F.
def stage_43():
    S.v2_dev_y = np.concatenate(S.train_df.nuclear_binding.apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
    S.V2_THR = v2_best_mcc_threshold(S.v2_dev_y, S.V2_OOF[S.V2_LOCKED])
    S.ext_rows = []
    for S.split in ['TestSet300', 'TestSetNew46']:
        S.y = np.concatenate(S.SPLIT_DF[S.split].nuclear_binding.apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
        S.p = v2_predict_external(S.split, S.V2_LOCKED)
        S.m05 = v2_metrics(S.y, S.p, 0.5)
        S.mdev = v2_metrics(S.y, S.p, S.V2_THR)
        S.row = {'dataset': S.split, 'variant': S.V2_LOCKED, 'Dev_CV_AUPRC': S.V2_LOCKED_SCORE, 'Dev_OOF_MCC_threshold': S.V2_THR}
        S.row.update({f'{k}_fixed0.5': v for k, v in S.m05.items()})
        S.row.update({f'{k}_OOFthr': v for k, v in S.mdev.items()})
        S.ext_rows.append(S.row)
        pd.DataFrame({'prot_id': np.concatenate([np.repeat(str(r.prot_id), len(r.sequence)) for r in S.SPLIT_DF[S.split].itertuples()]), 'residue_index': np.concatenate([np.arange(len(r.sequence)) for r in S.SPLIT_DF[S.split].itertuples()]), 'y_true': S.y, 'probability': S.p}).to_csv(S.V2_RUN_ROOT / f'predictions_{S.split}.csv.gz', index=False)
    S.v2_external = pd.DataFrame(S.ext_rows)
    display(S.v2_external)
    S.v2_external.to_csv(S.V2_RUN_ROOT / 'FINAL_V2_EXTERNAL.csv', index=False)
    S.comparison_rows = []
    for S._, S.r in S.v2_external.iterrows():
        S.comparison_rows.append({'dataset': S.r['dataset'], 'method': f'V2 {S.V2_LOCKED}', 'AUPRC': S.r['AUPRC_fixed0.5'], 'AUROC': S.r['AUROC_fixed0.5'], 'F1': S.r['F1_fixed0.5'], 'MCC': S.r['MCC_fixed0.5']})
    if 'headline' in globals() and len(S.headline):
        for S._, S.r in S.headline.iterrows():
            S.comparison_rows.append({'dataset': S.r['dataset'], 'method': f'V1 {S.GLOBAL_MODEL}', 'AUPRC': S.r['AUPRC_fixed0.5'], 'AUROC': S.r['AUROC_fixed0.5'], 'F1': S.r['F1_fixed0.5'], 'MCC': S.r['MCC_fixed0.5']})
    S.baseline_fp = S.BASELINE_WORK / 'results' / 'FULL_DIRECT_BASELINE_METRICS_300_46.csv'
    if S.baseline_fp.exists():
        S.bb = pd.read_csv(S.baseline_fp)
        if 'label' in S.bb.columns:
            S.bb = S.bb[S.bb.label.eq('nuclear_binding')].copy()
        if 'operating_point' in S.bb.columns:
            S.bb = S.bb[S.bb.operating_point.eq('fixed_0.5')].copy()
        for S._, S.r in S.bb.iterrows():
            if str(S.r.method) not in {'bindNode24', 'bindEmbed21DL'}:
                continue
            S.apcol = 'AP_AUPRC' if 'AP_AUPRC' in S.bb.columns else 'AUPRC'
            S.comparison_rows.append({'dataset': str(S.r.dataset), 'method': str(S.r.method), 'AUPRC': float(S.r[S.apcol]), 'AUROC': float(S.r.AUROC), 'F1': float(S.r.global_positive_F1), 'MCC': float(S.r.global_MCC)})
    S.v2_compare = pd.DataFrame(S.comparison_rows)
    display(S.v2_compare.sort_values(['dataset', 'AUPRC'], ascending=[True, False]))
    S.v2_compare.to_csv(S.V2_RUN_ROOT / 'FINAL_V2_COMPARISON.csv', index=False)
    print('\nV2 RESULT DIRECTORY:')
    print(S.V2_RUN_ROOT)

# NUCLEAR FINAL EXPERIMENT
def stage_44():
    assert 'V2_PATCH_RAM' in globals(), 'Run Nuclear V2-C first to create V2_PATCH_RAM.'
    assert 'NuclearPatchHypergraph' in globals()
    assert 'PatchProteinDataset' in globals()
    assert 'run_patch_cv' in globals()
    S.NUC_MS_ROOT = S.RESULTS_ROOT / 'NUCLEAR_V1_ATTENTION_V2_MULTISCALE_PATCH'
    S.NUC_MS_ROOT.mkdir(parents=True, exist_ok=True)
    S._OLD_PATCH_RAM = S.PATCH_RAM
    S._OLD_PATCH_FACTORY = S.PATCH_FACTORY
    S._OLD_PATCH_STATE_ROOT = S.PATCH_STATE_ROOT
    S._OLD_COLLATE_PATCH = S.collate_patch
    S.PATCH_RAM = S.V2_PATCH_RAM
    S.PATCH_FACTORY = {'NA_PatchHG_Attention_MS': lambda dims: NuclearPatchHypergraphMS(dims, mode='attention', latent=S.PATCH_LATENT, drop=0.2)}
    S.PATCH_STATE_ROOT = S.NUC_MS_ROOT / 'states'
    S.PATCH_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    S.collate_patch = collate_patch_ms
    try:
        print('=' * 100)
        print('NUCLEAR: V1 ATTENTION + V2 SELECTIVE MULTI-SCALE PATCHES')
        print('=' * 100)
        S.ms_summary, S.ms_folds, S.ms_oof, S.ms_states = run_patch_cv('NA_PatchHG_Attention_MS')
        S.ms_summary_df = pd.DataFrame([S.ms_summary])
        display(S.ms_summary_df)
        display(S.ms_folds)
        S.ms_summary_df.to_csv(S.NUC_MS_ROOT / 'dev_cv_summary.csv', index=False)
        S.ms_folds.to_csv(S.NUC_MS_ROOT / 'dev_cv_folds.csv', index=False)
        S.MS_DEV = float(S.ms_summary['CV_AUPRC_mean'])
        S.V1_DEV = float(S.patch_summary.loc[S.patch_summary.architecture.eq('NA_PatchHG_Attention'), 'CV_AUPRC_mean'].iloc[0])
        print(f'\nV1 Attention Dev mean AUPRC : {S.V1_DEV:.6f}')
        print(f'MS Attention Dev mean AUPRC : {S.MS_DEV:.6f}')
        print(f'Delta MS - V1               : {S.MS_DEV - S.V1_DEV:+.6f}')
        S.MS_WINS = S.MS_DEV - S.V1_DEV >= 0.002
        print('MS becomes new headline?     :', S.MS_WINS)
        S.dev_y = np.concatenate(S.train_df.nuclear_binding.apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
        S.ms_thr, S.ms_oof_mcc = best_mcc_threshold(S.dev_y, S.ms_oof)
        print('Frozen Dev threshold:', S.ms_thr)
        print('Dev OOF MCC:', S.ms_oof_mcc)

        def _predict_ms_external(split):
            ds = PatchProteinDataset(split, S.SPLIT_DF[split])
            preds = []
            for fold, state in S.ms_states.items():
                model = S.PATCH_FACTORY['NA_PatchHG_Attention_MS'](S.DIMS).to(S.DEVICE)
                model.load_state_dict(state)
                p, _ = predict_patch(model, ds)
                preds.append(p)
                del model
                if S.DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
            return np.mean(np.stack(preds), axis=0)
        S.ext_rows = []
        for S.split in ['TestSet300', 'TestSetNew46']:
            S.y = np.concatenate(S.SPLIT_DF[S.split].nuclear_binding.apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
            S.p = _predict_ms_external(S.split)
            S.m05 = binary_metrics(S.y, S.p, 0.5)
            S.mdev = binary_metrics(S.y, S.p, S.ms_thr)
            S.row = {'dataset': S.split, 'method': 'NA_PatchHG_Attention_MS', 'Dev_CV_AUPRC': S.MS_DEV, 'Dev_OOF_MCC_threshold': S.ms_thr}
            S.row.update({f'{k}_fixed0.5': v for k, v in S.m05.items()})
            S.row.update({f'{k}_OOFthr': v for k, v in S.mdev.items()})
            S.ext_rows.append(S.row)
        S.ms_external = pd.DataFrame(S.ext_rows)
        display(S.ms_external)
        S.ms_external.to_csv(S.NUC_MS_ROOT / 'external_metrics.csv', index=False)
        S.cmp_rows = []
        for S._, S.r in S.ms_external.iterrows():
            S.cmp_rows.append({'dataset': S.r['dataset'], 'method': 'MS selective PatchHG Attention', 'AUPRC': S.r['AUPRC_fixed0.5'], 'AUROC': S.r['AUROC_fixed0.5'], 'F1': S.r['F1_fixed0.5'], 'MCC': S.r['MCC_fixed0.5']})
        for S._, S.r in S.headline.iterrows():
            S.cmp_rows.append({'dataset': S.r['dataset'], 'method': f'V1 {S.GLOBAL_MODEL}', 'AUPRC': S.r['AUPRC_fixed0.5'], 'AUROC': S.r['AUROC_fixed0.5'], 'F1': S.r['F1_fixed0.5'], 'MCC': S.r['MCC_fixed0.5']})
        S.baseline_fp = S.BASELINE_WORK / 'results' / 'FULL_DIRECT_BASELINE_METRICS_300_46.csv'
        if S.baseline_fp.exists():
            S.bb = pd.read_csv(S.baseline_fp)
            if 'label' in S.bb.columns:
                S.bb = S.bb[S.bb.label.eq('nuclear_binding')].copy()
            if 'operating_point' in S.bb.columns:
                S.bb = S.bb[S.bb.operating_point.eq('fixed_0.5')].copy()
            for S._, S.r in S.bb.iterrows():
                if str(S.r.method) not in {'bindNode24', 'bindEmbed21DL'}:
                    continue
                S.apcol = 'AP_AUPRC' if 'AP_AUPRC' in S.bb.columns else 'AUPRC'
                S.cmp_rows.append({'dataset': str(S.r.dataset), 'method': str(S.r.method), 'AUPRC': float(S.r[S.apcol]), 'AUROC': float(S.r.AUROC), 'F1': float(S.r.global_positive_F1), 'MCC': float(S.r.global_MCC)})
        S.ms_compare = pd.DataFrame(S.cmp_rows).sort_values(['dataset', 'AUPRC'], ascending=[True, False])
        print('\nFINAL NUCLEAR COMPARISON')
        display(S.ms_compare)
        S.ms_compare.to_csv(S.NUC_MS_ROOT / 'FINAL_COMPARISON.csv', index=False)
    finally:
        S.PATCH_RAM = S._OLD_PATCH_RAM
        S.PATCH_FACTORY = S._OLD_PATCH_FACTORY
        S.PATCH_STATE_ROOT = S._OLD_PATCH_STATE_ROOT
        S.collate_patch = S._OLD_COLLATE_PATCH
    print('\nSaved:', S.NUC_MS_ROOT)

# NUCLEAR FINAL DEV-LOCKED ENSEMBLE
def stage_45():
    S.ENS_ROOT = S.RESULTS_ROOT / 'NUCLEAR_FINAL_V1_MS_ENSEMBLE'
    S.ENS_ROOT.mkdir(parents=True, exist_ok=True)
    S.V1_OOF = np.asarray(S.PATCH_OOF['NA_PatchHG_Attention'], dtype=np.float32)
    S.MS_OOF = np.asarray(S.ms_oof, dtype=np.float32)
    assert S.V1_OOF.shape == S.MS_OOF.shape
    assert np.all(np.isfinite(S.V1_OOF))
    assert np.all(np.isfinite(S.MS_OOF))
    S.DEV_Y = np.concatenate(S.train_df.nuclear_binding.apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
    assert len(S.DEV_Y) == len(S.V1_OOF)
    S.offsets = {}
    S.cursor = 0
    for S.r in S.train_df.itertuples():
        S.L = len(S.r.sequence)
        S.offsets[str(S.r.prot_id)] = (S.cursor, S.cursor + S.L)
        S.cursor += S.L
    S.FOLD_RES_IDX = {}
    for S.fold, S.ids in S.OFFICIAL_FOLDS.items():
        S.idx = []
        for S.pid in S.ids:
            S.a, S.b = S.offsets[str(S.pid)]
            S.idx.extend(range(S.a, S.b))
        S.FOLD_RES_IDX[S.fold] = np.asarray(S.idx, dtype=np.int64)
    S.rows = []
    for S.w in np.linspace(0.0, 1.0, 21):
        S.pred = S.w * S.V1_OOF + (1.0 - S.w) * S.MS_OOF
        S.fold_ap = []
        for S.fold in sorted(S.FOLD_RES_IDX):
            S.idx = S.FOLD_RES_IDX[S.fold]
            S.fold_ap.append(float(average_precision_score(S.DEV_Y[S.idx], S.pred[S.idx])))
        S.rows.append({'w_V1': float(S.w), 'w_MS': float(1.0 - S.w), 'CV_AUPRC_mean': float(np.mean(S.fold_ap)), 'CV_AUPRC_std': float(np.std(S.fold_ap, ddof=1)), 'OOF_AUPRC': float(average_precision_score(S.DEV_Y, S.pred))})
    S.weight_search = pd.DataFrame(S.rows)
    display(S.weight_search.sort_values('CV_AUPRC_mean', ascending=False).head(10))
    S.weight_search.to_csv(S.ENS_ROOT / 'dev_weight_search.csv', index=False)
    S.best = S.weight_search.sort_values('CV_AUPRC_mean', ascending=False).iloc[0]
    S.ENS_W_V1 = float(S.best['w_V1'])
    S.ENS_W_MS = float(S.best['w_MS'])
    S.ENS_DEV = float(S.best['CV_AUPRC_mean'])
    S.V1_DEV = float(S.patch_summary.loc[S.patch_summary.architecture.eq('NA_PatchHG_Attention'), 'CV_AUPRC_mean'].iloc[0])
    print('\n' + '=' * 90)
    print('DEV-LOCKED ENSEMBLE')
    print('=' * 90)
    print('V1 weight:', S.ENS_W_V1)
    print('MS weight:', S.ENS_W_MS)
    print('V1 Dev mean AUPRC:', S.V1_DEV)
    print('Ensemble Dev mean AUPRC:', S.ENS_DEV)
    print('Delta:', S.ENS_DEV - S.V1_DEV)
    S.ENSEMBLE_PASS = S.ENS_DEV - S.V1_DEV >= 0.002
    print('Promote ensemble?', S.ENSEMBLE_PASS)
    S.ENS_OOF = S.ENS_W_V1 * S.V1_OOF + S.ENS_W_MS * S.MS_OOF
    S.ENS_THR, S.ENS_MCC = best_mcc_threshold(S.DEV_Y, S.ENS_OOF)
    print('Frozen Dev threshold:', S.ENS_THR)
    print('Dev OOF MCC:', S.ENS_MCC)
    S.external_rows = []
    for S.split in ['TestSet300', 'TestSetNew46']:
        S.y = np.concatenate(S.SPLIT_DF[S.split].nuclear_binding.apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
        S.p_v1 = predict_patch_external(S.split)
        S.p_ms = predict_ms_again(S.split)
        S.p_ens = S.ENS_W_V1 * S.p_v1 + S.ENS_W_MS * S.p_ms
        for S.name, S.p, S.thr in [('V1_Attention', S.p_v1, S.PATCH_THR), ('MS_Attention', S.p_ms, S.ms_thr), ('DEV_LOCKED_ENSEMBLE', S.p_ens, S.ENS_THR)]:
            S.m05 = _metric(S.y, S.p, 0.5)
            S.mdev = _metric(S.y, S.p, S.thr)
            S.row = {'dataset': S.split, 'method': S.name, 'AUPRC': S.m05['AUPRC'], 'AUROC': S.m05['AUROC'], 'precision_fixed0.5': S.m05['precision'], 'recall_fixed0.5': S.m05['recall'], 'F1_fixed0.5': S.m05['F1'], 'MCC_fixed0.5': S.m05['MCC'], 'frozen_threshold': S.thr, 'F1_OOFthr': S.mdev['F1'], 'MCC_OOFthr': S.mdev['MCC']}
            S.external_rows.append(S.row)
    S.ext = pd.DataFrame(S.external_rows)
    S.baseline_fp = S.BASELINE_WORK / 'results' / 'FULL_DIRECT_BASELINE_METRICS_300_46.csv'
    if S.baseline_fp.exists():
        S.bb = pd.read_csv(S.baseline_fp)
        S.bb = S.bb[S.bb.label.eq('nuclear_binding')].copy()
        if 'operating_point' in S.bb.columns:
            S.bb = S.bb[S.bb.operating_point.eq('fixed_0.5')]
        S.base_rows = []
        for S._, S.r in S.bb.iterrows():
            if str(S.r.method) not in {'bindNode24', 'bindEmbed21DL'}:
                continue
            S.apcol = 'AP_AUPRC' if 'AP_AUPRC' in S.bb.columns else 'AUPRC'
            S.base_rows.append({'dataset': str(S.r.dataset), 'method': str(S.r.method), 'AUPRC': float(S.r[S.apcol]), 'AUROC': float(S.r.AUROC), 'precision_fixed0.5': np.nan, 'recall_fixed0.5': np.nan, 'F1_fixed0.5': float(S.r.global_positive_F1), 'MCC_fixed0.5': float(S.r.global_MCC), 'frozen_threshold': 0.5, 'F1_OOFthr': np.nan, 'MCC_OOFthr': np.nan})
        S.ext = pd.concat([S.ext, pd.DataFrame(S.base_rows)], ignore_index=True)
    S.ext = S.ext.sort_values(['dataset', 'AUPRC'], ascending=[True, False])
    print('\nFINAL NUCLEAR ENSEMBLE COMPARISON')
    display(S.ext)
    S.ext.to_csv(S.ENS_ROOT / 'FINAL_COMPARISON.csv', index=False)
    S.summary = {'w_V1': S.ENS_W_V1, 'w_MS': S.ENS_W_MS, 'V1_Dev_AUPRC': S.V1_DEV, 'Ensemble_Dev_AUPRC': S.ENS_DEV, 'Dev_gain': S.ENS_DEV - S.V1_DEV, 'Promote': bool(S.ENSEMBLE_PASS)}
    (S.ENS_ROOT / 'FINAL_DECISION.json').write_text(json.dumps(S.summary, indent=2))
    print('\nDecision:', S.summary)
    print('\nSaved:', S.ENS_ROOT)

# NUCLEAR FINAL ONE-CELL
def stage_46():
    S.REQ = ['PATCH_OOF', 'PATCH_STATES', 'V2_PATCH_RAM', 'ms_oof', 'ms_states', 'NuclearPatchHypergraphMS', 'collate_patch_ms', 'PatchProteinDataset', 'predict_patch', 'PATCH_FACTORY', 'PATCH_RAM', 'collate_patch', 'train_df', 'SPLIT_DF', 'OFFICIAL_FOLDS', 'BASELINE_WORK']
    S.missing = [x for x in S.REQ if x not in globals()]
    if S.missing:
        raise RuntimeError('Run the completed Nuclear V1 + MS cells first. Missing: ' + ', '.join(S.missing))
    S.OUT = S.RESULTS_ROOT / 'NUCLEAR_FINAL_ONECELL'
    S.OUT.mkdir(parents=True, exist_ok=True)
    S.V1_NAME = 'NA_PatchHG_Attention'
    S.v1_oof = np.asarray(S.PATCH_OOF[S.V1_NAME], dtype=np.float32)
    S.ms_oof_ = np.asarray(S.ms_oof, dtype=np.float32)
    S.y_dev = np.concatenate(S.train_df.nuclear_binding.apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
    assert S.v1_oof.shape == S.ms_oof_.shape == S.y_dev.shape
    S.offsets, S.cur = ({}, 0)
    for S.r in S.train_df.itertuples():
        S.L = len(S.r.sequence)
        S.offsets[str(S.r.prot_id)] = (S.cur, S.cur + S.L)
        S.cur += S.L
    S.fold_idx = {}
    for S.fold, S.ids in S.OFFICIAL_FOLDS.items():
        S.z = []
        for S.pid in S.ids:
            S.a, S.b = S.offsets[str(S.pid)]
            S.z.extend(range(S.a, S.b))
        S.fold_idx[S.fold] = np.asarray(S.z, dtype=np.int64)
    S.rows = []
    for S.w in np.linspace(0, 1, 21):
        S.p = S.w * S.v1_oof + (1 - S.w) * S.ms_oof_
        S.aps = [average_precision_score(S.y_dev[idx], S.p[idx]) for _, idx in sorted(S.fold_idx.items())]
        S.rows.append({'w_V1': float(S.w), 'w_MS': float(1 - S.w), 'CV_AUPRC_mean': float(np.mean(S.aps)), 'CV_AUPRC_std': float(np.std(S.aps, ddof=1)), 'OOF_AUPRC': float(average_precision_score(S.y_dev, S.p))})
    S.weight_search = pd.DataFrame(S.rows).sort_values('CV_AUPRC_mean', ascending=False).reset_index(drop=True)
    display(S.weight_search.head(10))
    S.weight_search.to_csv(S.OUT / 'DEV_WEIGHT_SEARCH.csv', index=False)
    S.best = S.weight_search.iloc[0]
    S.w1, S.wm = (float(S.best.w_V1), float(S.best.w_MS))
    S.ens_dev = float(S.best.CV_AUPRC_mean)
    S.v1_dev = float(S.patch_summary.loc[S.patch_summary.architecture.eq(S.V1_NAME), 'CV_AUPRC_mean'].iloc[0])
    S.PROMOTE = S.ens_dev - S.v1_dev >= 0.002
    if not S.PROMOTE:
        S.w1, S.wm = (1.0, 0.0)
        S.ens_dev = S.v1_dev
    S.ens_oof = S.w1 * S.v1_oof + S.wm * S.ms_oof_
    S.ens_thr, S.ens_oof_mcc = best_mcc_thr(S.y_dev, S.ens_oof)
    print('\nDEV FINAL')
    print('w_V1 =', S.w1, 'w_MS =', S.wm)
    print('V1 Dev =', S.v1_dev)
    print('Final Dev =', S.ens_dev)
    print('Delta =', S.ens_dev - S.v1_dev)
    print('Promote ensemble =', S.PROMOTE)
    print('Frozen Dev threshold =', S.ens_thr)
    print('OOF MCC =', S.ens_oof_mcc)
    S.our_rows = []
    S.pred_cache = {}
    for S.split in ['TestSet300', 'TestSetNew46']:
        S.df = S.SPLIT_DF[S.split]
        S.y = np.concatenate(S.df.nuclear_binding.apply(lambda x: np.asarray(x, dtype=int)).tolist())
        S.pv1 = predict_v1(S.split)
        S.pms = predict_ms(S.split)
        S.pens = S.w1 * S.pv1 + S.wm * S.pms
        S.pred_cache[S.split] = S.pens
        for S.name, S.p, S.thr in [('V1_Attention', S.pv1, 0.5), ('MS_Attention', S.pms, 0.5), ('FINAL_DEV_LOCKED', S.pens, 0.5)]:
            S.g = global_metrics(S.y, S.p, S.thr)
            S.pm = protein_metrics(S.df, S.p, S.thr)
            S.our_rows.append({'dataset': S.split, 'method': S.name})
        S.pids = []
        S.ridx = []
        for S.r in S.df.itertuples():
            S.pids.extend([str(S.r.prot_id)] * len(S.r.sequence))
            S.ridx.extend(range(len(S.r.sequence)))
        pd.DataFrame({'prot_id': S.pids, 'residue_index': S.ridx, 'y_true': S.y, 'probability': S.pens}).to_csv(S.OUT / f'FINAL_predictions_{S.split}.csv.gz', index=False)
    S.ours = pd.DataFrame(S.our_rows)
    S.bfp = S.BASELINE_WORK / 'results' / 'FULL_DIRECT_BASELINE_METRICS_300_46.csv'
    S.bb = pd.read_csv(S.bfp)
    S.bb = S.bb[S.bb.label.eq('nuclear_binding')].copy()
    if 'operating_point' in S.bb.columns:
        S.bb = S.bb[S.bb.operating_point.eq('fixed_0.5')].copy()
    S.base_rows = []
    for S._, S.r in S.bb.iterrows():
        if str(S.r.method) not in {'bindNode24', 'bindEmbed21DL'}:
            continue
        S.apcol = 'AP_AUPRC' if 'AP_AUPRC' in S.bb.columns else 'AUPRC'
        S.base_rows.append({'dataset': str(S.r.dataset), 'method': str(S.r.method), 'AUPRC': float(S.r[S.apcol]), 'AUROC': float(S.r.AUROC), 'precision': np.nan, 'recall': np.nan, 'F1': float(S.r.global_positive_F1), 'MCC': float(S.r.global_MCC), 'protein_mean_F1': float(S.r.protein_mean_F1), 'protein_mean_MCC': float(S.r.protein_mean_MCC)})
    S.final = pd.concat([S.ours, pd.DataFrame(S.base_rows)], ignore_index=True).sort_values(['dataset', 'AUPRC'], ascending=[True, False])
    print('\nFINAL NUCLEAR — ALL AVAILABLE METRICS')
    display(S.final)
    S.final.to_csv(S.OUT / 'FINAL_ALL_METRICS.csv', index=False)
    S.decision = {'w_V1': S.w1, 'w_MS': S.wm, 'V1_Dev_mean_AUPRC': S.v1_dev, 'Final_Dev_mean_AUPRC': S.ens_dev, 'Dev_gain': S.ens_dev - S.v1_dev, 'Promoted': bool(S.PROMOTE), 'Frozen_Dev_MCC_threshold': S.ens_thr}
    (S.OUT / 'FINAL_DECISION.json').write_text(json.dumps(S.decision, indent=2))
    print('\nDecision:', S.decision)
    print('Saved:', S.OUT)

# NUCLEAR FINAL PAPER-READY REPORT — ONE CELL
def stage_47():
    S.TASK_NAME = 'NUCLEAR'
    S.LABEL_COL = 'nuclear_binding'
    S.V1_NAME = 'NA_PatchHG_Attention'
    S.MS_NAME = 'NA_PatchHG_Attention_MS'
    S.FINAL_NAME = 'FINAL_DEV_LOCKED_ENSEMBLE'
    S.MIN_DEV_GAIN = 0.002
    S.WEIGHT_GRID = np.linspace(0.0, 1.0, 21)
    S.OUT = S.RESULTS_ROOT / 'NUCLEAR_FINAL_PAPER_READY'
    S.OUT.mkdir(parents=True, exist_ok=True)
    S.REQ = ['PATCH_OOF', 'PATCH_STATES', 'V2_PATCH_RAM', 'ms_oof', 'ms_states', 'NuclearPatchHypergraphMS', 'collate_patch_ms', 'PatchProteinDataset', 'predict_patch', 'PATCH_FACTORY', 'PATCH_RAM', 'collate_patch', 'train_df', 'SPLIT_DF', 'OFFICIAL_FOLDS', 'BASELINE_WORK', 'patch_summary', 'DIMS', 'DEVICE', 'PATCH_LATENT']
    S.missing = [x for x in S.REQ if x not in globals()]
    if S.missing:
        raise RuntimeError('Run the completed V1 + multi-scale cells first. Missing: ' + ', '.join(S.missing))
    S.V1_OOF = np.asarray(S.PATCH_OOF[S.V1_NAME], dtype=np.float64)
    S.MS_OOF = np.asarray(S.ms_oof, dtype=np.float64)
    S.Y_DEV = np.concatenate(S.train_df[S.LABEL_COL].apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
    assert S.V1_OOF.shape == S.MS_OOF.shape == S.Y_DEV.shape
    assert np.all(np.isfinite(S.V1_OOF))
    assert np.all(np.isfinite(S.MS_OOF))
    S.offsets, S.cur = ({}, 0)
    for S.r in S.train_df.itertuples():
        S.L = len(S.r.sequence)
        S.offsets[str(S.r.prot_id)] = (S.cur, S.cur + S.L)
        S.cur += S.L
    S.FOLD_IDX = {}
    for S.fold, S.ids in S.OFFICIAL_FOLDS.items():
        S.z = []
        for S.pid in S.ids:
            S.a, S.b = S.offsets[str(S.pid)]
            S.z.extend(range(S.a, S.b))
        S.FOLD_IDX[S.fold] = np.asarray(S.z, dtype=np.int64)
    S.weight_rows = []
    for S.w in S.WEIGHT_GRID:
        S.p = S.w * S.V1_OOF + (1.0 - S.w) * S.MS_OOF
        S.fold_ap = []
        for S.fold in sorted(S.FOLD_IDX):
            S.idx = S.FOLD_IDX[S.fold]
            S.fold_ap.append(float(average_precision_score(S.Y_DEV[S.idx], S.p[S.idx])))
        S.weight_rows.append({'w_V1': float(S.w), 'w_MS': float(1.0 - S.w), 'CV_AUPRC_mean': float(np.mean(S.fold_ap)), 'CV_AUPRC_std': float(np.std(S.fold_ap, ddof=1)), 'OOF_AUPRC': float(average_precision_score(S.Y_DEV, S.p))})
    S.weight_search = pd.DataFrame(S.weight_rows).sort_values(['CV_AUPRC_mean', 'w_V1'], ascending=[False, False]).reset_index(drop=True)
    S.best = S.weight_search.iloc[0]
    S.W_V1_CAND = float(S.best.w_V1)
    S.W_MS_CAND = float(S.best.w_MS)
    S.ENS_DEV_CAND = float(S.best.CV_AUPRC_mean)
    S.V1_DEV = float(S.patch_summary.loc[S.patch_summary.architecture.eq(S.V1_NAME), 'CV_AUPRC_mean'].iloc[0])
    S.PROMOTE = S.ENS_DEV_CAND - S.V1_DEV >= S.MIN_DEV_GAIN
    if S.PROMOTE:
        S.W_V1, S.W_MS = (S.W_V1_CAND, S.W_MS_CAND)
        S.FINAL_DEV = S.ENS_DEV_CAND
    else:
        S.W_V1, S.W_MS = (1.0, 0.0)
        S.FINAL_DEV = S.V1_DEV
    S.FINAL_OOF = S.W_V1 * S.V1_OOF + S.W_MS * S.MS_OOF
    print('=' * 100)
    print(f'{S.TASK_NAME} DEV ENSEMBLE LOCK')
    print('=' * 100)
    display(S.weight_search.head(10))
    print('Candidate weights:', S.W_V1_CAND, S.W_MS_CAND)
    print('Candidate Dev mean AUPRC:', S.ENS_DEV_CAND)
    print('V1 Dev mean AUPRC:', S.V1_DEV)
    print('Candidate gain:', S.ENS_DEV_CAND - S.V1_DEV)
    print('Promote by >=0.002 rule:', S.PROMOTE)
    print('FINAL weights:', S.W_V1, S.W_MS)
    print('FINAL Dev mean AUPRC:', S.FINAL_DEV)
    S.weight_search.to_csv(S.OUT / 'DEV_ENSEMBLE_WEIGHT_SEARCH.csv', index=False)
    S.cf_rows = []
    for S.held_fold in sorted(S.FOLD_IDX):
        S.train_folds = [f for f in sorted(S.FOLD_IDX) if f != S.held_fold]
        S.cand = []
        for S.w in S.WEIGHT_GRID:
            S.p = S.w * S.V1_OOF + (1.0 - S.w) * S.MS_OOF
            S.aps = [average_precision_score(S.Y_DEV[S.FOLD_IDX[f]], S.p[S.FOLD_IDX[f]]) for f in S.train_folds]
            S.cand.append((float(np.mean(S.aps)), float(S.w)))
        S.cand.sort(key=lambda x: (x[0], x[1]), reverse=True)
        S._, S.wcf = S.cand[0]
        S.pm = 1.0 - S.wcf
        S.idx = S.FOLD_IDX[S.held_fold]
        S.pcf = S.wcf * S.V1_OOF[S.idx] + S.pm * S.MS_OOF[S.idx]
        S.cf_rows.append({'held_fold': S.held_fold, 'selected_w_V1_from_other_4_folds': S.wcf, 'selected_w_MS_from_other_4_folds': S.pm, 'held_fold_AUPRC_crossfit_ensemble': float(average_precision_score(S.Y_DEV[S.idx], S.pcf)), 'held_fold_AUPRC_V1': float(average_precision_score(S.Y_DEV[S.idx], S.V1_OOF[S.idx])), 'held_fold_AUPRC_MS': float(average_precision_score(S.Y_DEV[S.idx], S.MS_OOF[S.idx]))})
    S.crossfit = pd.DataFrame(S.cf_rows)
    S.crossfit.to_csv(S.OUT / 'DEV_CROSSFIT_ENSEMBLE_AUDIT.csv', index=False)
    print('\nCross-fitted ensemble audit:')
    display(S.crossfit)
    print('Crossfit mean AUPRC ensemble =', S.crossfit.held_fold_AUPRC_crossfit_ensemble.mean())
    print('Crossfit mean AUPRC V1       =', S.crossfit.held_fold_AUPRC_V1.mean())
    S.DEV_PROBS = {'V1_Attention': S.V1_OOF, 'MS_Attention': S.MS_OOF, S.FINAL_NAME: S.FINAL_OOF}
    S.THRESHOLDS = {}
    S.threshold_rows = []
    S.dev_metric_rows = []
    for S.method, S.p in S.DEV_PROBS.items():
        S.THRESHOLDS[S.method] = optimize_thresholds_exact(S.Y_DEV, S.p)
        for S.rule, S.thr in S.THRESHOLDS[S.method].items():
            S.threshold_rows.append({'method': S.method, 'threshold_rule': S.rule, 'threshold': float(S.thr), 'threshold_selected_on': 'fixed' if S.rule == 'fixed_0.5' else 'DevSet1010 OOF only'})
            S.m = full_global_metrics(S.Y_DEV, S.p, S.thr)
            S.pm = protein_mean_metrics_baseline_convention(S.train_df, S.p, S.thr)
            S.dev_metric_rows.append({'dataset': 'DevSet1010_OOF', 'method': S.method, 'threshold_rule': S.rule, 'threshold': float(S.thr)})
    S.threshold_df = pd.DataFrame(S.threshold_rows)
    S.dev_metrics = pd.DataFrame(S.dev_metric_rows)
    S.threshold_df.to_csv(S.OUT / 'DEV_THRESHOLD_LOCK.csv', index=False)
    S.dev_metrics.to_csv(S.OUT / 'DEV_OOF_FULL_METRICS.csv', index=False)
    print('\nDEV threshold locks:')
    display(S.threshold_df)
    S.PRED_CACHE = S.OUT / 'prediction_cache'
    S.PRED_CACHE.mkdir(parents=True, exist_ok=True)
    S.external_rows = []
    S.final_prediction_files = {}
    for S.split in ['TestSet300', 'TestSetNew46']:
        S.df = S.SPLIT_DF[S.split]
        S.y = np.concatenate(S.df[S.LABEL_COL].apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
        S.p_v1 = predict_v1_external(S.split)
        S.p_ms = predict_ms_external(S.split)
        S.p_final = S.W_V1 * S.p_v1 + S.W_MS * S.p_ms
        S.pred_map = {'V1_Attention': S.p_v1, 'MS_Attention': S.p_ms, S.FINAL_NAME: S.p_final}
        for S.method, S.p in S.pred_map.items():
            for S.rule, S.thr in S.THRESHOLDS[S.method].items():
                S.gm = full_global_metrics(S.y, S.p, S.thr)
                S.pm = protein_mean_metrics_baseline_convention(S.df, S.p, S.thr)
                S.external_rows.append({'dataset': S.split, 'method': S.method, 'threshold_rule': S.rule, 'threshold': float(S.thr), 'threshold_selected_on': 'fixed' if S.rule == 'fixed_0.5' else 'DevSet1010 OOF only', 'directly_comparable_to_baseline_fixed0.5': bool(S.rule == 'fixed_0.5')})
        S.pids, S.ridx = ([], [])
        for S.r in S.df.itertuples():
            S.L = len(S.r.sequence)
            S.pids.extend([str(S.r.prot_id)] * S.L)
            S.ridx.extend(range(S.L))
        S.pred_df = pd.DataFrame({'prot_id': S.pids, 'residue_index': S.ridx, 'y_true': S.y, 'prob_V1': S.p_v1, 'prob_MS': S.p_ms, 'prob_FINAL': S.p_final})
        S.pred_fp = S.OUT / f'FINAL_RESIDUE_PREDICTIONS_{S.split}.csv.gz'
        S.pred_df.to_csv(S.pred_fp, index=False)
        S.final_prediction_files[S.split] = str(S.pred_fp)
    S.external_full = pd.DataFrame(S.external_rows)
    S.baseline_fp = S.BASELINE_WORK / 'results' / 'FULL_DIRECT_BASELINE_METRICS_300_46.csv'
    S.bb = pd.read_csv(S.baseline_fp)
    S.bb = S.bb[S.bb.label.eq(S.LABEL_COL)].copy()
    if 'operating_point' in S.bb.columns:
        S.bb = S.bb[S.bb.operating_point.eq('fixed_0.5')].copy()
    S.baseline_rows = []
    for S._, S.r in S.bb.iterrows():
        if str(S.r.method) not in {'bindNode24', 'bindEmbed21DL'}:
            continue
        S.ap = _first_existing(S.r, ['AP_AUPRC', 'AUPRC'])
        S.auroc = _first_existing(S.r, ['AUROC'])
        S.acc = _first_existing(S.r, ['accuracy', 'Accuracy', 'global_accuracy'])
        S.prec = _first_existing(S.r, ['global_positive_precision', 'precision', 'Precision'])
        S.rec = _first_existing(S.r, ['global_positive_recall', 'recall', 'Recall'])
        S.f1p = _first_existing(S.r, ['global_positive_F1', 'F1', 'f1'])
        S.mcc = _first_existing(S.r, ['global_MCC', 'MCC'])
        S.pmf1 = _first_existing(S.r, ['protein_mean_F1'])
        S.pmmcc = _first_existing(S.r, ['protein_mean_MCC'])
        S.baseline_rows.append({'dataset': str(S.r.dataset), 'method': str(S.r.method), 'threshold_rule': 'fixed_0.5', 'threshold': 0.5, 'threshold_selected_on': 'reproduced baseline operating point', 'directly_comparable_to_baseline_fixed0.5': True, 'n_residues': np.nan, 'n_positive': np.nan, 'prevalence': np.nan, 'TP': np.nan, 'TN': np.nan, 'FP': np.nan, 'FN': np.nan, 'AUPRC_global_residue': S.ap, 'AUROC_global_residue': S.auroc, 'Accuracy_global_residue': S.acc, 'Balanced_accuracy_global_residue': np.nan, 'Precision_positive_global_residue': S.prec, 'Recall_positive_global_residue': S.rec, 'Sensitivity_global_residue': S.rec, 'Specificity_global_residue': np.nan, 'NPV_global_residue': np.nan, 'F1_positive_global_residue': S.f1p, 'MCC_global_residue': S.mcc, 'F1_micro_all_classes': S.acc, 'F1_macro_two_classes': np.nan, 'protein_mean_F1_baseline_convention': S.pmf1, 'protein_mean_MCC_baseline_convention': S.pmmcc, 'protein_n_total': np.nan, 'protein_n_included_baseline_convention': np.nan, 'protein_n_excluded_TN_only': np.nan})
    S.all_metrics = pd.concat([S.external_full, pd.DataFrame(S.baseline_rows)], ignore_index=True, sort=False)
    S.all_metrics.to_csv(S.OUT / 'EXTERNAL_FULL_METRICS_LONG.csv', index=False)
    S.headline_cols = ['dataset', 'method', 'threshold', 'AUPRC_global_residue', 'AUROC_global_residue', 'Accuracy_global_residue', 'Precision_positive_global_residue', 'Recall_positive_global_residue', 'F1_positive_global_residue', 'MCC_global_residue', 'F1_micro_all_classes', 'F1_macro_two_classes', 'protein_mean_F1_baseline_convention', 'protein_mean_MCC_baseline_convention']
    S.headline = S.all_metrics[S.all_metrics.threshold_rule.eq('fixed_0.5')][S.headline_cols].copy()
    S.headline = S.headline.sort_values(['dataset', 'AUPRC_global_residue'], ascending=[True, False])
    S.headline.to_csv(S.OUT / 'HEADLINE_FIXED05_COMPARISON.csv', index=False)
    S.final_threshold_report = S.external_full[S.external_full.method.eq(S.FINAL_NAME)].copy()
    S.final_threshold_report = S.final_threshold_report[['dataset', 'method', 'threshold_rule', 'threshold', 'AUPRC_global_residue', 'AUROC_global_residue', 'Accuracy_global_residue', 'Balanced_accuracy_global_residue', 'Precision_positive_global_residue', 'Recall_positive_global_residue', 'Specificity_global_residue', 'F1_positive_global_residue', 'MCC_global_residue', 'F1_micro_all_classes', 'F1_macro_two_classes', 'protein_mean_F1_baseline_convention', 'protein_mean_MCC_baseline_convention', 'protein_n_included_baseline_convention']]
    S.final_threshold_report.to_csv(S.OUT / 'FINAL_MODEL_ALL_THRESHOLDS.csv', index=False)
    S.decision = {'task': S.TASK_NAME, 'final_method': S.FINAL_NAME, 'V1_name': S.V1_NAME, 'MS_name': S.MS_NAME, 'candidate_w_V1': S.W_V1_CAND, 'candidate_w_MS': S.W_MS_CAND, 'final_w_V1': S.W_V1, 'final_w_MS': S.W_MS, 'V1_Dev_mean_AUPRC': S.V1_DEV, 'candidate_ensemble_Dev_mean_AUPRC': S.ENS_DEV_CAND, 'final_Dev_mean_AUPRC': S.FINAL_DEV, 'candidate_Dev_gain_vs_V1': S.ENS_DEV_CAND - S.V1_DEV, 'promoted_by_0.002_rule': bool(S.PROMOTE), 'crossfit_mean_AUPRC_ensemble': float(S.crossfit.held_fold_AUPRC_crossfit_ensemble.mean()), 'crossfit_mean_AUPRC_V1': float(S.crossfit.held_fold_AUPRC_V1.mean()), 'metric_naming': {'F1_positive_global_residue': 'binary positive-class F1 after pooling all residues', 'MCC_global_residue': 'MCC from one pooled residue-level confusion matrix', 'F1_micro_all_classes': 'formal sklearn-style micro-F1 across both classes; equals accuracy in this binary single-label task', 'protein_mean_metrics': 'per-protein metrics averaged after excluding TN-only proteins (TP+FP+FN=0), matching reproduced baseline convention'}, 'threshold_rules': {method: {k: float(v) for k, v in rules.items()} for method, rules in S.THRESHOLDS.items()}, 'prediction_files': S.final_prediction_files}
    (S.OUT / 'FINAL_DECISION_AND_DEFINITIONS.json').write_text(json.dumps(S.decision, indent=2))
    print('\n' + '=' * 100)
    print(f'{S.TASK_NAME} — HEADLINE FIXED 0.5 (DIRECT BASELINE COMPARISON)')
    print('=' * 100)
    display(S.headline)
    print('\n' + '=' * 100)
    print(f'{S.TASK_NAME} — FINAL MODEL AT ALL DEV-LOCKED THRESHOLDS')
    print('=' * 100)
    display(S.final_threshold_report)
    print('\nMetric naming:')
    print('  F1_positive_global_residue = pooled binary positive-class F1')
    print('  MCC_global_residue         = pooled/global residue-level MCC')
    print('  F1_micro_all_classes       = formal micro-F1; equals Accuracy here')
    print('  protein_mean_*             = baseline convention, TN-only excluded')
    print('\nSaved:', S.OUT)
    print('Key files:')
    for S.fn in ['DEV_ENSEMBLE_WEIGHT_SEARCH.csv', 'DEV_CROSSFIT_ENSEMBLE_AUDIT.csv', 'DEV_THRESHOLD_LOCK.csv', 'DEV_OOF_FULL_METRICS.csv', 'EXTERNAL_FULL_METRICS_LONG.csv', 'HEADLINE_FIXED05_COMPARISON.csv', 'FINAL_MODEL_ALL_THRESHOLDS.csv', 'FINAL_DECISION_AND_DEFINITIONS.json']:
        print(' -', S.OUT / S.fn)

STAGES = [stage_38, stage_39, stage_40, stage_42, stage_43, stage_44, stage_45, stage_46, stage_47]

