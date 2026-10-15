"""Extracted from SMALL_ULTIMATE_V1_DATA_DRIVEN_POCKET-3.ipynb. Section: pocket."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Chain, Counter, DataLoader, Dataset, F, Model, PDBIO, PDBParser, Path, PocketProteinDataset, PocketResidueEncoder, SmallPocketHypergraph, Structure, Subset, accuracy_score, ast, average_precision_score, balanced_accuracy_score, cKDTree, concat_blocks, deepcopy, defaultdict, f1_score, hashlib, json, math, matthews_corrcoef, nn, np, os, pd, platform, precision_score, recall_score, representative_coord, residue_sequence, roc_auc_score, seq1, shutil, standard_residues, subprocess, sys, tarfile, torch
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
    from lc_bind.utils.pdb import residue_sequence, standard_residues
    from lc_bind.models.small.features import representative_coord
    from lc_bind.models.small.screen import concat_blocks

    class PocketProteinDataset(Dataset):
    
        def __init__(self, split, df):
            self.split = split
            self.df = df.reset_index(drop=True)
            self.store = S.ctx.stores[S.STORE_KEY[split]]
    
        def __len__(self):
            return len(self.df)
    
        def __getitem__(self, i):
            r = self.df.iloc[i]
            pid = str(r.prot_id)
            mods = {m: concat_blocks(self.store, pid, blocks).astype(np.float32, copy=False) for m, blocks in S.MODALITY_SPEC.items()}
            er, ep, pf = S.POCKET_RAM[self.split, pid]
            return {'pid': pid, 'mods': mods, 'y': np.asarray(r.small_binding, dtype=np.float32), 'edge_res': er, 'edge_pocket': ep, 'pocket_feat': pf}

    class PocketResidueEncoder(nn.Module):
    
        def __init__(self, dims, mode='gated', latent=192, drop=0.2):
            super().__init__()
            self.keys = sorted(dims)
            self.mode = mode
            self.enc = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, latent), nn.GELU(), nn.Dropout(drop)) for k, d in dims.items()})
            if mode == 'gated':
                self.gate = nn.ModuleDict({k: nn.Linear(latent, 1) for k in dims})
            elif mode == 'attention':
                self.attn = nn.MultiheadAttention(latent, 6, dropout=drop, batch_first=True)
            else:
                raise ValueError(mode)
    
        def forward(self, x):
            Z = torch.stack([self.enc[k](x[k]) for k in self.keys], dim=1)
            if self.mode == 'gated':
                G = torch.cat([self.gate[k](Z[:, j, :]) for j, k in enumerate(self.keys)], dim=1)
                G = torch.softmax(G, dim=1).unsqueeze(-1)
                return (Z * G).sum(dim=1)
            A, _ = self.attn(Z, Z, Z, need_weights=False)
            return (Z + A).mean(dim=1)

    class SmallPocketHypergraph(nn.Module):
    
        def __init__(self, dims, mode='gated', latent=192, moe=False, experts=4, drop=0.2):
            super().__init__()
            self.encoder = PocketResidueEncoder(dims, mode=mode, latent=latent, drop=drop)
            self.pocket_static = nn.Sequential(nn.LayerNorm(22), nn.Linear(22, 64), nn.GELU())
            self.pocket_mlp = nn.Sequential(nn.Linear(latent * 2 + 64, latent), nn.GELU(), nn.Dropout(drop), nn.Linear(latent, latent), nn.GELU())
            self.message_mlp = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, latent), nn.GELU(), nn.Dropout(drop), nn.Linear(latent, latent))
            self.alpha_logit = nn.Parameter(torch.tensor(-2.944, dtype=torch.float32))
            self.moe = moe
            if not moe:
                self.head = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))
            else:
                self.experts = nn.ModuleList([nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1)) for _ in range(experts)])
                self.expert_gate = nn.Linear(latent, experts)
    
        def index_mean(self, x, index, n):
            out = x.new_zeros((n, x.shape[-1]))
            cnt = x.new_zeros((n, 1))
            if len(index):
                out.index_add_(0, index, x)
                cnt.index_add_(0, index, x.new_ones((len(index), 1)))
            return out / cnt.clamp_min(1.0)
    
        def index_max(self, x, index, n):
            if n == 0:
                return x.new_zeros((0, x.shape[-1]))
            base = x.new_full((n, x.shape[-1]), -1000000000.0)
            if len(index):
                idx = index[:, None].expand(-1, x.shape[-1])
                out = base.scatter_reduce(0, idx, x, reduce='amax', include_self=True)
            else:
                out = base
            out = torch.where(out < -100000000.0, torch.zeros_like(out), out)
            return out
    
        def forward(self, mods, edge_res, edge_pocket, pocket_feat):
            h = self.encoder(mods)
            nres = h.shape[0]
            npocket = pocket_feat.shape[0]
            alpha = torch.sigmoid(self.alpha_logit)
            if npocket and edge_res.numel():
                mean = self.index_mean(h[edge_res], edge_pocket, npocket)
                mx = self.index_max(h[edge_res], edge_pocket, npocket)
                ps = self.pocket_static(pocket_feat)
                hp = self.pocket_mlp(torch.cat([mean, mx, ps], dim=-1))
                msg = self.index_mean(hp[edge_pocket], edge_res, nres)
                incident = h.new_zeros((nres, 1))
                incident.index_add_(0, edge_res, h.new_ones((len(edge_res), 1)))
                delta = self.message_mlp(msg)
                delta = delta * (incident > 0).float()
                h = h + alpha * delta
            if not self.moe:
                return self.head(h).squeeze(-1)
            logits = torch.cat([e(h) for e in self.experts], dim=1)
            w = torch.softmax(self.expert_gate(h), dim=1)
            return (logits * w).sum(dim=1)

def structure_coords(split, pid, seq):
    path = S.PDB_NORM_ROOT / split / f'{pid}.pdb'
    st = PDBParser(QUIET=True).get_structure('p', str(path))
    m = next(st.get_models())
    rr = standard_residues(m['A'])
    if residue_sequence(rr) != seq:
        raise RuntimeError(f'{split}/{pid}: normalized PDB mismatch')
    return np.stack([representative_coord(r) for r in rr], axis=0).astype(np.float32)

def pocket_static_features(seq, coords, members):
    members = np.asarray(members, dtype=int)
    aa = [seq[i] for i in members]
    cc = coords[members]
    n = len(members)
    centroid = cc.mean(axis=0)
    radial = np.linalg.norm(cc - centroid[None, :], axis=1)
    pair = []
    if n >= 2:
        for i in range(n):
            for j in range(i + 1, n):
                pair.append(float(np.linalg.norm(cc[i] - cc[j])))
    pair = np.asarray(pair if pair else [0.0], dtype=np.float32)
    hydro = np.asarray([S.HYDROPATHY.get(a, 0.0) / 4.5 for a in aa], dtype=np.float32)
    comp = Counter(aa)
    probs = np.asarray([x / n for x in comp.values()], dtype=np.float32)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / math.log(20))
    f = np.zeros(22, dtype=np.float32)
    f[0] = min(n, 40) / 40.0
    f[1] = sum((x in S.GROUPS['AROM'] for x in aa)) / n
    f[2] = sum((x in S.GROUPS['SULFUR'] for x in aa)) / n
    f[3] = sum((x in S.GROUPS['HYDRO'] for x in aa)) / n
    f[4] = sum((x in S.GROUPS['POLAR'] for x in aa)) / n
    f[5] = sum((x in S.GROUPS['ACIDIC'] for x in aa)) / n
    f[6] = sum((x in S.GROUPS['HBD'] for x in aa)) / n
    f[7] = sum((x in S.GROUPS['HBA'] for x in aa)) / n
    f[8] = float(np.mean(hydro))
    f[9] = sum((x in S.GROUPS['GP'] for x in aa)) / n
    f[10] = min(float(radial.mean()) / 12.0, 1.0)
    f[11] = min(float(radial.std()) / 8.0, 1.0)
    f[12] = min(float(radial.max()) / 15.0, 1.0)
    f[13] = min(float(pair.mean()) / 15.0, 1.0)
    f[14] = min(float(pair.std()) / 10.0, 1.0)
    f[15] = min((members.max() - members.min()) / 100.0, 1.0)
    f[16] = entropy
    if n >= 2:
        seqsep = []
        for i in range(n):
            for j in range(i + 1, n):
                seqsep.append(abs(int(members[i]) - int(members[j])))
        seqsep = np.asarray(seqsep, dtype=float)
        f[17] = float(np.mean(seqsep > 10))
        f[18] = min(float(seqsep.mean()) / 50.0, 1.0)
    if n >= 3:
        centered = cc - cc.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(n - 1, 1)
        eig = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
        total = float(eig.sum() + 1e-08)
        f[19] = np.clip((eig[0] - eig[1]) / (eig[0] + 1e-08), 0, 1)
        f[20] = np.clip((eig[1] - eig[2]) / (eig[0] + 1e-08), 0, 1)
        f[21] = np.clip(eig[2] / total, 0, 1)
    return f

def build_candidate_pockets(split, pid, seq):
    coords = structure_coords(split, pid, seq)
    tree = cKDTree(coords)
    candidates = {}
    for seed, a in enumerate(seq):
        if a not in S.POCKET_SEEDS:
            continue
        members = tuple(sorted(tree.query_ball_point(coords[seed], S.POCKET_RADIUS_A)))
        if len(members) < 4:
            continue
        feat = pocket_static_features(seq, coords, members)
        score = 1.4 * feat[1] + 0.8 * feat[2] + 0.8 * feat[3] + 0.5 * feat[6] + 0.5 * feat[7] - 0.5 * feat[5] + 0.25 * feat[0] + 0.15 * feat[17]
        old = candidates.get(members)
        if old is None or score > old[0]:
            candidates[members] = (score, feat)
    ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)[:S.POCKET_MAX_PER_PROTEIN]
    edge_res = []
    edge_pocket = []
    pocket_feat = []
    for pidx, (members, (_, feat)) in enumerate(ranked):
        pocket_feat.append(feat)
        for r in members:
            edge_res.append(int(r))
            edge_pocket.append(int(pidx))
    return (np.asarray(edge_res, dtype=np.int64), np.asarray(edge_pocket, dtype=np.int64), np.asarray(pocket_feat, dtype=np.float32).reshape(-1, 22))

def collate_pocket(batch):
    keys = sorted(batch[0]['mods'])
    mods = {k: torch.from_numpy(np.concatenate([b['mods'][k] for b in batch], axis=0)).float() for k in keys}
    y = torch.from_numpy(np.concatenate([b['y'] for b in batch])).float()
    pocket_feat = []
    edge_res = []
    edge_pocket = []
    roff = 0
    poff = 0
    for b in batch:
        L = len(b['y'])
        P = len(b['pocket_feat'])
        if P:
            pocket_feat.append(torch.from_numpy(b['pocket_feat']).float())
        if len(b['edge_res']):
            edge_res.append(torch.from_numpy(b['edge_res']).long() + roff)
            edge_pocket.append(torch.from_numpy(b['edge_pocket']).long() + poff)
        roff += L
        poff += P
    pocket_feat = torch.cat(pocket_feat, dim=0) if pocket_feat else torch.zeros((0, 22), dtype=torch.float32)
    edge_res = torch.cat(edge_res, dim=0) if edge_res else torch.zeros((0,), dtype=torch.long)
    edge_pocket = torch.cat(edge_pocket, dim=0) if edge_pocket else torch.zeros((0,), dtype=torch.long)
    return {'mods': mods, 'y': y, 'edge_res': edge_res, 'edge_pocket': edge_pocket, 'pocket_feat': pocket_feat}

def move_pocket_batch(b):
    return {'mods': {k: v.to(S.DEVICE, non_blocking=True) for k, v in b['mods'].items()}, 'y': b['y'].to(S.DEVICE, non_blocking=True), 'edge_res': b['edge_res'].to(S.DEVICE, non_blocking=True), 'edge_pocket': b['edge_pocket'].to(S.DEVICE, non_blocking=True), 'pocket_feat': b['pocket_feat'].to(S.DEVICE, non_blocking=True)}

def predict_pocket(model, dataset):
    loader = DataLoader(dataset, batch_size=S.POCKET_PROTEIN_BATCH, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_pocket)
    pp = []
    yy = []
    model.eval()
    with torch.no_grad():
        for b in loader:
            b = move_pocket_batch(b)
            p = torch.sigmoid(model(b['mods'], b['edge_res'], b['edge_pocket'], b['pocket_feat']))
            pp.append(p.detach().cpu().numpy())
            yy.append(b['y'].detach().cpu().numpy())
    return (np.concatenate(pp), np.concatenate(yy))

def train_pocket_fold(arch, tr_ds, va_ds, fold):
    torch.manual_seed(S.SEED + 100 + fold)
    np.random.seed(S.SEED + 100 + fold)
    model = S.POCKET_FACTORY[arch](S.DIMS).to(S.DEVICE)
    ytr = np.concatenate(tr_ds.df.small_binding.apply(lambda x: np.asarray(x, dtype=np.float32)).tolist())
    npos = max(float(ytr.sum()), 1.0)
    nneg = max(float(len(ytr) - ytr.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.sqrt(nneg / npos), dtype=torch.float32, device=S.DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=S.POCKET_LR, weight_decay=S.POCKET_WEIGHT_DECAY)
    loader = DataLoader(tr_ds, batch_size=S.POCKET_PROTEIN_BATCH, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_pocket)
    best_ap = -1.0
    best_state = None
    bad = 0
    for epoch in range(1, S.POCKET_EPOCHS + 1):
        model.train()
        losses = []
        for b in loader:
            b = move_pocket_batch(b)
            opt.zero_grad(set_to_none=True)
            logits = model(b['mods'], b['edge_res'], b['edge_pocket'], b['pocket_feat'])
            loss = criterion(logits, b['y'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        vp, vy = predict_pocket(model, va_ds)
        val_ap = S.ap(vy, vp)
        if val_ap > best_ap + 1e-05:
            best_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch == 1 or epoch % 5 == 0:
            print(arch, 'fold', fold, 'epoch', epoch, 'loss', round(float(np.mean(losses)), 4), 'val_AP', round(val_ap, 5), 'best', round(best_ap, 5), 'alpha', round(float(torch.sigmoid(model.alpha_logit).detach().cpu()), 4))
        if bad >= S.POCKET_PATIENCE:
            break
    model.load_state_dict(best_state)
    return (model, best_state)

def run_pocket_cv(arch):
    offsets = {}
    cursor = 0
    for r in S.train_df.itertuples():
        L = len(r.sequence)
        offsets[str(r.prot_id)] = (cursor, cursor + L)
        cursor += L
    oof = np.full(cursor, np.nan, dtype=np.float32)
    rows = []
    states = {}
    adir = S.POCKET_STATE_ROOT / arch
    adir.mkdir(parents=True, exist_ok=True)
    for fold, val_ids in S.OFFICIAL_FOLDS.items():
        val_set = set(val_ids)
        tr_df = S.train_df[~S.train_df.prot_id.isin(val_set)].copy()
        va_df = S.train_df[S.train_df.prot_id.isin(val_set)].copy()
        tr_ds = PocketProteinDataset('DevSet1010', tr_df)
        va_ds = PocketProteinDataset('DevSet1010', va_df)
        state_fp = adir / f'fold{fold}.pt'
        pred_fp = adir / f'fold{fold}_val.npy'
        meta_fp = adir / f'fold{fold}.json'
        expected = int(va_df.sequence.str.len().sum())
        reuse = False
        if state_fp.exists() and pred_fp.exists() and meta_fp.exists():
            try:
                meta = json.loads(meta_fp.read_text())
                pp = np.load(pred_fp)
                reuse = meta.get('representation') == S.BEST_REP_NAME and meta.get('n_val') == expected and (pp.shape == (expected,)) and np.all(np.isfinite(pp))
            except Exception:
                reuse = False
        if reuse:
            print(arch, 'fold', fold, 'REUSE')
            state = torch.load(state_fp, map_location='cpu')
            pred = np.load(pred_fp).astype(np.float32)
        else:
            print(arch, 'fold', fold, 'TRAIN')
            model, state = train_pocket_fold(arch, tr_ds, va_ds, fold)
            pred, _ = predict_pocket(model, va_ds)
            pred = pred.astype(np.float32)
            torch.save(state, state_fp)
            np.save(pred_fp, pred)
            meta_fp.write_text(json.dumps({'representation': S.BEST_REP_NAME, 'blocks': S.BEST_REP_BLOCKS, 'n_val': expected}, indent=2))
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
        yv = np.concatenate(va_df.small_binding.apply(lambda x: np.asarray(x, dtype=np.float32)).tolist())
        rows.append({'architecture': arch, 'fold': fold, 'AUPRC': S.ap(yv, pred), 'n_val': len(pred), 'n_positive': int(yv.sum())})
    if not np.all(np.isfinite(oof)):
        raise RuntimeError(f'{arch}: incomplete pocket OOF')
    f = pd.DataFrame(rows)
    yall = np.concatenate(S.train_df.small_binding.apply(lambda x: np.asarray(x, dtype=np.float32)).tolist())
    return ({'architecture': arch, 'CV_AUPRC_mean': float(f.AUPRC.mean()), 'CV_AUPRC_std': float(f.AUPRC.std(ddof=1)), 'OOF_AUPRC': S.ap(yall, oof)}, f, oof, states)

# 14. BUILD / CACHE CANDIDATE POCKET HYPERGRAPHS
def stage_23():
    S.POCKET_CACHE = S.CACHE_ROOT / 'pocket_hypergraph_v1'
    S.POCKET_CACHE.mkdir(parents=True, exist_ok=True)
    S.POCKET_SEEDS = set('WFYHCM')
    S.POCKET_RAM = {}
    S.POCKET_AUDIT = []
    for S.split, S.df in S.SPLIT_DF.items():
        S.outdir = S.POCKET_CACHE / S.split
        S.outdir.mkdir(parents=True, exist_ok=True)
        for S.n, S.r in enumerate(S.df.itertuples(), start=1):
            S.fp = S.outdir / f'{S.r.prot_id}.npz'
            S.valid = False
            if S.fp.exists() and (not S.FORCE_REBUILD_POCKETS):
                try:
                    S.z = np.load(S.fp)
                    S.valid = S.z['pocket_feat'].ndim == 2 and S.z['pocket_feat'].shape[1] == 22 and (len(S.z['edge_res']) == len(S.z['edge_pocket']))
                except Exception:
                    S.valid = False
            if not S.valid:
                S.er, S.ep, S.pf = build_candidate_pockets(S.split, str(S.r.prot_id), S.r.sequence)
                np.savez_compressed(S.fp, edge_res=S.er, edge_pocket=S.ep, pocket_feat=S.pf)
            S.z = np.load(S.fp)
            S.er = S.z['edge_res'].astype(np.int64)
            S.ep = S.z['edge_pocket'].astype(np.int64)
            S.pf = S.z['pocket_feat'].astype(np.float32)
            S.POCKET_RAM[S.split, str(S.r.prot_id)] = (S.er, S.ep, S.pf)
            S.covered = len(np.unique(S.er)) if len(S.er) else 0
            S.POCKET_AUDIT.append({'dataset': S.split, 'prot_id': str(S.r.prot_id), 'n_residues': len(S.r.sequence), 'n_candidate_pockets': len(S.pf), 'n_memberships': len(S.er), 'covered_residue_fraction': S.covered / max(len(S.r.sequence), 1)})
            if S.n % 250 == 0:
                print(S.split, S.n, '/', len(S.df))
    S.pocket_audit = pd.DataFrame(S.POCKET_AUDIT)
    S.pocket_audit.to_csv(S.RUN_ROOT / 'pocket_audit.csv', index=False)
    display(S.pocket_audit.groupby('dataset').agg(proteins=('prot_id', 'nunique'), mean_pockets=('n_candidate_pockets', 'mean'), median_pockets=('n_candidate_pockets', 'median'), mean_memberships=('n_memberships', 'mean'), mean_covered_fraction=('covered_residue_fraction', 'mean')).reset_index())
    print('Pocket hypergraph cache: PASS')

# 15. POCKET-HYPERGRAPH DATASET / MODEL
def stage_24():
    S.POCKET_FACTORY = {'SM_PocketHG_Gated': lambda dims: SmallPocketHypergraph(dims, mode='gated', latent=S.POCKET_LATENT, moe=False), 'SM_PocketHG_Attention': lambda dims: SmallPocketHypergraph(dims, mode='attention', latent=S.POCKET_LATENT, moe=False), 'SM_PocketHG_GatedMoE': lambda dims: SmallPocketHypergraph(dims, mode='gated', latent=S.POCKET_LATENT, moe=True, experts=4)}
    print('Pocket-hypergraph architectures:', list(S.POCKET_FACTORY))

# 16. POCKET-HYPERGRAPH CV TRAINER — RESUME SAFE
def stage_25():
    S.POCKET_STATE_ROOT = S.RUN_ROOT / 'pocket_states'
    S.POCKET_STATE_ROOT.mkdir(parents=True, exist_ok=True)

# 17. RUN POCKET-HYPERGRAPH SCREEN — DEV ONLY
def stage_26():
    S.POCKET_RESULTS = {}
    S.POCKET_FOLDS = []
    S.POCKET_OOF = {}
    S.POCKET_STATES = {}
    if S.RUN_POCKET_HYPERGRAPH:
        for S.arch in S.POCKET_FACTORY:
            print('\n' + '=' * 80)
            print('POCKET HYPERGRAPH:', S.arch)
            print('=' * 80)
            S.s, S.f, S.o, S.st = run_pocket_cv(S.arch)
            S.POCKET_RESULTS[S.arch] = S.s
            S.POCKET_FOLDS.append(S.f)
            S.POCKET_OOF[S.arch] = S.o
            S.POCKET_STATES[S.arch] = S.st
    S.pocket_summary = pd.DataFrame(S.POCKET_RESULTS.values()).sort_values('CV_AUPRC_mean', ascending=False).reset_index(drop=True)
    S.pocket_folds = pd.concat(S.POCKET_FOLDS, ignore_index=True)
    S.pocket_summary.to_csv(S.RUN_ROOT / 'pocket_hypergraph_cv.csv', index=False)
    S.pocket_folds.to_csv(S.RUN_ROOT / 'pocket_hypergraph_folds.csv', index=False)
    display(S.pocket_summary)
    S.BEST_POCKET = str(S.pocket_summary.iloc[0].architecture)
    S.BEST_POCKET_SCORE = float(S.pocket_summary.iloc[0].CV_AUPRC_mean)
    S.alphas = []
    for S.fold, S.state in S.POCKET_STATES[S.BEST_POCKET].items():
        if 'alpha_logit' in S.state:
            S.alphas.append(float(torch.sigmoid(S.state['alpha_logit']).item()))
    print('BEST POCKET MODEL:', S.BEST_POCKET)
    print('Dev CV AUPRC:', S.BEST_POCKET_SCORE)
    print('Residual alpha:', S.alphas)
    print('Mean alpha:', float(np.mean(S.alphas)) if S.alphas else None)
    (S.RUN_ROOT / 'locked_pocket_model.json').write_text(json.dumps({'architecture': S.BEST_POCKET, 'dev_cv_auprc': S.BEST_POCKET_SCORE, 'alpha_by_fold': S.alphas, 'alpha_mean': float(np.mean(S.alphas)) if S.alphas else None}, indent=2))

STAGES = [stage_23, stage_24, stage_25, stage_26]

