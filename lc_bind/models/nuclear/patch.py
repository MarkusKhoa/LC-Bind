"""Extracted from NUCLEAR_ULTIMATE_V1_DATA_DRIVEN_PATCH-4.ipynb. Section: patch."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Chain, Counter, DataLoader, Dataset, F, Model, NuclearPatchHypergraph, PDBIO, PDBParser, PatchProteinDataset, PatchResidueEncoder, Path, Structure, Subset, accuracy_score, ast, average_precision_score, balanced_accuracy_score, cKDTree, concat_blocks, deepcopy, defaultdict, f1_score, hashlib, json, math, matthews_corrcoef, nn, np, os, pd, platform, precision_score, recall_score, representative_coord, residue_sequence, roc_auc_score, seq1, shutil, standard_residues, subprocess, sys, tarfile, torch
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
    from collections import defaultdict, Counter
    import json, math, numpy as np, pandas as pd
    import json, numpy as np, pandas as pd, torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
    from lc_bind.utils.pdb import residue_sequence, standard_residues
    from lc_bind.models.nuclear.features import representative_coord
    from lc_bind.models.nuclear.screen import concat_blocks

    class PatchProteinDataset(Dataset):
    
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
            er, ep, pf = S.PATCH_RAM[self.split, pid]
            return {'pid': pid, 'mods': mods, 'y': np.asarray(r.nuclear_binding, dtype=np.float32), 'edge_res': er, 'edge_patch': ep, 'patch_feat': pf}

    class PatchResidueEncoder(nn.Module):
    
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

    class NuclearPatchHypergraph(nn.Module):
    
        def __init__(self, dims, mode='gated', latent=192, moe=False, experts=4, drop=0.2):
            super().__init__()
            self.encoder = PatchResidueEncoder(dims, mode=mode, latent=latent, drop=drop)
            self.patch_static = nn.Sequential(nn.LayerNorm(18), nn.Linear(18, 64), nn.GELU())
            self.patch_mlp = nn.Sequential(nn.Linear(latent * 2 + 64, latent), nn.GELU(), nn.Dropout(drop), nn.Linear(latent, latent), nn.GELU())
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
    
        def forward(self, mods, edge_res, edge_patch, patch_feat):
            h = self.encoder(mods)
            nres = h.shape[0]
            npatch = patch_feat.shape[0]
            alpha = torch.sigmoid(self.alpha_logit)
            if npatch and edge_res.numel():
                mean = self.index_mean(h[edge_res], edge_patch, npatch)
                mx = self.index_max(h[edge_res], edge_patch, npatch)
                ps = self.patch_static(patch_feat)
                hp = self.patch_mlp(torch.cat([mean, mx, ps], dim=-1))
                msg = self.index_mean(hp[edge_patch], edge_res, nres)
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

def patch_static_features(seq, coords, members):
    members = np.asarray(members, dtype=int)
    aa = [seq[i] for i in members]
    cc = coords[members]
    n = len(members)
    q = np.asarray([S.CHARGE.get(x, 0.0) for x in aa], dtype=np.float32)
    centroid = cc.mean(axis=0)
    radial = np.linalg.norm(cc - centroid[None, :], axis=1)
    pair = []
    if n >= 2:
        for i in range(n):
            for j in range(i + 1, n):
                pair.append(float(np.linalg.norm(cc[i] - cc[j])))
    pair = np.asarray(pair if pair else [0.0], dtype=np.float32)
    probs = []
    cnt = Counter(aa)
    for x in cnt.values():
        probs.append(x / n)
    entropy = -sum((p * math.log(p + 1e-12) for p in probs)) / math.log(20)
    f = np.zeros(18, dtype=np.float32)
    f[0] = min(n, 40) / 40.0
    f[1] = sum((x in S.GROUPS['KR'] for x in aa)) / n
    f[2] = sum((x in S.GROUPS['KRH'] for x in aa)) / n
    f[3] = sum((x in S.GROUPS['DE'] for x in aa)) / n
    f[4] = sum((x in S.GROUPS['POLAR'] for x in aa)) / n
    f[5] = sum((x in S.GROUPS['AROM'] for x in aa)) / n
    f[6] = sum((x in S.GROUPS['HYDRO'] for x in aa)) / n
    f[7] = float(np.mean(q))
    f[8] = sum((x in S.GROUPS['HBD'] for x in aa)) / n
    f[9] = sum((x in S.GROUPS['HBA'] for x in aa)) / n
    f[10] = min(float(radial.mean()) / 12.0, 1.0)
    f[11] = min(float(radial.std()) / 8.0, 1.0)
    f[12] = min(float(radial.max()) / 15.0, 1.0)
    f[13] = min((members.max() - members.min()) / 100.0, 1.0)
    f[14] = min(float(pair.mean()) / 15.0, 1.0)
    f[15] = min(float(pair.std()) / 10.0, 1.0)
    f[16] = float(entropy)
    f[17] = float(np.tanh((sum((x in S.GROUPS['KRH'] for x in aa)) - sum((x in S.GROUPS['DE'] for x in aa))) / 5.0))
    return f

def build_patch_graph(split, pid, seq):
    coords = structure_coords(split, pid, seq)
    tree = cKDTree(coords)
    candidates = {}
    for seed, a in enumerate(seq):
        if a not in S.PATCH_SEEDS:
            continue
        members = tuple(sorted(tree.query_ball_point(coords[seed], S.PATCH_RADIUS_A)))
        if len(members) < 4:
            continue
        if not any((seq[j] in S.GROUPS['KRH'] for j in members)):
            continue
        f = patch_static_features(seq, coords, members)
        score = 2.0 * f[2] + 0.7 * f[4] - 0.7 * f[3] + 0.2 * f[0]
        old = candidates.get(members)
        if old is None or score > old[0]:
            candidates[members] = (score, f)
    ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)[:S.PATCH_MAX_PER_PROTEIN]
    edge_res = []
    edge_patch = []
    patch_feat = []
    for pidx, (members, (_, feat)) in enumerate(ranked):
        patch_feat.append(feat)
        for r in members:
            edge_res.append(int(r))
            edge_patch.append(int(pidx))
    return (np.asarray(edge_res, dtype=np.int64), np.asarray(edge_patch, dtype=np.int64), np.asarray(patch_feat, dtype=np.float32).reshape(-1, 18))

def collate_patch(batch):
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
    patch_feat = torch.cat(patch_feat, dim=0) if patch_feat else torch.zeros((0, 18), dtype=torch.float32)
    edge_res = torch.cat(edge_res, dim=0) if edge_res else torch.zeros((0,), dtype=torch.long)
    edge_patch = torch.cat(edge_patch, dim=0) if edge_patch else torch.zeros((0,), dtype=torch.long)
    return {'mods': mods, 'y': y, 'edge_res': edge_res, 'edge_patch': edge_patch, 'patch_feat': patch_feat}

def move_patch_batch(b):
    return {'mods': {k: v.to(S.DEVICE, non_blocking=True) for k, v in b['mods'].items()}, 'y': b['y'].to(S.DEVICE, non_blocking=True), 'edge_res': b['edge_res'].to(S.DEVICE, non_blocking=True), 'edge_patch': b['edge_patch'].to(S.DEVICE, non_blocking=True), 'patch_feat': b['patch_feat'].to(S.DEVICE, non_blocking=True)}

def predict_patch(model, dataset):
    loader = DataLoader(dataset, batch_size=S.PATCH_PROTEIN_BATCH, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_patch)
    pp = []
    yy = []
    model.eval()
    with torch.no_grad():
        for b in loader:
            b = move_patch_batch(b)
            p = torch.sigmoid(model(b['mods'], b['edge_res'], b['edge_patch'], b['patch_feat']))
            pp.append(p.detach().cpu().numpy())
            yy.append(b['y'].detach().cpu().numpy())
    return (np.concatenate(pp), np.concatenate(yy))

def train_patch_fold(arch, tr_ds, va_ds, fold):
    torch.manual_seed(S.SEED + 100 + fold)
    np.random.seed(S.SEED + 100 + fold)
    model = S.PATCH_FACTORY[arch](S.DIMS).to(S.DEVICE)
    ytr = np.concatenate(tr_ds.df.nuclear_binding.apply(lambda x: np.asarray(x, dtype=np.float32)).tolist())
    npos = max(float(ytr.sum()), 1.0)
    nneg = max(float(len(ytr) - ytr.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.sqrt(nneg / npos), dtype=torch.float32, device=S.DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=S.PATCH_LR, weight_decay=S.PATCH_WEIGHT_DECAY)
    loader = DataLoader(tr_ds, batch_size=S.PATCH_PROTEIN_BATCH, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_patch)
    best_ap = -1.0
    best_state = None
    bad = 0
    for epoch in range(1, S.PATCH_EPOCHS + 1):
        model.train()
        losses = []
        for b in loader:
            b = move_patch_batch(b)
            opt.zero_grad(set_to_none=True)
            logits = model(b['mods'], b['edge_res'], b['edge_patch'], b['patch_feat'])
            loss = criterion(logits, b['y'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        vp, vy = predict_patch(model, va_ds)
        val_ap = S.ap(vy, vp)
        if val_ap > best_ap + 1e-05:
            best_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch == 1 or epoch % 5 == 0:
            print(arch, 'fold', fold, 'epoch', epoch, 'loss', round(float(np.mean(losses)), 4), 'val_AP', round(val_ap, 5), 'best', round(best_ap, 5), 'alpha', round(float(torch.sigmoid(model.alpha_logit).detach().cpu()), 4))
        if bad >= S.PATCH_PATIENCE:
            break
    model.load_state_dict(best_state)
    return (model, best_state)

def run_patch_cv(arch):
    offsets = {}
    cursor = 0
    for r in S.train_df.itertuples():
        L = len(r.sequence)
        offsets[str(r.prot_id)] = (cursor, cursor + L)
        cursor += L
    oof = np.full(cursor, np.nan, dtype=np.float32)
    rows = []
    states = {}
    adir = S.PATCH_STATE_ROOT / arch
    adir.mkdir(parents=True, exist_ok=True)
    for fold, val_ids in S.OFFICIAL_FOLDS.items():
        val_set = set(val_ids)
        tr_df = S.train_df[~S.train_df.prot_id.isin(val_set)].copy()
        va_df = S.train_df[S.train_df.prot_id.isin(val_set)].copy()
        tr_ds = PatchProteinDataset('DevSet1010', tr_df)
        va_ds = PatchProteinDataset('DevSet1010', va_df)
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
            model, state = train_patch_fold(arch, tr_ds, va_ds, fold)
            pred, _ = predict_patch(model, va_ds)
            torch.save(state, state_fp)
            np.save(pred_fp, pred)
            meta_fp.write_text(json.dumps({'representation': S.BEST_REP_NAME, 'blocks': S.BEST_REP_BLOCKS, 'n_val': expected}, indent=2))
            del model
            torch.cuda.empty_cache()
        states[fold] = state
        pos = 0
        for r in va_df.itertuples():
            L = len(r.sequence)
            a, b = offsets[str(r.prot_id)]
            oof[a:b] = pred[pos:pos + L]
            pos += L
        yv = np.concatenate(va_df.nuclear_binding.apply(lambda x: np.asarray(x, dtype=np.float32)).tolist())
        rows.append({'architecture': arch, 'fold': fold, 'AUPRC': S.ap(yv, pred), 'n_val': len(pred), 'n_positive': int(yv.sum())})
    if not np.all(np.isfinite(oof)):
        raise RuntimeError(f'{arch}: incomplete patch OOF')
    f = pd.DataFrame(rows)
    yall = np.concatenate(S.train_df.nuclear_binding.apply(lambda x: np.asarray(x, dtype=np.float32)).tolist())
    return ({'architecture': arch, 'CV_AUPRC_mean': float(f.AUPRC.mean()), 'CV_AUPRC_std': float(f.AUPRC.std(ddof=1)), 'OOF_AUPRC': S.ap(yall, oof)}, f, oof, states)

# 14. BUILD / CACHE NUCLEAR CANDIDATE PATCH HYPERGRAPHS
def stage_23():
    S.PATCH_CACHE = S.CACHE_ROOT / 'patch_hypergraph_v1'
    S.PATCH_CACHE.mkdir(parents=True, exist_ok=True)
    S.PATCH_SEEDS = set('KRHNQSTY')
    S.PATCH_RAM = {}
    S.PATCH_AUDIT = []
    for S.split, S.df in S.SPLIT_DF.items():
        S.outdir = S.PATCH_CACHE / S.split
        S.outdir.mkdir(parents=True, exist_ok=True)
        for S.n, S.r in enumerate(S.df.itertuples(), start=1):
            S.fp = S.outdir / f'{S.r.prot_id}.npz'
            S.valid = False
            if S.fp.exists() and (not S.FORCE_REBUILD_PATCHES):
                try:
                    S.z = np.load(S.fp)
                    S.valid = S.z['patch_feat'].ndim == 2 and S.z['patch_feat'].shape[1] == 18 and (len(S.z['edge_res']) == len(S.z['edge_patch']))
                except Exception:
                    S.valid = False
            if not S.valid:
                S.er, S.ep, S.pf = build_patch_graph(S.split, str(S.r.prot_id), S.r.sequence)
                np.savez_compressed(S.fp, edge_res=S.er, edge_patch=S.ep, patch_feat=S.pf)
            S.z = np.load(S.fp)
            S.er = S.z['edge_res'].astype(np.int64)
            S.ep = S.z['edge_patch'].astype(np.int64)
            S.pf = S.z['patch_feat'].astype(np.float32)
            S.PATCH_RAM[S.split, str(S.r.prot_id)] = (S.er, S.ep, S.pf)
            S.covered = len(np.unique(S.er)) if len(S.er) else 0
            S.PATCH_AUDIT.append({'dataset': S.split, 'prot_id': str(S.r.prot_id), 'n_residues': len(S.r.sequence), 'n_candidate_patches': len(S.pf), 'n_memberships': len(S.er), 'covered_residue_fraction': S.covered / max(len(S.r.sequence), 1)})
            if S.n % 250 == 0:
                print(S.split, S.n, '/', len(S.df))
    S.patch_audit = pd.DataFrame(S.PATCH_AUDIT)
    S.patch_audit.to_csv(S.RUN_ROOT / 'patch_audit.csv', index=False)
    display(S.patch_audit.groupby('dataset').agg(proteins=('prot_id', 'nunique'), mean_patches=('n_candidate_patches', 'mean'), median_patches=('n_candidate_patches', 'median'), mean_memberships=('n_memberships', 'mean'), mean_covered_fraction=('covered_residue_fraction', 'mean')).reset_index())
    print('Patch hypergraph cache: PASS')

# 15. PATCH-HYPERGRAPH DATASET / MODEL
def stage_24():
    S.PATCH_FACTORY = {'NA_PatchHG_Gated': lambda dims: NuclearPatchHypergraph(dims, mode='gated', latent=S.PATCH_LATENT, moe=False), 'NA_PatchHG_Attention': lambda dims: NuclearPatchHypergraph(dims, mode='attention', latent=S.PATCH_LATENT, moe=False), 'NA_PatchHG_GatedMoE': lambda dims: NuclearPatchHypergraph(dims, mode='gated', latent=S.PATCH_LATENT, moe=True, experts=4)}
    print('Patch-hypergraph architectures:', list(S.PATCH_FACTORY))

# 16. PATCH-HYPERGRAPH CV TRAINER — RESUME SAFE
def stage_25():
    S.PATCH_STATE_ROOT = S.RUN_ROOT / 'patch_states'
    S.PATCH_STATE_ROOT.mkdir(parents=True, exist_ok=True)

# 17. RUN PATCH-HYPERGRAPH SCREEN — DEV ONLY
def stage_26():
    S.PATCH_RESULTS = {}
    S.PATCH_FOLDS = []
    S.PATCH_OOF = {}
    S.PATCH_STATES = {}
    if S.RUN_PATCH_HYPERGRAPH:
        for S.arch in S.PATCH_FACTORY:
            print('\n' + '=' * 80)
            print('PATCH HYPERGRAPH:', S.arch)
            print('=' * 80)
            S.s, S.f, S.o, S.st = run_patch_cv(S.arch)
            S.PATCH_RESULTS[S.arch] = S.s
            S.PATCH_FOLDS.append(S.f)
            S.PATCH_OOF[S.arch] = S.o
            S.PATCH_STATES[S.arch] = S.st
    S.patch_summary = pd.DataFrame(S.PATCH_RESULTS.values()).sort_values('CV_AUPRC_mean', ascending=False).reset_index(drop=True)
    S.patch_folds = pd.concat(S.PATCH_FOLDS, ignore_index=True)
    S.patch_summary.to_csv(S.RUN_ROOT / 'patch_hypergraph_cv.csv', index=False)
    S.patch_folds.to_csv(S.RUN_ROOT / 'patch_hypergraph_folds.csv', index=False)
    display(S.patch_summary)
    S.BEST_PATCH = str(S.patch_summary.iloc[0].architecture)
    S.BEST_PATCH_SCORE = float(S.patch_summary.iloc[0].CV_AUPRC_mean)
    S.alphas = []
    for S.fold, S.state in S.PATCH_STATES[S.BEST_PATCH].items():
        if 'alpha_logit' in S.state:
            S.alphas.append(float(torch.sigmoid(S.state['alpha_logit']).item()))
    print('BEST PATCH MODEL:', S.BEST_PATCH)
    print('Dev CV AUPRC:', S.BEST_PATCH_SCORE)
    print('Residual alpha:', S.alphas)
    print('Mean alpha:', float(np.mean(S.alphas)) if S.alphas else None)
    (S.RUN_ROOT / 'locked_patch_model.json').write_text(json.dumps({'architecture': S.BEST_PATCH, 'dev_cv_auprc': S.BEST_PATCH_SCORE, 'alpha_by_fold': S.alphas, 'alpha_mean': float(np.mean(S.alphas)) if S.alphas else None}, indent=2))

STAGES = [stage_23, stage_24, stage_25, stage_26]

