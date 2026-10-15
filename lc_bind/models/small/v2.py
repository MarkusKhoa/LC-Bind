"""Extracted from SMALL_ULTIMATE_V1_DATA_DRIVEN_POCKET-3.ipynb. Section: v2."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Chain, Counter, DataLoader, Dataset, F, Model, PDBIO, PDBParser, Path, PocketProteinDataset, SMV2Dataset, SMV2ResidueEncoder, SmallPocketHypergraph, SmallPocketHypergraphMS, SmallPocketV2, Structure, Subset, accuracy_score, ast, average_precision_score, balanced_accuracy_score, best_mcc_threshold, binary_metrics, cKDTree, concat_blocks, deepcopy, defaultdict, f1_score, hashlib, json, math, matthews_corrcoef, nn, np, os, pd, platform, pocket_static_features, precision_score, predict_pocket, recall_score, roc_auc_score, run_pocket_cv, seq1, shutil, structure_coords, subprocess, sys, tarfile, torch
    import math
    from pathlib import Path
    from collections import Counter
    import json
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from scipy.spatial import cKDTree
    import json, numpy as np, pandas as pd, torch
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, matthews_corrcoef, precision_score, recall_score
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
    import hashlib
    import os
    from lc_bind.utils.metrics import best_mcc_threshold, binary_metrics
    from lc_bind.models.small.pocket import PocketProteinDataset, SmallPocketHypergraph, pocket_static_features, predict_pocket, run_pocket_cv, structure_coords
    from lc_bind.models.small.screen import concat_blocks

    class SMV2ResidueEncoder(nn.Module):
    
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

    class SmallPocketV2(nn.Module):
    
        def __init__(self, dims, message_mode='none', latent=192, heads=6, drop=0.2):
            super().__init__()
            assert message_mode in {'none', 'mean', 'cross'}
            self.message_mode = message_mode
            self.latent = latent
            self.residue_encoder = SMV2ResidueEncoder(dims, latent=latent, heads=heads, drop=drop)
            self.pocket_static = nn.Sequential(nn.LayerNorm(22), nn.Linear(22, 64), nn.GELU())
            self.pocket_encoder = nn.Sequential(nn.Linear(latent * 2 + 64, latent), nn.GELU(), nn.Dropout(drop), nn.Linear(latent, latent), nn.GELU())
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
    
        def forward(self, mods, edge_res, edge_pocket, pocket_feat, return_aux=False):
            h = self.residue_encoder(mods)
            gate_mean = h.new_tensor(0.0)
            if self.message_mode != 'none' and pocket_feat.shape[0] > 0 and (edge_res.numel() > 0):
                nres = h.shape[0]
                npocket = pocket_feat.shape[0]
                mean = self.index_mean(h[edge_res], edge_pocket, npocket)
                mx = self.index_max(h[edge_res], edge_pocket, npocket)
                stat = self.pocket_static(pocket_feat)
                hp = self.pocket_encoder(torch.cat([mean, mx, stat], dim=-1))
                if self.message_mode == 'mean':
                    msg = self.index_mean(hp[edge_pocket], edge_res, nres)
                else:
                    q = self.q(h[edge_res])
                    k = self.k(hp[edge_pocket])
                    score = (q * k).sum(dim=-1) / math.sqrt(self.latent)
                    base = score.new_full((nres,), -1000000000.0)
                    max_r = base.scatter_reduce(0, edge_res, score, reduce='amax', include_self=True)
                    max_r = torch.where(max_r < -100000000.0, torch.zeros_like(max_r), max_r)
                    w = torch.exp(score - max_r[edge_res])
                    denom = w.new_zeros((nres, 1)).index_add(0, edge_res, w[:, None])
                    weighted = self.v(hp[edge_pocket]) * w[:, None]
                    msg = h.new_zeros((nres, self.latent)).index_add(0, edge_res, weighted)
                    msg = msg / denom.clamp_min(1e-08)
                incident = h.new_zeros((nres, 1)).index_add(0, edge_res, h.new_ones((len(edge_res), 1)))
                g = torch.sigmoid(self.gate(torch.cat([h, msg], dim=-1)))
                g = g * (incident > 0).float()
                h = h + g * self.delta(msg)
                active = incident.squeeze(-1) > 0
                if active.any():
                    gate_mean = g[active].mean()
            logits = self.head(h).squeeze(-1)
            if return_aux:
                return (logits, {'gate_mean': gate_mean})
            return logits

    class SMV2Dataset(Dataset):
    
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

    class SmallPocketHypergraphMS(SmallPocketHypergraph):
    
        def __init__(self, dims, mode='attention', latent=192, drop=0.2):
            super().__init__(dims, mode=mode, latent=latent, moe=False, drop=drop)
            self.pocket_static = nn.Sequential(nn.LayerNorm(23), nn.Linear(23, 64), nn.GELU())

def smv2_collate(batch):
    keys = sorted(batch[0]['mods'])
    mods = {k: torch.from_numpy(np.concatenate([b['mods'][k] for b in batch], axis=0)).float() for k in keys}
    y = torch.from_numpy(np.concatenate([b['y'] for b in batch])).float()
    protein_index = []
    pf_all = []
    er_all = []
    ep_all = []
    roff = 0
    poff = 0
    for pidx, b in enumerate(batch):
        L = len(b['y'])
        P = len(b['pocket_feat'])
        protein_index.append(torch.full((L,), pidx, dtype=torch.long))
        if P:
            pf_all.append(torch.from_numpy(b['pocket_feat']).float())
        if len(b['edge_res']):
            er_all.append(torch.from_numpy(b['edge_res']).long() + roff)
            ep_all.append(torch.from_numpy(b['edge_pocket']).long() + poff)
        roff += L
        poff += P
    return {'mods': mods, 'y': y, 'protein_index': torch.cat(protein_index, dim=0), 'edge_res': torch.cat(er_all, dim=0) if er_all else torch.zeros((0,), dtype=torch.long), 'edge_pocket': torch.cat(ep_all, dim=0) if ep_all else torch.zeros((0,), dtype=torch.long), 'pocket_feat': torch.cat(pf_all, dim=0) if pf_all else torch.zeros((0, 22), dtype=torch.float32)}

def smv2_move(b):
    return {'mods': {k: v.to(S.DEVICE, non_blocking=True) for k, v in b['mods'].items()}, 'y': b['y'].to(S.DEVICE, non_blocking=True), 'protein_index': b['protein_index'].to(S.DEVICE, non_blocking=True), 'edge_res': b['edge_res'].to(S.DEVICE, non_blocking=True), 'edge_pocket': b['edge_pocket'].to(S.DEVICE, non_blocking=True), 'pocket_feat': b['pocket_feat'].to(S.DEVICE, non_blocking=True)}

def smv2_loss(logits, y, protein_index, pos_weight, hardneg):
    raw = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction='none')
    nprot = int(protein_index.max().item()) + 1
    sums = raw.new_zeros((nprot,)).index_add(0, protein_index, raw)
    counts = raw.new_zeros((nprot,)).index_add(0, protein_index, raw.new_ones(len(raw)))
    base = (sums / counts.clamp_min(1.0)).mean()
    if not hardneg:
        return (base, raw.new_tensor(0.0))
    terms = []
    for p in range(nprot):
        mask = protein_index == p
        lp = logits[mask]
        yp = y[mask] > 0.5
        if not yp.any() or (~yp).sum() == 0:
            continue
        pos_logits = lp[yp]
        neg_logits = lp[~yp]
        k = min(max(8, int(2 * len(pos_logits))), 48, len(neg_logits))
        hard = torch.topk(neg_logits, k=k).values
        pos_anchor = pos_logits.mean()
        terms.append(F.softplus(hard - pos_anchor + S.SMV2_HARDNEG_MARGIN).mean())
    rank = torch.stack(terms).mean() if terms else base.new_tensor(0.0)
    return (base + S.SMV2_HARDNEG_LAMBDA * rank, rank)

def smv2_predict(model, ds):
    loader = DataLoader(ds, batch_size=S.SMV2_BATCH, shuffle=False, num_workers=0, pin_memory=S.DEVICE.type == 'cuda', collate_fn=smv2_collate)
    pp = []
    yy = []
    gg = []
    model.eval()
    with torch.no_grad():
        for b in loader:
            b = smv2_move(b)
            logits, aux = model(b['mods'], b['edge_res'], b['edge_pocket'], b['pocket_feat'], return_aux=True)
            pp.append(torch.sigmoid(logits).cpu().numpy())
            yy.append(b['y'].cpu().numpy())
            gg.append(float(aux['gate_mean'].detach().cpu()))
    return (np.concatenate(pp), np.concatenate(yy), float(np.mean(gg)))

def smv2_train_fold(variant, tr_ds, va_ds, fold):
    cfg = S.SMV2_VARIANTS[variant]
    torch.manual_seed(S.SEED + 900 + fold)
    np.random.seed(S.SEED + 900 + fold)
    model = SmallPocketV2(S.DIMS, message_mode=cfg['mode'], latent=192, heads=6, drop=0.2).to(S.DEVICE)
    ytr = np.concatenate(tr_ds.df.small_binding.apply(lambda x: np.asarray(x, dtype=np.float32)).tolist())
    npos = max(float(ytr.sum()), 1.0)
    nneg = max(float(len(ytr) - ytr.sum()), 1.0)
    pos_weight = torch.tensor(np.sqrt(nneg / npos), dtype=torch.float32, device=S.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=S.SMV2_LR, weight_decay=S.SMV2_WD)
    loader = DataLoader(tr_ds, batch_size=S.SMV2_BATCH, shuffle=True, num_workers=0, pin_memory=S.DEVICE.type == 'cuda', collate_fn=smv2_collate)
    best_ap = -1.0
    best_state = None
    bad = 0
    for epoch in range(1, S.SMV2_EPOCHS + 1):
        model.train()
        losses = []
        ranks = []
        for b in loader:
            b = smv2_move(b)
            opt.zero_grad(set_to_none=True)
            logits = model(b['mods'], b['edge_res'], b['edge_pocket'], b['pocket_feat'])
            loss, rank = smv2_loss(logits, b['y'], b['protein_index'], pos_weight, cfg['hardneg'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            ranks.append(float(rank.detach().cpu()))
        vp, vy, gate = smv2_predict(model, va_ds)
        vap = float(average_precision_score(vy, vp))
        if vap > best_ap + 1e-05:
            best_ap = vap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch == 1 or epoch % 5 == 0:
            print(variant, 'fold', fold, 'epoch', epoch, 'loss', round(float(np.mean(losses)), 4), 'rank', round(float(np.mean(ranks)), 4), 'val_AP', round(vap, 5), 'gate', round(gate, 4))
        if bad >= S.SMV2_PATIENCE:
            break
    model.load_state_dict(best_state)
    return (model, best_state)

def smv2_run_cv(variant):
    offsets = {}
    cursor = 0
    for r in S.train_df.itertuples():
        L = len(r.sequence)
        offsets[str(r.prot_id)] = (cursor, cursor + L)
        cursor += L
    oof = np.full(cursor, np.nan, dtype=np.float32)
    rows = []
    states = {}
    adir = S.SMV2_STATE_ROOT / variant
    adir.mkdir(parents=True, exist_ok=True)
    for fold, val_ids in S.OFFICIAL_FOLDS.items():
        val_set = set(val_ids)
        tr_df = S.train_df[~S.train_df.prot_id.isin(val_set)].copy()
        va_df = S.train_df[S.train_df.prot_id.isin(val_set)].copy()
        tr_ds = SMV2Dataset('DevSet1010', tr_df)
        va_ds = SMV2Dataset('DevSet1010', va_df)
        state_fp = adir / f'fold{fold}.pt'
        pred_fp = adir / f'fold{fold}_val.npy'
        meta_fp = adir / f'fold{fold}.json'
        expected = int(va_df.sequence.str.len().sum())
        reuse = False
        if state_fp.exists() and pred_fp.exists() and meta_fp.exists():
            try:
                meta = json.loads(meta_fp.read_text())
                pred = np.load(pred_fp)
                reuse = meta.get('code_version') == S.SMV2_CODE_VERSION and meta.get('n_val') == expected and (pred.shape == (expected,)) and np.all(np.isfinite(pred))
            except Exception:
                reuse = False
        if reuse:
            print(variant, 'fold', fold, 'REUSE')
            state = torch.load(state_fp, map_location='cpu')
            pred = np.load(pred_fp).astype(np.float32)
            model = SmallPocketV2(S.DIMS, message_mode=S.SMV2_VARIANTS[variant]['mode']).to(S.DEVICE)
            model.load_state_dict(state)
            _, _, gate = smv2_predict(model, va_ds)
            del model
        else:
            print(variant, 'fold', fold, 'TRAIN')
            model, state = smv2_train_fold(variant, tr_ds, va_ds, fold)
            pred, _, gate = smv2_predict(model, va_ds)
            torch.save(state, state_fp)
            np.save(pred_fp, pred.astype(np.float32))
            meta_fp.write_text(json.dumps({'code_version': S.SMV2_CODE_VERSION, 'n_val': expected, 'message_mode': S.SMV2_VARIANTS[variant]['mode'], 'hardneg': S.SMV2_VARIANTS[variant]['hardneg']}, indent=2))
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
        rows.append({'variant': variant, 'fold': fold, 'AUPRC': float(average_precision_score(yv, pred)), 'n_positive': int(yv.sum()), 'gate_mean': gate})
    if not np.all(np.isfinite(oof)):
        raise RuntimeError(f'{variant}: incomplete OOF')
    f = pd.DataFrame(rows)
    yall = np.concatenate(S.train_df.small_binding.apply(lambda x: np.asarray(x, dtype=np.float32)).tolist())
    return ({'variant': variant, 'CV_AUPRC_mean': float(f.AUPRC.mean()), 'CV_AUPRC_std': float(f.AUPRC.std(ddof=1)), 'OOF_AUPRC': float(average_precision_score(yall, oof)), 'gate_mean': float(f.gate_mean.mean())}, f, oof, states)

def smv2score(name):
    return float(S.smv2_summary.loc[S.smv2_summary.variant.eq(name), 'CV_AUPRC_mean'].iloc[0])

def smv2_best_mcc_threshold(y, p):
    best = (-999, None)
    for t in np.linspace(0.01, 0.99, 199):
        pred = (p >= t).astype(int)
        m = matthews_corrcoef(y, pred)
        if m > best[0]:
            best = (m, float(t))
    return best[1]

def smv2_predict_external(split):
    ds = SMV2Dataset(split, S.SPLIT_DF[split])
    pp = []
    for fold, state in S.SMV2_STATES[S.SMV2_LOCKED].items():
        model = SmallPocketV2(S.DIMS, message_mode=S.SMV2_VARIANTS[S.SMV2_LOCKED]['mode'], latent=192, heads=6, drop=0.2).to(S.DEVICE)
        model.load_state_dict(state)
        p, _, _ = smv2_predict(model, ds)
        pp.append(p)
        del model
        if S.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    return np.mean(np.stack(pp), axis=0)

def smv2_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {'AUPRC': float(average_precision_score(y, p)), 'AUROC': float(roc_auc_score(y, p)), 'precision': float(precision_score(y, pred, zero_division=0)), 'recall': float(recall_score(y, pred, zero_division=0)), 'F1': float(f1_score(y, pred, zero_division=0)), 'MCC': float(matthews_corrcoef(y, pred))}

def _jaccard(a, b):
    a = set(a)
    b = set(b)
    return len(a & b) / max(len(a | b), 1)

def _build_multiscale_pockets(split, pid, seq):
    coords = structure_coords(split, pid, seq)
    tree = cKDTree(coords)
    candidates = []
    for seed, aa in enumerate(seq):
        if aa not in S.POCKET_SEEDS:
            continue
        for radius in S.SMMS_RADII:
            members = tuple(sorted(tree.query_ball_point(coords[seed], radius)))
            if len(members) < 4:
                continue
            feat22 = pocket_static_features(seq, coords, members)
            feat23 = np.concatenate([feat22.astype(np.float32), np.array([radius / 12.0], dtype=np.float32)])
            score = 1.4 * feat22[1] + 0.8 * feat22[2] + 0.8 * feat22[3] + 0.5 * feat22[6] + 0.5 * feat22[7] - 0.5 * feat22[5] + 0.25 * feat22[0] + 0.15 * feat22[17] - 0.05 * (radius / 12.0)
            candidates.append((float(score), members, feat23))
    unique = {}
    for score, members, feat in candidates:
        if members not in unique or score > unique[members][0]:
            unique[members] = (score, feat)
    ranked = sorted([(s, m, f) for m, (s, f) in unique.items()], key=lambda x: x[0], reverse=True)
    selected = []
    for item in ranked:
        score, members, feat = item
        if any((_jaccard(members, old_members) >= S.SMMS_NMS for _, old_members, _ in selected)):
            continue
        selected.append(item)
        if len(selected) >= S.SMMS_MAX:
            break
    edge_res = []
    edge_pocket = []
    pocket_feat = []
    for pidx, (_, members, feat) in enumerate(selected):
        pocket_feat.append(feat)
        for rid in members:
            edge_res.append(int(rid))
            edge_pocket.append(int(pidx))
    return (np.asarray(edge_res, dtype=np.int64), np.asarray(edge_pocket, dtype=np.int64), np.asarray(pocket_feat, dtype=np.float32).reshape(-1, 23))

def collate_pocket_ms(batch):
    keys = sorted(batch[0]['mods'])
    mods = {k: torch.from_numpy(np.concatenate([b['mods'][k] for b in batch], axis=0)).float() for k in keys}
    y = torch.from_numpy(np.concatenate([b['y'] for b in batch])).float()
    pf_all = []
    er_all = []
    ep_all = []
    roff = 0
    poff = 0
    for b in batch:
        L = len(b['y'])
        P = len(b['pocket_feat'])
        if P:
            pf_all.append(torch.from_numpy(b['pocket_feat']).float())
        if len(b['edge_res']):
            er_all.append(torch.from_numpy(b['edge_res']).long() + roff)
            ep_all.append(torch.from_numpy(b['edge_pocket']).long() + poff)
        roff += L
        poff += P
    return {'mods': mods, 'y': y, 'edge_res': torch.cat(er_all, 0) if er_all else torch.zeros((0,), dtype=torch.long), 'edge_pocket': torch.cat(ep_all, 0) if ep_all else torch.zeros((0,), dtype=torch.long), 'pocket_feat': torch.cat(pf_all, 0) if pf_all else torch.zeros((0, 23), dtype=torch.float32)}

def best_mcc_thr(y, p):
    best = (-999, 0.5)
    for t in np.linspace(0.01, 0.99, 199):
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > best[0]:
            best = (m, float(t))
    return (best[1], best[0])

def predict_v1(split):
    ds = PocketProteinDataset(split, S.SPLIT_DF[split])
    pred = []
    for fold, state in S.POCKET_STATES[S.V1_NAME].items():
        model = S.POCKET_FACTORY[S.V1_NAME](S.DIMS).to(S.DEVICE)
        model.load_state_dict(state)
        p, _ = predict_pocket(model, ds)
        pred.append(p)
        del model
        if S.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    return np.mean(np.stack(pred), axis=0)

def predict_ms(split):
    old_ram, old_fac, old_col = (S.POCKET_RAM, S.POCKET_FACTORY, S.collate_pocket)
    try:
        S.POCKET_RAM = S.SMMS_RAM
        S.POCKET_FACTORY = {'SM_PocketHG_Attention_MS': lambda dims: SmallPocketHypergraphMS(dims, mode='attention', latent=S.POCKET_LATENT, drop=0.2)}
        S.collate_pocket = collate_pocket_ms
        ds = PocketProteinDataset(split, S.SPLIT_DF[split])
        pred = []
        for fold, state in S.ms_states.items():
            model = S.POCKET_FACTORY['SM_PocketHG_Attention_MS'](S.DIMS).to(S.DEVICE)
            model.load_state_dict(state)
            p, _ = predict_pocket(model, ds)
            pred.append(p)
            del model
            if S.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        return np.mean(np.stack(pred), axis=0)
    finally:
        S.POCKET_RAM, S.POCKET_FACTORY, S.collate_pocket = (old_ram, old_fac, old_col)

def global_metrics(y, p, t=0.5):
    pr = (p >= t).astype(int)
    return {'AUPRC': float(average_precision_score(y, p)), 'AUROC': float(roc_auc_score(y, p)), 'precision': float(precision_score(y, pr, zero_division=0)), 'recall': float(recall_score(y, pr, zero_division=0)), 'F1': float(f1_score(y, pr, zero_division=0)), 'MCC': float(matthews_corrcoef(y, pr))}

def protein_metrics(df, p, t=0.5):
    f1s, mccs = ([], [])
    pos = 0
    for r in df.itertuples():
        L = len(r.sequence)
        y = np.asarray(r.small_binding, dtype=int)
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
    ds = PocketProteinDataset(split, S.SPLIT_DF[split])
    preds = []
    for fold, state in S.POCKET_STATES[S.V1_NAME].items():
        model = S.POCKET_FACTORY[S.V1_NAME](S.DIMS).to(S.DEVICE)
        model.load_state_dict(state)
        p, _ = predict_pocket(model, ds)
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
    old_ram = S.POCKET_RAM
    old_factory = S.POCKET_FACTORY
    old_collate = S.collate_pocket
    try:
        S.POCKET_RAM = S.SMMS_RAM
        S.POCKET_FACTORY = {S.MS_NAME: lambda dims: SmallPocketHypergraphMS(dims, mode='attention', latent=S.POCKET_LATENT, drop=0.2)}
        S.collate_pocket = collate_pocket_ms
        ds = PocketProteinDataset(split, S.SPLIT_DF[split])
        preds = []
        for fold, state in S.ms_states.items():
            model = S.POCKET_FACTORY[S.MS_NAME](S.DIMS).to(S.DEVICE)
            model.load_state_dict(state)
            p, _ = predict_pocket(model, ds)
            preds.append(p)
            del model
            if S.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        p = np.mean(np.stack(preds), axis=0)
        _save_cached('MS_Attention', split, p)
        return p.astype(np.float64)
    finally:
        S.POCKET_RAM = old_ram
        S.POCKET_FACTORY = old_factory
        S.collate_pocket = old_collate

def _first_existing(row, candidates, default=np.nan):
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return float(row[c])
    return default

# SMALL V2-A.
def stage_37():
    S.SMV2_RUN_ROOT = S.RESULTS_ROOT / 'SMALL_V2_MATCHED_POCKET_REASONING'
    S.SMV2_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    S.rows = []
    S.split = 'DevSet1010'
    S.df = S.SPLIT_DF[S.split]
    S.all_y = []
    S.all_cov = []
    S.per_protein = []
    for S.r in S.df.itertuples():
        S.pid = str(S.r.prot_id)
        S.y = np.asarray(S.r.small_binding, dtype=np.int8)
        S.er, S.ep, S.pf = S.POCKET_RAM[S.split, S.pid]
        S.covered = np.zeros(len(S.y), dtype=bool)
        if len(S.er):
            S.covered[np.unique(S.er)] = True
        S.all_y.append(S.y)
        S.all_cov.append(S.covered)
        if S.y.sum() > 0:
            S.per_protein.append({'prot_id': S.pid, 'n_positive': int(S.y.sum()), 'positive_recall': float(S.covered[S.y == 1].mean()), 'overall_coverage': float(S.covered.mean())})
    S.y = np.concatenate(S.all_y)
    S.cov = np.concatenate(S.all_cov)
    S.prevalence = float(S.y.mean())
    S.covered_prevalence = float(S.y[S.cov].mean())
    S.positive_coverage = float(S.cov[S.y == 1].mean())
    S.negative_coverage = float(S.cov[S.y == 0].mean())
    S.audit = pd.DataFrame([{'dataset': 'DevSet1010', 'global_prevalence': S.prevalence, 'overall_residue_coverage': float(S.cov.mean()), 'positive_residue_coverage': S.positive_coverage, 'negative_residue_coverage': S.negative_coverage, 'P_positive_given_covered': S.covered_prevalence, 'covered_enrichment_vs_global': S.covered_prevalence / max(S.prevalence, 1e-12), 'coverage_odds_ratio_pos_vs_neg': S.positive_coverage / max(1 - S.positive_coverage, 1e-12) / (S.negative_coverage / max(1 - S.negative_coverage, 1e-12))}])
    S.positive_protein_audit = pd.DataFrame(S.per_protein)
    print('CURRENT POCKET CANDIDATE AUDIT')
    display(S.audit)
    print('Positive-protein pocket recall:')
    display(S.positive_protein_audit[['positive_recall', 'overall_coverage']].describe())
    S.audit.to_csv(S.SMV2_RUN_ROOT / 'current_pocket_candidate_audit.csv', index=False)
    S.positive_protein_audit.to_csv(S.SMV2_RUN_ROOT / 'current_pocket_positive_protein_audit.csv', index=False)

# SMALL V2-C.
def stage_39():
    S.SMV2_HARDNEG_LAMBDA = 0.15
    S.SMV2_HARDNEG_MARGIN = 0.5

# SMALL V2-D.
def stage_40():
    S.SMV2_EPOCHS = 36
    S.SMV2_PATIENCE = 6
    S.SMV2_BATCH = 4
    S.SMV2_LR = 0.0003
    S.SMV2_WD = 0.0001
    S.SMV2_CODE_VERSION = 'small_v2_matched_pocket_reasoning_r1'
    S.SMV2_VARIANTS = {'NoPocket_BCE': {'mode': 'none', 'hardneg': False}, 'NoPocket_HardNeg': {'mode': 'none', 'hardneg': True}, 'MeanPocket_BCE': {'mode': 'mean', 'hardneg': False}, 'MeanPocket_HardNeg': {'mode': 'mean', 'hardneg': True}, 'CrossPocket_BCE': {'mode': 'cross', 'hardneg': False}, 'CrossPocket_HardNeg': {'mode': 'cross', 'hardneg': True}}
    S.SMV2_STATE_ROOT = S.SMV2_RUN_ROOT / 'states'
    S.SMV2_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    S.SMV2_RESULTS = {}
    S.SMV2_FOLDS = []
    S.SMV2_OOF = {}
    S.SMV2_STATES = {}
    for S.variant in S.SMV2_VARIANTS:
        print('\n' + '=' * 88)
        print('SMALL V2:', S.variant)
        print('=' * 88)
        S.s, S.f, S.o, S.st = smv2_run_cv(S.variant)
        S.SMV2_RESULTS[S.variant] = S.s
        S.SMV2_FOLDS.append(S.f)
        S.SMV2_OOF[S.variant] = S.o
        S.SMV2_STATES[S.variant] = S.st
    S.smv2_summary = pd.DataFrame(S.SMV2_RESULTS.values()).sort_values('CV_AUPRC_mean', ascending=False).reset_index(drop=True)
    S.smv2_folds = pd.concat(S.SMV2_FOLDS, ignore_index=True)
    display(S.smv2_summary)
    S.smv2_summary.to_csv(S.SMV2_RUN_ROOT / 'v2_cv_summary.csv', index=False)
    S.smv2_folds.to_csv(S.SMV2_RUN_ROOT / 'v2_cv_folds.csv', index=False)
    S.matched = pd.DataFrame([{'comparison': 'Mean pocket effect under BCE', 'delta_AUPRC': smv2score('MeanPocket_BCE') - smv2score('NoPocket_BCE')}, {'comparison': 'Cross-attention pocket effect under BCE', 'delta_AUPRC': smv2score('CrossPocket_BCE') - smv2score('NoPocket_BCE')}, {'comparison': 'Hard-negative effect without pocket', 'delta_AUPRC': smv2score('NoPocket_HardNeg') - smv2score('NoPocket_BCE')}, {'comparison': 'Hard-negative effect with mean pocket', 'delta_AUPRC': smv2score('MeanPocket_HardNeg') - smv2score('MeanPocket_BCE')}, {'comparison': 'Hard-negative effect with cross pocket', 'delta_AUPRC': smv2score('CrossPocket_HardNeg') - smv2score('CrossPocket_BCE')}, {'comparison': 'Cross vs mean under BCE', 'delta_AUPRC': smv2score('CrossPocket_BCE') - smv2score('MeanPocket_BCE')}, {'comparison': 'Cross vs mean under HardNeg', 'delta_AUPRC': smv2score('CrossPocket_HardNeg') - smv2score('MeanPocket_HardNeg')}])
    display(S.matched)
    S.matched.to_csv(S.SMV2_RUN_ROOT / 'matched_ablation.csv', index=False)
    S.complexity = {'NoPocket_BCE': 0, 'NoPocket_HardNeg': 1, 'MeanPocket_BCE': 2, 'MeanPocket_HardNeg': 3, 'CrossPocket_BCE': 4, 'CrossPocket_HardNeg': 5}
    S.ordered = sorted(S.SMV2_VARIANTS, key=lambda x: S.complexity[x])
    S.winner = S.ordered[0]
    for S.name in S.ordered[1:]:
        if smv2score(S.name) - smv2score(S.winner) >= 0.002:
            S.winner = S.name
    S.SMV2_LOCKED = S.winner
    S.SMV2_LOCKED_SCORE = smv2score(S.winner)
    print('\nSMALL V2 LOCKED:', S.SMV2_LOCKED, S.SMV2_LOCKED_SCORE)

# SMALL V2-E.
def stage_41():
    S.dev_y_smv2 = np.concatenate(S.train_df.small_binding.apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
    S.SMV2_THR = smv2_best_mcc_threshold(S.dev_y_smv2, S.SMV2_OOF[S.SMV2_LOCKED])
    S.ext = []
    for S.split in ['TestSet300', 'TestSetNew46']:
        S.y = np.concatenate(S.SPLIT_DF[S.split].small_binding.apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
        S.p = smv2_predict_external(S.split)
        S.m05 = smv2_metrics(S.y, S.p, 0.5)
        S.mdev = smv2_metrics(S.y, S.p, S.SMV2_THR)
        S.row = {'dataset': S.split, 'variant': S.SMV2_LOCKED, 'Dev_CV_AUPRC': S.SMV2_LOCKED_SCORE, 'Dev_threshold': S.SMV2_THR}
        S.row.update({f'{k}_fixed0.5': v for k, v in S.m05.items()})
        S.row.update({f'{k}_OOFthr': v for k, v in S.mdev.items()})
        S.ext.append(S.row)
    S.smv2_external = pd.DataFrame(S.ext)
    display(S.smv2_external)
    S.smv2_external.to_csv(S.SMV2_RUN_ROOT / 'FINAL_V2_EXTERNAL.csv', index=False)
    S.compare = []
    for S._, S.r in S.headline.iterrows():
        S.compare.append({'dataset': S.r['dataset'], 'method': f'V1 {S.GLOBAL_MODEL}', 'AUPRC': float(S.r['AUPRC_fixed0.5']), 'AUROC': float(S.r['AUROC_fixed0.5']), 'F1': float(S.r['F1_fixed0.5']), 'MCC': float(S.r['MCC_fixed0.5'])})
    for S._, S.r in S.headline.iterrows():
        S.compare.append({'dataset': S.r['dataset'], 'method': f'V1 {S.GLOBAL_MODEL}', 'AUPRC': float(S.r['AUPRC_fixed0.5']), 'AUROC': float(S.r['AUROC_fixed0.5']), 'F1': float(S.r['F1_fixed0.5']), 'MCC': float(S.r['MCC_fixed0.5'])})
    S.baseline_fp = S.BASELINE_WORK / 'results' / 'FULL_DIRECT_BASELINE_METRICS_300_46.csv'
    if S.baseline_fp.exists():
        S.bb = pd.read_csv(S.baseline_fp)
        if 'label' in S.bb.columns:
            S.bb = S.bb[S.bb.label.eq('small_binding')].copy()
        if 'operating_point' in S.bb.columns:
            S.bb = S.bb[S.bb.operating_point.eq('fixed_0.5')].copy()
        for S._, S.r in S.bb.iterrows():
            if str(S.r.method) not in {'bindNode24', 'bindEmbed21DL'}:
                continue
            S.apcol = 'AP_AUPRC' if 'AP_AUPRC' in S.bb.columns else 'AUPRC'
            S.compare.append({'dataset': str(S.r.dataset), 'method': str(S.r.method), 'AUPRC': float(S.r[S.apcol]), 'AUROC': float(S.r.AUROC), 'F1': float(S.r.global_positive_F1), 'MCC': float(S.r.global_MCC)})
    S.smv2_compare = pd.DataFrame(S.compare).sort_values(['dataset', 'AUPRC'], ascending=[True, False])
    display(S.smv2_compare)
    S.smv2_compare.to_csv(S.SMV2_RUN_ROOT / 'FINAL_V2_COMPARISON.csv', index=False)
    print('\nSaved:', S.SMV2_RUN_ROOT)

# SMALL FINAL EXPERIMENT
def stage_42():
    S.SMMS_ROOT = S.RESULTS_ROOT / 'SMALL_V1_ATTENTION_MULTISCALE_POCKETS'
    S.SMMS_ROOT.mkdir(parents=True, exist_ok=True)
    S.SMMS_CACHE = S.CACHE_ROOT / 'small_multiscale_pockets_v1'
    S.SMMS_CACHE.mkdir(parents=True, exist_ok=True)
    S.SMMS_RADII = (6.0, 8.0, 10.0, 12.0)
    S.SMMS_MAX = 64
    S.SMMS_NMS = 0.65
    S.SMMS_FORCE = False
    S.SMMS_RAM = {}
    S.audit_rows = []
    for S.split, S.df in S.SPLIT_DF.items():
        S.od = S.SMMS_CACHE / S.split
        S.od.mkdir(parents=True, exist_ok=True)
        for S.n, S.r in enumerate(S.df.itertuples(), start=1):
            S.fp = S.od / f'{S.r.prot_id}.npz'
            S.good = False
            if S.fp.exists() and (not S.SMMS_FORCE):
                try:
                    S.z = np.load(S.fp)
                    S.good = S.z['pocket_feat'].ndim == 2 and S.z['pocket_feat'].shape[1] == 23 and (len(S.z['edge_res']) == len(S.z['edge_pocket']))
                except Exception:
                    S.good = False
            if not S.good:
                S.er, S.ep, S.pf = _build_multiscale_pockets(S.split, str(S.r.prot_id), S.r.sequence)
                np.savez_compressed(S.fp, edge_res=S.er, edge_pocket=S.ep, pocket_feat=S.pf)
            S.z = np.load(S.fp)
            S.er = S.z['edge_res'].astype(np.int64)
            S.ep = S.z['edge_pocket'].astype(np.int64)
            S.pf = S.z['pocket_feat'].astype(np.float32)
            S.SMMS_RAM[S.split, str(S.r.prot_id)] = (S.er, S.ep, S.pf)
            S.covered = len(np.unique(S.er)) if len(S.er) else 0
            S.audit_rows.append({'dataset': S.split, 'prot_id': str(S.r.prot_id), 'n_pockets': len(S.pf), 'coverage': S.covered / max(len(S.r.sequence), 1), 'memberships': len(S.er)})
            if S.n % 250 == 0:
                print(S.split, S.n, '/', len(S.df))
    S.smms_audit = pd.DataFrame(S.audit_rows)
    print('\nMULTI-SCALE POCKET AUDIT')
    display(S.smms_audit.groupby('dataset').agg(mean_pockets=('n_pockets', 'mean'), median_pockets=('n_pockets', 'median'), mean_coverage=('coverage', 'mean'), mean_memberships=('memberships', 'mean')).reset_index())
    S.smms_audit.to_csv(S.SMMS_ROOT / 'pocket_audit.csv', index=False)
    S._OLD_POCKET_RAM = S.POCKET_RAM
    S._OLD_POCKET_FACTORY = S.POCKET_FACTORY
    S._OLD_POCKET_STATE_ROOT = S.POCKET_STATE_ROOT
    S._OLD_COLLATE_POCKET = S.collate_pocket
    S.POCKET_RAM = S.SMMS_RAM
    S.POCKET_FACTORY = {'SM_PocketHG_Attention_MS': lambda dims: SmallPocketHypergraphMS(dims, mode='attention', latent=S.POCKET_LATENT, drop=0.2)}
    S.POCKET_STATE_ROOT = S.SMMS_ROOT / 'states'
    S.POCKET_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    S.collate_pocket = collate_pocket_ms
    try:
        print('=' * 100)
        print('SMALL: V1 ATTENTION + MULTI-SCALE POCKET BANK')
        print('=' * 100)
        S.ms_summary, S.ms_folds, S.ms_oof, S.ms_states = run_pocket_cv('SM_PocketHG_Attention_MS')
        S.ms_summary_df = pd.DataFrame([S.ms_summary])
        display(S.ms_summary_df)
        display(S.ms_folds)
        S.ms_summary_df.to_csv(S.SMMS_ROOT / 'dev_cv_summary.csv', index=False)
        S.ms_folds.to_csv(S.SMMS_ROOT / 'dev_cv_folds.csv', index=False)
        S.MS_DEV = float(S.ms_summary['CV_AUPRC_mean'])
        S.V1_DEV = float(S.pocket_summary.loc[S.pocket_summary.architecture.eq('SM_PocketHG_Attention'), 'CV_AUPRC_mean'].iloc[0])
        print(f'\nV1 Attention Dev mean AUPRC : {S.V1_DEV:.6f}')
        print(f'MS Attention Dev mean AUPRC : {S.MS_DEV:.6f}')
        print(f'Delta MS - V1               : {S.MS_DEV - S.V1_DEV:+.6f}')
        S.MS_WINS = S.MS_DEV - S.V1_DEV >= 0.002
        print('MS becomes new headline?     :', S.MS_WINS)
        S.dev_y = np.concatenate(S.train_df.small_binding.apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
        S.ms_thr, S.ms_oof_mcc = best_mcc_threshold(S.dev_y, S.ms_oof)

        def _predict_ms_external(split):
            ds = PocketProteinDataset(split, S.SPLIT_DF[split])
            preds = []
            for fold, state in S.ms_states.items():
                model = S.POCKET_FACTORY['SM_PocketHG_Attention_MS'](S.DIMS).to(S.DEVICE)
                model.load_state_dict(state)
                p, _ = predict_pocket(model, ds)
                preds.append(p)
                del model
                if S.DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
            return np.mean(np.stack(preds), axis=0)
        S.ext_rows = []
        for S.split in ['TestSet300', 'TestSetNew46']:
            S.y = np.concatenate(S.SPLIT_DF[S.split].small_binding.apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
            S.p = _predict_ms_external(S.split)
            S.m05 = binary_metrics(S.y, S.p, 0.5)
            S.mdev = binary_metrics(S.y, S.p, S.ms_thr)
            S.row = {'dataset': S.split, 'method': 'SM_PocketHG_Attention_MS', 'Dev_CV_AUPRC': S.MS_DEV, 'Dev_OOF_MCC_threshold': S.ms_thr}
            S.row.update({f'{k}_fixed0.5': v for k, v in S.m05.items()})
            S.row.update({f'{k}_OOFthr': v for k, v in S.mdev.items()})
            S.ext_rows.append(S.row)
        S.ms_external = pd.DataFrame(S.ext_rows)
        display(S.ms_external)
        S.ms_external.to_csv(S.SMMS_ROOT / 'external_metrics.csv', index=False)
        S.cmp_rows = []
        for S._, S.r in S.ms_external.iterrows():
            S.cmp_rows.append({'dataset': S.r['dataset'], 'method': 'MS PocketHG Attention', 'AUPRC': S.r['AUPRC_fixed0.5'], 'AUROC': S.r['AUROC_fixed0.5'], 'F1': S.r['F1_fixed0.5'], 'MCC': S.r['MCC_fixed0.5']})
        for S._, S.r in S.headline.iterrows():
            S.cmp_rows.append({'dataset': S.r['dataset'], 'method': f'V1 {S.GLOBAL_MODEL}', 'AUPRC': S.r['AUPRC_fixed0.5'], 'AUROC': S.r['AUROC_fixed0.5'], 'F1': S.r['F1_fixed0.5'], 'MCC': S.r['MCC_fixed0.5']})
        S.baseline_fp = S.BASELINE_WORK / 'results' / 'FULL_DIRECT_BASELINE_METRICS_300_46.csv'
        if S.baseline_fp.exists():
            S.bb = pd.read_csv(S.baseline_fp)
            if 'label' in S.bb.columns:
                S.bb = S.bb[S.bb.label.eq('small_binding')].copy()
            if 'operating_point' in S.bb.columns:
                S.bb = S.bb[S.bb.operating_point.eq('fixed_0.5')].copy()
            for S._, S.r in S.bb.iterrows():
                if str(S.r.method) not in {'bindNode24', 'bindEmbed21DL'}:
                    continue
                S.apcol = 'AP_AUPRC' if 'AP_AUPRC' in S.bb.columns else 'AUPRC'
                S.cmp_rows.append({'dataset': str(S.r.dataset), 'method': str(S.r.method), 'AUPRC': float(S.r[S.apcol]), 'AUROC': float(S.r.AUROC), 'F1': float(S.r.global_positive_F1), 'MCC': float(S.r.global_MCC)})
        S.ms_compare = pd.DataFrame(S.cmp_rows).sort_values(['dataset', 'AUPRC'], ascending=[True, False])
        print('\nFINAL SMALL COMPARISON')
        display(S.ms_compare)
        S.ms_compare.to_csv(S.SMMS_ROOT / 'FINAL_COMPARISON.csv', index=False)
    finally:
        S.POCKET_RAM = S._OLD_POCKET_RAM
        S.POCKET_FACTORY = S._OLD_POCKET_FACTORY
        S.POCKET_STATE_ROOT = S._OLD_POCKET_STATE_ROOT
        S.collate_pocket = S._OLD_COLLATE_POCKET
    print('\nSaved:', S.SMMS_ROOT)

# SMALL FINAL ONE-CELL
def stage_43():
    S.REQ = ['POCKET_OOF', 'POCKET_STATES', 'SMMS_RAM', 'ms_oof', 'ms_states', 'SmallPocketHypergraphMS', 'collate_pocket_ms', 'PocketProteinDataset', 'predict_pocket', 'POCKET_FACTORY', 'POCKET_RAM', 'collate_pocket', 'train_df', 'SPLIT_DF', 'OFFICIAL_FOLDS', 'BASELINE_WORK']
    S.missing = [x for x in S.REQ if x not in globals()]
    if S.missing:
        raise RuntimeError('Run the completed Small V1 + MS cells first. Missing: ' + ', '.join(S.missing))
    S.OUT = S.RESULTS_ROOT / 'SMALL_FINAL_ONECELL'
    S.OUT.mkdir(parents=True, exist_ok=True)
    S.V1_NAME = 'SM_PocketHG_Attention'
    S.v1_oof = np.asarray(S.POCKET_OOF[S.V1_NAME], dtype=np.float32)
    S.ms_oof_ = np.asarray(S.ms_oof, dtype=np.float32)
    S.y_dev = np.concatenate(S.train_df.small_binding.apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
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
    S.v1_dev = float(S.pocket_summary.loc[S.pocket_summary.architecture.eq(S.V1_NAME), 'CV_AUPRC_mean'].iloc[0])
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
    for S.split in ['TestSet300', 'TestSetNew46']:
        S.df = S.SPLIT_DF[S.split]
        S.y = np.concatenate(S.df.small_binding.apply(lambda x: np.asarray(x, dtype=int)).tolist())
        S.pv1 = predict_v1(S.split)
        S.pms = predict_ms(S.split)
        S.pens = S.w1 * S.pv1 + S.wm * S.pms
        for S.name, S.p in [('V1_Attention', S.pv1), ('MS_Attention', S.pms), ('FINAL_DEV_LOCKED', S.pens)]:
            S.g = global_metrics(S.y, S.p, 0.5)
            S.pm = protein_metrics(S.df, S.p, 0.5)
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
    S.bb = S.bb[S.bb.label.eq('small_binding')].copy()
    if 'operating_point' in S.bb.columns:
        S.bb = S.bb[S.bb.operating_point.eq('fixed_0.5')].copy()
    S.base_rows = []
    for S._, S.r in S.bb.iterrows():
        if str(S.r.method) not in {'bindNode24', 'bindEmbed21DL'}:
            continue
        S.apcol = 'AP_AUPRC' if 'AP_AUPRC' in S.bb.columns else 'AUPRC'
        S.base_rows.append({'dataset': str(S.r.dataset), 'method': str(S.r.method), 'AUPRC': float(S.r[S.apcol]), 'AUROC': float(S.r.AUROC), 'precision': np.nan, 'recall': np.nan, 'F1': float(S.r.global_positive_F1), 'MCC': float(S.r.global_MCC), 'protein_mean_F1': float(S.r.protein_mean_F1), 'protein_mean_MCC': float(S.r.protein_mean_MCC)})
    S.final = pd.concat([S.ours, pd.DataFrame(S.base_rows)], ignore_index=True).sort_values(['dataset', 'AUPRC'], ascending=[True, False])
    print('\nFINAL SMALL — ALL AVAILABLE METRICS')
    display(S.final)
    S.final.to_csv(S.OUT / 'FINAL_ALL_METRICS.csv', index=False)
    S.decision = {'w_V1': S.w1, 'w_MS': S.wm, 'V1_Dev_mean_AUPRC': S.v1_dev, 'Final_Dev_mean_AUPRC': S.ens_dev, 'Dev_gain': S.ens_dev - S.v1_dev, 'Promoted': bool(S.PROMOTE), 'Frozen_Dev_MCC_threshold': S.ens_thr}
    (S.OUT / 'FINAL_DECISION.json').write_text(json.dumps(S.decision, indent=2))
    print('\nDecision:', S.decision)
    print('Saved:', S.OUT)

# SMALL FINAL PAPER-READY REPORT — ONE CELL
def stage_44():
    S.TASK_NAME = 'SMALL'
    S.LABEL_COL = 'small_binding'
    S.V1_NAME = 'SM_PocketHG_Attention'
    S.MS_NAME = 'SM_PocketHG_Attention_MS'
    S.FINAL_NAME = 'FINAL_DEV_LOCKED_ENSEMBLE'
    S.MIN_DEV_GAIN = 0.002
    S.WEIGHT_GRID = np.linspace(0.0, 1.0, 21)
    S.OUT = S.RESULTS_ROOT / 'SMALL_FINAL_PAPER_READY'
    S.OUT.mkdir(parents=True, exist_ok=True)
    S.REQ = ['POCKET_OOF', 'POCKET_STATES', 'SMMS_RAM', 'ms_oof', 'ms_states', 'SmallPocketHypergraphMS', 'collate_pocket_ms', 'PocketProteinDataset', 'predict_pocket', 'POCKET_FACTORY', 'POCKET_RAM', 'collate_pocket', 'train_df', 'SPLIT_DF', 'OFFICIAL_FOLDS', 'BASELINE_WORK', 'pocket_summary', 'DIMS', 'DEVICE', 'POCKET_LATENT']
    S.missing = [x for x in S.REQ if x not in globals()]
    if S.missing:
        raise RuntimeError('Run the completed V1 + multi-scale cells first. Missing: ' + ', '.join(S.missing))
    S.V1_OOF = np.asarray(S.POCKET_OOF[S.V1_NAME], dtype=np.float64)
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
    S.V1_DEV = float(S.pocket_summary.loc[S.pocket_summary.architecture.eq(S.V1_NAME), 'CV_AUPRC_mean'].iloc[0])
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

STAGES = [stage_37, stage_39, stage_40, stage_41, stage_42, stage_43, stage_44]

