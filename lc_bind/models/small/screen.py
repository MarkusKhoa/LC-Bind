"""Extracted from SMALL_ULTIMATE_V1_DATA_DRIVEN_POCKET-3.ipynb. Section: screen."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Chain, ConcatMLP, Counter, DataLoader, Dataset, F, GatedFusion, LatentLigandMoE, ModalityAttention, Model, MultiArrayDataset, PDBIO, PDBParser, Path, PocketResidualFusion, ProjectedConcat, Structure, Subset, accuracy_score, ap, ast, average_precision_score, balanced_accuracy_score, cKDTree, collate_multi, deepcopy, defaultdict, f1_score, get_block, hashlib, json, math, matthews_corrcoef, nn, np, os, pd, platform, precision_score, predict_flat, recall_score, roc_auc_score, seq1, shutil, subprocess, sys, tarfile, torch
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
    from lc_bind.models.shared import ConcatMLP, GatedFusion, ModalityAttention, MultiArrayDataset, ProjectedConcat, ap, collate_multi, predict_flat
    from lc_bind.utils.cache import get_block

    class PocketResidualFusion(nn.Module):
    
        def __init__(self, dims, latent=192, drop=0.2):
            super().__init__()
            if 'POCKET' not in dims:
                raise ValueError('PocketResidualFusion requires POCKET modality')
            self.base_keys = sorted([k for k in dims if k != 'POCKET'])
            self.base_enc = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, latent), nn.GELU(), nn.Dropout(drop)) for k, d in dims.items() if k != 'POCKET'})
            self.pocket_enc = nn.Sequential(nn.LayerNorm(dims['POCKET']), nn.Linear(dims['POCKET'], latent), nn.GELU(), nn.Dropout(drop), nn.Linear(latent, latent))
            self.base_gate = nn.ModuleDict({k: nn.Linear(latent, 1) for k in self.base_keys})
            self.alpha_logit = nn.Parameter(torch.tensor(-2.944, dtype=torch.float32))
            self.head = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, 192), nn.GELU(), nn.Dropout(drop), nn.Linear(192, 1))
    
        def forward(self, x):
            zz = []
            gg = []
            for k in self.base_keys:
                z = self.base_enc[k](x[k])
                zz.append(z)
                gg.append(self.base_gate[k](z))
            Z = torch.stack(zz, dim=1)
            G = torch.softmax(torch.cat(gg, dim=1), dim=1).unsqueeze(-1)
            h = (Z * G).sum(dim=1)
            pocket = self.pocket_enc(x['POCKET'])
            alpha = torch.sigmoid(self.alpha_logit)
            h = h + alpha * pocket
            return self.head(h).squeeze(-1)

    class LatentLigandMoE(nn.Module):
    
        def __init__(self, dims, latent=192, experts=4, drop=0.2):
            super().__init__()
            self.keys = sorted(dims)
            self.enc = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, latent), nn.GELU(), nn.Dropout(drop)) for k, d in dims.items()})
            self.mod_gate = nn.ModuleDict({k: nn.Linear(latent, 1) for k in dims})
            self.shared = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, latent), nn.GELU())
            self.experts = nn.ModuleList([nn.Sequential(nn.Linear(latent, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1)) for _ in range(experts)])
            self.expert_gate = nn.Linear(latent, experts)
    
        def forward(self, x):
            zz = []
            gg = []
            for k in self.keys:
                z = self.enc[k](x[k])
                zz.append(z)
                gg.append(self.mod_gate[k](z))
            Z = torch.stack(zz, dim=1)
            G = torch.softmax(torch.cat(gg, dim=1), dim=1).unsqueeze(-1)
            h = self.shared((Z * G).sum(dim=1))
            logits = torch.cat([e(h) for e in self.experts], dim=1)
            w = torch.softmax(self.expert_gate(h), dim=1)
            return (logits * w).sum(dim=1)

def modality_spec(blocks):
    blocks = list(blocks)
    spec = {}
    if 'ESM' in blocks:
        spec['ESM'] = ['ESM']
    pocket = [b for b in ['ACCESS', 'TOPO', 'SM_POCKET3D44', 'SM_TOPO16'] if b in blocks]
    if pocket:
        spec['POCKET'] = pocket
    chem = [b for b in ['CHEM', 'SOLV', 'ATOM', 'SM_LIGCHEM20'] if b in blocks]
    if chem:
        spec['CHEM'] = chem
    arom = [b for b in ['SM_SEQCHEM32', 'SM_AROMSTACK12'] if b in blocks]
    if arom:
        spec['AROM_HYDRO'] = arom
    bio = [b for b in ['EVO', 'ESMIF', 'SEQMOTIF', 'SM_PROTCTX8', 'SPATIAL', 'LOCAL'] if b in blocks]
    if bio:
        spec['BIO'] = bio
    assigned = {x for v in spec.values() for x in v}
    other = [b for b in blocks if b not in assigned]
    if other:
        spec['OTHER'] = other
    return spec

def concat_blocks(store, pid, blocks):
    xs = [get_block(store, pid, b).astype(np.float32, copy=False) for b in blocks]
    lens = {x.shape[0] for x in xs}
    if len(lens) != 1:
        raise RuntimeError(f'{pid}: block length mismatch')
    return np.concatenate(xs, axis=1)

def build_pack(split):
    df = S.SPLIT_DF[split]
    store = S.ctx.stores[S.STORE_KEY[split]]
    mods = {k: [] for k in S.MODALITY_SPEC}
    yy = []
    pids = []
    ridx = []
    for r in df.itertuples():
        L = len(r.sequence)
        for m, blocks in S.MODALITY_SPEC.items():
            mods[m].append(concat_blocks(store, r.prot_id, blocks))
        yy.append(np.asarray(r.small_binding, dtype=np.float32))
        pids.extend([str(r.prot_id)] * L)
        ridx.extend(range(L))
    mods = {k: np.concatenate(v, axis=0).astype(np.float16 if k in {'ESM', 'BIO'} else np.float32, copy=False) for k, v in mods.items()}
    return {'mods': mods, 'y': np.concatenate(yy).astype(np.float32), 'pids': np.asarray(pids, dtype=object), 'ridx': np.asarray(ridx, dtype=np.int32)}

def train_one_arch_fold(arch, pack, train_idx, val_idx, fold):
    torch.manual_seed(S.SEED + fold)
    np.random.seed(S.SEED + fold)
    ds = MultiArrayDataset(pack['mods'], pack['y'])
    train_ds = Subset(ds, train_idx.tolist())
    model = S.ARCH_FACTORY[arch](S.DIMS).to(S.DEVICE)
    ytr = pack['y'][train_idx]
    npos = max(float(ytr.sum()), 1.0)
    nneg = max(float(len(ytr) - ytr.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.sqrt(nneg / npos), dtype=torch.float32, device=S.DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=S.CUSTOM_LR, weight_decay=S.CUSTOM_WEIGHT_DECAY)
    loader = DataLoader(train_ds, batch_size=S.CUSTOM_BATCH, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_multi)
    best_ap = -1.0
    best_state = None
    bad = 0
    for epoch in range(1, S.CUSTOM_EPOCHS + 1):
        model.train()
        losses = []
        for x, y in loader:
            x = {k: v.to(S.DEVICE, non_blocking=True) for k, v in x.items()}
            y = y.to(S.DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        vp = predict_flat(model, ds, val_idx)
        vy = pack['y'][val_idx]
        val_ap = S.ap(vy, vp)
        if val_ap > best_ap + 1e-05:
            best_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch == 1 or epoch % 5 == 0:
            print(arch, 'fold', fold, 'epoch', epoch, 'loss', round(float(np.mean(losses)), 4), 'val_AP', round(val_ap, 5), 'best', round(best_ap, 5))
        if bad >= S.CUSTOM_PATIENCE:
            break
    model.load_state_dict(best_state)
    return (model, best_state)

def run_arch_cv(arch):
    pack = S.PACKS['DevSet1010']
    pids = pack['pids']
    y = pack['y']
    ds = MultiArrayDataset(pack['mods'], pack['y'])
    oof = np.full(len(y), np.nan, dtype=np.float32)
    rows = []
    states = {}
    adir = S.CUSTOM_STATE_ROOT / arch
    adir.mkdir(parents=True, exist_ok=True)
    for fold, val_ids in S.OFFICIAL_FOLDS.items():
        val_set = set(val_ids)
        val_idx = np.flatnonzero(np.array([p in val_set for p in pids]))
        train_idx = np.flatnonzero(np.array([p not in val_set for p in pids]))
        state_fp = adir / f'fold{fold}.pt'
        pred_fp = adir / f'fold{fold}_val.npy'
        meta_fp = adir / f'fold{fold}.json'
        reuse = False
        if state_fp.exists() and pred_fp.exists() and meta_fp.exists():
            try:
                meta = json.loads(meta_fp.read_text())
                pp = np.load(pred_fp)
                reuse = meta.get('representation') == S.BEST_REP_NAME and meta.get('n_val') == len(val_idx) and (pp.shape == (len(val_idx),)) and np.all(np.isfinite(pp))
            except Exception:
                reuse = False
        if reuse:
            print(arch, 'fold', fold, 'REUSE')
            state = torch.load(state_fp, map_location='cpu')
            pred = np.load(pred_fp).astype(np.float32)
        else:
            print(arch, 'fold', fold, 'TRAIN')
            model, state = train_one_arch_fold(arch, pack, train_idx, val_idx, fold)
            pred = predict_flat(model, ds, val_idx).astype(np.float32)
            torch.save(state, state_fp)
            np.save(pred_fp, pred)
            meta_fp.write_text(json.dumps({'representation': S.BEST_REP_NAME, 'blocks': S.BEST_REP_BLOCKS, 'n_val': int(len(val_idx))}, indent=2))
            del model
            if S.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        states[fold] = state
        oof[val_idx] = pred
        rows.append({'architecture': arch, 'fold': fold, 'AUPRC': S.ap(y[val_idx], pred), 'n_val': len(val_idx), 'n_positive': int(y[val_idx].sum())})
    if not np.all(np.isfinite(oof)):
        raise RuntimeError(f'{arch}: incomplete OOF')
    f = pd.DataFrame(rows)
    return ({'architecture': arch, 'CV_AUPRC_mean': float(f.AUPRC.mean()), 'CV_AUPRC_std': float(f.AUPRC.std(ddof=1)), 'OOF_AUPRC': S.ap(y, oof)}, f, oof, states)

# 8. SINGLE-FAMILY MLP SCREEN — DEV ONLY
def stage_15():
    S.FAMILY_GRID = {'ESM': ['ESM'], 'ESM+ACCESS': ['ESM', 'ACCESS'], 'ESM+TOPO': ['ESM', 'TOPO'], 'ESM+ATOM': ['ESM', 'ATOM'], 'ESM+CHEM': ['ESM', 'CHEM'], 'ESM+SOLV': ['ESM', 'SOLV'], 'ESM+SEQMOTIF': ['ESM', 'SEQMOTIF'], 'ESM+Structure': ['ESM', 'SPATIAL', 'LOCAL'], 'ESM+SM_SEQCHEM': ['ESM', 'SM_SEQCHEM32'], 'ESM+SM_POCKET3D': ['ESM', 'SM_POCKET3D44'], 'ESM+SM_LIGCHEM': ['ESM', 'SM_LIGCHEM20'], 'ESM+SM_TOPO': ['ESM', 'SM_TOPO16'], 'ESM+SM_AROMSTACK': ['ESM', 'SM_AROMSTACK12'], 'ESM+SM_PROTCTX': ['ESM', 'SM_PROTCTX8']}
    if S.EVO_AVAILABLE:
        S.FAMILY_GRID['ESM+EVO'] = ['ESM', 'EVO']
    if S.ESMIF_AVAILABLE:
        S.FAMILY_GRID['ESM+ESMIF'] = ['ESM', 'ESMIF']
    if S.RUN_FAMILY_SCREEN:
        S.family_cv, S.family_folds = S.suite.run_dev_cv_only(S.ctx, S.FAMILY_GRID, models=('MLP',), run_version='small_v1_01_family_mlp')
        S.family_cv.to_csv(S.RUN_ROOT / 'family_cv.csv', index=False)
        S.family_folds.to_csv(S.RUN_ROOT / 'family_folds.csv', index=False)
    else:
        S.family_cv = pd.read_csv(S.RUN_ROOT / 'family_cv.csv')
        S.family_folds = pd.read_csv(S.RUN_ROOT / 'family_folds.csv')
    S.base_score = float(S.family_cv.loc[S.family_cv.representation.eq('ESM'), 'CV_AUPRC'].iloc[0])
    S.family_rank = S.family_cv.copy()
    S.family_rank['delta_vs_ESM'] = S.family_rank.CV_AUPRC - S.base_score
    S.family_rank = S.family_rank.sort_values('CV_AUPRC', ascending=False)
    display(S.family_rank)
    S.family_rank.to_csv(S.RUN_ROOT / 'family_ranked.csv', index=False)
    print('ESM baseline:', S.base_score)

# 9. MECHANISM COMBINATION LADDER — DEV ONLY
def stage_16():
    S.MECHANISM_GRID = {'ESM': ['ESM'], 'ESM+ACCESS+TOPO': ['ESM', 'ACCESS', 'TOPO'], 'ESM+POCKET_GEOMETRY': ['ESM', 'ACCESS', 'SM_POCKET3D44', 'SM_TOPO16'], 'ESM+AROM_HYDRO': ['ESM', 'SM_SEQCHEM32', 'SM_AROMSTACK12'], 'ESM+LIGAND_CHEMISTRY': ['ESM', 'CHEM', 'SOLV', 'SM_LIGCHEM20'], 'ESM+POCKET_CHEM': ['ESM', 'ACCESS', 'SM_POCKET3D44', 'SM_TOPO16', 'SM_SEQCHEM32', 'SM_AROMSTACK12', 'SM_LIGCHEM20'], 'ESM+POCKET_CHEM+SOLV': ['ESM', 'ACCESS', 'SOLV', 'SM_POCKET3D44', 'SM_TOPO16', 'SM_SEQCHEM32', 'SM_AROMSTACK12', 'SM_LIGCHEM20'], 'ESM+POCKET_CHEM+ATOM_CHEM_CONTROL': ['ESM', 'ACCESS', 'SOLV', 'ATOM', 'CHEM', 'SM_POCKET3D44', 'SM_TOPO16', 'SM_SEQCHEM32', 'SM_AROMSTACK12', 'SM_LIGCHEM20']}
    if S.EVO_AVAILABLE:
        S.MECHANISM_GRID['ESM+POCKET_CHEM+EVO_CONTROL'] = ['ESM', 'ACCESS', 'SM_POCKET3D44', 'SM_TOPO16', 'SM_SEQCHEM32', 'SM_AROMSTACK12', 'SM_LIGCHEM20', 'EVO']
    if S.ESMIF_AVAILABLE:
        S.MECHANISM_GRID['ESM+POCKET_CHEM+ESMIF'] = ['ESM', 'ACCESS', 'SM_POCKET3D44', 'SM_TOPO16', 'SM_SEQCHEM32', 'SM_AROMSTACK12', 'SM_LIGCHEM20', 'ESMIF']
    if S.RUN_COMBINATION_LADDER:
        S.combo_cv, S.combo_folds = S.suite.run_dev_cv_only(S.ctx, S.MECHANISM_GRID, models=('MLP',), run_version='small_v1_02_mechanism_combo_mlp')
        S.combo_cv.to_csv(S.RUN_ROOT / 'mechanism_combo_cv.csv', index=False)
        S.combo_folds.to_csv(S.RUN_ROOT / 'mechanism_combo_folds.csv', index=False)
    else:
        S.combo_cv = pd.read_csv(S.RUN_ROOT / 'mechanism_combo_cv.csv')
        S.combo_folds = pd.read_csv(S.RUN_ROOT / 'mechanism_combo_folds.csv')
    S.combo_rank = S.combo_cv.sort_values('CV_AUPRC', ascending=False).reset_index(drop=True)
    display(S.combo_rank)
    S.BEST_REP_NAME = str(S.combo_rank.iloc[0].representation)
    S.BEST_REP_BLOCKS = list(S.MECHANISM_GRID[S.BEST_REP_NAME])
    S.BEST_REP_SCORE = float(S.combo_rank.iloc[0].CV_AUPRC)
    print('LOCKED REPRESENTATION:', S.BEST_REP_NAME)
    print('Blocks:', S.BEST_REP_BLOCKS)
    print('Dev CV AUPRC:', S.BEST_REP_SCORE)
    (S.RUN_ROOT / 'locked_representation.json').write_text(json.dumps({'representation': S.BEST_REP_NAME, 'blocks': S.BEST_REP_BLOCKS, 'dev_cv_auprc': S.BEST_REP_SCORE, 'selection_source': 'DevSet1010 only'}, indent=2))

# 10. PACK LOCKED REPRESENTATION INTO SEMANTIC MODALITIES
def stage_18():
    S.MODALITY_SPEC = modality_spec(S.BEST_REP_BLOCKS)
    print('MODALITY_SPEC =', S.MODALITY_SPEC)
    S.PACKS = {s: build_pack(s) for s in S.SPLIT_DF}
    S.DIMS = {k: v.shape[1] for k, v in S.PACKS['DevSet1010']['mods'].items()}
    for S.split, S.p in S.PACKS.items():
        print(S.split, {k: v.shape for k, v in S.p['mods'].items()}, 'y=', S.p['y'].shape, 'positive=', int(S.p['y'].sum()))
    print('Modality dims:', S.DIMS)

# 11. MULTIMODAL MODELS
def stage_19():
    S.ARCH_FACTORY = {'ConcatMLP': lambda dims: ConcatMLP(dims), 'ProjectedConcat': lambda dims: ProjectedConcat(dims), 'GatedFusion': lambda dims: GatedFusion(dims), 'ModalityAttention': lambda dims: ModalityAttention(dims), 'LatentLigandMoE': lambda dims: LatentLigandMoE(dims)}
    if 'POCKET' in S.DIMS:
        S.ARCH_FACTORY['PocketResidualFusion'] = lambda dims: PocketResidualFusion(dims)
    print('Architectures:', list(S.ARCH_FACTORY))

# 12. CUSTOM CV TRAINER — OFFICIAL FOLDS, RESUME SAFE
def stage_20():
    S.CUSTOM_STATE_ROOT = S.RUN_ROOT / 'custom_states'
    S.CUSTOM_STATE_ROOT.mkdir(parents=True, exist_ok=True)

# 13. RUN MULTIMODAL ARCHITECTURE SCREEN — DEV ONLY
def stage_21():
    S.ARCH_RESULTS = {}
    S.ARCH_FOLDS = []
    S.ARCH_OOF = {}
    S.ARCH_STATES = {}
    if S.RUN_CUSTOM_ARCHITECTURES:
        for S.arch in S.ARCH_FACTORY:
            print('\n' + '=' * 80)
            print('ARCH:', S.arch)
            print('=' * 80)
            S.s, S.f, S.o, S.st = run_arch_cv(S.arch)
            S.ARCH_RESULTS[S.arch] = S.s
            S.ARCH_FOLDS.append(S.f)
            S.ARCH_OOF[S.arch] = S.o
            S.ARCH_STATES[S.arch] = S.st
    S.arch_summary = pd.DataFrame(S.ARCH_RESULTS.values()).sort_values('CV_AUPRC_mean', ascending=False).reset_index(drop=True)
    S.arch_folds = pd.concat(S.ARCH_FOLDS, ignore_index=True)
    S.arch_summary.to_csv(S.RUN_ROOT / 'architecture_cv.csv', index=False)
    S.arch_folds.to_csv(S.RUN_ROOT / 'architecture_folds.csv', index=False)
    display(S.arch_summary)
    S.concat_score = float(S.arch_summary.loc[S.arch_summary.architecture.eq('ConcatMLP'), 'CV_AUPRC_mean'].iloc[0])
    S.best_arch = str(S.arch_summary.iloc[0].architecture)
    S.best_arch_score = float(S.arch_summary.iloc[0].CV_AUPRC_mean)
    S.LOCKED_ARCH = S.best_arch if S.best_arch_score - S.concat_score >= S.MIN_MEAN_AUPRC_GAIN else 'ConcatMLP'
    S.LOCKED_ARCH_SCORE = float(S.arch_summary.loc[S.arch_summary.architecture.eq(S.LOCKED_ARCH), 'CV_AUPRC_mean'].iloc[0])
    print('LOCKED ARCH:', S.LOCKED_ARCH, S.LOCKED_ARCH_SCORE)
    (S.RUN_ROOT / 'locked_architecture.json').write_text(json.dumps({'architecture': S.LOCKED_ARCH, 'dev_cv_auprc': S.LOCKED_ARCH_SCORE, 'concat_cv_auprc': S.concat_score, 'best_raw_architecture': S.best_arch, 'best_raw_cv_auprc': S.best_arch_score}, indent=2))

STAGES = [stage_15, stage_16, stage_18, stage_19, stage_20, stage_21]

