"""Extracted from NUCLEAR_STANDALONE_EVO_XTASK_ADDIN-3.ipynb. Section: nuclear_addin."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global DataLoader, Dataset, EvoPatchDataset, NuclearPatchHypergraphFinal, PatchResidueEncoderFinal, Path, accuracy_score, ast, average_precision_score, f1_score, get_block, get_record, gzip, json, matthews_corrcoef, nn, np, os, parse_list, pd, precision_score, recall_score, roc_auc_score, set_block, subprocess, sys, time, torch
    from pathlib import Path
    import os, sys, json, ast, time, subprocess
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import average_precision_score
    import os
    import gzip, time, json
    import json
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score
    from lc_bind.io.datasets import parse_list
    from lc_bind.utils.cache import get_block, get_record, set_block

    class EvoPatchDataset(Dataset):
    
        def __init__(self, split, df, modality_spec):
            self.split = split
            self.df = df.reset_index(drop=True)
            self.store = S.ctx.stores[S.STORE_KEY[split]]
            self.modality_spec = modality_spec
    
        def __len__(self):
            return len(self.df)
    
        def __getitem__(self, i):
            r = self.df.iloc[i]
            pid = str(r.prot_id)
            mods = {m: concat_blocks_local(self.store, pid, blocks).astype(np.float32, copy=False) for m, blocks in self.modality_spec.items()}
            er, ep, pf = S.PATCH_RAM_FINAL[self.split, pid]
            return {'pid': pid, 'mods': mods, 'y': np.asarray(r.nuclear_binding, dtype=np.float32), 'edge_res': er, 'edge_patch': ep, 'patch_feat': pf}

    class PatchResidueEncoderFinal(nn.Module):
    
        def __init__(self, dims, latent=192, drop=0.2):
            super().__init__()
            self.keys = sorted(dims)
            self.enc = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, latent), nn.GELU(), nn.Dropout(drop)) for k, d in dims.items()})
            self.attn = nn.MultiheadAttention(latent, 6, dropout=drop, batch_first=True)
    
        def forward(self, x):
            Z = torch.stack([self.enc[k](x[k]) for k in self.keys], dim=1)
            A, _ = self.attn(Z, Z, Z, need_weights=False)
            return (Z + A).mean(dim=1)

    class NuclearPatchHypergraphFinal(nn.Module):
    
        def __init__(self, dims, latent=192, drop=0.2):
            super().__init__()
            self.encoder = PatchResidueEncoderFinal(dims, latent=latent, drop=drop)
            self.patch_static = nn.Sequential(nn.LayerNorm(18), nn.Linear(18, 64), nn.GELU())
            self.patch_mlp = nn.Sequential(nn.Linear(latent * 2 + 64, latent), nn.GELU(), nn.Dropout(drop), nn.Linear(latent, latent), nn.GELU())
            self.message_mlp = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, latent), nn.GELU(), nn.Dropout(drop), nn.Linear(latent, latent))
            self.alpha_logit = nn.Parameter(torch.tensor(-2.944, dtype=torch.float32))
            self.head = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))
    
        @staticmethod
        def index_mean(x, index, n):
            out = x.new_zeros((n, x.shape[-1]))
            cnt = x.new_zeros((n, 1))
            if len(index):
                out.index_add_(0, index, x)
                cnt.index_add_(0, index, x.new_ones((len(index), 1)))
            return out / cnt.clamp_min(1.0)
    
        @staticmethod
        def index_max(x, index, n):
            if n == 0:
                return x.new_zeros((0, x.shape[-1]))
            base = x.new_full((n, x.shape[-1]), -1000000000.0)
            if len(index):
                idx = index[:, None].expand(-1, x.shape[-1])
                out = base.scatter_reduce(0, idx, x, reduce='amax', include_self=True)
            else:
                out = base
            return torch.where(out < -100000000.0, torch.zeros_like(out), out)
    
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
            return self.head(h).squeeze(-1)

def load_df(path):
    df = pd.read_csv(path)
    df['prot_id'] = df['prot_id'].astype(str)
    df['sequence'] = df['sequence'].astype(str)
    for col in ['metal_binding', 'nuclear_binding', 'small_binding', 'any_ligand_binding_sites']:
        if col in df.columns:
            df[col] = df[col].apply(parse_list)
    return df

def reconstruct_oof_from_fold_predictions(pred_dir, label_col):
    """
    Saved foldX_val.npy was produced on:
      va_df = train_df[train_df.prot_id.isin(val_set)]
    preserving train_df order.
    """
    offsets = {}
    cursor = 0
    for r in S.train_df.itertuples():
        L = len(r.sequence)
        offsets[str(r.prot_id)] = (cursor, cursor + L)
        cursor += L
    oof = np.full(cursor, np.nan, dtype=np.float32)
    for fold, val_ids in S.OFFICIAL_FOLDS.items():
        val_set = set(val_ids)
        va_df = S.train_df[S.train_df.prot_id.isin(val_set)].copy()
        fp = Path(pred_dir) / f'fold{fold}_val.npy'
        if not fp.exists():
            raise FileNotFoundError(fp)
        pred = np.load(fp).astype(np.float32)
        expected = int(va_df.sequence.str.len().sum())
        if pred.shape != (expected,):
            raise RuntimeError(f'{fp}: shape {pred.shape}, expected {(expected,)}')
        pos = 0
        for r in va_df.itertuples():
            L = len(r.sequence)
            a, b = offsets[str(r.prot_id)]
            oof[a:b] = pred[pos:pos + L]
            pos += L
    if not np.all(np.isfinite(oof)):
        raise RuntimeError('Incomplete reconstructed OOF')
    y = np.concatenate(S.train_df[label_col].apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
    return (oof, y)

def official_fold_indices():
    offsets = {}
    cursor = 0
    for r in S.train_df.itertuples():
        L = len(r.sequence)
        offsets[str(r.prot_id)] = (cursor, cursor + L)
        cursor += L
    out = {}
    for fold, ids in S.OFFICIAL_FOLDS.items():
        z = []
        for pid in ids:
            a, b = offsets[str(pid)]
            z.extend(range(a, b))
        out[fold] = np.asarray(z, dtype=np.int64)
    return out

def weight_search(y, p1, p2):
    rows = []
    for w in np.linspace(0, 1, 21):
        p = w * p1 + (1 - w) * p2
        aps = [average_precision_score(y[idx], p[idx]) for _, idx in sorted(S.FOLD_IDX.items())]
        rows.append({'w_V1': float(w), 'w_MS': float(1 - w), 'CV_AUPRC_mean': float(np.mean(aps)), 'CV_AUPRC_std': float(np.std(aps, ddof=1)), 'OOF_AUPRC': float(average_precision_score(y, p))})
    return pd.DataFrame(rows).sort_values(['CV_AUPRC_mean', 'w_V1'], ascending=[False, False]).reset_index(drop=True)

def _read_maybe_gzip_csv(fp):
    with open(fp, 'rb') as f:
        magic = f.read(2)
    if magic == b'\x1f\x8b':
        return pd.read_csv(fp)
    print('WARNING: counterpart has .gz suffix but is plain CSV; reading safely as plain CSV.')
    return pd.read_csv(fp, compression=None)

def concat_blocks_local(store, pid, blocks):
    xs = [np.asarray(get_block(store, pid, b), dtype=np.float32) for b in blocks]
    lens = {x.shape[0] for x in xs}
    if len(lens) != 1:
        raise RuntimeError(f'{pid}: block length mismatch {blocks}')
    return np.concatenate(xs, axis=1)

def load_patch_ram(split):
    """
    Reuse exact V1 candidate-patch cache.
    No PDB parsing.
    No graph rebuilding.
    """
    df = S.SPLIT_DF[split]
    outdir = S.PATCH_CACHE / split
    if not outdir.exists():
        raise FileNotFoundError(outdir)
    ram = {}
    for n, r in enumerate(df.itertuples(), start=1):
        fp = outdir / f'{r.prot_id}.npz'
        if not fp.exists():
            raise FileNotFoundError(fp)
        z = np.load(fp)
        er = z['edge_res'].astype(np.int64)
        ep = z['edge_patch'].astype(np.int64)
        pf = z['patch_feat'].astype(np.float32)
        if pf.ndim != 2 or pf.shape[1] != 18:
            raise RuntimeError(f'{split}/{r.prot_id}: unexpected patch_feat {pf.shape}')
        if len(er) != len(ep):
            raise RuntimeError(f'{split}/{r.prot_id}: edge length mismatch')
        ram[split, str(r.prot_id)] = (er, ep, pf)
        if n % 250 == 0:
            print('patch cache', split, n, '/', len(df))
    return ram

def collate_patch_final(batch):
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

def modality_dims(spec):
    store = S.ctx.stores[S.STORE_KEY['DevSet1010']]
    r = S.train_df.iloc[0]
    return {m: int(concat_blocks_local(store, r.prot_id, blocks).shape[1]) for m, blocks in spec.items()}

def move_batch_final(b):
    return {'mods': {k: v.to(S.DEVICE, non_blocking=True) for k, v in b['mods'].items()}, 'y': b['y'].to(S.DEVICE, non_blocking=True), 'edge_res': b['edge_res'].to(S.DEVICE, non_blocking=True), 'edge_patch': b['edge_patch'].to(S.DEVICE, non_blocking=True), 'patch_feat': b['patch_feat'].to(S.DEVICE, non_blocking=True)}

def predict_final(model, dataset):
    loader = DataLoader(dataset, batch_size=S.PATCH_PROTEIN_BATCH, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_patch_final)
    pp = []
    yy = []
    model.eval()
    with torch.no_grad():
        for b in loader:
            b = move_batch_final(b)
            p = torch.sigmoid(model(b['mods'], b['edge_res'], b['edge_patch'], b['patch_feat']))
            pp.append(p.detach().cpu().numpy())
            yy.append(b['y'].detach().cpu().numpy())
    return (np.concatenate(pp), np.concatenate(yy))

def train_fold_final(rep_name, modality_spec, tr_ds, va_ds, fold):
    torch.manual_seed(S.SEED_FINAL + 100 + fold)
    np.random.seed(S.SEED_FINAL + 100 + fold)
    dims = modality_dims(modality_spec)
    model = NuclearPatchHypergraphFinal(dims, latent=S.PATCH_LATENT, drop=0.2).to(S.DEVICE)
    ytr = np.concatenate(tr_ds.df.nuclear_binding.apply(lambda x: np.asarray(x, dtype=np.float32)).tolist())
    npos = max(float(ytr.sum()), 1.0)
    nneg = max(float(len(ytr) - ytr.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.sqrt(nneg / npos), dtype=torch.float32, device=S.DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=S.PATCH_LR, weight_decay=S.PATCH_WEIGHT_DECAY)
    loader = DataLoader(tr_ds, batch_size=S.PATCH_PROTEIN_BATCH, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_patch_final)
    best_ap = -1.0
    best_state = None
    bad = 0
    for epoch in range(1, S.PATCH_EPOCHS + 1):
        model.train()
        losses = []
        for b in loader:
            b = move_batch_final(b)
            opt.zero_grad(set_to_none=True)
            logits = model(b['mods'], b['edge_res'], b['edge_patch'], b['patch_feat'])
            loss = criterion(logits, b['y'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        vp, vy = predict_final(model, va_ds)
        val_ap = float(average_precision_score(vy, vp))
        if val_ap > best_ap + 1e-05:
            best_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch == 1 or epoch % 5 == 0:
            print(rep_name, 'fold', fold, 'epoch', epoch, 'loss', round(float(np.mean(losses)), 4), 'val_AP', round(val_ap, 5), 'best', round(best_ap, 5), 'alpha', round(float(torch.sigmoid(model.alpha_logit).detach().cpu()), 4))
        if bad >= S.PATCH_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f'{rep_name} fold {fold}: no state')
    model.load_state_dict(best_state)
    return (model, best_state)

def run_cv_final(rep_name, modality_spec):
    offsets = {}
    cursor = 0
    for r in S.train_df.itertuples():
        L = len(r.sequence)
        offsets[str(r.prot_id)] = (cursor, cursor + L)
        cursor += L
    oof = np.full(cursor, np.nan, dtype=np.float32)
    rows = []
    states = {}
    rep_dir = S.STATE_ROOT / rep_name
    rep_dir.mkdir(parents=True, exist_ok=True)
    for fold, val_ids in S.OFFICIAL_FOLDS.items():
        val_set = set(map(str, val_ids))
        tr_df = S.train_df[~S.train_df.prot_id.astype(str).isin(val_set)].copy()
        va_df = S.train_df[S.train_df.prot_id.astype(str).isin(val_set)].copy()
        tr_ds = EvoPatchDataset('DevSet1010', tr_df, modality_spec)
        va_ds = EvoPatchDataset('DevSet1010', va_df, modality_spec)
        state_fp = rep_dir / f'fold{fold}.pt'
        pred_fp = rep_dir / f'fold{fold}_val.npy'
        meta_fp = rep_dir / f'fold{fold}.json'
        expected = int(va_df.sequence.str.len().sum())
        reuse = False
        if state_fp.exists() and pred_fp.exists() and meta_fp.exists():
            try:
                meta = json.loads(meta_fp.read_text())
                pred = np.load(pred_fp)
                reuse = meta.get('representation') == rep_name and meta.get('n_val') == expected and (pred.shape == (expected,)) and np.all(np.isfinite(pred))
            except Exception:
                reuse = False
        if reuse:
            print(rep_name, 'fold', fold, 'REUSE')
            state = torch.load(state_fp, map_location='cpu')
            pred = np.load(pred_fp).astype(np.float32)
        else:
            print(rep_name, 'fold', fold, 'TRAIN')
            model, state = train_fold_final(rep_name, modality_spec, tr_ds, va_ds, fold)
            pred, _ = predict_final(model, va_ds)
            torch.save(state, state_fp)
            np.save(pred_fp, pred.astype(np.float32))
            meta_fp.write_text(json.dumps({'representation': rep_name, 'modality_spec': modality_spec, 'n_val': expected}, indent=2))
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
        rows.append({'representation': rep_name, 'fold': fold, 'AUPRC': float(average_precision_score(yv, pred)), 'n_val': len(pred), 'n_positive': int(yv.sum())})
    if not np.all(np.isfinite(oof)):
        raise RuntimeError(f'{rep_name}: incomplete OOF')
    f = pd.DataFrame(rows)
    return ({'representation': rep_name, 'CV_AUPRC_mean': float(f.AUPRC.mean()), 'CV_AUPRC_std': float(f.AUPRC.std(ddof=1)), 'OOF_AUPRC': float(average_precision_score(S.Y_DEV, oof))}, f, oof, states)

def fold_indices_final():
    offsets = {}
    cursor = 0
    for r in S.train_df.itertuples():
        L = len(r.sequence)
        offsets[str(r.prot_id)] = (cursor, cursor + L)
        cursor += L
    out = {}
    for fold, ids in S.OFFICIAL_FOLDS.items():
        idx = []
        for pid in ids:
            a, b = offsets[str(pid)]
            idx.extend(range(a, b))
        out[fold] = np.asarray(idx, dtype=np.int64)
    return out

def best_mcc_threshold_local(y, p):
    best_mcc = -999.0
    best_t = 0.5
    for t in np.linspace(0.01, 0.99, 199):
        pred = (p >= t).astype(int)
        m = float(matthews_corrcoef(y, pred))
        if m > best_mcc:
            best_mcc = m
            best_t = float(t)
    return (best_t, best_mcc)

def metrics_full(y, p, t):
    pred = (p >= t).astype(int)
    return {'AUPRC': float(average_precision_score(y, p)), 'AUROC': float(roc_auc_score(y, p)), 'Accuracy': float(accuracy_score(y, pred)), 'Precision_positive': float(precision_score(y, pred, zero_division=0)), 'Recall_positive': float(recall_score(y, pred, zero_division=0)), 'F1_positive': float(f1_score(y, pred, zero_division=0)), 'MCC': float(matthews_corrcoef(y, pred))}

# STANDALONE BOOTSTRAP — fresh Colab runtime
def stage_01():
    S.BASE = Path(str(S.BASE))
    S.ULTIMATE_CANDIDATES = [S.GEAR_DIR, S.BASE / 'GEAR_ULTIMATE_OTHER_BINDINGS', S.BASE / 'GEAR_ULTIMATE']
    S.ULTIMATE_DIR = next((p for p in S.ULTIMATE_CANDIDATES if p is not None and (Path(p) / 'gear_ultimate').exists()), None)
    if S.ULTIMATE_DIR is None:
        raise FileNotFoundError('GEAR backend not found under GEAR_ULTIMATE_OTHER_BINDINGS or GEAR_ULTIMATE')
    S.req = S.ULTIMATE_DIR / 'requirements.txt'
    if S.req.exists():
        pass
    sys.path.insert(0, str(S.ULTIMATE_DIR))
    from gear_ultimate import gear_evolution as evo
    S.evo = evo
    from gear_ultimate import gear_suite as suite
    S.suite = suite
    from gear_ultimate import gear_reporting as reporting
    S.reporting = reporting
    S.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if S.DEVICE.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    print('BASE:', S.BASE)
    print('GEAR backend:', S.ULTIMATE_DIR)
    print('DEVICE:', S.DEVICE)

# Standalone settings
def stage_02():
    print('Standalone add-in runtime started.')

# DATA / OFFICIAL FOLDS / STORE HELPERS
def stage_03():
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
        S.BN24 = Path(str(S.CACHE_ROOT / 'bindnode24_standalone_addin'))
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
        raise RuntimeError(f'Missing official Dev IDs: {sorted(S.missing)[:20]}')
    S.train_df = S.train_lookup.loc[S.official_order].reset_index(drop=True)
    assert len(S.train_df) == 1010
    assert len(S.test300_df) == 300
    assert len(S.test46_df) == 46
    S.SPLIT_DF = {'DevSet1010': S.train_df, 'TestSet300': S.test300_df, 'TestSetNew46': S.test46_df}
    S.STORE_KEY = {'DevSet1010': 'train', 'TestSet300': 'test300', 'TestSetNew46': 'test46'}
    S.FOLD_IDX = official_fold_indices()
    print('Dataset/fold bootstrap: PASS')

# NUCLEAR STANDALONE CONTEXT + CACHED EXPLICIT BLOCKS + EVO
def stage_04():
    S.ADDIN_ROOT = S.RESULTS_ROOT / 'ADDIN_STANDALONE_NUCLEAR_EVO_XTASK'
    S.ADDIN_ROOT.mkdir(parents=True, exist_ok=True)
    S.XTASK_ROOT = S.RESULTS_ROOT / 'CROSSTASK_EXPORTS'
    S.XTASK_ROOT.mkdir(parents=True, exist_ok=True)
    S.ctx = S.suite.prepare_context(base=S.BASE, run_version='standalone_nuclear_addin_context', label_column='nuclear_binding', models=('MLP',), ensure_base=True, add_mech=True, add_multi=True, add_seqctx=True, esm_batch_size=8, device=S.DEVICE)
    for S.split, S.df in S.SPLIT_DF.items():
        S.store_ids = set(map(str, S.ctx.stores[S.STORE_KEY[S.split]].keys()))
        S.csv_ids = set(S.df.prot_id.astype(str))
        if S.store_ids != S.csv_ids:
            raise RuntimeError(f'{S.split}: store/CSV mismatch')
    S.FEATURE_CACHE = S.CACHE_ROOT / 'nuclear_ultimate_v1' / 'nuclear_explicit_v1'
    S.NEW_BLOCKS = {'NA_SEQELECTRO32': 32, 'NA_PATCH3D40': 40, 'NA_HBOND16': 16, 'NA_TOPO12': 12, 'NA_PROTCTX8': 8, 'NA_AROMCTRL8': 8}
    for S.split, S.df in S.SPLIT_DF.items():
        S.store = S.ctx.stores[S.STORE_KEY[S.split]]
        S.outdir = S.FEATURE_CACHE / S.split
        for S.r in S.df.itertuples():
            S.fp = S.outdir / f'{S.r.prot_id}.npz'
            if not S.fp.exists():
                raise FileNotFoundError(f'Missing cached nuclear feature file: {S.fp}')
            S.z = np.load(S.fp)
            for S.k, S.d in S.NEW_BLOCKS.items():
                S.a = S.z[S.k].astype(np.float32, copy=False)
                if S.a.shape != (len(S.r.sequence), S.d):
                    raise RuntimeError(f'{S.split}/{S.r.prot_id}/{S.k}: {S.a.shape}')
                set_block(S.store, S.r.prot_id, S.k, S.a)
    print('Cached nuclear explicit blocks attached.')
    S.EVO_AVAILABLE = False
    try:
        S.coverage = S.evo.attach_evo(S.ctx.stores, S.BASE, splits=('train', 'test300', 'test46'), require_msa=True)
        display(S.coverage)
        S.EVO_AVAILABLE = True
        print('EVO attached.')
    except Exception as e:
        print('EVO unavailable -> experiment will continue without EVO.')
        print(repr(e))
    display(S.reporting.dimension_table(S.ctx.stores['train']))

# RECONSTRUCT NUCLEAR V1/MS OOF FROM SAVED FOLD PREDICTIONS
def stage_05():
    S.V1_PRED_DIR = S.RESULTS_ROOT / 'NUCLEAR_ULTIMATE_V1_DATA_DRIVEN_PATCH' / 'patch_states' / 'NA_PatchHG_Attention'
    S.MS_PRED_DIR = S.RESULTS_ROOT / 'NUCLEAR_V1_ATTENTION_V2_MULTISCALE_PATCH' / 'states' / 'NA_PatchHG_Attention_MS'
    S.V1_OOF, S.Y_DEV = reconstruct_oof_from_fold_predictions(S.V1_PRED_DIR, 'nuclear_binding')
    S.MS_OOF, S.Y2 = reconstruct_oof_from_fold_predictions(S.MS_PRED_DIR, 'nuclear_binding')
    assert np.array_equal(S.Y_DEV, S.Y2)
    S.ws = weight_search(S.Y_DEV, S.V1_OOF, S.MS_OOF)
    display(S.ws.head(10))
    S.ws.to_csv(S.ADDIN_ROOT / 'DEV_WEIGHT_SEARCH.csv', index=False)
    S.best = S.ws.iloc[0]
    S.W_V1 = float(S.best.w_V1)
    S.W_MS = float(S.best.w_MS)
    S.FINAL_OOF = S.W_V1 * S.V1_OOF + S.W_MS * S.MS_OOF
    print('Locked weights:', S.W_V1, S.W_MS)
    print('Dev mean AUPRC:', float(S.best.CV_AUPRC_mean))
    S.rows = []
    S.pos = 0
    for S.r in S.train_df.itertuples():
        S.L = len(S.r.sequence)
        S.y = np.asarray(S.r.nuclear_binding, dtype=np.int8)
        for S.i in range(S.L):
            S.rows.append({'dataset': 'DevSet1010', 'prot_id': str(S.r.prot_id), 'residue_index': S.i, 'y_nuclear': int(S.y[S.i]), 'prob_FINAL': float(S.FINAL_OOF[S.pos + S.i]), 'prob_V1': float(S.V1_OOF[S.pos + S.i]), 'prob_MS': float(S.MS_OOF[S.pos + S.i])})
        S.pos += S.L
    S.export = pd.DataFrame(S.rows)
    S.export_fp = S.XTASK_ROOT / 'NUCLEAR_DEV_OOF.csv.gz'
    S.tmp_fp = S.XTASK_ROOT / 'NUCLEAR_DEV_OOF.csv.gz.tmp'
    S.export.to_csv(S.tmp_fp, index=False)
    os.replace(S.tmp_fp, S.export_fp)
    print('Nuclear OOF export:', S.export_fp)

# FIX NUCLEAR OOF EXPORT — WRITE REAL GZIP
def stage_07():
    S.XTASK_ROOT = S.RESULTS_ROOT / 'CROSSTASK_EXPORTS'
    S.export_fp = S.XTASK_ROOT / 'NUCLEAR_DEV_OOF.csv.gz'
    S.tmp_fp = S.XTASK_ROOT / 'NUCLEAR_DEV_OOF.tmp.csv.gz'
    assert 'export' in globals(), 'Need the previous Nuclear OOF export cell state.'
    S.export.to_csv(S.tmp_fp, index=False, compression='gzip')
    os.replace(S.tmp_fp, S.export_fp)
    S._check = pd.read_csv(S.export_fp)
    print('FIXED:', S.export_fp)
    print('rows:', len(S._check))
    print('columns:', S._check.columns.tolist())
    print(S._check.head())

# WAIT FOR SMALL EXPORT -> ATTACH X-TASK + EVO RESIDUAL -> SCREEN
def stage_08():
    S.COUNTERPART = S.XTASK_ROOT / 'SMALL_DEV_OOF.csv.gz'
    S.WAIT_SECONDS = 900
    S.POLL_SECONDS = 10
    S.t0 = time.time()
    while not S.COUNTERPART.exists() and time.time() - S.t0 < S.WAIT_SECONDS:
        print(f'Waiting for Small export... {int(time.time() - S.t0)}s / {S.WAIT_SECONDS}s', end='\r')
        time.sleep(S.POLL_SECONDS)
    S.XTASK_SMALL_AVAILABLE = S.COUNTERPART.exists()
    print('\nSmall cross-task export available:', S.XTASK_SMALL_AVAILABLE)
    if S.XTASK_SMALL_AVAILABLE:
        S.xdf = pd.read_csv(S.COUNTERPART)
        S.xdf['prot_id'] = S.xdf['prot_id'].astype(str)
        S.store = S.ctx.stores['train']
        for S.r in S.train_df.itertuples():
            S.pid = str(S.r.prot_id)
            S.sub = S.xdf[S.xdf.prot_id.eq(S.pid)].sort_values('residue_index')
            if len(S.sub) != len(S.r.sequence):
                raise RuntimeError(f'{S.pid}: cross-task length mismatch')
            S.arr = S.sub[['prob_FINAL', 'prob_V1', 'prob_MS']].to_numpy(np.float32)
            set_block(S.store, S.pid, 'XTASK_SMALL3', S.arr)
    if S.EVO_AVAILABLE:
        for S.split, S.df in S.SPLIT_DF.items():
            S.store = S.ctx.stores[S.STORE_KEY[S.split]]
            for S.r in S.df.itertuples():
                S.E = np.asarray(get_block(S.store, S.r.prot_id, 'EVO'), dtype=np.float64)
                S.A = np.asarray(get_block(S.store, S.r.prot_id, 'ACCESS'), dtype=np.float64)
                if S.A.ndim == 1:
                    S.A = S.A[:, None]
                S.X = np.concatenate([np.ones((len(S.A), 1)), S.A, S.A * S.A], axis=1)
                if len(S.E) > S.X.shape[1] + 2:
                    S.beta = np.linalg.lstsq(S.X, S.E, rcond=None)[0]
                    S.R = S.E - S.X @ S.beta
                else:
                    S.R = S.E - S.E.mean(axis=0, keepdims=True)
                S.sd = S.R.std(axis=0, keepdims=True)
                S.R = S.R / np.where(S.sd < 1e-06, 1.0, S.sd)
                set_block(S.store, S.r.prot_id, 'EVO_ACCESS_RES', S.R.astype(np.float32))
    S.BASE_BLOCKS = ['ESM', 'ACCESS', 'NA_PATCH3D40', 'NA_TOPO12']
    S.GRID = {'N_BASE_PATCH_ORGANIZATION': S.BASE_BLOCKS}
    if S.EVO_AVAILABLE:
        S.GRID['N_BASE_PLUS_EVO'] = S.BASE_BLOCKS + ['EVO']
        S.GRID['N_BASE_PLUS_EVO_ACCESS_RES'] = S.BASE_BLOCKS + ['EVO_ACCESS_RES']
        S.GRID['N_BASE_PLUS_EVO_BOTH'] = S.BASE_BLOCKS + ['EVO', 'EVO_ACCESS_RES']
    if S.XTASK_SMALL_AVAILABLE:
        S.GRID['N_BASE_PLUS_XTASK_SMALL'] = S.BASE_BLOCKS + ['XTASK_SMALL3']
        if S.EVO_AVAILABLE:
            S.GRID['N_BASE_PLUS_EVORES_XTASK_SMALL'] = S.BASE_BLOCKS + ['EVO_ACCESS_RES', 'XTASK_SMALL3']
    print('Screening:')
    for S.k, S.v in S.GRID.items():
        print(' ', S.k, '=', S.v)
    S.cv, S.folds = S.suite.run_dev_cv_only(S.ctx, S.GRID, models=('MLP',), run_version='standalone_nuclear_evo_xtask_screen')
    S.base = float(S.cv.loc[S.cv.representation.eq('N_BASE_PATCH_ORGANIZATION'), 'CV_AUPRC'].iloc[0])
    S.cv = S.cv.copy()
    S.cv['delta_vs_base'] = S.cv['CV_AUPRC'] - S.base
    S.cv = S.cv.sort_values('CV_AUPRC', ascending=False).reset_index(drop=True)
    display(S.cv)
    display(S.folds)
    S.cv.to_csv(S.ADDIN_ROOT / 'REPRESENTATION_SCREEN.csv', index=False)
    S.folds.to_csv(S.ADDIN_ROOT / 'REPRESENTATION_SCREEN_FOLDS.csv', index=False)
    S.best = S.cv.iloc[0]
    S.decision = {'base_dev_auprc': S.base, 'best_representation': str(S.best.representation), 'best_dev_auprc': float(S.best.CV_AUPRC), 'gain_vs_base': float(S.best.delta_vs_base), 'promote_if_gain_ge_0.002': bool(S.best.delta_vs_base >= 0.002), 'EVO_available': bool(S.EVO_AVAILABLE), 'XTASK_small_available': bool(S.XTASK_SMALL_AVAILABLE)}
    (S.ADDIN_ROOT / 'DECISION.json').write_text(json.dumps(S.decision, indent=2))
    print('\nFINAL DECISION')
    print(json.dumps(S.decision, indent=2))
    print('Saved:', S.ADDIN_ROOT)

# Attach clean cross-task OOF:
def stage_10():
    print('=' * 100)
    print('NUCLEAR CLEAN V3 — EVO + TRUE SMALL -> NUCLEAR CROSS-TASK SCREEN')
    print('=' * 100)
    S.COUNTERPART = S.XTASK_ROOT / 'SMALL_DEV_OOF.csv.gz'
    S.WAIT_SECONDS = 300
    S.POLL_SECONDS = 5
    S.t0 = time.time()
    while not S.COUNTERPART.exists() and time.time() - S.t0 < S.WAIT_SECONDS:
        print(f'Waiting for Small export... {int(time.time() - S.t0)}s', end='\r')
        time.sleep(S.POLL_SECONDS)
    if not S.COUNTERPART.exists():
        raise FileNotFoundError(S.COUNTERPART)
    S.xdf = _read_maybe_gzip_csv(S.COUNTERPART)
    S.xdf['prot_id'] = S.xdf['prot_id'].astype(str)
    S.xdf['residue_index'] = S.xdf['residue_index'].astype(int)
    S.required = {'prot_id', 'residue_index', 'y_small', 'prob_V1', 'prob_MS'}
    S.missing = S.required - set(S.xdf.columns)
    if S.missing:
        raise RuntimeError(f'Wrong/old Small export. Missing columns: {sorted(S.missing)}; got {S.xdf.columns.tolist()}')
    if 'y_nuclear' in S.xdf.columns:
        raise RuntimeError('WRONG FILE: Small export contains y_nuclear. Repair SMALL_DEV_OOF first.')
    if S.xdf.duplicated(['prot_id', 'residue_index']).any():
        raise RuntimeError('Duplicate (prot_id,residue_index) in Small OOF export.')
    S.expected_rows = int(S.train_df.sequence.str.len().sum())
    if len(S.xdf) != S.expected_rows:
        raise RuntimeError(f'Small OOF row count mismatch: {len(S.xdf)} vs {S.expected_rows}')
    if set(S.xdf.prot_id) != set(S.train_df.prot_id.astype(str)):
        raise RuntimeError('Small OOF protein IDs do not match DevSet1010.')
    print('Small export integrity: PASS')
    print('rows:', len(S.xdf))
    print('columns:', S.xdf.columns.tolist())
    S.xmap = {pid: g.sort_values('residue_index') for pid, g in S.xdf.groupby('prot_id', sort=False)}
    S.train_store = S.ctx.stores[S.STORE_KEY['DevSet1010']]
    for S.r in S.train_df.itertuples():
        S.pid = str(S.r.prot_id)
        S.sub = S.xmap[S.pid]
        S.L = len(S.r.sequence)
        if len(S.sub) != S.L:
            raise RuntimeError(f'{S.pid}: Small OOF length {len(S.sub)} != sequence length {S.L}')
        if not np.array_equal(S.sub.residue_index.to_numpy(), np.arange(S.L)):
            raise RuntimeError(f'{S.pid}: residue_index is not exactly 0..L-1')
        S.arr = S.sub[['prob_V1', 'prob_MS']].to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(S.arr)):
            raise RuntimeError(f'{S.pid}: non-finite Small OOF probability')
        set_block(S.train_store, S.pid, 'XTASK_SMALL2_CLEAN', S.arr)
    print('Attached XTASK_SMALL2_CLEAN: PASS')
    S.EVO_OK = bool(globals().get('EVO_AVAILABLE', False))
    if S.EVO_OK:
        for S.r in S.train_df.itertuples():
            S.E = np.asarray(get_block(S.train_store, S.r.prot_id, 'EVO'), dtype=np.float64)
            S.A = np.asarray(get_block(S.train_store, S.r.prot_id, 'ACCESS'), dtype=np.float64)
            if S.A.ndim == 1:
                S.A = S.A[:, None]
            S.X = np.concatenate([np.ones((len(S.A), 1)), S.A, S.A * S.A], axis=1)
            if len(S.E) > S.X.shape[1] + 2:
                S.beta = np.linalg.lstsq(S.X, S.E, rcond=None)[0]
                S.R = S.E - S.X @ S.beta
            else:
                S.R = S.E - S.E.mean(axis=0, keepdims=True)
            S.sd = S.R.std(axis=0, keepdims=True)
            S.R = S.R / np.where(S.sd < 1e-06, 1.0, S.sd)
            set_block(S.train_store, S.r.prot_id, 'EVO_ACCESS_RES_CLEAN', S.R.astype(np.float32))
        print('Attached EVO_ACCESS_RES_CLEAN: PASS')
    S.BASE_BLOCKS = ['ESM', 'ACCESS', 'NA_PATCH3D40', 'NA_TOPO12']
    S.GRID = {'N_BASE_PATCH_ORGANIZATION_CLEANV3': S.BASE_BLOCKS, 'N_BASE_PLUS_XTASK_SMALL2_CLEANV3': S.BASE_BLOCKS + ['XTASK_SMALL2_CLEAN']}
    if S.EVO_OK:
        S.GRID['N_BASE_PLUS_EVO_CLEANV3'] = S.BASE_BLOCKS + ['EVO']
        S.GRID['N_BASE_PLUS_EVO_ACCESS_RES_CLEANV3'] = S.BASE_BLOCKS + ['EVO_ACCESS_RES_CLEAN']
        S.GRID['N_BASE_PLUS_EVO_XTASK_SMALL2_CLEANV3'] = S.BASE_BLOCKS + ['EVO', 'XTASK_SMALL2_CLEAN']
    print('\nFresh representations:')
    for S.k, S.v in S.GRID.items():
        print(' -', S.k, '=', S.v)
    S.RUN_VERSION = 'standalone_nuclear_evo_xtask_screen_CLEAN_V3_20260914'
    S.cv, S.folds = S.suite.run_dev_cv_only(S.ctx, S.GRID, models=('MLP',), run_version=S.RUN_VERSION)
    S.base_name = 'N_BASE_PATCH_ORGANIZATION_CLEANV3'
    S.base = float(S.cv.loc[S.cv.representation.eq(S.base_name), 'CV_AUPRC'].iloc[0])
    S.cv = S.cv.copy()
    S.cv['delta_vs_base'] = S.cv['CV_AUPRC'] - S.base
    S.cv = S.cv.sort_values('CV_AUPRC', ascending=False).reset_index(drop=True)
    print('\nCLEAN V3 NUCLEAR RESULT')
    display(S.cv)
    display(S.folds)
    S.cv.to_csv(S.ADDIN_ROOT / 'REPRESENTATION_SCREEN_CLEAN_V3.csv', index=False)
    S.folds.to_csv(S.ADDIN_ROOT / 'REPRESENTATION_SCREEN_FOLDS_CLEAN_V3.csv', index=False)
    S.best = S.cv.iloc[0]
    S.xt = S.cv[S.cv.representation.eq('N_BASE_PLUS_XTASK_SMALL2_CLEANV3')]
    S.xt_score = float(S.xt.CV_AUPRC.iloc[0])
    if S.EVO_OK:
        S.evo_row = S.cv[S.cv.representation.eq('N_BASE_PLUS_EVO_CLEANV3')]
        S.evo_score = float(S.evo_row.CV_AUPRC.iloc[0])
        S.combo_row = S.cv[S.cv.representation.eq('N_BASE_PLUS_EVO_XTASK_SMALL2_CLEANV3')]
        S.combo_score = float(S.combo_row.CV_AUPRC.iloc[0])
    else:
        S.evo_score = None
        S.combo_score = None
    S.decision = {'run_version': S.RUN_VERSION, 'base_dev_auprc': S.base, 'xtask_small2_dev_auprc': S.xt_score, 'xtask_small2_gain_vs_base': S.xt_score - S.base, 'evo_dev_auprc': S.evo_score, 'evo_gain_vs_base': None if S.evo_score is None else S.evo_score - S.base, 'evo_plus_xtask_dev_auprc': S.combo_score, 'evo_plus_xtask_gain_vs_base': None if S.combo_score is None else S.combo_score - S.base, 'best_representation': str(S.best.representation), 'best_dev_auprc': float(S.best.CV_AUPRC), 'best_gain_vs_base': float(S.best.delta_vs_base), 'promote_best_if_gain_ge_0.002': bool(S.best.delta_vs_base >= 0.002), 'EVO_available': S.EVO_OK, 'cross_task_source_columns': ['prob_V1', 'prob_MS']}
    (S.ADDIN_ROOT / 'DECISION_CLEAN_V3.json').write_text(json.dumps(S.decision, indent=2))
    print('\nDECISION')
    print(json.dumps(S.decision, indent=2))
    print('Saved:', S.ADDIN_ROOT)

# NUCLEAR FINAL ADD-IN — PATCHHG ATTENTION + EVO48
def stage_11():
    print('=' * 110)
    print('NUCLEAR FINAL TEST — PATCHHG ATTENTION + EVO48')
    print('=' * 110)
    S.REQ = ['BASE', 'ctx', 'train_df', 'SPLIT_DF', 'STORE_KEY', 'OFFICIAL_FOLDS', 'get_block', 'DEVICE', 'V1_OOF', 'MS_OOF', 'Y_DEV']
    S.missing = [x for x in S.REQ if x not in globals()]
    if S.missing:
        raise RuntimeError('Run Nuclear standalone notebook through CLEAN V3 first. Missing: ' + ', '.join(S.missing))
    if not bool(globals().get('EVO_AVAILABLE', False)):
        raise RuntimeError('EVO_AVAILABLE=False. This final experiment requires EVO48.')
    S._ = get_block(S.ctx.stores[S.STORE_KEY['DevSet1010']], S.train_df.iloc[0].prot_id, 'EVO')
    S.FINAL_ROOT = S.RESULTS_ROOT / 'ADDIN_STANDALONE_NUCLEAR_EVO_XTASK' / 'FINAL_PATCHHG_EVO48'
    S.FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    S.STATE_ROOT = S.FINAL_ROOT / 'states'
    S.STATE_ROOT.mkdir(parents=True, exist_ok=True)
    S.PATCH_CACHE = S.CACHE_ROOT / 'nuclear_ultimate_v1' / 'patch_hypergraph_v1'
    S.SEED_FINAL = 2026
    S.PATCH_LATENT = 192
    S.PATCH_EPOCHS = 32
    S.PATCH_PATIENCE = 6
    S.PATCH_PROTEIN_BATCH = 4
    S.PATCH_LR = 0.0003
    S.PATCH_WEIGHT_DECAY = 0.0001
    S.MIN_GAIN = 0.002
    S.BASE_REP_NAME = 'PATCHHG_BASE_RECHECK_FINALV1'
    S.EVO_REP_NAME = 'PATCHHG_EVO48_FINALV1'
    S.BASE_MODALITY_SPEC = {'ESM': ['ESM'], 'ELECTRO': ['ACCESS'], 'PATCH': ['NA_PATCH3D40', 'NA_TOPO12']}
    S.EVO_MODALITY_SPEC = {'ESM': ['ESM'], 'ELECTRO': ['ACCESS'], 'PATCH': ['NA_PATCH3D40', 'NA_TOPO12'], 'BIO': ['EVO']}
    print('\nLoading V1 patch graph cache for Dev only...')
    S.PATCH_RAM_FINAL = load_patch_ram('DevSet1010')
    print('Dev patch cache loaded:', len(S.PATCH_RAM_FINAL))
    print('\n' + '=' * 110)
    print('MATCHED BASE PATCHHG')
    print('=' * 110)
    S.base_summary, S.base_folds, S.base_oof, S.base_states = run_cv_final(S.BASE_REP_NAME, S.BASE_MODALITY_SPEC)
    print('\n' + '=' * 110)
    print('PATCHHG + EVO48')
    print('=' * 110)
    S.evo_summary, S.evo_folds, S.evo_oof, S.evo_states = run_cv_final(S.EVO_REP_NAME, S.EVO_MODALITY_SPEC)
    S.matched = pd.DataFrame([S.base_summary, S.evo_summary])
    S.base_score = float(S.base_summary['CV_AUPRC_mean'])
    S.evo_score = float(S.evo_summary['CV_AUPRC_mean'])
    S.evo_gain = S.evo_score - S.base_score
    S.fold_cmp = S.base_folds[['fold', 'AUPRC']].rename(columns={'AUPRC': 'AUPRC_BASE'}).merge(S.evo_folds[['fold', 'AUPRC']].rename(columns={'AUPRC': 'AUPRC_EVO'}), on='fold')
    S.fold_cmp['delta_EVO_minus_BASE'] = S.fold_cmp['AUPRC_EVO'] - S.fold_cmp['AUPRC_BASE']
    print('\nMATCHED DEV RESULT')
    display(S.matched)
    display(S.fold_cmp)
    print('Base PatchHG mean AUPRC:', S.base_score)
    print('EVO PatchHG mean AUPRC:', S.evo_score)
    print('EVO gain:', S.evo_gain)
    print('EVO wins folds:', int((S.fold_cmp['delta_EVO_minus_BASE'] > 0).sum()), '/5')
    S.matched.to_csv(S.FINAL_ROOT / 'MATCHED_PATCHHG_BASE_VS_EVO.csv', index=False)
    S.fold_cmp.to_csv(S.FINAL_ROOT / 'MATCHED_PATCHHG_FOLDS.csv', index=False)
    S.FIDX_FINAL = fold_indices_final()
    S.ensemble_rows = []
    for S.w in np.linspace(0.0, 1.0, 21):
        S.p = S.w * S.evo_oof + (1.0 - S.w) * S.MS_OOF
        S.aps = [float(average_precision_score(S.Y_DEV[idx], S.p[idx])) for _, idx in sorted(S.FIDX_FINAL.items())]
        S.ensemble_rows.append({'w_EVO_PATCHHG': float(S.w), 'w_MS': float(1.0 - S.w), 'CV_AUPRC_mean': float(np.mean(S.aps)), 'CV_AUPRC_std': float(np.std(S.aps, ddof=1)), 'OOF_AUPRC': float(average_precision_score(S.Y_DEV, S.p))})
    S.ens_search = pd.DataFrame(S.ensemble_rows).sort_values(['CV_AUPRC_mean', 'w_EVO_PATCHHG'], ascending=[False, False]).reset_index(drop=True)
    display(S.ens_search.head(10))
    S.ens_search.to_csv(S.FINAL_ROOT / 'EVO_PATCHHG_MS_WEIGHT_SEARCH.csv', index=False)
    S.best_ens = S.ens_search.iloc[0]
    S.W_EVO = float(S.best_ens['w_EVO_PATCHHG'])
    S.W_MS = float(S.best_ens['w_MS'])
    S.NEW_FINAL_DEV = float(S.best_ens['CV_AUPRC_mean'])
    S.NEW_FINAL_OOF = S.W_EVO * S.evo_oof + S.W_MS * S.MS_OOF
    S.old_rows = []
    for S.w in np.linspace(0.0, 1.0, 21):
        S.p = S.w * S.V1_OOF + (1.0 - S.w) * S.MS_OOF
        S.aps = [float(average_precision_score(S.Y_DEV[idx], S.p[idx])) for _, idx in sorted(S.FIDX_FINAL.items())]
        S.old_rows.append(float(np.mean(S.aps)))
    S.OLD_FINAL_DEV = float(np.max(S.old_rows))
    S.gain_vs_old_final = S.NEW_FINAL_DEV - S.OLD_FINAL_DEV
    S.PROMOTE_EVO_MODEL = S.evo_gain >= S.MIN_GAIN
    S.PROMOTE_NEW_FINAL = S.gain_vs_old_final >= S.MIN_GAIN
    print('\n' + '=' * 110)
    print('DEV FINAL DECISION')
    print('=' * 110)
    print('Matched EVO PatchHG gain vs matched base:', S.evo_gain)
    print('Promote EVO architecture?', S.PROMOTE_EVO_MODEL)
    print('Best new ensemble:', f'{S.W_EVO:.2f} EVO-PatchHG + {S.W_MS:.2f} MS')
    print('Old final V1+MS Dev:', S.OLD_FINAL_DEV)
    print('New EVO-PatchHG+MS Dev:', S.NEW_FINAL_DEV)
    print('Gain new vs old final:', S.gain_vs_old_final)
    print('Promote new final ensemble?', S.PROMOTE_NEW_FINAL)
    S.NEW_THR, S.NEW_OOF_MCC = best_mcc_threshold_local(S.Y_DEV, S.NEW_FINAL_OOF)
    print('New final frozen Dev-MCC threshold:', S.NEW_THR)
    print('New final Dev OOF MCC:', S.NEW_OOF_MCC)
    S.external_rows = []
    if S.PROMOTE_NEW_FINAL:
        print('\nDev rule passed. Running external evaluation...')
        for S.split in ['TestSet300', 'TestSetNew46']:
            S.df = S.SPLIT_DF[S.split]
            S.store = S.ctx.stores[S.STORE_KEY[S.split]]
            try:
                S._ = get_block(S.store, S.df.iloc[0].prot_id, 'EVO')
            except Exception as e:
                raise RuntimeError(f'{S.split}: EVO48 not attached. Attach cached EVO to external sets before rerunning this cell.') from e
            S.ext_ram = load_patch_ram(S.split)
            S.PATCH_RAM_FINAL.update(S.ext_ram)
            S.ds = EvoPatchDataset(S.split, S.df, S.EVO_MODALITY_SPEC)
            S.evo_preds = []
            for S.fold, S.state in S.evo_states.items():
                S.model = NuclearPatchHypergraphFinal(modality_dims(S.EVO_MODALITY_SPEC), latent=S.PATCH_LATENT, drop=0.2).to(S.DEVICE)
                S.model.load_state_dict(S.state)
                S.p, S._ = predict_final(S.model, S.ds)
                S.evo_preds.append(S.p)
                del S.model
                if S.DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
            S.p_evo = np.mean(np.stack(S.evo_preds), axis=0)
            S.paper_pred_fp = S.RESULTS_ROOT / 'NUCLEAR_FINAL_PAPER_READY' / f'FINAL_RESIDUE_PREDICTIONS_{S.split}.csv.gz'
            if not S.paper_pred_fp.exists():
                raise FileNotFoundError(S.paper_pred_fp)
            S.oldpred = pd.read_csv(S.paper_pred_fp)
            if 'prob_MS' not in S.oldpred.columns:
                raise RuntimeError(f'{S.paper_pred_fp} lacks prob_MS')
            S.p_ms = S.oldpred['prob_MS'].to_numpy(dtype=np.float64)
            if len(S.p_ms) != len(S.p_evo):
                raise RuntimeError(f'{S.split}: EVO/MS prediction length mismatch')
            S.p_new = S.W_EVO * S.p_evo + S.W_MS * S.p_ms
            S.y = np.concatenate(S.df.nuclear_binding.apply(lambda x: np.asarray(x, dtype=np.int8)).tolist())
            for S.rule, S.thr in [('fixed_0.5', 0.5), ('Dev_OOF_MCC', S.NEW_THR)]:
                S.m = metrics_full(S.y, S.p_new, S.thr)
                S.external_rows.append({'dataset': S.split, 'method': f'{S.W_EVO:.2f} EVO-PatchHG + {S.W_MS:.2f} MS', 'threshold_rule': S.rule, 'threshold': S.thr})
            S.out = S.oldpred[['prot_id', 'residue_index', 'y_true']].copy()
            S.out['prob_EVO_PatchHG'] = S.p_evo
            S.out['prob_MS'] = S.p_ms
            S.out['prob_NEW_FINAL'] = S.p_new
            S.out.to_csv(S.FINAL_ROOT / f'NEW_FINAL_PREDICTIONS_{S.split}.csv.gz', index=False)
        S.external_df = pd.DataFrame(S.external_rows)
        print('\nEXTERNAL NEW FINAL')
        display(S.external_df)
        S.external_df.to_csv(S.FINAL_ROOT / 'EXTERNAL_NEW_FINAL.csv', index=False)
    else:
        print('\nNew ensemble did not beat old final by >=0.002 Dev AUPRC. External evaluation intentionally skipped.')
    S.decision = {'matched_base_patchhg_dev_auprc': S.base_score, 'evo_patchhg_dev_auprc': S.evo_score, 'evo_patchhg_gain_vs_matched_base': S.evo_gain, 'evo_patchhg_fold_wins': int((S.fold_cmp['delta_EVO_minus_BASE'] > 0).sum()), 'promote_evo_architecture': bool(S.PROMOTE_EVO_MODEL), 'old_final_v1_ms_dev_auprc': S.OLD_FINAL_DEV, 'new_w_EVO_PatchHG': S.W_EVO, 'new_w_MS': S.W_MS, 'new_final_dev_auprc': S.NEW_FINAL_DEV, 'new_final_gain_vs_old_final': S.gain_vs_old_final, 'promote_new_final': bool(S.PROMOTE_NEW_FINAL), 'new_final_dev_mcc_threshold': S.NEW_THR, 'new_final_dev_oof_mcc': S.NEW_OOF_MCC}
    (S.FINAL_ROOT / 'FINAL_DECISION.json').write_text(json.dumps(S.decision, indent=2))
    print('\nFINAL DECISION')
    print(json.dumps(S.decision, indent=2))
    print('\nSaved:', S.FINAL_ROOT)

STAGES = [stage_01, stage_02, stage_03, stage_04, stage_05, stage_07, stage_08, stage_10, stage_11]


import sys
from lc_bind.paths import bind_session


def _apply(overrides):
    for key, value in overrides.items():
        if value is not None:
            setattr(S, key, value)


def run(paths, **overrides):
    bind_session(paths, S)
    gear = paths.gear_dir
    if gear is not None and (gear / 'gear_ultimate').exists():
        text = str(gear.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)
    else:
        paths.require_gear()
    prepare()
    _apply(overrides)
    stage_01()
    _apply(overrides)
    stage_02()
    _apply(overrides)
    stage_03()
    _apply(overrides)
    stage_04()
    _apply(overrides)
    stage_05()
    _apply(overrides)
    stage_07()
    _apply(overrides)
    stage_08()
    _apply(overrides)
    stage_10()
    _apply(overrides)
    stage_11()
    _apply(overrides)
    S.RESULT_KEY = 'nuclear_xtask'
    return S

