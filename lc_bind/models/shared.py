"""Fusion models and the residue-level trainer pieces shared by Nuclear and Small."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset, Subset

from lc_bind.session import S

def prepare():
    return None

class ConcatMLP(nn.Module):

    def __init__(self, dims, drop=0.2):
        super().__init__()
        self.keys = sorted(dims)
        self.norm = nn.ModuleDict({k: nn.LayerNorm(d) for k, d in dims.items()})
        d = sum(dims.values())
        self.head = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(drop), nn.Linear(512, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))

    def forward(self, x):
        z = torch.cat([self.norm[k](x[k]) for k in self.keys], dim=-1)
        return self.head(z).squeeze(-1)

class ProjectedConcat(nn.Module):

    def __init__(self, dims, latent=192, drop=0.2):
        super().__init__()
        self.keys = sorted(dims)
        self.enc = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, latent), nn.GELU(), nn.Dropout(drop)) for k, d in dims.items()})
        self.head = nn.Sequential(nn.Linear(latent * len(dims), 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 1))

    def forward(self, x):
        z = torch.cat([self.enc[k](x[k]) for k in self.keys], dim=-1)
        return self.head(z).squeeze(-1)

class GatedFusion(nn.Module):

    def __init__(self, dims, latent=192, drop=0.2):
        super().__init__()
        self.keys = sorted(dims)
        self.enc = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, latent), nn.GELU(), nn.Dropout(drop)) for k, d in dims.items()})
        self.gate = nn.ModuleDict({k: nn.Linear(latent, 1) for k in dims})
        self.head = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, 192), nn.GELU(), nn.Dropout(drop), nn.Linear(192, 1))

    def encode(self, x):
        zz = []
        gg = []
        for k in self.keys:
            z = self.enc[k](x[k])
            zz.append(z)
            gg.append(self.gate[k](z))
        Z = torch.stack(zz, dim=1)
        G = torch.softmax(torch.cat(gg, dim=1), dim=1).unsqueeze(-1)
        return ((Z * G).sum(dim=1), G.squeeze(-1))

    def forward(self, x):
        h, _ = self.encode(x)
        return self.head(h).squeeze(-1)

class ModalityAttention(nn.Module):

    def __init__(self, dims, latent=192, heads=6, drop=0.2):
        super().__init__()
        assert latent % heads == 0
        self.keys = sorted(dims)
        self.enc = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, latent), nn.GELU(), nn.Dropout(drop)) for k, d in dims.items()})
        self.attn = nn.MultiheadAttention(latent, heads, dropout=drop, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, 192), nn.GELU(), nn.Dropout(drop), nn.Linear(192, 1))

    def forward(self, x):
        Z = torch.stack([self.enc[k](x[k]) for k in self.keys], dim=1)
        A, _ = self.attn(Z, Z, Z, need_weights=False)
        h = (Z + A).mean(dim=1)
        return self.head(h).squeeze(-1)

class MultiArrayDataset(Dataset):

    def __init__(self, mods, y):
        self.mods = mods
        self.y = y
        self.keys = sorted(mods)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = {k: torch.from_numpy(np.asarray(self.mods[k][i], dtype=np.float32)) for k in self.keys}
        return (x, torch.tensor(self.y[i], dtype=torch.float32))

def collate_multi(batch):
    keys = sorted(batch[0][0])
    x = {k: torch.stack([b[0][k] for b in batch], dim=0) for k in keys}
    y = torch.stack([b[1] for b in batch])
    return (x, y)

def ap(y, p):
    return float(average_precision_score(y, p))

def predict_flat(model, dataset, indices=None, batch_size=8192):
    ds = dataset if indices is None else Subset(dataset, indices.tolist() if isinstance(indices, np.ndarray) else indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_multi)
    out = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = {k: v.to(S.DEVICE, non_blocking=True) for k, v in x.items()}
            out.append(torch.sigmoid(model(x)).detach().cpu().numpy())
    return np.concatenate(out)

