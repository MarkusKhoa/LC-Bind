"""Extracted from NUCLEAR_ULTIMATE_V1_DATA_DRIVEN_PATCH-4.ipynb. Section: pipeline_stages."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Chain, Counter, DataLoader, Dataset, F, Model, PDBIO, PDBParser, Path, Structure, Subset, accuracy_score, ast, average_precision_score, balanced_accuracy_score, cKDTree, deepcopy, defaultdict, f1_score, hashlib, json, math, matthews_corrcoef, nn, np, os, pd, platform, precision_score, recall_score, roc_auc_score, seq1, shutil, subprocess, sys, tarfile, torch
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

# 0. DRIVE / INSTALL / IMPORT
def stage_01():
    S.BASE = Path(str(S.BASE))
    S.ULTIMATE_CANDIDATES = [S.GEAR_DIR, S.BASE / 'GEAR_ULTIMATE_OTHER_BINDINGS', S.BASE / 'GEAR_ULTIMATE']
    S.ULTIMATE_DIR = next((p for p in S.ULTIMATE_CANDIDATES if p is not None and (Path(p) / 'gear_ultimate').exists()), None)
    if S.ULTIMATE_DIR is None:
        raise FileNotFoundError('GEAR backend not found under GEAR_ULTIMATE_OTHER_BINDINGS or GEAR_ULTIMATE.')
    S.req = S.ULTIMATE_DIR / 'requirements.txt'
    if S.req.exists():
        pass
    sys.path.insert(0, str(S.ULTIMATE_DIR))
    try:
        from gear_ultimate import GEAR_ULTIMATE_VERSION as GEAR_VERSION
        S.GEAR_VERSION = GEAR_VERSION
    except Exception:
        from gear_ultimate import __version__ as GEAR_VERSION
        S.GEAR_VERSION = GEAR_VERSION
    from gear_ultimate import gear_core as core
    S.core = core
    from gear_ultimate import gear_structure as structure
    S.structure = structure
    from gear_ultimate import gear_mechanistic as mech
    S.mech = mech
    from gear_ultimate import gear_graph as graph
    S.graph = graph
    from gear_ultimate import gear_evolution as evo
    S.evo = evo
    from gear_ultimate import gear_suite as suite
    S.suite = suite
    from gear_ultimate import gear_reporting as reporting
    S.reporting = reporting
    S.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if S.DEVICE.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    S.RUN_ROOT = S.RESULTS_ROOT / 'NUCLEAR_ULTIMATE_V1_DATA_DRIVEN_PATCH'
    S.RUN_ROOT.mkdir(parents=True, exist_ok=True)
    S.CACHE_ROOT = S.CACHE_ROOT / 'nuclear_ultimate_v1'
    S.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    print('BASE:', S.BASE)
    print('GEAR backend:', S.ULTIMATE_DIR)
    print('GEAR version:', S.GEAR_VERSION)
    print('Device:', S.DEVICE)
    print('Run root:', S.RUN_ROOT)

# 1. MASTER SWITCHES
def stage_03():
    S.ESM_BATCH_SIZE = 8
    S.RUN_FEATURE_GENERATION = True
    S.RUN_FAMILY_SCREEN = True
    S.RUN_COMBINATION_LADDER = True
    S.RUN_CUSTOM_ARCHITECTURES = True
    S.RUN_PATCH_HYPERGRAPH = True
    S.RUN_FINAL_EXTERNAL = True
    S.ATTACH_EVO_IF_CACHED = True
    S.ATTACH_ESMIF_IF_CACHED = True
    S.FORCE_REBUILD_NA_FEATURES = False
    S.FORCE_REBUILD_PATCHES = False
    S.MIN_MEAN_AUPRC_GAIN = 0.002
    S.GLOBAL_COMPLEXITY_MARGIN = 0.002
    S.CUSTOM_BATCH = 4096
    S.CUSTOM_EPOCHS = 45
    S.CUSTOM_PATIENCE = 7
    S.CUSTOM_LR = 0.0003
    S.CUSTOM_WEIGHT_DECAY = 0.0001
    S.SEED = 2026
    S.PATCH_RADIUS_A = 10.0
    S.PATCH_MAX_PER_PROTEIN = 128
    S.PATCH_PROTEIN_BATCH = 4
    S.PATCH_EPOCHS = 32
    S.PATCH_PATIENCE = 6
    S.PATCH_LR = 0.0003
    S.PATCH_WEIGHT_DECAY = 0.0001
    S.PATCH_LATENT = 192
    print('Full run:', {'feature_generation': S.RUN_FEATURE_GENERATION, 'family_screen': S.RUN_FAMILY_SCREEN, 'combination_ladder': S.RUN_COMBINATION_LADDER, 'custom_architectures': S.RUN_CUSTOM_ARCHITECTURES, 'patch_hypergraph': S.RUN_PATCH_HYPERGRAPH, 'external': S.RUN_FINAL_EXTERNAL})

# 2. PREPARE / REUSE NUCLEAR GEAR CONTEXT
def stage_04():
    S.ctx = S.suite.prepare_context(base=S.BASE, run_version='nuclear_ultimate_v1_context', label_column='nuclear_binding', models=('MLP',), ensure_base=True, add_mech=True, add_multi=True, add_seqctx=True, esm_batch_size=S.ESM_BATCH_SIZE, device=S.DEVICE)
    display(S.suite.audit_context(S.ctx))
    display(S.reporting.dimension_table(S.ctx.stores['train']))
    assert len(S.ctx.stores['train']) == 1010
    assert len(S.ctx.stores['test300']) == 300
    assert len(S.ctx.stores['test46']) == 46
    print('Context sizes: PASS')

STAGES = [stage_01, stage_03, stage_04]

