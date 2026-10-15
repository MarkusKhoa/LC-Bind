"""Extracted from MASTER_BINDING_BASELINE_REPRODUCTION_300_46_FULL_v10_BINDNODE24_FIXED-2.ipynb. Section: setup."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Bio, Chain, Model, PDBIO, PDBParser, Path, SeqIO, Structure, T5EncoderModel, T5Tokenizer, accuracy_score, ast, auc, average_precision_score, balanced_accuracy_score, brier_score_loss, deepcopy, f1_score, fbeta_score, h5py, hashlib, importlib, jaccard_score, json, matthews_corrcoef, np, os, pd, pickle, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve, seq1, shutil, student_t, subprocess, sys, time, torch, tqdm
    from pathlib import Path
    import os, sys, json, ast, shutil, subprocess, pickle, hashlib, time
    import numpy as np
    import pandas as pd
    import importlib.util
    import subprocess
    import sys
    from Bio import SeqIO
    import Bio
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score, fbeta_score, matthews_corrcoef, precision_score, recall_score, accuracy_score, balanced_accuracy_score, jaccard_score, brier_score_loss, auc, roc_curve
    from scipy.stats import t as student_t
    import h5py, torch
    from transformers import T5EncoderModel, T5Tokenizer
    from tqdm.auto import tqdm
    from copy import deepcopy
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.SeqUtils import seq1
    import shutil
    import json
    import hashlib

def sh(cmd, cwd=None, check=True, capture=False):
    print('+', cmd)
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check, text=True, stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)

def env_path(name):
    return S.LOCAL_MAMBA_ROOT / 'envs' / name

def env_healthy(name):
    """
    Check that an existing local environment can actually execute Python.
    A directory merely existing is not enough.
    """
    p = env_path(name)
    if not p.exists():
        return False
    r = subprocess.run(f'MAMBA_ROOT_PREFIX="{S.LOCAL_MAMBA_ROOT}" micromamba run -n {name} python -c "import sys; print(sys.executable)"', shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        print(f'BROKEN ENV {name}: return code {r.returncode}')
        print(r.stderr[-2000:])
        return False
    return True

def create_env(name, python='3.10'):
    if env_healthy(name):
        print('REUSE LOCAL ENV', name)
        return
    p = env_path(name)
    if p.exists():
        print('REMOVE BROKEN LOCAL ENV', name)
        sh(f'rm -rf "{p}"')
    sh(f'MAMBA_ROOT_PREFIX="{S.LOCAL_MAMBA_ROOT}" micromamba create -y -n {name} -c conda-forge python={python} pip')
    if not env_healthy(name):
        raise RuntimeError(f'Environment {name} was created but cannot execute Python.')

def mex(name, cmd, cwd=None, check=True, capture=False):
    """
    Execute a command inside a LOCAL micromamba environment.
    Use `python -m pip` rather than the pip entry-point whenever possible.
    """
    prefix = f'MAMBA_ROOT_PREFIX="{S.LOCAL_MAMBA_ROOT}" micromamba run -n {name}'
    return sh(f'{prefix} {cmd}', cwd=cwd, check=check, capture=capture)

def ensure_kernel_package(import_name, pip_spec):
    if importlib.util.find_spec(import_name) is not None:
        print(f'Kernel dependency OK: {import_name}')
        return
    print(f'Kernel dependency missing: {import_name}')
    print(f'Installing into active notebook kernel: {pip_spec}')
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pip_spec], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise RuntimeError(f'Failed to install {pip_spec} into notebook kernel (return code {r.returncode})')
    importlib.invalidate_caches()
    if importlib.util.find_spec(import_name) is None:
        raise RuntimeError(f'{pip_spec} installed without an obvious pip error, but import {import_name} is still unavailable in this kernel.')
    print(f'Installed kernel dependency: {pip_spec}')

# 0. COLAB + DRIVE
def stage_02():
    S.BASE = Path(str(S.BASE))
    S.WORK = S.BASE / 'baseline_reproduction_master_v6'
    S.REPOS = S.WORK / 'repos'
    S.ENVS = S.WORK / 'envs'
    S.CACHE = S.WORK / 'cache'
    S.PRED = S.WORK / 'predictions'
    S.RESULTS = S.WORK / 'results'
    S.MANIFEST = S.WORK / 'manifest'
    S.CANON = S.WORK / 'canonical_inputs'
    for S.p in [S.WORK, S.REPOS, S.ENVS, S.CACHE, S.PRED, S.RESULTS, S.MANIFEST, S.CANON]:
        S.p.mkdir(parents=True, exist_ok=True)
    S.TRAIN_CSV = S.DATA_ROOT / 'development_set/full_grouped_train_binding_sites_df.csv'
    S.TEST300_CSV = S.DATA_ROOT / 'development_set/full_grouped_test_binding_sites_df.csv'
    S.TEST46_CSV = S.DATA_ROOT / 'independent_set/grouped_test_46_new_binding_sites.csv'
    print('BASE   =', S.BASE)
    print('WORK   =', S.WORK)
    print('RESULTS=', S.RESULTS)

# 1. MASTER SWITCHES — v6 ONE-CLICK DEFAULTS
def stage_03():
    S.RUN_BIND21 = True
    S.RUN_BINDNODE24 = True
    S.RUN_OOF_THRESHOLDS = True
    S.RUN_ESMBIND_TRANSFER = True
    S.RUN_METALBIND_TRANSFER = True
    S.RUN_METALBIND_NATIVE = False
    S.RUN_METALBIND_NATIVE_REFERENCE_CHECK = False
    S.REBUILD_PROTT5 = False
    S.REBUILD_ESMBIND_ESM2 = False
    S.REBUILD_ESMBIND_ESMIF = False
    S.REBUILD_METALBIND_ESMC = False
    S.REBUILD_METALBIND_SURFACE = False
    S.FIXED_THRESHOLD = 0.5
    S.BOOTSTRAP_REPS = 2000
    S.RANDOM_SEED = 42
    S.REPRO_TOLERANCE_ABS = 0.04
    S.REQUIRE_GPU_FOR_METALBIND = True

# 2A. CLONE + PIN
def stage_06():
    S.REPO_SPECS = {'bind21': ('https://github.com/Rostlab/bindPredict.git', 'c9b12e7369641a9cbc3afbef85ed2e55ea433aff'), 'bindnode24': ('https://github.com/erckert/bindNode24.git', '86b85818dc5ec6567ca425a0c98dc4930ea62394'), 'esmbind': ('https://github.com/Structurebiology-BNL/ESMBind.git', '5d81fb5b3965f973bcbbdde443132b31bef07348'), 'metalbind': ('https://github.com/zpliulab/MetalBind.git', '6d2d13c4566477a1b79ac8e22ac1aaa6840c25a0')}
    for S.name, (S.url, S.sha) in S.REPO_SPECS.items():
        S.dst = S.REPOS / S.name
        if not S.dst.exists():
            sh(f'git clone {S.url} "{S.dst}"')
        sh('git fetch --all --tags', cwd=S.dst)
        sh(f'git checkout --detach {S.sha}', cwd=S.dst)
        S.got = subprocess.check_output('git rev-parse HEAD', cwd=S.dst, shell=True, text=True).strip()
        assert S.got == S.sha, (S.name, S.got, S.sha)
        print(S.name, '=>', S.got)
    (S.MANIFEST / 'repo_pins.json').write_text(json.dumps({k: {'url': v[0], 'sha': v[1]} for k, v in S.REPO_SPECS.items()}, indent=2))

# 3A. MICROMAMBA HELPERS — LOCAL /content ENVIRONMENTS
def stage_08():
    S.MAMBA = Path('/usr/local/bin/micromamba')
    if not S.MAMBA.exists():
        sh('curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /tmp bin/micromamba')
        sh('cp /tmp/bin/micromamba /usr/local/bin/micromamba')
        sh('chmod +x /usr/local/bin/micromamba')
    S.LOCAL_MAMBA_ROOT = Path(str(S.CACHE_ROOT / 'micromamba_binding_baselines'))
    S.LOCAL_MAMBA_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ['MAMBA_ROOT_PREFIX'] = str(S.LOCAL_MAMBA_ROOT)
    print('Micromamba executable :', S.MAMBA)
    print('Micromamba env root   :', S.LOCAL_MAMBA_ROOT)
    print('Persistent results    :', S.RESULTS)

# 4B.1 NOTEBOOK-KERNEL DEPENDENCY GUARD
def stage_15():
    ensure_kernel_package('Bio', 'biopython>=1.83,<2')
    print('Biopython:', Bio.__version__)
    print('Bio.SeqIO import: OK')

STAGES = [stage_02, stage_03, stage_06, stage_08, stage_15]

