"""Extracted from MASTER_BINDING_BASELINE_REPRODUCTION_300_46_FULL_v10_BINDNODE24_FIXED-2.ipynb. Section: esmbind."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Bio, Chain, Model, PDBIO, PDBParser, Path, SeqIO, Structure, T5EncoderModel, T5Tokenizer, accuracy_score, ast, auc, average_precision_score, balanced_accuracy_score, brier_score_loss, create_env, deepcopy, evaluate_probability_dict, f1_score, fbeta_score, h5py, hashlib, importlib, jaccard_score, json, matthews_corrcoef, mex, np, os, pd, pickle, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve, save_probs, seq1, shutil, student_t, subprocess, sys, time, torch, tqdm
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
    from lc_bind.models.baselines.metrics import evaluate_probability_dict, save_probs
    from lc_bind.models.baselines.setup import create_env, mex

def _write_esmbind_fasta(df, path):
    """
    Full FASTA for ESM-IF and final ESMBind inference.
    Keep ALL protein IDs, including exact duplicate sequences.
    """
    with open(path, 'w') as f:
        for r in df.itertuples():
            f.write(f'>sp|{r.prot_id}|{r.prot_id}\n{r.sequence}\n')

def _valid_npy(path, shape):
    path = Path(path)
    if not path.exists():
        return False
    try:
        a = np.load(path, mmap_mode='r')
        return a.shape == shape and np.all(np.isfinite(a))
    except Exception:
        return False

def _prepare_unique_esm2_fasta(dataset, df, inp, rebuild=False):
    """
    Official ESMBind ESM2 extractor rejects duplicate sequences.

    ESM2 is sequence-only, therefore exact duplicate amino-acid
    sequences legitimately share the same ESM2 representation.

    We:
      1. retain ALL benchmark proteins;
      2. generate ESM2 once per unique sequence;
      3. broadcast the embedding back to every original protein ID.
    """
    esm_dir = inp / 'esm'
    esm_dir.mkdir(parents=True, exist_ok=True)
    seq_groups = {}
    for r in df.itertuples():
        seq_groups.setdefault(str(r.sequence), []).append(str(r.prot_id))
    duplicate_groups = {seq: ids for seq, ids in seq_groups.items() if len(ids) > 1}
    print(f'{dataset}: proteins={len(df)}, unique_sequences={len(seq_groups)}, duplicate_groups={len(duplicate_groups)}')
    reps_to_generate = []
    for seq, ids in seq_groups.items():
        expected = (len(seq), 1280)
        source = None
        if not rebuild:
            for pid in ids:
                fp = esm_dir / f'{pid}.npy'
                if _valid_npy(fp, expected):
                    source = fp
                    break
        if source is not None:
            arr = np.load(source).astype(np.float32, copy=False)
            for pid in ids:
                dst = esm_dir / f'{pid}.npy'
                if not _valid_npy(dst, expected):
                    np.save(dst, arr)
        else:
            reps_to_generate.append((ids[0], seq, ids))
    unique_fasta = inp / f'{dataset}_unique_for_esm2.fasta'
    with open(unique_fasta, 'w') as f:
        for rep_pid, seq, _ in reps_to_generate:
            f.write(f'>sp|{rep_pid}|{rep_pid}\n{seq}\n')
    return (unique_fasta, reps_to_generate)

def _broadcast_generated_esm2(dataset, inp, reps_to_generate):
    esm_dir = inp / 'esm'
    for rep_pid, seq, ids in reps_to_generate:
        src = esm_dir / f'{rep_pid}.npy'
        expected = (len(seq), 1280)
        if not _valid_npy(src, expected):
            raise RuntimeError(f'{dataset}/{rep_pid}: representative ESM2 missing/invalid after official extraction; expected {expected}')
        arr = np.load(src).astype(np.float32, copy=False)
        for pid in ids:
            if pid == rep_pid:
                continue
            dst = esm_dir / f'{pid}.npy'
            np.save(dst, arr)

def _audit_esmbind_features(dataset, df, inp):
    bad = []
    for r in df.itertuples():
        pid = str(r.prot_id)
        e1 = inp / 'esm' / f'{pid}.npy'
        e2 = inp / 'esm_if' / f'{pid}.npy'
        if not e1.exists() or not e2.exists():
            bad.append((pid, 'missing feature file'))
            continue
        a = np.load(e1, mmap_mode='r')
        b = np.load(e2, mmap_mode='r')
        if a.shape != (len(r.sequence), 1280):
            bad.append((pid, f'ESM2 {a.shape}'))
        elif not np.all(np.isfinite(a)):
            bad.append((pid, 'ESM2 non-finite'))
        if b.shape != (len(r.sequence), 512):
            bad.append((pid, f'ESMIF {b.shape}'))
        elif not np.all(np.isfinite(b)):
            bad.append((pid, 'ESMIF non-finite'))
    if bad:
        raise RuntimeError(f'{dataset}: ESMBind feature audit failed: {bad[:10]}')

def run_esmbind_transfer(dataset, df):
    print('\n' + '=' * 72)
    print('ESMBind:', dataset)
    print('=' * 72)
    inp = S.CACHE / f'esmbind_{dataset}'
    for sub in ['esm', 'esm_if', 'pdb']:
        (inp / sub).mkdir(parents=True, exist_ok=True)
    fasta = inp / f'{dataset}_esmbind.fasta'
    _write_esmbind_fasta(df, fasta)
    for r in df.itertuples():
        pid = str(r.prot_id)
        src = S.NORMALIZED_EXTERNAL_PDB[dataset][pid]
        dst = inp / 'pdb' / f'{pid}.pdb'
        if not dst.exists():
            shutil.copy2(src, dst)
    unique_fasta, reps_to_generate = _prepare_unique_esm2_fasta(dataset, df, inp, rebuild=S.REBUILD_ESMBIND_ESM2)
    if reps_to_generate:
        too_long = [(pid, len(seq)) for pid, seq, _ in reps_to_generate if len(seq) > 1022]
        if too_long:
            raise RuntimeError(f'{dataset}: sequence(s) >1022 aa exceed ESM2 t33 650M limit: {too_long[:10]}')
        print(f'{dataset}: generating ESM2 for {len(reps_to_generate)} unique missing sequence(s)')
        rr = mex('esmbind', f'python multi_modal_binding/get_esm_embedding.py "{unique_fasta}" "{inp / 'esm'}" --batch_size 1 --device cuda', cwd=S.ESMB, check=False, capture=True)
        if rr.stdout:
            print(rr.stdout)
        if rr.returncode != 0:
            raise RuntimeError(f'{dataset}: ESM2 generation failed with return code {rr.returncode}. Full subprocess log printed above.')
        _broadcast_generated_esm2(dataset, inp, reps_to_generate)
    bad_esm2 = []
    for r in df.itertuples():
        pid = str(r.prot_id)
        fp = inp / 'esm' / f'{pid}.npy'
        if not _valid_npy(fp, (len(r.sequence), 1280)):
            bad_esm2.append(pid)
    if bad_esm2:
        raise RuntimeError(f'{dataset}: missing/invalid ESM2: {bad_esm2[:10]}')
    print(f'{dataset}: ESM2 complete for {len(df)}/{len(df)} proteins.')
    failed = inp / 'esm_if' / 'failed_ids.txt'
    need_esmif = S.REBUILD_ESMBIND_ESMIF or any((not _valid_npy(inp / 'esm_if' / f'{r.prot_id}.npy', (len(r.sequence), 512)) for r in df.itertuples()))
    if need_esmif:
        if failed.exists():
            failed.unlink()
        rr = mex('esmbind', f'python multi_modal_binding/get_esm_if_embedding.py "{fasta}" "{inp / 'esm_if'}" "{inp / 'pdb'}" --chain_id A', cwd=S.ESMB, check=False, capture=True)
        if rr.stdout:
            print(rr.stdout)
        if rr.returncode != 0:
            raise RuntimeError(f'{dataset}: ESM-IF process failed with return code {rr.returncode}. Full subprocess log printed above.')
    if failed.exists():
        bad = [x.strip() for x in failed.read_text().splitlines() if x.strip()]
        if bad:
            raise RuntimeError(f'{dataset}: ESM-IF failed proteins cannot be dropped: {bad[:10]}')
    _audit_esmbind_features(dataset, df, inp)
    print(f'{dataset}: ESM2 + ESM-IF feature audit PASS.')
    conf = {'general': {'seed': 1, 'gpu_id': 0, 'debug': False}, 'inference': {'ensemble_path': str(S.ESMB / 'multi_modal_binding' / 'model' / 'trained_weights'), 'fasta_path': str(fasta), 'precomputed_feature': str(inp)}}
    confp = S.WORK / f'esmbind_{dataset}.json'
    confp.write_text(json.dumps(conf, indent=2))
    started = time.time()
    rr = mex('esmbind', f'python multi_modal_binding/inference.py --config "{confp}"', cwd=S.ESMB, check=False, capture=True)
    if rr.stdout:
        print(rr.stdout)
    if rr.returncode != 0:
        raise RuntimeError(f'{dataset}: official ESMBind inference failed with return code {rr.returncode}. Full subprocess log printed above.')
    cand = [p for p in S.ESMB.rglob('predictions.pkl') if p.stat().st_mtime >= started - 2]
    cand = sorted(cand, key=lambda p: p.stat().st_mtime, reverse=True)
    if not cand:
        raise RuntimeError(f'{dataset}: no fresh predictions.pkl produced')
    out_copy = S.PRED / 'ESMBind' / dataset
    out_copy.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cand[0], out_copy / 'predictions.pkl')
    with open(cand[0], 'rb') as f:
        z = pickle.load(f)
    ions = ['MG', 'FE', 'CU', 'CO', 'CA', 'MN', 'ZN']
    for key in ions + ['threshold_f1', 'threshold_mcc']:
        if key not in z:
            raise KeyError(f'{dataset}: ESMBind output missing key {key}')
    ion_probs = {ion: {} for ion in ions}
    for r in df.itertuples():
        pid = str(r.prot_id)
        for ion in ions:
            a = np.asarray(z[ion][pid], dtype=float)
            if a.shape != (len(r.sequence),):
                raise RuntimeError(f'{dataset}/{pid}/{ion}: prediction shape {a.shape}')
            if not np.all(np.isfinite(a)) or np.min(a) < 0 or np.max(a) > 1:
                raise RuntimeError(f'{dataset}/{pid}/{ion}: invalid probabilities')
            ion_probs[ion][pid] = a
    for ion, d in ion_probs.items():
        save_probs(d, out_copy / f'{ion}.csv.gz')
    pooled = {}
    for r in df.itertuples():
        pid = str(r.prot_id)
        pooled[pid] = np.maximum.reduce([ion_probs[ion][pid] for ion in ions])
    thresholds = {'threshold_f1': {k: float(v) for k, v in z['threshold_f1'].items()}, 'threshold_mcc': {k: float(v) for k, v in z['threshold_mcc'].items()}}
    (out_copy / 'author_native_ion_thresholds.json').write_text(json.dumps(thresholds, indent=2))
    print(f'{dataset}: official ESMBind inference PASS.')
    return (pooled, ion_probs, thresholds)

# CELL 1 — FINAL ESM-IF1 ENVIRONMENT FIX + SMOKE TEST
def stage_54():
    S.rr = mex('esmbind', 'python - <<\'PY\'\nimport sys\nimport subprocess\n\nsubprocess.check_call([\n    sys.executable,\n    "-m",\n    "pip",\n    "install",\n    "-q",\n    "--force-reinstall",\n    "numpy<2",\n    "biotite==0.41.2"\n])\n\nprint("Biotite 0.41.2 install PASS")\nPY', check=False, capture=True)
    print(S.rr.stdout)
    if S.rr.returncode != 0:
        raise RuntimeError(f'Biotite installation failed: return code {S.rr.returncode}')
    S.rr = mex('esmbind', 'python - <<\'PY\'\nimport sys\nimport subprocess\nimport torch\n\nprint("torch:", torch.__version__)\nprint("torch CUDA:", torch.version.cuda)\n\ntry:\n    import torch_scatter\n    print("Existing torch_scatter:", torch_scatter.__version__)\nexcept Exception:\n    cuda = torch.version.cuda\n\n    if cuda is None:\n        tag = "cpu"\n    elif cuda.startswith("12.4"):\n        tag = "cu124"\n    elif cuda.startswith("12.1"):\n        tag = "cu121"\n    elif cuda.startswith("11.8"):\n        tag = "cu118"\n    else:\n        raise RuntimeError(\n            f"Unsupported CUDA for automatic torch_scatter wheel: {cuda}"\n        )\n\n    wheel_url = (\n        f"https://data.pyg.org/whl/"\n        f"torch-2.4.0+{tag}.html"\n    )\n\n    print("Installing torch_scatter from:", wheel_url)\n\n    subprocess.check_call([\n        sys.executable,\n        "-m",\n        "pip",\n        "install",\n        "-q",\n        "torch-scatter==2.1.2",\n        "-f",\n        wheel_url\n    ])\n\n    import torch_scatter\n    print("Installed torch_scatter:", torch_scatter.__version__)\nPY', check=False, capture=True)
    print(S.rr.stdout)
    if S.rr.returncode != 0:
        raise RuntimeError(f'torch_scatter setup failed: return code {S.rr.returncode}')
    S.rr = mex('esmbind', 'python - <<\'PY\'\nimport torch\nimport torch_scatter\nimport torch_geometric\nimport biotite\n\nfrom biotite.structure import filter_backbone\n\nimport esm\nimport esm.inverse_folding\n\nprint("torch:", torch.__version__)\nprint("CUDA:", torch.version.cuda)\nprint("cuda available:", torch.cuda.is_available())\nprint("torch_scatter:", torch_scatter.__version__)\nprint("torch_geometric:", torch_geometric.__version__)\nprint("biotite:", biotite.__version__)\nprint("filter_backbone import: PASS")\n\nassert torch.cuda.is_available(), "CUDA unavailable"\n\nmodel, alphabet = (\n    esm.pretrained.esm_if1_gvp4_t16_142M_UR50()\n)\n\nmodel = model.eval().cuda()\n\nprint("ESM-IF1 MODEL LOAD PASS")\nPY', check=False, capture=True)
    print(S.rr.stdout)
    if S.rr.returncode != 0:
        raise RuntimeError(f'ESM-IF1 smoke test failed: return code {S.rr.returncode}')
    print('\nFINAL ESM-IF ENVIRONMENT: PASS')

# 17A. ESMBind ISOLATED ENVIRONMENT
def stage_55():
    S.ESMB = S.REPOS / 'esmbind'
    if S.RUN_ESMBIND_TRANSFER:
        create_env('esmbind', '3.10')
        mex('esmbind', "python -m pip install -q --upgrade 'pip<25' setuptools wheel")
        mex('esmbind', "python -m pip install -q 'numpy<2' scipy scikit-learn biopython tqdm ml-collections 'torchmetrics<1.5' 'torch==2.4.1' 'fair-esm==2.0.0'")
        mex('esmbind', 'python -c "import torch,esm,torchmetrics,ml_collections; print(\'torch\',torch.__version__,\'cuda\',torch.cuda.is_available()); assert torch.cuda.is_available(), \'ESMBind feature extraction is configured for GPU\'"')

# CELL 2 / 17B
def stage_56():
    print('17B helper definitions loaded.')

# CELL 3 / 17C
def stage_57():
    S.esmb_rows = []
    S.esmbind_pooled_probs = {}
    S.esmbind_ion_probs = {}
    S.esmbind_author_thresholds = {}
    if S.RUN_ESMBIND_TRANSFER:
        for S.dataset, S.df in S.DATASETS.items():
            print('\nSTART ESMBind external evaluation:', S.dataset)
            S.pooled, S.ion_probs, S.thresholds = run_esmbind_transfer(S.dataset, S.df)
            S.esmbind_pooled_probs[S.dataset] = S.pooled
            S.esmbind_ion_probs[S.dataset] = S.ion_probs
            S.esmbind_author_thresholds[S.dataset] = S.thresholds
            save_probs(S.pooled, S.PRED / f'ESMBind_7ion_max_TRANSFER__{S.dataset}__metal_binding.csv.gz')
            S.out, S.pt = evaluate_probability_dict(S.df, S.pooled, 'metal_binding', 'ESMBind_7ion_max_TRANSFER', S.dataset, 'fixed_0.5', 0.5, 'C_cross_benchmark_transfer')
            S.esmb_rows.append(S.out)
            S.pt.to_csv(S.RESULTS / f'ESMBind_{S.dataset}_per_protein.csv', index=False)
            print('FINISHED ESMBind external evaluation:', S.dataset)
        S.esmb_df = pd.DataFrame(S.esmb_rows)
        S.esmb_df.to_csv(S.RESULTS / 'ESMBind_TRANSFER_FULL_METRICS_300_46.csv', index=False)
        display(S.esmb_df)
        S.expected = set(S.DATASETS.keys())
        S.done = set(S.esmb_df['dataset'].astype(str))
        if S.done != S.expected:
            raise RuntimeError(f'ESMBind completion mismatch: expected {S.expected}, got {S.done}')
        print('\nESMBind Test300 + Test46 FULL-SCORE PASS.')

# ESM-IF1 ONE-PROTEIN FULL TRACE
def stage_58():
    S.dataset = 'TestSet300'
    S.pid = 'A0L5S6'
    S.inp = S.CACHE / f'esmbind_{S.dataset}'
    S.pdb_path = S.inp / 'pdb' / f'{S.pid}.pdb'
    S.rr = mex('esmbind', f'''python - <<'PY'\nimport torch\nimport esm\nimport esm.inverse_folding\n\npdb_path = r"{S.pdb_path}"\n\nprint("torch:", torch.__version__)\nprint("cuda:", torch.version.cuda)\nprint("cuda available:", torch.cuda.is_available())\n\nmodel, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()\nmodel = model.eval().cuda()\n\nprint("model device:", next(model.parameters()).device)\n\nstructure = esm.inverse_folding.util.load_structure(\n    pdb_path,\n    "A"\n)\n\ncoords, seq = (\n    esm.inverse_folding.util.extract_coords_from_structure(\n        structure\n    )\n)\n\nprint("sequence length:", len(seq))\nprint("coords shape:", coords.shape)\n\n# IMPORTANT:\n# no try/except here -> full traceback\nrep = esm.inverse_folding.util.get_encoder_output(\n    model,\n    alphabet,\n    coords\n)\n\nprint("REP:", rep.shape, rep.device)\nprint("ONE PROTEIN ESM-IF PASS")\nPY''', check=False, capture=True)
    print(S.rr.stdout)
    print('return code:', S.rr.returncode)

# HOTFIX FINAL — fair-esm v2.0.0 ESM-IF1 DEVICE BUG
def stage_59():
    S.rr = mex('esmbind', 'python - <<\'PY\'\nfrom pathlib import Path\nimport esm.inverse_folding.util as util\n\np = Path(util.__file__)\ntext = p.read_text()\n\nmarker = "# ESMIF_DEVICE_PATCH_V2"\n\nif marker not in text:\n    patch_lines = [\n        "",\n        "",\n        marker,\n        "def get_encoder_output(model, alphabet, coords):",\n        "    device = next(model.parameters()).device",\n        "    batch_converter = CoordBatchConverter(alphabet)",\n        "    batch = [(coords, None, None)]",\n        "    coords, confidence, _, _, padding_mask = batch_converter(",\n        "        batch, device=device",\n        "    )",\n        "    encoder_out = model.encoder.forward(",\n        "        coords,",\n        "        padding_mask,",\n        "        confidence,",\n        "        return_all_hiddens=False",\n        "    )",\n        "    return encoder_out[\'encoder_out\'][0][1:-1, 0]",\n        "",\n    ]\n\n    p.write_text(\n        text + "\\n".join(patch_lines)\n    )\n\n    print("Applied ESM-IF device patch to:")\n    print(p)\n\nelse:\n    print("ESM-IF device patch already present:")\n    print(p)\nPY', check=False, capture=True)
    print(S.rr.stdout)
    if S.rr.returncode != 0:
        raise RuntimeError(f'ESM-IF device patch failed: return code {S.rr.returncode}')
    S.dataset = 'TestSet300'
    S.pid = 'A0L5S6'
    S.inp = S.CACHE / f'esmbind_{S.dataset}'
    S.pdb_path = S.inp / 'pdb' / f'{S.pid}.pdb'
    S.rr = mex('esmbind', f'''python - <<'PY'\nimport torch\nimport esm\nimport esm.inverse_folding\n\npdb_path = r"{S.pdb_path}"\n\nmodel, alphabet = (\n    esm.pretrained.esm_if1_gvp4_t16_142M_UR50()\n)\n\nmodel = model.eval().cuda()\n\nprint("model device:", next(model.parameters()).device)\n\nstructure = esm.inverse_folding.util.load_structure(\n    pdb_path,\n    "A"\n)\n\ncoords, seq = (\n    esm.inverse_folding.util.extract_coords_from_structure(\n        structure\n    )\n)\n\nprint("sequence length:", len(seq))\nprint("coords shape:", coords.shape)\n\nrep = esm.inverse_folding.util.get_encoder_output(\n    model,\n    alphabet,\n    coords\n)\n\nprint("representation shape:", tuple(rep.shape))\nprint("representation device:", rep.device)\n\nassert tuple(rep.shape) == (len(seq), 512)\n\nprint("ONE PROTEIN ESM-IF PASS")\nPY''', check=False, capture=True)
    print(S.rr.stdout)
    if S.rr.returncode != 0:
        raise RuntimeError(f'ESM-IF one-protein smoke test failed: return code {S.rr.returncode}')
    print('\nESM-IF DEVICE FIX: PASS')

STAGES = [stage_54, stage_55, stage_56, stage_57, stage_58, stage_59]

