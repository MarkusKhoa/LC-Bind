"""Extracted from MASTER_BINDING_BASELINE_REPRODUCTION_300_46_FULL_v10_BINDNODE24_FIXED-2.ipynb. Section: bindembed21."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Bio, Chain, Model, PDBIO, PDBParser, Path, SeqIO, Structure, T5EncoderModel, T5Tokenizer, accuracy_score, ast, auc, average_precision_score, balanced_accuracy_score, brier_score_loss, create_env, deepcopy, f1_score, fbeta_score, h5py, hashlib, importlib, jaccard_score, json, matthews_corrcoef, mex, np, os, pd, pickle, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve, save_probs, seq1, shutil, student_t, subprocess, sys, thresholds_from_oof, time, torch, tqdm
    import h5py, torch
    from transformers import T5EncoderModel, T5Tokenizer
    from tqdm.auto import tqdm
    from Bio import SeqIO
    from pathlib import Path
    import os, sys, json, ast, shutil, subprocess, pickle, hashlib, time
    import numpy as np
    import pandas as pd
    import importlib.util
    import subprocess
    import sys
    import Bio
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score, fbeta_score, matthews_corrcoef, precision_score, recall_score, accuracy_score, balanced_accuracy_score, jaccard_score, brier_score_loss, auc, roc_curve
    from scipy.stats import t as student_t
    from copy import deepcopy
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.SeqUtils import seq1
    import shutil
    import json
    import hashlib
    from lc_bind.models.baselines.metrics import save_probs, thresholds_from_oof
    from lc_bind.models.baselines.setup import create_env, mex

def generate_prott5_h5(df, path, rebuild=False):
    path = Path(path)
    if path.exists() and (not rebuild):
        with h5py.File(path, 'r') as h5:
            good = len(h5.keys()) == len(df) and all((str(r.prot_id) in h5 and h5[str(r.prot_id)].shape == (len(r.sequence), 1024) for r in df.itertuples()))
        if good:
            print('REUSE', path)
            return path
        print('Existing H5 failed audit; rebuilding', path)
        path.unlink()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = T5Tokenizer.from_pretrained(S.PROTT5_MODEL, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(S.PROTT5_MODEL, torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32).to(device).eval()
    with h5py.File(path, 'w') as h5:
        for r in tqdm(df.itertuples(), total=len(df)):
            seq = ''.join(('X' if aa in 'UZOB' else aa for aa in str(r.sequence)))
            enc = tok(' '.join(seq), return_tensors='pt', add_special_tokens=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                z = model(**enc).last_hidden_state[0][:len(seq)].float().cpu().numpy()
            assert z.shape == (len(seq), 1024), (r.prot_id, z.shape, len(seq))
            h5.create_dataset(str(r.prot_id), data=z.astype(np.float32), compression='gzip')
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return path

def write_bind21_external_adapter(dataset, df, h5):
    outdir = S.PRED / 'bindEmbed21DL' / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    script = S.B21 / f'run_{dataset}_fullprecision.py'
    script.write_text(f'''import sys\nfrom pathlib import Path\nimport numpy as np\nimport h5py\nimport torch\n\nsys.path.insert(0,r"{S.B21}")\n\nfrom config import FileSetter\nfrom bindEmbed21DL import BindEmbed21DL\nfrom data_preparation import ProteinResults\n\nH5 = r"{h5}"\nIDS = r"{S.CANON / f'{dataset}_ids.txt'}"\nFASTA = r"{S.CANON / f'{dataset}.fasta'}"\nREPO = Path(r"{S.B21}")\nOUT = Path(r"{outdir}")\n\nprint("=== bindEmbed21 adapter preflight ===")\nprint("python", sys.version)\nprint("torch", torch.__version__)\nprint("numpy", np.__version__)\nprint("cuda", torch.cuda.is_available())\nprint("H5", H5)\n\nids=[x.strip() for x in open(IDS) if x.strip()]\nprint("n ids", len(ids))\n\n# Strict embedding audit inside the SAME environment that will run inference.\nwith h5py.File(H5,'r') as fh:\n    missing=[pid for pid in ids if pid not in fh]\n    if missing:\n        raise RuntimeError(f"Missing {{len(missing)}} embeddings; first={{missing[:10]}}")\n    for pid in ids:\n        shape=fh[pid].shape\n        if len(shape)!=2 or shape[1]!=1024:\n            raise RuntimeError(f"{{pid}}: bad ProtT5 shape {{shape}}, expected (L,1024)")\nprint("embedding key/dimension audit PASS")\n\n# IMPORTANT:\n# Released ProteinResults rounds probabilities to 3 decimals internally.\n# For threshold-free common metrics we remove ONLY that rounding.\ndef _as_numpy(x):\n    if isinstance(x, torch.Tensor):\n        return x.detach().cpu().numpy()\n    return np.asarray(x)\n\ndef _set_full(self,predictions):\n    arr=_as_numpy(predictions)\n    self.predictions=np.asarray(arr.T,dtype=np.float64)\n\ndef _add_full(self,predictions):\n    arr=_as_numpy(predictions)\n    self.predictions=np.add(self.predictions,np.asarray(arr,dtype=np.float64))\n\ndef _norm_full(self,norm_factor):\n    self.predictions=self.predictions/float(norm_factor)\n\nProteinResults.set_predictions=_set_full\nProteinResults.add_predictions=_add_full\nProteinResults.normalize_predictions=_norm_full\n\nFileSetter.embeddings_input=staticmethod(lambda:H5)\n\n# Check that all five released checkpoints exist before loading any model.\nfor k in range(1,6):\n    ckpt=REPO/"trained_models"/f"checkpoint{{k}}.pt"\n    if not ckpt.exists():\n        raise FileNotFoundError(ckpt)\nprint("5 released checkpoints present")\n\nproteins=BindEmbed21DL.prediction_pipeline(\n    model_prefix=str(REPO/"trained_models"/"checkpoint"),\n    cutoff=.5,\n    result_folder=None,\n    ids=ids,\n    fasta_file=FASTA,\n    ri=False\n)\n\nOUT.mkdir(parents=True,exist_ok=True)\nfor pid,obj in proteins.items():\n    arr=np.asarray(obj.predictions,dtype=np.float64).squeeze()\n    if arr.ndim!=2 or arr.shape[1]!=3:\n        raise RuntimeError(f"{{pid}}: unexpected prediction shape {{arr.shape}}")\n    if not np.all(np.isfinite(arr)):\n        raise RuntimeError(f"{{pid}}: non-finite probabilities")\n    np.save(OUT/f"{{pid}}.npy",arr.astype(np.float32))\n\nif len(proteins)!=len(ids):\n    raise RuntimeError(f"Predicted {{len(proteins)}}/{{len(ids)}} proteins")\n\nprint("saved",len(proteins),"proteins to",OUT)\nprint("=== bindEmbed21 adapter PASS ===")\n''')
    return (script, outdir)

def load_three_class_npy(df, folder):
    metal = {}
    nuc = {}
    small = {}
    for r in df.itertuples():
        arr = np.load(Path(folder) / f'{r.prot_id}.npy')
        assert arr.shape == (len(r.sequence), 3), (r.prot_id, arr.shape, len(r.sequence))
        metal[r.prot_id] = arr[:, 0]
        nuc[r.prot_id] = arr[:, 1]
        small[r.prot_id] = arr[:, 2]
    anyp = {pid: np.maximum.reduce([metal[pid], nuc[pid], small[pid]]) for pid in metal}
    return {'metal_binding': metal, 'nuclear_binding': nuc, 'small_binding': small, 'any_ligand_binding_sites': anyp}

def read_fasta_dict(path):
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(str(path), 'fasta')}

def read_binding_file(path):
    d = {}
    with open(path) as f:
        for line in f:
            s = line.strip().split()
            if len(s) > 1:
                d[s[0]] = [int(x) for x in s[1].split(',') if x != '']
    return d

def build_public_bind21_dev():
    seqs = read_fasta_dict(S.B21 / 'data/development_set/all.fasta')
    ids = []
    fold_map = {}
    for k in range(1, 6):
        z = [x.strip() for x in open(S.B21 / f'data/development_set/ids_split{k}.txt') if x.strip()]
        ids += z
        for x in z:
            fold_map[x] = k
    bind = {lab: read_binding_file(S.B21 / f'data/development_set/binding_residues_2.5_{short}.txt') for lab, short in [('metal_binding', 'metal'), ('nuclear_binding', 'nuclear'), ('small_binding', 'small')]}
    rows = []
    for pid in ids:
        seq = seqs[pid]
        rec = {'prot_id': pid, 'sequence': seq, 'fold': fold_map[pid]}
        for lab in ['metal_binding', 'nuclear_binding', 'small_binding']:
            y = np.zeros(len(seq), dtype=np.int8)
            for pos in bind[lab].get(pid, []):
                if 1 <= pos <= len(seq):
                    y[pos - 1] = 1
            rec[lab] = y.tolist()
        rec['any_ligand_binding_sites'] = np.maximum.reduce([rec['metal_binding'], rec['nuclear_binding'], rec['small_binding']]).astype(int).tolist()
        rows.append(rec)
    return pd.DataFrame(rows)

# 7A. PROTT5 GENERATOR
def stage_22():
    S.PROTT5_MODEL = 'Rostlab/prot_t5_xl_half_uniref50-enc'
    S.PROTT5 = {}
    for S.name, S.df in S.DATASETS.items():
        S.PROTT5[S.name] = generate_prott5_h5(S.df, S.CACHE / f'{S.name}_ProtT5.h5', S.REBUILD_PROTT5)

# 8A. bindEmbed21 ENV — LEGACY-COMPATIBLE PINNED STACK
def stage_26():
    S.B21 = S.REPOS / 'bind21'
    if S.RUN_BIND21 or S.RUN_OOF_THRESHOLDS:
        create_env('bind21', '3.10')
        S.bind21_specs = ['numpy==1.26.4', 'pandas==2.0.3', 'scipy==1.11.4', 'scikit-learn==1.3.2', 'h5py==3.10.0', 'torch==1.13.1']
        S.r = mex('bind21', 'python -m pip install --upgrade --force-reinstall ' + ' '.join([f'"{x}"' for x in S.bind21_specs]), check=False, capture=True)
        print(S.r.stdout)
        if S.r.returncode != 0:
            raise RuntimeError(f'bind21 dependency installation failed (return code {S.r.returncode}). See the pip output immediately above.')
        S.check = mex('bind21', "python - <<'PY'\nimport torch,numpy,pandas,scipy,sklearn,h5py\nprint('torch',torch.__version__)\nprint('numpy',numpy.__version__)\nprint('pandas',pandas.__version__)\nprint('scipy',scipy.__version__)\nprint('sklearn',sklearn.__version__)\nprint('h5py',h5py.__version__)\nprint('cuda',torch.cuda.is_available())\nx=torch.tensor([1.,2.])\na=x.numpy()\ny=torch.from_numpy(a)\nprint('torch<->numpy bridge OK',a.tolist(),y.tolist())\nPY", capture=True)
        print(S.check.stdout)

# 8B. bindEmbed21 EXTERNAL ADAPTER FOR BOTH SETS
def stage_27():
    S.B21_OUT = {}
    if S.RUN_BIND21:
        for S.dataset, S.df in S.DATASETS.items():
            S.script, S.outdir = write_bind21_external_adapter(S.dataset, S.df, S.PROTT5[S.dataset])
            S.r = mex('bind21', f'python "{S.script.name}"', cwd=S.B21, check=False, capture=True)
            print(S.r.stdout)
            if S.r.returncode != 0:
                raise RuntimeError(f'bindEmbed21 inference failed on {S.dataset} (return code {S.r.returncode}). The complete inner traceback is printed immediately above.')
            S.B21_OUT[S.dataset] = S.outdir

# 8C. NORMALIZE bindEmbed21 OUTPUTS
def stage_28():
    S.bind21_probs = {}
    if S.RUN_BIND21:
        for S.dataset, S.df in S.DATASETS.items():
            S.bind21_probs[S.dataset] = load_three_class_npy(S.df, S.B21_OUT[S.dataset])
            for S.label, S.d in S.bind21_probs[S.dataset].items():
                save_probs(S.d, S.PRED / f'bindEmbed21DL__{S.dataset}__{S.label}.csv.gz')

# 9A. BUILD bindEmbed21 PUBLIC DEV DATAFRAME
def stage_30():
    S.bind21_dev_df = build_public_bind21_dev() if S.RUN_OOF_THRESHOLDS else None
    if S.bind21_dev_df is not None:
        display(S.bind21_dev_df.groupby('fold').size())

# 9B. PROTT5 FOR bindEmbed21 PUBLIC DEV + OOF ADAPTER
def stage_31():
    if S.RUN_OOF_THRESHOLDS:
        S.B21_DEV_H5 = generate_prott5_h5(S.bind21_dev_df, S.CACHE / 'bindEmbed21_DevSet1014_ProtT5.h5', S.REBUILD_PROTT5)
        S.oofdir = S.PRED / 'bindEmbed21DL' / 'DevOOF'
        S.oofdir.mkdir(parents=True, exist_ok=True)
        S.script = S.B21 / 'run_dev_oof_fullprecision.py'
        S.script.write_text(f'''import sys\nfrom pathlib import Path\nimport numpy as np\nsys.path.insert(0,r"{S.B21}")\nfrom config import FileSetter,FileManager\nfrom data_preparation import ProteinInformation,ProteinResults\nfrom ml_predictor import MLPredictor\n# Disable the author code's 3-decimal probability rounding for full-precision common metrics.\ndef _set_full(self,predictions): self.predictions=np.asarray(predictions).T.astype(np.float64,copy=False)\nProteinResults.set_predictions=_set_full\nFileSetter.embeddings_input=staticmethod(lambda:r"{S.B21_DEV_H5}")\nemb=FileManager.read_embeddings(r"{S.B21_DEV_H5}")\nseqs=FileManager.read_fasta(r"{S.B21 / 'data/development_set/all.fasta'}")\nout=Path(r"{S.oofdir}");out.mkdir(parents=True,exist_ok=True)\nfor k in range(1,6):\n ids=FileManager.read_ids(r"{S.B21}/data/development_set/ids_split"+str(k)+".txt")\n model=FileManager.load_classifier_torch(r"{S.B21}/trained_models/checkpoint"+str(k)+".pt")\n # labels are not needed for inference; get_data_predictions creates compatible ProteinInformation objects.\n sequences,max_length,labels=ProteinInformation.get_data_predictions(ids,r"{S.B21 / 'data/development_set/all.fasta'}")\n pred=MLPredictor(model).predict_per_protein(ids,sequences,emb,labels,max_length)\n for pid,obj in pred.items():\n  arr=np.asarray(obj.predictions).squeeze()\n  assert arr.ndim==2 and arr.shape[1]==3,(pid,arr.shape)\n  np.save(out/f"{{pid}}.npy",arr.astype(np.float32))\nprint("done")\n''')
        mex('bind21', f'python "{S.script.name}"', cwd=S.B21)

# 9C. SELECT bindEmbed21 OOF THRESHOLDS
def stage_32():
    S.bind21_thresholds = None
    if S.RUN_OOF_THRESHOLDS:
        S.b21_oof_probs = load_three_class_npy(S.bind21_dev_df, S.PRED / 'bindEmbed21DL' / 'DevOOF')
        S.y_by = {lab: np.concatenate([np.asarray(x, np.int8) for x in S.bind21_dev_df[lab]]) for lab in S.LABELS}
        S.p_by = {lab: np.concatenate([np.asarray(S.b21_oof_probs[lab][pid], float) for pid in S.bind21_dev_df.prot_id]) for lab in S.LABELS}
        S.bind21_thresholds, S.b21_thr_df = thresholds_from_oof(S.y_by, S.p_by)
        display(S.b21_thr_df)
        S.b21_thr_df.to_csv(S.RESULTS / 'bindEmbed21DL_OOF_thresholds.csv', index=False)

STAGES = [stage_22, stage_26, stage_27, stage_28, stage_30, stage_31, stage_32]

