"""Extracted from MASTER_BINDING_BASELINE_REPRODUCTION_300_46_FULL_v10_BINDNODE24_FIXED-2.ipynb. Section: metalbind."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Bio, Chain, Model, PDBIO, PDBParser, Path, SeqIO, Structure, T5EncoderModel, T5Tokenizer, accuracy_score, ast, auc, average_precision_score, balanced_accuracy_score, brier_score_loss, create_env, deepcopy, evaluate_probability_dict, f1_score, fbeta_score, h5py, hashlib, importlib, jaccard_score, json, matthews_corrcoef, mex, np, os, pd, pickle, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve, save_probs, seq1, sh, shutil, student_t, subprocess, sys, time, torch, tqdm
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
    from lc_bind.models.baselines.setup import create_env, mex, sh

def write_esmc_generation_script(dataset, df, outdir):
    fasta = S.CANON / f'{dataset}.fasta'
    script = S.WORK / f'generate_metalbind_esmc_{dataset}.py'
    script.write_text(f'\nimport os\nfrom pathlib import Path\n\nimport numpy as np\nimport torch\n\nfrom Bio import SeqIO\nfrom transformers import AutoTokenizer, AutoModel\n\n\nMODEL_ID = {S.METALB_ESMC_HF_MODEL!r}\n\nFASTA = Path(r"{fasta}")\nOUT = Path(r"{outdir}")\n\nOUT.mkdir(\n    parents=True,\n    exist_ok=True\n)\n\n\n# ------------------------------------------------------------\n# GPU guard\n# ------------------------------------------------------------\n\nif not torch.cuda.is_available():\n    raise RuntimeError(\n        "MetalBind ESM-C extraction requires CUDA"\n    )\n\n\n# ------------------------------------------------------------\n# Optional HuggingFace authentication\n# ------------------------------------------------------------\n\ntoken = (\n    os.environ.get("HF_TOKEN")\n    or\n    os.environ.get("HUGGINGFACE_TOKEN")\n)\n\nkwargs = (\n    {{"token": token}}\n    if token\n    else {{}}\n)\n\n\n# ------------------------------------------------------------\n# Official HF-compatible Biohub ESM-C 600M\n# ------------------------------------------------------------\n\nprint(\n    "Loading ESM-C:",\n    MODEL_ID\n)\n\ntok = AutoTokenizer.from_pretrained(\n    MODEL_ID,\n    **kwargs\n)\n\nmodel = AutoModel.from_pretrained(\n    MODEL_ID,\n    dtype=torch.float16,\n    **kwargs\n).to(\n    "cuda"\n).eval()\n\n\nprint(\n    "ESM-C d_model:",\n    model.config.d_model\n)\n\nif model.config.d_model != 1152:\n    raise RuntimeError(\n        "Unexpected ESM-C hidden dimension: "\n        f"{{model.config.d_model}}"\n    )\n\n\n# ------------------------------------------------------------\n# Generate one residue-level embedding per protein\n# ------------------------------------------------------------\n\nrecords = list(\n    SeqIO.parse(\n        str(FASTA),\n        "fasta"\n    )\n)\n\nprint(\n    "Proteins:",\n    len(records)\n)\n\n\nfor idx, rec in enumerate(\n    records,\n    start=1\n):\n\n    pid = str(rec.id)\n    seq = str(rec.seq)\n\n    fp = (\n        OUT\n        /\n        f"{{pid}}.npy"\n    )\n\n\n    # --------------------------------------------------------\n    # Cache reuse\n    # --------------------------------------------------------\n\n    if fp.exists():\n\n        try:\n\n            a = np.load(\n                fp,\n                mmap_mode="r"\n            )\n\n            if (\n                a.shape\n                ==\n                (\n                    len(seq),\n                    1152\n                )\n                and\n                np.all(\n                    np.isfinite(a)\n                )\n            ):\n                print(\n                    f"[{{idx}}/{{len(records)}}] "\n                    f"{{pid}} cached"\n                )\n\n                continue\n\n        except Exception:\n            pass\n\n\n    # --------------------------------------------------------\n    # Tokenize\n    # --------------------------------------------------------\n\n    enc = tok(\n        seq,\n        return_tensors="pt",\n        add_special_tokens=True,\n        return_special_tokens_mask=True,\n        truncation=False,\n    )\n\n    special = enc.pop(\n        "special_tokens_mask"\n    )[0].bool()\n\n\n    enc = {{\n        k: v.to("cuda")\n        for k, v\n        in enc.items()\n    }}\n\n\n    # --------------------------------------------------------\n    # Forward\n    # --------------------------------------------------------\n\n    with torch.inference_mode():\n\n        out = model(\n            **enc,\n            return_dict=True\n        )\n\n\n    if (\n        not hasattr(\n            out,\n            "last_hidden_state"\n        )\n        or\n        out.last_hidden_state\n        is None\n    ):\n        raise RuntimeError(\n            f"{{pid}}: "\n            "ESMCModel output missing "\n            "last_hidden_state"\n        )\n\n\n    h = (\n        out\n        .last_hidden_state[0]\n        .float()\n        .cpu()\n    )\n\n\n    # --------------------------------------------------------\n    # Remove CLS/EOS special tokens\n    # --------------------------------------------------------\n\n    special_cpu = (\n        special\n        .cpu()\n    )\n\n\n    if (\n        special_cpu.shape[0]\n        !=\n        h.shape[0]\n    ):\n        raise RuntimeError(\n            f"{{pid}}: "\n            f"special-token mask length "\n            f"{{special_cpu.shape[0]}} "\n            f"!= hidden length "\n            f"{{h.shape[0]}}"\n        )\n\n\n    h = h[\n        ~special_cpu\n    ]\n\n\n    arr = (\n        h.numpy()\n        .astype(\n            np.float32,\n            copy=False\n        )\n    )\n\n\n    # --------------------------------------------------------\n    # STRICT residue alignment\n    # --------------------------------------------------------\n\n    expected = (\n        len(seq),\n        1152\n    )\n\n\n    if arr.shape != expected:\n        raise RuntimeError(\n            f"{{pid}}: "\n            f"ESM-C shape "\n            f"{{arr.shape}}, "\n            f"expected "\n            f"{{expected}}"\n        )\n\n\n    if not np.all(\n        np.isfinite(arr)\n    ):\n        raise RuntimeError(\n            f"{{pid}}: "\n            "non-finite ESM-C embedding"\n        )\n\n\n    np.save(\n        fp,\n        arr\n    )\n\n\n    print(\n        f"[{{idx}}/{{len(records)}}] "\n        f"{{pid}} PASS "\n        f"{{arr.shape}}"\n    )\n\n\nprint(\n    "ESM-C generation complete:",\n    len(\n        list(\n            OUT.glob("*.npy")\n        )\n    )\n)\n')
    return script

def load_metalbind_ion_csv(path, df):
    z = pd.read_csv(path)
    need = {'prot_id', 'residue_index', 'probability', 'is_surface_residue'}
    if not need.issubset(z.columns):
        raise ValueError(f'{path}: need columns {sorted(need)}')
    z['prot_id'] = z['prot_id'].astype(str)
    out = {}
    for r in df.itertuples():
        q = z[z.prot_id.eq(r.prot_id)].sort_values('residue_index')
        if len(q) != len(r.sequence):
            raise RuntimeError(f'{path}: {r.prot_id}: {len(q)} rows != {len(r.sequence)} residues')
        idx = q.residue_index.to_numpy(int)
        if not np.array_equal(idx, np.arange(len(r.sequence))):
            raise RuntimeError(f'{path}: noncanonical indexing for {r.prot_id}')
        p = q.probability.to_numpy(float)
        if not np.all(np.isfinite(p)) or np.min(p) < 0 or np.max(p) > 1:
            raise RuntimeError(f'{path}: invalid probabilities for {r.prot_id}')
        out[r.prot_id] = p
    return out

# MetalBind ENV DIAGNOSTIC — CURRENT SESSION
def stage_61():
    S.tests = {'python': 'python --version', 'torch': 'python - <<\'PY\'\nimport torch\nprint("torch:", torch.__version__)\nprint("torch.version.cuda:", torch.version.cuda)\nprint("cuda available:", torch.cuda.is_available())\nif torch.cuda.is_available():\n    print("GPU:", torch.cuda.get_device_name(0))\n    x = torch.tensor([1.0], device="cuda")\n    print("CUDA tensor test:", x, x.device)\nPY', 'pymesh': 'python - <<\'PY\'\nimport pymesh\nprint("pymesh import PASS")\nprint("pymesh path:", pymesh.__file__)\nPY', 'torch_scatter': 'python - <<\'PY\'\nimport torch_scatter\nprint("torch_scatter:", torch_scatter.__version__)\nPY', 'torch_sparse': 'python - <<\'PY\'\nimport torch_sparse\nprint("torch_sparse:", torch_sparse.__version__)\nPY', 'torch_cluster': 'python - <<\'PY\'\nimport torch_cluster\nprint("torch_cluster:", torch_cluster.__version__)\nPY', 'torch_geometric': 'python - <<\'PY\'\nimport torch_geometric\nprint("torch_geometric:", torch_geometric.__version__)\nPY', 'pykeops': 'python - <<\'PY\'\nimport pykeops\nprint("pykeops:", pykeops.__version__)\nPY'}
    S.metalbind_diag = {}
    for S.name, S.cmd in S.tests.items():
        print('\n' + '=' * 70)
        print('TEST:', S.name)
        print('=' * 70)
        S.rr = mex('metalbind', S.cmd, check=False, capture=True)
        print(S.rr.stdout)
        if getattr(S.rr, 'stderr', None):
            print(S.rr.stderr)
        print('RETURN CODE:', S.rr.returncode)
        S.metalbind_diag[S.name] = S.rr.returncode
    print('\n' + '=' * 70)
    print('EXTERNAL BINARIES')
    print('=' * 70)
    for S.fp in [S.reduce_bin, S.msms_bin]:
        print(S.fp, 'exists =', S.fp.exists(), 'executable =', os.access(S.fp, os.X_OK))
    print('\nSUMMARY')
    print(S.metalbind_diag)

# METALBIND HOTFIX — CUDA 11.3 RUNTIME FOR torch_sparse
def stage_62():
    sh(f'MAMBA_ROOT_PREFIX="{S.LOCAL_MAMBA_ROOT}" micromamba install -y -n metalbind -c conda-forge "cudatoolkit=11.3"')
    S.metalbind_envlib = S.LOCAL_MAMBA_ROOT / 'envs' / 'metalbind' / 'lib'
    os.environ['LD_LIBRARY_PATH'] = str(S.metalbind_envlib) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    print('MetalBind LD_LIBRARY_PATH:')
    print(os.environ['LD_LIBRARY_PATH'])
    S.cusparse_files = sorted(S.metalbind_envlib.glob('libcusparse.so*'))
    print('\ncuSPARSE files:')
    for S.p in S.cusparse_files:
        print(' ', S.p)
    if not S.cusparse_files:
        raise RuntimeError('cudatoolkit installed but libcusparse.so* not found')
    S.rr = mex('metalbind', 'python - <<\'PY\'\nimport torch\nimport pymesh\nimport torch_scatter\nimport torch_sparse\nimport torch_cluster\nimport torch_geometric\nimport pykeops\n\nprint("torch:", torch.__version__)\nprint("torch CUDA:", torch.version.cuda)\nprint("CUDA available:", torch.cuda.is_available())\n\nprint("pymesh: PASS")\nprint("torch_scatter:", torch_scatter.__version__)\nprint("torch_sparse:", torch_sparse.__version__)\nprint("torch_cluster:", torch_cluster.__version__)\nprint("torch_geometric:", torch_geometric.__version__)\nprint("pykeops:", pykeops.__version__)\n\nassert torch.cuda.is_available()\n\n# ------------------------------------------------------------\n# Real CUDA torch_sparse operation\n# ------------------------------------------------------------\nfrom torch_sparse import SparseTensor\n\nrow = torch.tensor(\n    [0, 1],\n    dtype=torch.long,\n    device=\'cuda\'\n)\n\ncol = torch.tensor(\n    [1, 0],\n    dtype=torch.long,\n    device=\'cuda\'\n)\n\nvalue = torch.ones(\n    2,\n    device=\'cuda\'\n)\n\nA = SparseTensor(\n    row=row,\n    col=col,\n    value=value,\n    sparse_sizes=(2, 2)\n)\n\nx = torch.ones(\n    (2, 1),\n    device=\'cuda\'\n)\n\ny = A.matmul(x)\n\nprint("torch_sparse CUDA operation:", y.flatten().tolist())\nprint("torch_sparse CUDA operation: PASS")\n\nprint("\\nMETALBIND GPU STACK: PASS")\nPY', check=False, capture=True)
    print(S.rr.stdout)
    if getattr(S.rr, 'stderr', None):
        print(S.rr.stderr)
    if S.rr.returncode != 0:
        raise RuntimeError(f'MetalBind GPU-stack test failed: return code {S.rr.returncode}')

# 18A. MetalBind AUTHOR-COMPATIBLE ENVIRONMENT
def stage_63():
    S.METALB = S.REPOS / 'metalbind'
    S.METALB_IONS = ['CA', 'CD', 'CO', 'CU', 'CU1', 'FE', 'FE2', 'K', 'MG', 'MN', 'NA', 'NI', 'ZN']
    S.METALB_EXT = S.WORK / 'metalbind_external_v6'
    S.METALB_EXT.mkdir(parents=True, exist_ok=True)
    if S.RUN_METALBIND_TRANSFER or S.RUN_METALBIND_NATIVE:
        create_env('metalbind', '3.7')
        sh(f'MAMBA_ROOT_PREFIX="{S.LOCAL_MAMBA_ROOT}" micromamba install -y -n metalbind -c conda-forge -c bioconda scripps-msms=2.6.1 reduce "cudatoolkit=11.3"')
        mex('metalbind', "python -m pip install -q --upgrade 'pip==23.3.2' 'setuptools<69' wheel")
        mex('metalbind', "python -m pip install -q 'numpy==1.21.6' 'scipy==1.7.3' 'scikit-learn==1.0.2' 'biopython==1.78' 'pandas==1.3.5' 'tqdm<5'")
        mex('metalbind', "python -m pip install -q 'torch==1.10.1+cu113' -f https://download.pytorch.org/whl/torch_stable.html")
        mex('metalbind', "python -m pip install -q 'torch-scatter==2.0.9' 'torch-sparse==0.6.13' 'torch-cluster==1.5.9' 'torch-spline-conv==1.2.1' -f https://data.pyg.org/whl/torch-1.10.1+cu113.html")
        mex('metalbind', "python -m pip install -q 'torch-geometric==2.0.4' 'pykeops==2.1'")
        S.pymesh_wheel = S.CACHE / 'pymesh2-0.3-cp37-cp37m-linux_x86_64.whl'
        if not S.pymesh_wheel.exists():
            sh(f'curl -L --fail --retry 3 "https://sourceforge.net/projects/pymesh.mirror/files/v0.3/pymesh2-0.3-cp37-cp37m-linux_x86_64.whl/download" -o "{S.pymesh_wheel}"')
        mex('metalbind', f'python -m pip install -q "{S.pymesh_wheel}"')
        S.envbin = S.LOCAL_MAMBA_ROOT / 'envs' / 'metalbind' / 'bin'
        S.reduce_bin = S.envbin / 'reduce'
        S.msms_bin = S.envbin / 'msms'
        if not S.reduce_bin.exists() or not S.msms_bin.exists():
            raise RuntimeError(f'MetalBind binaries missing: reduce={S.reduce_bin.exists()}, msms={S.msms_bin.exists()}')
        (S.METALB / 'default_config/bin_path.py').write_text(f"bin_path = {{}}\nbin_path['REDUCE'] = r'{S.reduce_bin}'\nbin_path['MSMS'] = r'{S.msms_bin}'\n")
        S.check = mex('metalbind', 'python -c "import torch,pymesh,torch_geometric,pykeops; print(\'torch\',torch.__version__,\'cuda\',torch.cuda.is_available()); print(\'pyg\',torch_geometric.__version__); assert torch.cuda.is_available()"', capture=True)
        print(S.check.stdout)

# FINAL ESM-C 600M SMOKE TEST
def stage_64():
    S.rr = mex('esmc', 'python - <<\'PY\'\nimport torch\nfrom transformers import AutoTokenizer, AutoModel\n\nMODEL_ID = "biohub/ESMC-600M"\n\nprint("torch:", torch.__version__)\nprint("cuda:", torch.cuda.is_available())\n\ntok = AutoTokenizer.from_pretrained(MODEL_ID)\n\nmodel = AutoModel.from_pretrained(\n    MODEL_ID,\n    dtype=torch.float16\n).to("cuda").eval()\n\nprint("model:", MODEL_ID)\nprint("d_model:", model.config.d_model)\n\nassert model.config.d_model == 1152\n\nseq = "ACDEFGHIKLMNPQRSTVWY"\n\nenc = tok(\n    seq,\n    return_tensors="pt",\n    add_special_tokens=True,\n    return_special_tokens_mask=True,\n)\n\nspecial = enc.pop("special_tokens_mask")[0].bool()\n\nenc = {\n    k: v.to("cuda")\n    for k, v in enc.items()\n}\n\nwith torch.inference_mode():\n    out = model(\n        **enc,\n        return_dict=True\n    )\n\nh = out.last_hidden_state[0].float().cpu()\n\nprint("raw hidden:", tuple(h.shape))\n\nh = h[~special.cpu()]\n\nprint("residue hidden:", tuple(h.shape))\n\nassert h.shape == (len(seq), 1152)\nassert torch.isfinite(h).all()\n\nprint("ESM-C 600M / 1152-d SMOKE TEST PASS")\nPY', check=False, capture=True)
    print(S.rr.stdout)
    if getattr(S.rr, 'stderr', None):
        print(S.rr.stderr)
    if S.rr.returncode != 0:
        raise RuntimeError(f'ESM-C smoke test failed: {S.rr.returncode}')
    print('\nBIOHUB ESM-C ENVIRONMENT: PASS')

# 18B. MetalBind ESM-C 600M / 1152-d FEATURES — FINAL
def stage_65():
    S.BIOHUB_TRANSFORMERS_COMMIT = '3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf'
    S.METALB_ESMC_HF_MODEL = 'biohub/ESMC-600M'
    S.METALB_ESMC_ROOT = S.CACHE / 'metalbind_esmc600m_v6'
    S.METALB_ESMC_ROOT.mkdir(parents=True, exist_ok=True)
    if S.RUN_METALBIND_TRANSFER:
        create_env('esmc', '3.10')
        mex('esmc', "python -m pip install -q --upgrade 'pip<25' setuptools wheel")
        mex('esmc', "python -m pip install -q 'torch==2.4.1' 'numpy<2' biopython safetensors")
        S.rr = mex('esmc', f"python -m pip install -q --upgrade 'git+https://github.com/Biohub/transformers.git@{S.BIOHUB_TRANSFORMERS_COMMIT}' 'huggingface-hub>=0.34,<1' 'tokenizers>=0.22,<=0.23' 'safetensors>=0.4.3'", check=False, capture=True)
        if S.rr.stdout:
            print(S.rr.stdout)
        if getattr(S.rr, 'stderr', None):
            print(S.rr.stderr)
        if S.rr.returncode != 0:
            raise RuntimeError(f'Biohub Transformers installation failed: {S.rr.returncode}')
        if not (os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')):
            try:
                S._hf = userdata.get('HF_TOKEN')
                if S._hf:
                    os.environ['HF_TOKEN'] = S._hf
                    print('HF_TOKEN loaded from Colab Secrets.')
            except Exception:
                pass
        S.rr = mex('esmc', 'python - <<\'PY\'\nimport torch\nimport transformers\n\nfrom transformers.models.esmc.tokenization_esmc import ESMCTokenizer\nfrom transformers.models.esmc.modeling_esmc import ESMCModel\n\nprint("transformers:", transformers.__version__)\nprint("torch:", torch.__version__)\nprint("CUDA:", torch.cuda.is_available())\n\nassert torch.cuda.is_available()\n\nprint("ESMCTokenizer import: PASS")\nprint("ESMCModel import: PASS")\nprint("ESM-C environment: PASS")\nPY', check=False, capture=True)
        print(S.rr.stdout)
        if S.rr.returncode != 0:
            raise RuntimeError(f'ESM-C environment smoke test failed: {S.rr.returncode}')
    if S.RUN_METALBIND_TRANSFER:
        for S.dataset, S.df in S.DATASETS.items():
            print('\n' + '=' * 72)
            print('MetalBind ESM-C:', S.dataset)
            print('=' * 72)
            S.outdir = S.METALB_ESMC_ROOT / S.dataset
            S.outdir.mkdir(parents=True, exist_ok=True)
            S.need = S.REBUILD_METALBIND_ESMC
            if not S.need:
                for S.r in S.df.itertuples():
                    S.fp = S.outdir / f'{S.r.prot_id}.npy'
                    if not S.fp.exists():
                        S.need = True
                        break
                    try:
                        S.a = np.load(S.fp, mmap_mode='r')
                    except Exception:
                        S.need = True
                        break
                    if S.a.shape != (len(S.r.sequence), 1152) or not np.all(np.isfinite(S.a)):
                        S.need = True
                        break
            if S.need:
                S.script = write_esmc_generation_script(S.dataset, S.df, S.outdir)
                print('Generation script:', S.script)
                S.rr = mex('esmc', f'python "{S.script}"', check=False, capture=True)
                if S.rr.stdout:
                    print(S.rr.stdout)
                if getattr(S.rr, 'stderr', None):
                    print(S.rr.stderr)
                if S.rr.returncode != 0:
                    raise RuntimeError(f'{S.dataset}: ESM-C generation failed with return code {S.rr.returncode}. Full subprocess log printed above.')
            S.bad = []
            for S.r in S.df.itertuples():
                S.pid = str(S.r.prot_id)
                S.fp = S.outdir / f'{S.pid}.npy'
                if not S.fp.exists():
                    S.bad.append((S.pid, 'missing'))
                    continue
                try:
                    S.a = np.load(S.fp, mmap_mode='r')
                except Exception as e:
                    S.bad.append((S.pid, f'unreadable: {e}'))
                    continue
                S.expected = (len(S.r.sequence), 1152)
                if S.a.shape != S.expected:
                    S.bad.append((S.pid, f'shape={S.a.shape}, expected={S.expected}'))
                elif not np.all(np.isfinite(S.a)):
                    S.bad.append((S.pid, 'non-finite'))
            if S.bad:
                raise RuntimeError(f'{S.dataset}: ESM-C final audit failed: {S.bad[:10]}')
            print(f'{S.dataset}: ESM-C audit PASS {len(S.df)}/{len(S.df)} proteins.')
        print('\nMetalBind ESM-C 600M audit PASSED for TestSet300 + TestSetNew46.')

# 18C. CONVERT ESM-C TO AUTHOR TEXT FORMAT + PREPARE MANIFESTS
def stage_66():
    S.METALB_PREP_ROOT = S.METALB_EXT / 'prepared'
    S.METALB_PREP_ROOT.mkdir(parents=True, exist_ok=True)
    if S.RUN_METALBIND_TRANSFER:
        for S.dataset, S.df in S.DATASETS.items():
            S.root = S.METALB_PREP_ROOT / S.dataset
            S.raw_pdb = S.root / 'raw_pdb'
            S.esm_txt = S.root / '600M'
            S.raw_pdb.mkdir(parents=True, exist_ok=True)
            S.esm_txt.mkdir(parents=True, exist_ok=True)
            S.manifest_rows = []
            for S.r in S.df.itertuples():
                S.pid = str(S.r.prot_id)
                if ':' in S.pid or '/' in S.pid:
                    raise RuntimeError(f'MetalBind external adapter requires path-safe protein id, got {S.pid}')
                S.pdb_dst = S.raw_pdb / f'{S.pid}A.pdb'
                shutil.copy2(S.NORMALIZED_EXTERNAL_PDB[S.dataset][S.pid], S.pdb_dst)
                S.txt = S.esm_txt / f'{S.pid}A.txt'
                S.npy = S.METALB_ESMC_ROOT / S.dataset / f'{S.pid}.npy'
                if S.REBUILD_METALBIND_ESMC or not S.txt.exists():
                    S.arr = np.load(S.npy)
                    np.savetxt(S.txt, S.arr, fmt='%.10e')
                S.aa = np.loadtxt(S.txt, dtype=np.float32)
                if S.aa.shape != (len(S.r.sequence), 1152):
                    raise RuntimeError(f'{S.dataset}/{S.pid}: author ESM text shape {S.aa.shape}')
                S.manifest_rows.append([S.pid, S.r.sequence, str(S.pdb_dst), str(S.txt)])
            S.man = pd.DataFrame(S.manifest_rows, columns=['prot_id', 'sequence', 'pdb_path', 'esm_txt'])
            S.man.to_csv(S.root / 'manifest.csv', index=False)
            print(S.dataset, 'MetalBind manifest', len(S.man))

# 18D. WRITE + RUN LABEL-FREE ADAPTER AROUND AUTHORS' MetalBindProcessor
def stage_67():
    S.METALB_PREP_SCRIPT = S.METALB / 'external_prepare_v6.py'
    S.METALB_PREP_SCRIPT.write_text("\nimport argparse\nimport json\nimport traceback\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\n\nfrom MetalBindProcessor import MetalBindProcessor\n\n\n# ============================================================\n# Label-free inference adapters\n# ============================================================\n\ndef dummy_nucleic(self):\n    return []\n\n\ndef dummy_interface(self, anno):\n    # Inference only: labels/interfaces are placeholders and\n    # are not used to generate predictions.\n    return np.zeros(\n        self.mesh.vertices.shape[0],\n        dtype=np.float32\n    )\n\n\ndef dummy_site(self, anno):\n    # Preserve exact surface-residue ordering from author's processor.\n    s2i = {\n        res: i\n        for i, res in enumerate(self.surface_res)\n    }\n\n    point_res_id = np.zeros(\n        len(self.vert_info),\n        dtype=np.int64\n    )\n\n    for k, info in enumerate(self.vert_info):\n        fields = info.split('_')\n\n        if len(fields) < 2:\n            raise RuntimeError(\n                f'Unexpected vert_info entry: {info}'\n            )\n\n        rid = fields[0] + '_' + fields[1]\n\n        if rid not in s2i:\n            raise RuntimeError(\n                f'point residue {rid} missing from surface-residue map'\n            )\n\n        point_res_id[k] = s2i[rid]\n\n    self.site_by_biolip = set()\n\n    return (\n        np.zeros(\n            len(self.surface_res),\n            dtype=np.int64\n        ),\n        point_res_id\n    )\n\n\n# Patch labels only; structural preprocessing remains author's code.\nMetalBindProcessor._get_nucleic_acid_atom = dummy_nucleic\nMetalBindProcessor._get_interface_mental = dummy_interface\nMetalBindProcessor._get_site_biolip = dummy_site\n\n\n# ============================================================\n# CLI\n# ============================================================\n\nap = argparse.ArgumentParser()\nap.add_argument('--manifest', required=True)\nap.add_argument('--root', required=True)\nap.add_argument('--rebuild', action='store_true')\na = ap.parse_args()\n\nroot = Path(a.root)\nmanifest_path = Path(a.manifest)\n\nif not manifest_path.exists():\n    raise FileNotFoundError(\n        f'Manifest not found: {manifest_path}'\n    )\n\nmanifest = pd.read_csv(manifest_path)\n\nif len(manifest) == 0:\n    raise RuntimeError(\n        f'Empty manifest: {manifest_path}'\n    )\n\nrequired_cols = {\n    'prot_id',\n    'sequence',\n    'pdb_path',\n    'esm_txt',\n}\n\nmissing_cols = required_cols - set(manifest.columns)\n\nif missing_cols:\n    raise RuntimeError(\n        f'Manifest missing columns: {sorted(missing_cols)}'\n    )\n\n\n# ============================================================\n# Output directories\n# ============================================================\n\nfor d in [\n    'chain_pdb',\n    'protonated_pdb',\n    'xyzrn',\n    'msms',\n    'data_label',\n    'ligand_dummy',\n    'hmm',\n    'fasta',\n    'dssp',\n    'save',\n    'mesh',\n    'mesh_fix',\n    'ply',\n]:\n    (root / d).mkdir(\n        parents=True,\n        exist_ok=True\n    )\n\n\nraw_pdb = Path(\n    manifest.iloc[0].pdb_path\n).parent\n\nesm_dir = Path(\n    manifest.iloc[0].esm_txt\n).parent\n\n\nif not raw_pdb.exists():\n    raise FileNotFoundError(\n        f'raw_pdb directory not found: {raw_pdb}'\n    )\n\nif not esm_dir.exists():\n    raise FileNotFoundError(\n        f'ESM directory not found: {esm_dir}'\n    )\n\n\nopts = {\n    'base_dir': str(root),\n    'raw_pdb_dir': str(raw_pdb),\n    'ligand_dir': str(root / 'ligand_dummy'),\n    'hhm_dir': str(root / 'hmm'),\n    'fasta_dir': str(root / 'fasta'),\n    'dssp': str(root / 'dssp'),\n    'protonated_pdb_dir': str(root / 'protonated_pdb'),\n    'chain_pdb_dir': str(root / 'chain_pdb'),\n    'xyzrn_dir': str(root / 'xyzrn'),\n    'msms_dir': str(root / 'msms'),\n    'data_label': str(root / 'data_label'),\n    'save_dir': str(root / 'save'),\n    'mesh_dir': str(root / 'mesh'),\n    'mesh_fix_dir': str(root / 'mesh_fix'),\n    'ply_dir': str(root / 'ply'),\n    'esm_dir': str(esm_dir),\n}\n\n\nprint(\n    f'Preparing {len(manifest)} proteins'\n)\n\nprint(\n    'raw_pdb_dir =',\n    opts['raw_pdb_dir']\n)\n\nprint(\n    'esm_dir =',\n    opts['esm_dir']\n)\n\n\n# ============================================================\n# Per-protein preprocessing\n# ============================================================\n\nprepared = 0\ncached = 0\n\n\nfor i, row in enumerate(\n    manifest.itertuples(),\n    start=1\n):\n\n    pid = str(row.prot_id)\n    pair = f'{pid}:A:X1'\n\n    fp = (\n        root\n        /\n        'data_label'\n        /\n        pair\n    )\n\n    mp = (\n        root\n        /\n        'data_label'\n        /\n        f'{pair}.surface_map.json'\n    )\n\n\n    # --------------------------------------------------------\n    # Cache reuse\n    # --------------------------------------------------------\n\n    if (\n        fp.exists()\n        and\n        mp.exists()\n        and\n        not a.rebuild\n    ):\n        cached += 1\n\n        print(\n            f'[{i}/{len(manifest)}] {pid}: cached'\n        )\n\n        continue\n\n\n    print(\n        f'[{i}/{len(manifest)}] {pid}: START'\n    )\n\n\n    # --------------------------------------------------------\n    # Strict input checks\n    # --------------------------------------------------------\n\n    pdb_path = Path(row.pdb_path)\n    esm_path = Path(row.esm_txt)\n\n    if not pdb_path.exists():\n        raise FileNotFoundError(\n            f'{pid}: PDB missing: {pdb_path}'\n        )\n\n    if not esm_path.exists():\n        raise FileNotFoundError(\n            f'{pid}: ESM text missing: {esm_path}'\n        )\n\n\n    try:\n\n        # ----------------------------------------------------\n        # AUTHORS' processor\n        # ----------------------------------------------------\n\n        proc = MetalBindProcessor(\n            pair=pair,\n            anno='',\n            ligand_type='ZN',\n            # Metal branch -> 0.5-A probe radius in author's code\n            dir_opts=opts,\n            RQ_window=6.0,\n            # exact public prepare_one.sh setting\n        )\n\n        proc.get_data()\n\n\n        # ----------------------------------------------------\n        # Verify author's data object\n        # ----------------------------------------------------\n\n        if not fp.exists():\n            raise RuntimeError(\n                f'{pid}: proc.get_data() finished but expected '\n                f'data file was not created: {fp}'\n            )\n\n\n        # ----------------------------------------------------\n        # Surface residue -> canonical sequence position\n        # ----------------------------------------------------\n\n        mapping = []\n\n        for s in proc.surface_res:\n\n            fields = s.split('_')\n\n            if len(fields) < 2:\n                raise RuntimeError(\n                    f'{pid}: malformed surface residue: {s}'\n                )\n\n            if fields[0] != 'A':\n                raise RuntimeError(\n                    f'{pid}: unexpected surface chain {fields[0]}'\n                )\n\n            resid = int(fields[1])\n\n            if (\n                resid < 1\n                or\n                resid > len(row.sequence)\n            ):\n                raise RuntimeError(\n                    f'{pid}: surface residue number {resid} '\n                    f'outside sequence length {len(row.sequence)}'\n                )\n\n            mapping.append(\n                {\n                    'surface_residue': s,\n                    'canonical_index': resid - 1,\n                }\n            )\n\n\n        if len(mapping) == 0:\n            raise RuntimeError(\n                f'{pid}: no surface residues produced'\n            )\n\n\n        # ----------------------------------------------------\n        # Save exact remapping metadata\n        # ----------------------------------------------------\n\n        mp.write_text(\n            json.dumps(\n                {\n                    'prot_id': pid,\n                    'sequence_length': len(row.sequence),\n                    'surface_res': list(\n                        proc.surface_res\n                    ),\n                    'mapping': mapping,\n                },\n                indent=2\n            )\n        )\n\n\n        if not mp.exists():\n            raise RuntimeError(\n                f'{pid}: surface map was not written'\n            )\n\n\n        prepared += 1\n\n        print(\n            f'[{i}/{len(manifest)}] {pid}: PASS '\n            f'surface_res={len(mapping)}'\n        )\n\n\n    except Exception:\n\n        print(\n            '\\n'\n            + '=' * 80\n        )\n\n        print(\n            f'FAILED PROTEIN: {pid}'\n        )\n\n        print(\n            f'PAIR: {pair}'\n        )\n\n        print(\n            f'PDB: {pdb_path}'\n        )\n\n        print(\n            f'ESM: {esm_path}'\n        )\n\n        print(\n            '=' * 80\n        )\n\n        traceback.print_exc()\n\n        print(\n            '=' * 80\n            + '\\n'\n        )\n\n        raise\n\n\nprint(\n    f'PREPROCESS COMPLETE: '\n    f'prepared_now={prepared}, '\n    f'cached={cached}, '\n    f'total={len(manifest)}'\n)\n")
    if S.RUN_METALBIND_TRANSFER:
        for S.dataset, S.df in S.DATASETS.items():
            print('\n' + '=' * 80)
            print('MetalBind surface preprocessing:', S.dataset)
            print('=' * 80)
            S.root = S.METALB_PREP_ROOT / S.dataset
            S.manifest_path = S.root / 'manifest.csv'
            if not S.manifest_path.exists():
                raise FileNotFoundError(f'{S.dataset}: manifest missing: {S.manifest_path}. Run 18C first.')
            S.cmd = f'python "{S.METALB_PREP_SCRIPT}" --manifest "{S.manifest_path}" --root "{S.root}"' + (' --rebuild' if S.REBUILD_METALBIND_SURFACE else '')
            S.rr = mex('metalbind', S.cmd, cwd=S.METALB, check=False, capture=True)
            print('\n' + '-' * 80)
            print(f'OUTPUT: {S.dataset}')
            print('-' * 80)
            if S.rr.stdout:
                print(S.rr.stdout)
            print(f'\nRETURN CODE: {S.rr.returncode}')
            if S.rr.returncode != 0:
                raise RuntimeError(f'{S.dataset}: MetalBind surface preprocessing failed with return code {S.rr.returncode}. The real subprocess traceback is printed above.')
            S.missing = []
            S.bad_maps = []
            for S.r in S.df.itertuples():
                S.pid = str(S.r.prot_id)
                S.pair = f'{S.pid}:A:X1'
                S.data_fp = S.root / 'data_label' / S.pair
                S.map_fp = S.root / 'data_label' / f'{S.pair}.surface_map.json'
                if not S.data_fp.exists():
                    S.missing.append((S.pid, 'pickle'))
                if not S.map_fp.exists():
                    S.missing.append((S.pid, 'map'))
                    continue
                try:
                    S.z = json.loads(S.map_fp.read_text())
                except Exception as e:
                    S.bad_maps.append((S.pid, f'unreadable map: {e}'))
                    continue
                if int(S.z.get('sequence_length', -1)) != len(S.r.sequence):
                    S.bad_maps.append((S.pid, 'sequence_length mismatch'))
                S.mapping = S.z.get('mapping', [])
                if len(S.mapping) == 0:
                    S.bad_maps.append((S.pid, 'empty mapping'))
                    continue
                S.canonical_indices = [int(x['canonical_index']) for x in S.mapping]
                if any((x < 0 or x >= len(S.r.sequence) for x in S.canonical_indices)):
                    S.bad_maps.append((S.pid, 'canonical index out of range'))
            if S.missing:
                raise RuntimeError(f'{S.dataset}: MetalBind preprocessing incomplete: {S.missing[:10]}')
            if S.bad_maps:
                raise RuntimeError(f'{S.dataset}: MetalBind surface-map audit failed: {S.bad_maps[:10]}')
            print(f'{S.dataset}: MetalBind preprocessing PASS {len(S.df)}/{len(S.df)} proteins.')
        print('\nMetalBind surface preprocessing PASSED for TestSet300 + TestSetNew46.')

# 18D-0. FINAL MSMS 2.6.1 FIX
def stage_68():
    print('=' * 80)
    print('Replacing MSMS package with legacy Bioconda msms=2.6.1')
    print('=' * 80)
    sh(f'MAMBA_ROOT_PREFIX="{S.LOCAL_MAMBA_ROOT}" micromamba remove -y -n metalbind scripps-msms || true')
    sh(f'MAMBA_ROOT_PREFIX="{S.LOCAL_MAMBA_ROOT}" micromamba install -y -n metalbind -c bioconda -c conda-forge "msms=2.6.1"')
    S.envbin = S.LOCAL_MAMBA_ROOT / 'envs' / 'metalbind' / 'bin'
    S.reduce_bin = S.envbin / 'reduce'
    S.msms_bin = S.envbin / 'msms'
    print('reduce:', S.reduce_bin)
    print('msms:', S.msms_bin)
    print('reduce exists:', S.reduce_bin.exists())
    print('msms exists:', S.msms_bin.exists())
    if not S.reduce_bin.exists():
        raise RuntimeError(f'Reduce binary missing: {S.reduce_bin}')
    if not S.msms_bin.exists():
        raise RuntimeError(f'MSMS binary missing: {S.msms_bin}')
    (S.METALB / 'default_config' / 'bin_path.py').write_text(f"bin_path = {{}}\nbin_path['REDUCE'] = r'{S.reduce_bin}'\nbin_path['MSMS'] = r'{S.msms_bin}'\n")
    S.rr = mex('metalbind', f'"{S.msms_bin}" -h', check=False, capture=True)
    print('\nMSMS binary output:')
    print(S.rr.stdout)
    print('MSMS -h return code:', S.rr.returncode)
    S.apply_msms_path = S.METALB / 'compute_surface' / 'apply_msms.py'
    S.apply_msms_path.write_text("\nfrom default_config.bin_path import bin_path\nfrom compute_surface.read_msms import read_msms\n\nfrom subprocess import Popen, PIPE\n\nimport os\nimport shutil\nimport tempfile\n\n\ndef computeMSMS(pair, dir_opts, probe_radius):\n\n    # --------------------------------------------------------\n    # Input XYZRN generated by author's preprocessing\n    # --------------------------------------------------------\n\n    xyzrn = os.path.join(\n        dir_opts['xyzrn_dir'],\n        pair + '.xyzrn'\n    )\n\n    if not os.path.exists(xyzrn):\n        raise FileNotFoundError(\n            f'MSMS input XYZRN missing: {xyzrn}'\n        )\n\n    if os.path.getsize(xyzrn) == 0:\n        raise RuntimeError(\n            f'MSMS input XYZRN is empty: {xyzrn}'\n        )\n\n\n    # --------------------------------------------------------\n    # Run native MSMS on LOCAL scratch.\n    #\n    # Google Drive/FUSE is not used for native temporary\n    # .vert/.face generation.\n    # --------------------------------------------------------\n\n    scratch_parent = os.environ.get(\n        'METALBIND_MSMS_TMP',\n        '/content'\n    )\n\n    if not os.path.isdir(scratch_parent):\n        scratch_parent = '/tmp'\n\n\n    tmpdir = tempfile.mkdtemp(\n        prefix='metalbind_msms_',\n        dir=scratch_parent\n    )\n\n    msms_file_base = os.path.join(\n        tmpdir,\n        'surface'\n    )\n\n\n    try:\n\n        msms_bin = bin_path['MSMS']\n\n\n        # ----------------------------------------------------\n        # EXACT author MSMS settings\n        # ----------------------------------------------------\n\n        args = [\n            msms_bin,\n            '-density',\n            '3',\n            '-hdensity',\n            '3',\n            '-probe',\n            str(probe_radius),\n            '-if',\n            xyzrn,\n            '-of',\n            msms_file_base,\n            '-af',\n            msms_file_base,\n        ]\n\n\n        print(\n            'MSMS COMMAND:',\n            ' '.join(args)\n        )\n\n\n        p = Popen(\n            args,\n            stdout=PIPE,\n            stderr=PIPE\n        )\n\n        stdout, stderr = (\n            p.communicate()\n        )\n\n\n        stdout_text = stdout.decode(\n            'utf-8',\n            errors='replace'\n        )\n\n        stderr_text = stderr.decode(\n            'utf-8',\n            errors='replace'\n        )\n\n\n        # ----------------------------------------------------\n        # STRICT return-code check\n        # ----------------------------------------------------\n\n        if p.returncode != 0:\n\n            raise RuntimeError(\n                'MSMS failed.\\n'\n                f'Return code: {p.returncode}\\n'\n                f'STDOUT:\\n{stdout_text}\\n'\n                f'STDERR:\\n{stderr_text}'\n            )\n\n\n        vert_path = (\n            msms_file_base\n            +\n            '.vert'\n        )\n\n        face_path = (\n            msms_file_base\n            +\n            '.face'\n        )\n\n\n        # ----------------------------------------------------\n        # STRICT output existence\n        # ----------------------------------------------------\n\n        for fp in [\n            vert_path,\n            face_path\n        ]:\n\n            if not os.path.exists(fp):\n\n                raise RuntimeError(\n                    'MSMS returned success but '\n                    f'output is missing: {fp}\\n'\n                    f'STDOUT:\\n{stdout_text}\\n'\n                    f'STDERR:\\n{stderr_text}'\n                )\n\n\n            if os.path.getsize(fp) == 0:\n\n                raise RuntimeError(\n                    f'MSMS output is empty: {fp}'\n                )\n\n\n        # ----------------------------------------------------\n        # MSMS .vert/.face MUST be ASCII text.\n        #\n        # Do not silently decode binary/corrupt files.\n        # ----------------------------------------------------\n\n        for fp in [\n            vert_path,\n            face_path\n        ]:\n\n            with open(\n                fp,\n                'rb'\n            ) as f:\n\n                raw = f.read()\n\n\n            try:\n\n                raw.decode(\n                    'ascii'\n                )\n\n            except UnicodeDecodeError as e:\n\n                head = raw[:256]\n\n                raise RuntimeError(\n                    'MSMS generated non-ASCII output.\\n'\n                    f'File: {fp}\\n'\n                    f'Size: {len(raw)} bytes\\n'\n                    f'First 256 bytes: {head!r}\\n'\n                    f'Decode error: {e}\\n'\n                    f'MSMS STDOUT:\\n{stdout_text}\\n'\n                    f'MSMS STDERR:\\n{stderr_text}'\n                )\n\n\n        # ----------------------------------------------------\n        # Parse using AUTHOR'S read_msms()\n        # ----------------------------------------------------\n\n        vertices, faces, normalv, res_id = (\n            read_msms(\n                msms_file_base\n            )\n        )\n\n\n        # ----------------------------------------------------\n        # Basic geometry sanity checks\n        # ----------------------------------------------------\n\n        if len(vertices) == 0:\n\n            raise RuntimeError(\n                'MSMS produced zero vertices'\n            )\n\n\n        if len(faces) == 0:\n\n            raise RuntimeError(\n                'MSMS produced zero faces'\n            )\n\n\n        if len(normalv) != len(vertices):\n\n            raise RuntimeError(\n                'MSMS normal/vertex count mismatch'\n            )\n\n\n        if len(res_id) != len(vertices):\n\n            raise RuntimeError(\n                'MSMS residue-id/vertex count mismatch'\n            )\n\n\n        print(\n            'MSMS PASS:',\n            'vertices=',\n            len(vertices),\n            'faces=',\n            len(faces)\n        )\n\n\n        return (\n            vertices,\n            faces,\n            normalv,\n            res_id\n        )\n\n\n    finally:\n\n        shutil.rmtree(\n            tmpdir,\n            ignore_errors=True\n        )\n")
    print('\nPatched:', S.apply_msms_path)
    S.dataset = 'TestSet300'
    S.pid = 'A0L5S6'
    S.pair = f'{S.pid}:A:X1'
    S.root = S.METALB_PREP_ROOT / S.dataset
    S.xyzrn_fp = S.root / 'xyzrn' / f'{S.pair}.xyzrn'
    print('\n' + '=' * 80)
    print('ONE-PROTEIN MSMS SMOKE TEST')
    print('=' * 80)
    print('XYZRN:', S.xyzrn_fp)
    print('exists:', S.xyzrn_fp.exists())
    if not S.xyzrn_fp.exists():
        raise RuntimeError(f'The previous 18D run should have produced this XYZRN file, but it is missing: {S.xyzrn_fp}')
    S.rr = mex('metalbind', f'''python - <<'PY'\nfrom compute_surface.apply_msms import computeMSMS\n\npair = {S.pair!r}\n\nopts = {{\n    'xyzrn_dir': r"{S.root / 'xyzrn'}",\n    'msms_dir': r"{S.root / 'msms'}",\n}}\n\nv, f, n, rid = computeMSMS(\n    pair,\n    dir_opts=opts,\n    probe_radius=0.5\n)\n\nprint("vertices:", v.shape)\nprint("faces:", f.shape)\nprint("normals:", n.shape)\nprint("residue IDs:", len(rid))\n\nassert v.ndim == 2\nassert v.shape[1] == 3\nassert f.ndim == 2\nassert f.shape[1] == 3\nassert n.shape == v.shape\nassert len(rid) == len(v)\n\nprint("ONE-PROTEIN MSMS SMOKE TEST: PASS")\nPY''', cwd=S.METALB, check=False, capture=True)
    print(S.rr.stdout)
    if S.rr.returncode != 0:
        raise RuntimeError('One-protein MSMS smoke test failed. Full error is printed above.')
    print('\nFINAL MSMS 2.6.1 FIX: PASS')

# 18D. MetalBind surface preprocessing — FINAL
def stage_69():
    S.APPLY_MSMS = S.METALB / 'compute_surface' / 'apply_msms.py'
    if not S.APPLY_MSMS.exists():
        raise FileNotFoundError(f'MetalBind apply_msms.py not found: {S.APPLY_MSMS}. Run 18A first.')
    S.APPLY_MSMS.write_text('\nfrom default_config.bin_path import bin_path\nfrom compute_surface.read_msms import read_msms\n\nfrom subprocess import Popen, PIPE\nfrom pathlib import Path\n\nimport shutil\nimport tempfile\n\n\ndef computeMSMS(pair, dir_opts, probe_radius):\n\n    # --------------------------------------------------------\n    # Author-generated XYZRN input\n    # --------------------------------------------------------\n    source_xyzrn = (\n        Path(dir_opts[\'xyzrn_dir\'])\n        / f\'{pair}.xyzrn\'\n    )\n\n    if not source_xyzrn.exists():\n        raise FileNotFoundError(\n            f\'MSMS input XYZRN missing: {source_xyzrn}\'\n        )\n\n    if source_xyzrn.stat().st_size == 0:\n        raise RuntimeError(\n            f\'MSMS input XYZRN is empty: {source_xyzrn}\'\n        )\n\n    # --------------------------------------------------------\n    # SHORT LOCAL PATHS ONLY for the native legacy MSMS binary\n    # --------------------------------------------------------\n    content_dir = Path(\'/content\')\n    scratch_parent = (\n        content_dir\n        if content_dir.is_dir()\n        else Path(tempfile.gettempdir())\n    )\n\n    scratch = Path(\n        tempfile.mkdtemp(\n            prefix=\'mbmsms_\',\n            dir=str(scratch_parent),\n        )\n    )\n\n    local_xyzrn = scratch / \'in.xyzrn\'\n    output_base = scratch / \'surface\'\n\n    try:\n        shutil.copy2(\n            source_xyzrn,\n            local_xyzrn,\n        )\n\n        msms_bin = str(bin_path[\'MSMS\'])\n\n        args = [\n            msms_bin,\n            \'-density\', \'3\',\n            \'-hdensity\', \'3\',\n            \'-probe\', str(probe_radius),\n            \'-if\', str(local_xyzrn),\n            \'-of\', str(output_base),\n            \'-af\', str(output_base),\n        ]\n\n        print(\n            \'MSMS COMMAND:\',\n            \' \'.join(args),\n        )\n\n        p = Popen(\n            args,\n            stdout=PIPE,\n            stderr=PIPE,\n        )\n\n        # IMPORTANT: this must be one assignment.\n        stdout, stderr = p.communicate()\n\n        stdout_text = stdout.decode(\n            \'utf-8\',\n            errors=\'replace\',\n        )\n        stderr_text = stderr.decode(\n            \'utf-8\',\n            errors=\'replace\',\n        )\n\n        if p.returncode != 0:\n            raise RuntimeError(\n                \'MSMS failed.\\n\'\n                f\'Return code: {p.returncode}\\n\'\n                f\'Input: {local_xyzrn}\\n\'\n                f\'STDOUT:\\n{stdout_text}\\n\'\n                f\'STDERR:\\n{stderr_text}\'\n            )\n\n        vert = Path(str(output_base) + \'.vert\')\n        face = Path(str(output_base) + \'.face\')\n\n        for fp in (vert, face):\n            if not fp.exists():\n                raise RuntimeError(\n                    \'MSMS returned success but output is missing: \'\n                    f\'{fp}\\n\'\n                    f\'STDOUT:\\n{stdout_text}\\n\'\n                    f\'STDERR:\\n{stderr_text}\'\n                )\n            if fp.stat().st_size == 0:\n                raise RuntimeError(\n                    f\'MSMS output is empty: {fp}\'\n                )\n\n        # Legacy MSMS .vert/.face are supposed to be ASCII text.\n        for fp in (vert, face):\n            raw = fp.read_bytes()\n            try:\n                raw.decode(\'ascii\')\n            except UnicodeDecodeError as e:\n                raise RuntimeError(\n                    \'MSMS generated non-ASCII output despite \'\n                    \'short-path execution.\\n\'\n                    f\'File: {fp}\\n\'\n                    f\'Size: {len(raw)} bytes\\n\'\n                    f\'First 256 bytes: {raw[:256]!r}\\n\'\n                    f\'Decode error: {e}\\n\'\n                    f\'MSMS STDOUT:\\n{stdout_text}\\n\'\n                    f\'MSMS STDERR:\\n{stderr_text}\'\n                )\n\n        # AUTHORS\' original parser.\n        vertices, faces, normalv, res_id = read_msms(\n            str(output_base)\n        )\n\n        # Strict geometry checks.\n        if getattr(vertices, \'ndim\', None) != 2 or vertices.shape[1] != 3:\n            raise RuntimeError(\n                f\'MSMS vertices must be N x 3, got {getattr(vertices, "shape", None)}\'\n            )\n\n        if getattr(faces, \'ndim\', None) != 2 or faces.shape[1] != 3:\n            raise RuntimeError(\n                f\'MSMS faces must be M x 3, got {getattr(faces, "shape", None)}\'\n            )\n\n        if normalv.shape != vertices.shape:\n            raise RuntimeError(\n                \'MSMS normal/vertex shape mismatch: \'\n                f\'{normalv.shape} vs {vertices.shape}\'\n            )\n\n        if len(res_id) != len(vertices):\n            raise RuntimeError(\n                \'MSMS residue-ID/vertex count mismatch: \'\n                f\'{len(res_id)} vs {len(vertices)}\'\n            )\n\n        if len(vertices) == 0 or len(faces) == 0:\n            raise RuntimeError(\n                \'MSMS produced empty geometry\'\n            )\n\n        print(\n            \'MSMS PASS:\',\n            f\'vertices={len(vertices)}\',\n            f\'faces={len(faces)}\',\n        )\n\n        return (\n            vertices,\n            faces,\n            normalv,\n            res_id,\n        )\n\n    finally:\n        shutil.rmtree(\n            scratch,\n            ignore_errors=True,\n        )\n')
    print('Patched short-path MSMS:', S.APPLY_MSMS)
    S.METALB_PREP_SCRIPT = S.METALB / 'external_prepare_v6.py'
    S.METALB_PREP_SCRIPT.write_text("\nimport argparse\nimport json\nimport traceback\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\n\nfrom MetalBindProcessor import MetalBindProcessor\n\n\n# ============================================================\n# Label-free inference adapters\n# ============================================================\n\ndef dummy_nucleic(self):\n    return []\n\n\ndef dummy_interface(self, anno):\n    # Inference only: labels/interfaces are placeholders and\n    # are not used to generate predictions.\n    return np.zeros(\n        self.mesh.vertices.shape[0],\n        dtype=np.float32\n    )\n\n\ndef dummy_site(self, anno):\n    # Preserve exact surface-residue ordering from author's processor.\n    s2i = {\n        res: i\n        for i, res in enumerate(self.surface_res)\n    }\n\n    point_res_id = np.zeros(\n        len(self.vert_info),\n        dtype=np.int64\n    )\n\n    for k, info in enumerate(self.vert_info):\n        fields = info.split('_')\n\n        if len(fields) < 2:\n            raise RuntimeError(\n                f'Unexpected vert_info entry: {info}'\n            )\n\n        rid = fields[0] + '_' + fields[1]\n\n        if rid not in s2i:\n            raise RuntimeError(\n                f'point residue {rid} missing from surface-residue map'\n            )\n\n        point_res_id[k] = s2i[rid]\n\n    self.site_by_biolip = set()\n\n    return (\n        np.zeros(\n            len(self.surface_res),\n            dtype=np.int64\n        ),\n        point_res_id\n    )\n\n\n# Patch labels only; structural preprocessing remains author's code.\nMetalBindProcessor._get_nucleic_acid_atom = dummy_nucleic\nMetalBindProcessor._get_interface_mental = dummy_interface\nMetalBindProcessor._get_site_biolip = dummy_site\n\n\n# ============================================================\n# CLI\n# ============================================================\n\nap = argparse.ArgumentParser()\nap.add_argument('--manifest', required=True)\nap.add_argument('--root', required=True)\nap.add_argument('--rebuild', action='store_true')\na = ap.parse_args()\n\nroot = Path(a.root)\nmanifest_path = Path(a.manifest)\n\nif not manifest_path.exists():\n    raise FileNotFoundError(\n        f'Manifest not found: {manifest_path}'\n    )\n\nmanifest = pd.read_csv(manifest_path)\n\nif len(manifest) == 0:\n    raise RuntimeError(\n        f'Empty manifest: {manifest_path}'\n    )\n\nrequired_cols = {\n    'prot_id',\n    'sequence',\n    'pdb_path',\n    'esm_txt',\n}\n\nmissing_cols = required_cols - set(manifest.columns)\n\nif missing_cols:\n    raise RuntimeError(\n        f'Manifest missing columns: {sorted(missing_cols)}'\n    )\n\n\n# ============================================================\n# Output directories\n# ============================================================\n\nfor d in [\n    'chain_pdb',\n    'protonated_pdb',\n    'xyzrn',\n    'msms',\n    'data_label',\n    'ligand_dummy',\n    'hmm',\n    'fasta',\n    'dssp',\n    'save',\n    'mesh',\n    'mesh_fix',\n    'ply',\n]:\n    (root / d).mkdir(\n        parents=True,\n        exist_ok=True\n    )\n\n\nraw_pdb = Path(\n    manifest.iloc[0].pdb_path\n).parent\n\nesm_dir = Path(\n    manifest.iloc[0].esm_txt\n).parent\n\n\nif not raw_pdb.exists():\n    raise FileNotFoundError(\n        f'raw_pdb directory not found: {raw_pdb}'\n    )\n\nif not esm_dir.exists():\n    raise FileNotFoundError(\n        f'ESM directory not found: {esm_dir}'\n    )\n\n\nopts = {\n    'base_dir': str(root),\n    'raw_pdb_dir': str(raw_pdb),\n    'ligand_dir': str(root / 'ligand_dummy'),\n    'hhm_dir': str(root / 'hmm'),\n    'fasta_dir': str(root / 'fasta'),\n    'dssp': str(root / 'dssp'),\n    'protonated_pdb_dir': str(root / 'protonated_pdb'),\n    'chain_pdb_dir': str(root / 'chain_pdb'),\n    'xyzrn_dir': str(root / 'xyzrn'),\n    'msms_dir': str(root / 'msms'),\n    'data_label': str(root / 'data_label'),\n    'save_dir': str(root / 'save'),\n    'mesh_dir': str(root / 'mesh'),\n    'mesh_fix_dir': str(root / 'mesh_fix'),\n    'ply_dir': str(root / 'ply'),\n    'esm_dir': str(esm_dir),\n}\n\n\nprint(\n    f'Preparing {len(manifest)} proteins'\n)\n\nprint(\n    'raw_pdb_dir =',\n    opts['raw_pdb_dir']\n)\n\nprint(\n    'esm_dir =',\n    opts['esm_dir']\n)\n\n\n# ============================================================\n# Per-protein preprocessing\n# ============================================================\n\nprepared = 0\ncached = 0\n\n\nfor i, row in enumerate(\n    manifest.itertuples(),\n    start=1\n):\n\n    pid = str(row.prot_id)\n    pair = f'{pid}:A:X1'\n\n    fp = (\n        root\n        /\n        'data_label'\n        /\n        pair\n    )\n\n    mp = (\n        root\n        /\n        'data_label'\n        /\n        f'{pair}.surface_map.json'\n    )\n\n\n    # --------------------------------------------------------\n    # Cache reuse\n    # --------------------------------------------------------\n\n    if (\n        fp.exists()\n        and\n        mp.exists()\n        and\n        not a.rebuild\n    ):\n        cached += 1\n\n        print(\n            f'[{i}/{len(manifest)}] {pid}: cached'\n        )\n\n        continue\n\n\n    print(\n        f'[{i}/{len(manifest)}] {pid}: START'\n    )\n\n\n    # --------------------------------------------------------\n    # Strict input checks\n    # --------------------------------------------------------\n\n    pdb_path = Path(row.pdb_path)\n    esm_path = Path(row.esm_txt)\n\n    if not pdb_path.exists():\n        raise FileNotFoundError(\n            f'{pid}: PDB missing: {pdb_path}'\n        )\n\n    if not esm_path.exists():\n        raise FileNotFoundError(\n            f'{pid}: ESM text missing: {esm_path}'\n        )\n\n\n    try:\n\n        # ----------------------------------------------------\n        # AUTHORS' processor\n        # ----------------------------------------------------\n\n        proc = MetalBindProcessor(\n            pair=pair,\n            anno='',\n            ligand_type='ZN',\n            # Metal branch -> 0.5-A probe radius in author's code\n            dir_opts=opts,\n            RQ_window=6.0,\n            # exact public prepare_one.sh setting\n        )\n\n        proc.get_data()\n\n\n        # ----------------------------------------------------\n        # Verify author's data object\n        # ----------------------------------------------------\n\n        if not fp.exists():\n            raise RuntimeError(\n                f'{pid}: proc.get_data() finished but expected '\n                f'data file was not created: {fp}'\n            )\n\n\n        # ----------------------------------------------------\n        # Surface residue -> canonical sequence position\n        # ----------------------------------------------------\n\n        mapping = []\n\n        for s in proc.surface_res:\n\n            fields = s.split('_')\n\n            if len(fields) < 2:\n                raise RuntimeError(\n                    f'{pid}: malformed surface residue: {s}'\n                )\n\n            if fields[0] != 'A':\n                raise RuntimeError(\n                    f'{pid}: unexpected surface chain {fields[0]}'\n                )\n\n            resid = int(fields[1])\n\n            if (\n                resid < 1\n                or\n                resid > len(row.sequence)\n            ):\n                raise RuntimeError(\n                    f'{pid}: surface residue number {resid} '\n                    f'outside sequence length {len(row.sequence)}'\n                )\n\n            mapping.append(\n                {\n                    'surface_residue': s,\n                    'canonical_index': resid - 1,\n                }\n            )\n\n\n        if len(mapping) == 0:\n            raise RuntimeError(\n                f'{pid}: no surface residues produced'\n            )\n\n\n        # ----------------------------------------------------\n        # Save exact remapping metadata\n        # ----------------------------------------------------\n\n        mp.write_text(\n            json.dumps(\n                {\n                    'prot_id': pid,\n                    'sequence_length': len(row.sequence),\n                    'surface_res': list(\n                        proc.surface_res\n                    ),\n                    'mapping': mapping,\n                },\n                indent=2\n            )\n        )\n\n\n        if not mp.exists():\n            raise RuntimeError(\n                f'{pid}: surface map was not written'\n            )\n\n\n        prepared += 1\n\n        print(\n            f'[{i}/{len(manifest)}] {pid}: PASS '\n            f'surface_res={len(mapping)}'\n        )\n\n\n    except Exception:\n\n        print(\n            '\\n'\n            + '=' * 80\n        )\n\n        print(\n            f'FAILED PROTEIN: {pid}'\n        )\n\n        print(\n            f'PAIR: {pair}'\n        )\n\n        print(\n            f'PDB: {pdb_path}'\n        )\n\n        print(\n            f'ESM: {esm_path}'\n        )\n\n        print(\n            '=' * 80\n        )\n\n        traceback.print_exc()\n\n        print(\n            '=' * 80\n            + '\\n'\n        )\n\n        raise\n\n\nprint(\n    f'PREPROCESS COMPLETE: '\n    f'prepared_now={prepared}, '\n    f'cached={cached}, '\n    f'total={len(manifest)}'\n)\n")
    if S.RUN_METALBIND_TRANSFER:
        for S.dataset, S.df in S.DATASETS.items():
            print('\n' + '=' * 80)
            print('MetalBind surface preprocessing:', S.dataset)
            print('=' * 80)
            S.root = S.METALB_PREP_ROOT / S.dataset
            S.manifest_path = S.root / 'manifest.csv'
            if not S.manifest_path.exists():
                raise FileNotFoundError(f'{S.dataset}: manifest missing: {S.manifest_path}. Run 18C first.')
            S.cmd = f'python "{S.METALB_PREP_SCRIPT}" --manifest "{S.manifest_path}" --root "{S.root}"' + (' --rebuild' if S.REBUILD_METALBIND_SURFACE else '')
            S.rr = mex('metalbind', S.cmd, cwd=S.METALB, check=False, capture=True)
            print('\n' + '-' * 80)
            print(f'OUTPUT: {S.dataset}')
            print('-' * 80)
            if S.rr.stdout:
                print(S.rr.stdout)
            print(f'\nRETURN CODE: {S.rr.returncode}')
            if S.rr.returncode != 0:
                raise RuntimeError(f'{S.dataset}: MetalBind surface preprocessing failed with return code {S.rr.returncode}. The real subprocess traceback is printed above.')
            S.missing = []
            S.bad_maps = []
            for S.r in S.df.itertuples():
                S.pid = str(S.r.prot_id)
                S.pair = f'{S.pid}:A:X1'
                S.data_fp = S.root / 'data_label' / S.pair
                S.map_fp = S.root / 'data_label' / f'{S.pair}.surface_map.json'
                if not S.data_fp.exists():
                    S.missing.append((S.pid, 'pickle'))
                if not S.map_fp.exists():
                    S.missing.append((S.pid, 'map'))
                    continue
                try:
                    S.z = json.loads(S.map_fp.read_text())
                except Exception as e:
                    S.bad_maps.append((S.pid, f'unreadable map: {e}'))
                    continue
                if int(S.z.get('sequence_length', -1)) != len(S.r.sequence):
                    S.bad_maps.append((S.pid, 'sequence_length mismatch'))
                S.mapping = S.z.get('mapping', [])
                if len(S.mapping) == 0:
                    S.bad_maps.append((S.pid, 'empty mapping'))
                    continue
                S.canonical_indices = [int(x['canonical_index']) for x in S.mapping]
                if any((x < 0 or x >= len(S.r.sequence) for x in S.canonical_indices)):
                    S.bad_maps.append((S.pid, 'canonical index out of range'))
            if S.missing:
                raise RuntimeError(f'{S.dataset}: MetalBind preprocessing incomplete: {S.missing[:10]}')
            if S.bad_maps:
                raise RuntimeError(f'{S.dataset}: MetalBind surface-map audit failed: {S.bad_maps[:10]}')
            print(f'{S.dataset}: MetalBind preprocessing PASS {len(S.df)}/{len(S.df)} proteins.')
        print('\nMetalBind surface preprocessing PASSED for TestSet300 + TestSetNew46.')

# 18E. RUN ALL 13 OFFICIAL MetalBind CHECKPOINTS AND REMAP FULL SEQUENCES
def stage_70():
    S.METALB_INFER_SCRIPT = S.METALB / 'external_infer_v6.py'
    S.METALB_INFER_SCRIPT.write_text('\nimport argparse, ast, json\nfrom pathlib import Path\nimport _pickle as cPickle\nimport numpy as np\nimport pandas as pd\nimport torch\n\nfrom Arguments import parser as author_parser\nfrom models.MetalBind_model import MetalBind\n\nap=argparse.ArgumentParser()\nap.add_argument(\'--manifest\',required=True)\nap.add_argument(\'--prepared_root\',required=True)\nap.add_argument(\'--repo\',required=True)\nap.add_argument(\'--output_root\',required=True)\na=ap.parse_args()\n\nif not torch.cuda.is_available():\n    raise RuntimeError(\'MetalBind author forward path requires CUDA in this reproduction\')\n\nmanifest=pd.read_csv(a.manifest)\nprep=Path(a.prepared_root)\nrepo=Path(a.repo)\noutroot=Path(a.output_root); outroot.mkdir(parents=True,exist_ok=True)\n\nions=[\'CA\',\'CD\',\'CO\',\'CU\',\'CU1\',\'FE\',\'FE2\',\'K\',\'MG\',\'MN\',\'NA\',\'NI\',\'ZN\']\n\n\ndef tensor_data(data):\n    n_points=data[\'xyz\'].shape[0]\n    xyz_type=np.asarray(data[\'xyz_type\']).reshape(-1,1).astype(np.int64)\n    chemi=torch.zeros((n_points,6),dtype=torch.float32)\n    chemi.scatter_(1,torch.from_numpy(xyz_type),1.0)\n    return {\n        \'xyz\':torch.from_numpy(np.asarray(data[\'xyz\'],dtype=np.float32)),\n        \'nuv\':torch.from_numpy(np.asarray(data[\'nuv\'],dtype=np.float32)).contiguous(),\n        \'hmm\':torch.from_numpy(np.asarray(data[\'esm\'],dtype=np.float32)),\n        \'geo\':torch.from_numpy(np.asarray(data[\'pka\'],dtype=np.float32)),\n        \'chemi\':chemi,\n        \'y\':torch.zeros(n_points,dtype=torch.long),\n        \'site\':torch.zeros(len(data[\'site\']),dtype=torch.long),\n        \'site_point\':torch.from_numpy(np.asarray(data[\'site_point\'],dtype=np.int64)),\n        \'xyz_batch\':torch.zeros(n_points,dtype=torch.long),\n    }\n\nfor ion in ions:\n    ckpt=repo/\'Pretrained_model\'/ion\n    if not (ckpt/\'best.pth\').exists():\n        raise FileNotFoundError(f\'missing released checkpoint {ckpt/"best.pth"}\')\n\n    args=author_parser.parse_args([\n        \'--ligand\',ion,\n        \'--checkpoints_dir\',str(ckpt),\n        \'--device\',\'cuda:0\',\n        \'--radius\',\'6.0\',\n        \'--n_layers\',\'4\',\n        \'--input_feature_type\',\'esm\',\'chemi\',\'pka\',\n        \'--batch_size\',\'1\',\n        \'--lr\',\'0.001\',\n        \'--loss_type\',\'site\',\n    ])\n    # dir_opts is not consumed by model construction/inference once tensors are ready.\n    args.dir_opts={}\n    model=MetalBind(args)\n    model.load_network()\n    model.eval()\n\n    rows=[]\n    with torch.no_grad():\n        for row in manifest.itertuples():\n            pid=str(row.prot_id); L=len(row.sequence)\n            pair=f\'{pid}:A:X1\'\n            with open(prep/\'data_label\'/pair,\'rb\') as f:\n                data=cPickle.load(f)\n            meta=json.loads((prep/\'data_label\'/f\'{pair}.surface_map.json\').read_text())\n\n            td=tensor_data(data)\n            model.set_input(td)\n            surf=np.asarray(model.test().detach().cpu(),dtype=float).reshape(-1)\n            mapping=meta[\'mapping\']\n            if len(surf) != len(mapping):\n                raise RuntimeError(f\'{pid}/{ion}: model emitted {len(surf)} surface residues, map has {len(mapping)}\')\n            if not np.all(np.isfinite(surf)) or np.min(surf)<0 or np.max(surf)>1:\n                raise RuntimeError(f\'{pid}/{ion}: invalid surface probabilities\')\n\n            full=np.zeros(L,dtype=np.float32)\n            seen=set()\n            for prob,m in zip(surf,mapping):\n                idx=int(m[\'canonical_index\'])\n                if idx in seen:\n                    raise RuntimeError(f\'{pid}: duplicate canonical surface index {idx}\')\n                seen.add(idx)\n                full[idx]=float(prob)\n\n            for i,p in enumerate(full):\n                rows.append((pid,i,float(p),i in seen))\n\n    out=pd.DataFrame(rows,columns=[\'prot_id\',\'residue_index\',\'probability\',\'is_surface_residue\'])\n    out.to_csv(outroot/f\'{ion}.csv.gz\',index=False,compression=\'gzip\')\n    print(ion,\'saved\',len(out),\'residue rows\')\n    del model\n    torch.cuda.empty_cache()\n')
    S.METALBIND_EXTERNAL_ROOT = S.WORK / 'metalbind_external_predictions_v6'
    S.METALBIND_EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    if S.RUN_METALBIND_TRANSFER:
        for S.dataset, S.df in S.DATASETS.items():
            S.root = S.METALB_PREP_ROOT / S.dataset
            S.outroot = S.METALBIND_EXTERNAL_ROOT / S.dataset
            S.outroot.mkdir(parents=True, exist_ok=True)
            mex('metalbind', f'python "{S.METALB_INFER_SCRIPT}" --manifest "{S.root / 'manifest.csv'}" --prepared_root "{S.root}" --repo "{S.METALB}" --output_root "{S.outroot}"', cwd=S.METALB)

# 18F. STRICT FULL-LENGTH MetalBind INGEST + POOL + FULL METRICS
def stage_71():
    S.metalb_rows = []
    S.metalbind_pooled_probs = {}
    S.metalbind_ion_probs = {}
    if S.RUN_METALBIND_TRANSFER:
        S.surface_coverage = []
        for S.dataset, S.df in S.DATASETS.items():
            S.ion_probs = {}
            for S.ion in S.METALB_IONS:
                S.fp = S.METALBIND_EXTERNAL_ROOT / S.dataset / f'{S.ion}.csv.gz'
                if not S.fp.exists():
                    raise FileNotFoundError(f'MetalBind missing completed ion file: {S.fp}')
                S.ion_probs[S.ion] = load_metalbind_ion_csv(S.fp, S.df)
            S.pooled = {}
            for S.r in S.df.itertuples():
                S.pooled[S.r.prot_id] = np.maximum.reduce([S.ion_probs[ion][S.r.prot_id] for ion in S.METALB_IONS])
            S.metalbind_pooled_probs[S.dataset] = S.pooled
            S.metalbind_ion_probs[S.dataset] = S.ion_probs
            save_probs(S.pooled, S.PRED / f'MetalBind_13ion_max_TRANSFER__{S.dataset}__metal_binding.csv.gz')
            S.out, S.pt = evaluate_probability_dict(S.df, S.pooled, 'metal_binding', 'MetalBind_13ion_max_TRANSFER', S.dataset, 'fixed_0.5', 0.5, 'C_cross_benchmark_transfer')
            S.metalb_rows.append(S.out)
            S.pt.to_csv(S.RESULTS / f'MetalBind_{S.dataset}_per_protein.csv', index=False)
            S.z = pd.read_csv(S.METALBIND_EXTERNAL_ROOT / S.dataset / f'{S.METALB_IONS[0]}.csv.gz')
            for S.pid, S.q in S.z.groupby(S.z.prot_id.astype(str)):
                S.surface_coverage.append([S.dataset, S.pid, int(S.q.is_surface_residue.astype(bool).sum()), len(S.q), float(S.q.is_surface_residue.astype(bool).mean())])
        S.metalb_df = pd.DataFrame(S.metalb_rows)
        S.metalb_df.to_csv(S.RESULTS / 'MetalBind_TRANSFER_FULL_METRICS_300_46.csv', index=False)
        pd.DataFrame(S.surface_coverage, columns=['dataset', 'prot_id', 'surface_residues', 'total_residues', 'surface_fraction']).to_csv(S.RESULTS / 'MetalBind_surface_coverage_300_46.csv', index=False)
        display(S.metalb_df)

STAGES = [stage_61, stage_62, stage_63, stage_64, stage_65, stage_66, stage_67, stage_68, stage_69, stage_70, stage_71]

