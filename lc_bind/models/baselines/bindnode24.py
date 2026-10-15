"""Extracted from MASTER_BINDING_BASELINE_REPRODUCTION_300_46_FULL_v10_BINDNODE24_FIXED-2.ipynb. Section: bindnode24."""
from __future__ import annotations

from lc_bind.compat import display
from lc_bind.session import S

def prepare():
    global Bio, Chain, Model, PDBIO, PDBParser, Path, SeqIO, Structure, T5EncoderModel, T5Tokenizer, accuracy_score, ast, auc, average_precision_score, balanced_accuracy_score, brier_score_loss, create_env, deepcopy, env_path, evaluate_probability_dict, f1_score, fbeta_score, generate_prott5_h5, h5py, hashlib, importlib, jaccard_score, json, matthews_corrcoef, mex, np, os, pd, pickle, precision_recall_curve, precision_score, read_binding_file, read_fasta_dict, recall_score, roc_auc_score, roc_curve, save_probs, seq1, sh, shutil, student_t, subprocess, sys, thresholds_from_oof, time, torch, tqdm
    from copy import deepcopy
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.SeqUtils import seq1
    import shutil
    import json
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
    import hashlib
    from lc_bind.models.baselines.bindembed21 import generate_prott5_h5, read_binding_file, read_fasta_dict
    from lc_bind.models.baselines.metrics import evaluate_probability_dict, save_probs, thresholds_from_oof
    from lc_bind.models.baselines.setup import create_env, env_path, mex, sh

def find_project_pdb(pid):
    for root in S.PDB_CANDIDATES:
        if not root.exists():
            continue
        p = root / f'{pid}.pdb'
        if p.exists():
            return p
        hit = list(root.rglob(f'{pid}.pdb'))
        if hit:
            return hit[0]
    return None

def bindnode_config(result_dir, embeddings, id_list, fasta, pdb_dir, weight_dir, dssp_dir, cache_dir):
    """
    Prediction-only config matching the released bindNode24 architecture.

    use_cache=False is deliberate for external inference:
    it prevents stale distance matrices generated from a different structure
    directory from being silently reused.
    """
    return f'[DEFAULT]\nmode = predict\n\n[FILE_PATHS]\nresult_dir = {result_dir}\nembeddings = {embeddings}\nid_list = {id_list}\ncv_split_id_files_folder = {S.BN24 / 'dataset/cv_splits/predicted_structures'}\n3d_structure_dir = {pdb_dir}\nprecomputed_DSSP_feature_dir = {dssp_dir}\nfasta_file_path = {fasta}\ncache_folder = {cache_dir}\n\n[LABELS]\nmetal_label_file = {S.BN24 / 'dataset/binding_residues_2.5_metal.txt'}\nsmall_label_file = {S.BN24 / 'dataset/binding_residues_2.5_small.txt'}\nnuclear_label_file = {S.BN24 / 'dataset/binding_residues_2.5_nuclear.txt'}\n\n[UTILITY]\ndo_logging = yes\nuse_cache = False\n\n[MODEL]\nweight_folder = {weight_dir}\nmodel_type = SAGEConv\ncutoff = 0.5\nin_channels = 1024\nnr_additional_features = 20\nuse_dssp = False\nout_channels = 3\n\n[DATASET]\ncutoff_structure = 17\n\n[OUTPUT]\nwrite_ri = False\n\n[MODEL_PARAMETERS]\nfeatures = 128\ndropout = 0.7\nepochs = 100\nearly_stopping = True\nactivation = F.leaky_relu\nbatchsize = 406\ndropout_fcn = 0.8\n\n[LOSS_FUNCTION_PARAMETERS]\nweights = [8.9, 7.7, 4.4]\n\n[OPTIMIZER_PARAMETERS]\nlearning_rate = 0.01\nbetas = [0.9, 0.999]\nepsilon = 1e-8\nweight_decay = 0.0\n'

def write_bindnode24_fullprecision_adapter(dataset, cfg, expected_ids, expected_lengths):
    """
    Run released bindNode24 models but bypass the authors' text writer,
    which rounds probabilities to 3 decimals.

    Model forward pass, graph construction, five checkpoints, and ensemble
    averaging are unchanged. Only serialization is replaced by .npy float32.
    """
    outdir = S.PRED / 'bindNode24' / dataset / 'fullprecision_npy'
    outdir.mkdir(parents=True, exist_ok=True)
    expected_json = S.WORK / f'bindNode24_{dataset}_expected.json'
    import json
    expected_json.write_text(json.dumps({'ids': [str(x) for x in expected_ids], 'lengths': {str(k): int(v) for k, v in expected_lengths.items()}}))
    script = S.BN24 / f'run_{dataset}_fullprecision.py'
    script.write_text(f'''import sys, json\nfrom pathlib import Path\nimport numpy as np\n\nREPO = Path(r"{S.BN24}")\nsys.path.insert(0, str(REPO/'src'))\n\nfrom setup.configProcessor import (\n    validate_config, use_cache, get_model_parameter_dict, get_weight_dir\n)\nfrom misc.Cache import setup_chache\nfrom machine_learning.predict import run_prediction\nfrom data_processing.post_processing import get_averaged_predictions\n\nOUT = Path(r"{outdir}")\nEXPECTED = json.loads(Path(r"{expected_json}").read_text())\n\nprint("=== bindNode24 full-precision adapter preflight ===")\nvalidate_config()\n\nweights = sorted(Path(get_weight_dir()).glob('*.state_dict'))\nprint("released weight files:", [p.name for p in weights])\nif len(weights) != 5:\n    raise RuntimeError(f"Expected 5 bindNode24 state_dict files, found {{len(weights)}}")\n\nif use_cache():\n    setup_chache()\n\nparams = get_model_parameter_dict(only_first_value=True)\nprint("model parameters:", params)\n\npredictions = run_prediction(params)\navg = get_averaged_predictions(predictions)\n\nexpected_ids = EXPECTED['ids']\nlengths = EXPECTED['lengths']\n\nmissing = sorted(set(expected_ids) - set(avg))\nextra = sorted(set(avg) - set(expected_ids))\nif missing:\n    raise RuntimeError(f"Missing predictions for {{len(missing)}} proteins: {{missing[:10]}}")\nif extra:\n    raise RuntimeError(f"Unexpected proteins in prediction dict: {{extra[:10]}}")\n\nOUT.mkdir(parents=True, exist_ok=True)\nfor pid in expected_ids:\n    arr = np.asarray(avg[pid], dtype=np.float64)\n    L = int(lengths[pid])\n\n    if arr.shape != (L,3):\n        raise RuntimeError(\n            f"{{pid}}: prediction shape {{arr.shape}}, expected {{(L,3)}}"\n        )\n    if not np.all(np.isfinite(arr)):\n        raise RuntimeError(f"{{pid}}: non-finite probabilities")\n\n    np.save(OUT/f"{{pid}}.npy", arr.astype(np.float32))\n\nprint("saved", len(expected_ids), "proteins to", OUT)\nprint("=== bindNode24 full-precision adapter PASS ===")\n''')
    return (script, outdir)

def load_bindnode24_npy(df, folder):
    folder = Path(folder)
    metal, nuc, small = ({}, {}, {})
    for r in df.itertuples():
        pid = str(r.prot_id)
        fp = folder / f'{pid}.npy'
        if not fp.exists():
            raise FileNotFoundError(fp)
        arr = np.load(fp)
        if arr.shape != (len(r.sequence), 3):
            raise RuntimeError(f'{pid}: prediction shape {arr.shape}; expected {(len(r.sequence), 3)}')
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f'{pid}: non-finite bindNode24 probability')
        metal[pid] = arr[:, 0].astype(np.float64)
        nuc[pid] = arr[:, 1].astype(np.float64)
        small[pid] = arr[:, 2].astype(np.float64)
    anyp = {pid: np.maximum.reduce([metal[pid], nuc[pid], small[pid]]) for pid in metal}
    return {'metal_binding': metal, 'nuclear_binding': nuc, 'small_binding': small, 'any_ligand_binding_sites': anyp}

def run_or_recover_bindnode24(dataset, df, fasta, id_list, pdb_dir, dssp_dir):
    """
    Disk-state-safe bindNode24 inference.

    If all expected .npy predictions already exist with the correct shapes,
    inference is skipped. This makes Colab disconnects/restarts recoverable.
    """
    outdir = S.PRED / 'bindNode24' / dataset / 'fullprecision_npy'
    outdir.mkdir(parents=True, exist_ok=True)
    expected = {str(r.prot_id): len(r.sequence) for r in df.itertuples()}
    good = True
    for pid, L in expected.items():
        fp = outdir / f'{pid}.npy'
        if not fp.exists():
            good = False
            break
        try:
            a = np.load(fp, mmap_mode='r')
            if a.shape != (L, 3):
                good = False
                break
        except Exception:
            good = False
            break
    if good and len(list(outdir.glob('*.npy'))) >= len(expected):
        print(f'{dataset}: all {len(expected)} bindNode24 predictions already exist -> SKIP inference')
        return outdir
    result = S.PRED / 'bindNode24' / dataset
    result.mkdir(parents=True, exist_ok=True)
    cfg = S.WORK / f'bindNode24_{dataset}.ini'
    cfg.write_text(bindnode_config(result_dir=result, embeddings=S.PROTT5[dataset], id_list=id_list, fasta=fasta, pdb_dir=pdb_dir, weight_dir=S.BN24 / 'trained_model_dicts/bindNode24', dssp_dir=dssp_dir, cache_dir=S.CACHE / f'bindNode24_{dataset}_fresh'))
    script, outdir = write_bindnode24_fullprecision_adapter(dataset=dataset, cfg=cfg, expected_ids=list(expected.keys()), expected_lengths=expected)
    r = mex('bindnode24', f'python "{script.name}" -f "{cfg}"', cwd=S.BN24, check=False, capture=True)
    print(r.stdout)
    print('return code:', r.returncode)
    if r.returncode != 0:
        raise RuntimeError(f'bindNode24 inference failed on {dataset}. The complete INNER traceback is printed immediately above.')
    missing = []
    bad = []
    for pid, L in expected.items():
        fp = outdir / f'{pid}.npy'
        if not fp.exists():
            missing.append(pid)
            continue
        a = np.load(fp, mmap_mode='r')
        if a.shape != (L, 3):
            bad.append((pid, a.shape, L))
    if missing or bad:
        raise RuntimeError(f'{dataset}: incomplete bindNode24 output. missing={missing[:10]}, bad={bad[:10]}')
    print(f'{dataset}: bindNode24 output audit PASS ({len(expected)} proteins)')
    return outdir

def _bn24_standard_residues(chain):
    return [r for r in chain if r.id[0] == ' ' and r.resname.strip().upper() in S.AA3_BN24]

def _bn24_seq(residues):
    return ''.join((seq1(r.resname.strip().upper()) for r in residues))

def _normalize_bn24_external_pdb(src, sequence, dst):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('src', str(src))
    model = next(structure.get_models())
    matches = []
    for chain in model:
        residues = _bn24_standard_residues(chain)
        if _bn24_seq(residues) == sequence:
            matches.append((chain, residues))
    if not matches:
        raise RuntimeError(f'{src}: no chain exactly matches canonical sequence length {len(sequence)}')
    matches = sorted(matches, key=lambda x: (0 if str(x[0].id) == 'A' else 1, str(x[0].id)))
    chain, residues = matches[0]
    out_s = Structure('normalized')
    out_m = Model(0)
    out_c = Chain('A')
    out_s.add(out_m)
    out_m.add(out_c)
    for i, res in enumerate(residues, start=1):
        rr = deepcopy(res)
        rr.detach_parent()
        rr.id = (' ', i, ' ')
        out_c.add(rr)
    io = PDBIO()
    io.set_structure(out_s)
    io.save(str(dst))
    chk = PDBParser(QUIET=True).get_structure('chk', str(dst))
    cm = next(chk.get_models())
    seq2 = _bn24_seq(_bn24_standard_residues(cm['A']))
    if seq2 != sequence:
        raise RuntimeError(f'{dst}: normalized read-back sequence mismatch')

def _write_dummy_dssp_csv(pid, L, dst):
    """
    bindNode24's dataset loader requests DSSP arrays even when use_dssp=False.
    For Test46 no author DSSP files exist, so make shape-correct dummy files.
    The released SAGEConv checkpoint ignores these channels because
    use_dssp=False; they do not enter the forward computation.
    """
    rows = []
    for i in range(1, L + 1):
        rows.append([i, 'X', 'C', 0.0, 0.0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0])
    pd.DataFrame(rows).to_csv(dst, sep=';', header=False, index=False)

def build_bn24_dev_df():
    seqs = read_fasta_dict(S.BN24 / 'dataset/all.fasta')
    folds = {}
    ids = []
    for k in range(1, 6):
        z = [x.strip() for x in open(S.BN24 / f'dataset/cv_splits/predicted_structures/ids_split{k}.txt') if x.strip()]
        ids += z
        for x in z:
            folds[x] = k
    bind = {lab: read_binding_file(S.BN24 / f'dataset/binding_residues_2.5_{short}.txt') for lab, short in [('metal_binding', 'metal'), ('nuclear_binding', 'nuclear'), ('small_binding', 'small')]}
    rows = []
    for pid in ids:
        seq = seqs[pid]
        rec = {'prot_id': pid, 'sequence': seq, 'fold': folds[pid]}
        for lab in ['metal_binding', 'nuclear_binding', 'small_binding']:
            y = np.zeros(len(seq), dtype=np.int8)
            for pos in bind[lab].get(pid, []):
                if 1 <= pos <= len(seq):
                    y[pos - 1] = 1
            rec[lab] = y.tolist()
        rec['any_ligand_binding_sites'] = np.maximum.reduce([rec['metal_binding'], rec['nuclear_binding'], rec['small_binding']]).astype(int).tolist()
        rows.append(rec)
    return pd.DataFrame(rows)

def add_method(method, all_probs, thresholds, tier_by_dataset):
    for dataset, df in S.DATASETS.items():
        if dataset not in all_probs:
            continue
        for label in S.LABELS:
            ops = [('fixed_0.5', 0.5)]
            if thresholds is not None:
                ops += [('OOF_F1', thresholds[label]['OOF_F1']), ('OOF_MCC', thresholds[label]['OOF_MCC'])]
            for opname, t in ops:
                out, pt = evaluate_probability_dict(df, all_probs[dataset][label], label, method, dataset, opname, t, tier_by_dataset[dataset])
                S.all_rows.append(out)
                pt.insert(0, 'operating_point', opname)
                pt.insert(0, 'label', label)
                pt.insert(0, 'dataset', dataset)
                pt.insert(0, 'method', method)
                S.per_protein_frames.append(pt)

# 10A. bindNode24 ENV — AUTHOR-COMPATIBLE EXACT STACK
def stage_34():
    S.BN24 = S.REPOS / 'bindnode24'
    S.RESET_BINDNODE24_ENV = True
    if S.RUN_BINDNODE24 or S.RUN_OOF_THRESHOLDS:
        if S.RESET_BINDNODE24_ENV:
            S.p = env_path('bindnode24')
            if S.p.exists():
                print('Removing previous bindNode24 env:', S.p)
                sh(f'rm -rf "{S.p}"')
        create_env('bindnode24', '3.10')

        def _bn24_install(cmd, label):
            r = mex('bindnode24', cmd, check=False, capture=True)
            print(r.stdout)
            if r.returncode != 0:
                raise RuntimeError(f'bindNode24 {label} failed with return code {r.returncode}. Full installer output is printed immediately above.')
        _bn24_install('python -m pip install --upgrade "numpy==1.26.4" "pandas==2.2.0" "scipy==1.12.0" "scikit-learn==1.4.0" "h5py==3.10.0" "biopython==1.83" "fastapy==1.0.3" "colorama==0.4.6" "pathvalidate==3.2.0"', 'core dependency installation')
        _bn24_install('python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu118 "torch==2.2.0" "torchvision==0.17.0" "torchaudio==2.2.0"', 'PyTorch CUDA 11.8 installation')
        _bn24_install('python -m pip install --upgrade -f https://data.pyg.org/whl/torch-2.2.0+cu118.html "torch-scatter==2.1.2" "torch-sparse==0.6.18" "torch-cluster==1.6.3" "torch-spline-conv==1.2.2"', 'PyG compiled extension installation')
        _bn24_install('python -m pip install --upgrade "torch-geometric==2.5.0"', 'torch-geometric installation')
        S.r = mex('bindnode24', 'python -c "\nimport torch, numpy, pandas, scipy, sklearn, h5py\nimport torch_geometric\nimport torch_scatter, torch_sparse, torch_cluster, torch_spline_conv\nfrom torch_geometric.nn import SAGEConv\nprint(\'torch=\', torch.__version__)\nprint(\'torch CUDA build=\', torch.version.cuda)\nprint(\'cuda available=\', torch.cuda.is_available())\nprint(\'torch_geometric=\', torch_geometric.__version__)\nprint(\'numpy=\', numpy.__version__)\ndevice=\'cuda:0\' if torch.cuda.is_available() else \'cpu\'\nx=torch.randn(4,8,device=device)\nedge_index=torch.tensor([[0,1,2],[1,2,3]],dtype=torch.long,device=device)\nm=SAGEConv(8,4).to(device)\ny=m(x,edge_index)\nprint(\'SAGEConv smoke test=\', tuple(y.shape), y.device)\n" ', check=False, capture=True)
        print(S.r.stdout)
        if S.r.returncode != 0:
            raise RuntimeError('bindNode24 environment smoke test failed. See complete output immediately above.')
        print('bindNode24 environment: PASS')

# 10B. FIND PROJECT STRUCTURES FOR TestSetNew46
def stage_35():
    S.PDB_CANDIDATES = [S.BASE / 'pdb/testset300', S.BASE / 'pdb/testset46', S.BASE / 'esmFold_pdb_files', S.BASE / 'pdb/all']
    S.PROJECT_PDB = {name: {pid: find_project_pdb(pid) for pid in df.prot_id} for name, df in S.DATASETS.items()}
    for S.name, S.d in S.PROJECT_PDB.items():
        S.missing = [k for k, v in S.d.items() if v is None]
        print(S.name, 'project PDB found', len(S.d) - len(S.missing), '/', len(S.d), 'missing', S.missing[:5])

# 10C. PATCH OUTPUT PRECISION ONLY
def stage_36():
    if S.RUN_BINDNODE24 or S.RUN_OOF_THRESHOLDS:
        S.writer = S.BN24 / 'src/output/outputFileWriting.py'
        S.txt = S.writer.read_text()
        if ':.3f' in S.txt:
            S.writer.write_text(S.txt.replace(':.3f', ':.9f'))
            print('Patched text serialization 3 -> 9 decimals only.')

# 10E. RUN / RECOVER bindNode24 ON TestSet300
def stage_38():
    S.bind24_probs = {}
    if S.RUN_BINDNODE24:
        S.out300 = run_or_recover_bindnode24(dataset='TestSet300', df=S.test300_df, fasta=S.BN24 / 'dataset/all.fasta', id_list=S.BN24 / 'dataset/TestSet300_ids.txt', pdb_dir=S.BN24 / 'dataset/AlphaFold_predictions', dssp_dir=S.BN24 / 'dataset/DSSP_features')
        S.bind24_probs['TestSet300'] = load_bindnode24_npy(S.test300_df, S.out300)
        for S.label, S.d in S.bind24_probs['TestSet300'].items():
            save_probs(S.d, S.PRED / f'bindNode24__TestSet300__{S.label}.csv.gz')
        print('bindNode24 TestSet300: COMPLETE')

# 10F. PREPARE TestSetNew46 STRUCTURES + DUMMY DSSP
def stage_39():
    S.AA3_BN24 = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
    S.BN24_46_PDB = S.CACHE / 'bindNode24_TestSetNew46_normalized_pdb'
    S.BN24_46_DSSP = S.CACHE / 'bindNode24_TestSetNew46_dummy_dssp'
    S.BN24_46_PDB.mkdir(parents=True, exist_ok=True)
    S.BN24_46_DSSP.mkdir(parents=True, exist_ok=True)
    if S.RUN_BINDNODE24:
        for S.r in S.test46_df.itertuples():
            S.pid = str(S.r.prot_id)
            S.src = S.PROJECT_PDB.get('TestSetNew46', {}).get(S.pid)
            if S.src is None or not Path(S.src).exists():
                raise FileNotFoundError(f'TestSetNew46/{S.pid}: project PDB not found')
            S.dst = S.BN24_46_PDB / f'{S.pid}.pdb'
            _normalize_bn24_external_pdb(S.src, S.r.sequence, S.dst)
            S.dssp = S.BN24_46_DSSP / f'{S.pid}.csv'
            _write_dummy_dssp_csv(S.pid, len(S.r.sequence), S.dssp)
        assert len(list(S.BN24_46_PDB.glob('*.pdb'))) >= len(S.test46_df)
        assert len(list(S.BN24_46_DSSP.glob('*.csv'))) >= len(S.test46_df)
        S.out46 = run_or_recover_bindnode24(dataset='TestSetNew46', df=S.test46_df, fasta=S.CANON / 'TestSetNew46.fasta', id_list=S.CANON / 'TestSetNew46_ids.txt', pdb_dir=S.BN24_46_PDB, dssp_dir=S.BN24_46_DSSP)
        S.bind24_probs['TestSetNew46'] = load_bindnode24_npy(S.test46_df, S.out46)
        for S.label, S.d in S.bind24_probs['TestSetNew46'].items():
            save_probs(S.d, S.PRED / f'bindNode24__TestSetNew46__{S.label}.csv.gz')
        print('bindNode24 TestSetNew46: COMPLETE')

# 11A. PUBLIC DevSet1010 DATAFRAME FOR bindNode24
def stage_41():
    S.d
    S.bn24_dev_df = build_bn24_dev_df() if S.RUN_OOF_THRESHOLDS else None
    if S.bn24_dev_df is not None:
        display(S.bn24_dev_df.groupby('fold').size())

# 11B. Dev ProtT5 + SINGLE-FOLD bindNode24 OOF RUNS
def stage_42():
    S.bind24_thresholds = None
    if S.RUN_OOF_THRESHOLDS:
        S.BN24_DEV_H5 = generate_prott5_h5(S.bn24_dev_df, S.CACHE / 'bindNode24_DevSet1010_ProtT5.h5', S.REBUILD_PROTT5)
        S.oofroot = S.PRED / 'bindNode24' / 'DevOOF_fullprecision'
        S.oofroot.mkdir(parents=True, exist_ok=True)
        S.local_weight_root = Path(str(S.CACHE_ROOT / 'bindnode24_oof_single_weights'))
        S.local_weight_root.mkdir(parents=True, exist_ok=True)
        S.local_cache_root = Path(str(S.CACHE_ROOT / 'bindnode24_oof_cache'))
        S.local_cache_root.mkdir(parents=True, exist_ok=True)

        def run_bn24_oof_fold(k):
            fold_df = S.bn24_dev_df[S.bn24_dev_df['fold'] == k].copy().reset_index(drop=True)
            expected = {str(r.prot_id): len(r.sequence) for r in fold_df.itertuples()}
            fold_out = S.oofroot / f'fold{k}'
            npy_out = fold_out / 'fullprecision_npy'
            npy_out.mkdir(parents=True, exist_ok=True)
            complete = True
            for pid, L in expected.items():
                fp = npy_out / f'{pid}.npy'
                if not fp.exists():
                    complete = False
                    break
                try:
                    a = np.load(fp, mmap_mode='r')
                    if a.shape != (L, 3):
                        complete = False
                        break
                    if not np.all(np.isfinite(a)):
                        complete = False
                        break
                except Exception:
                    complete = False
                    break
            if complete:
                print(f'Fold {k}: {len(expected)} valid OOF predictions already exist -> SKIP inference')
                return npy_out
            wdir = S.local_weight_root / f'fold{k}'
            if wdir.exists():
                shutil.rmtree(wdir)
            wdir.mkdir(parents=True, exist_ok=True)
            src = S.BN24 / 'trained_model_dicts' / 'bindNode24' / f'trained_model_{k - 1}.state_dict'
            if not src.exists():
                raise FileNotFoundError(f'Missing released checkpoint: {src}')
            dst = wdir / src.name
            shutil.copy2(src, dst)
            copied = list(wdir.glob('*.state_dict'))
            if len(copied) != 1:
                raise RuntimeError(f'Fold {k}: expected exactly 1 checkpoint; found {len(copied)}')
            if copied[0].stat().st_size != src.stat().st_size:
                raise RuntimeError(f'Fold {k}: copied checkpoint size mismatch')
            print(f'Fold {k}: local checkpoint =', copied[0])
            result = fold_out / 'native_result'
            result.mkdir(parents=True, exist_ok=True)
            cfg = S.WORK / f'bindNode24_DevOOF_fold{k}.ini'
            cfg.write_text(bindnode_config(result_dir=result, embeddings=S.BN24_DEV_H5, id_list=S.BN24 / 'dataset' / 'cv_splits' / 'predicted_structures' / f'ids_split{k}.txt', fasta=S.BN24 / 'dataset' / 'all.fasta', pdb_dir=S.BN24 / 'dataset' / 'AlphaFold_predictions', weight_dir=wdir, dssp_dir=S.BN24 / 'dataset' / 'DSSP_features', cache_dir=S.local_cache_root / f'fold{k}'))
            expected_json = S.WORK / f'bindNode24_DevOOF_fold{k}_expected.json'
            expected_json.write_text(json.dumps({'ids': list(expected.keys()), 'lengths': expected}))
            script = S.BN24 / f'run_DevOOF_fold{k}_fullprecision.py'
            script.write_text(f'''import sys\nimport json\n\nfrom pathlib import Path\n\nimport numpy as np\n\nREPO = Path(r"{S.BN24}")\n\nsys.path.insert(\n    0,\n    str(REPO / 'src')\n)\n\nfrom setup.configProcessor import (\n    validate_config,\n    use_cache,\n    get_model_parameter_dict,\n    get_weight_dir\n)\n\nfrom misc.Cache import (\n    setup_chache\n)\n\nfrom machine_learning.predict import (\n    run_prediction\n)\n\nfrom data_processing.post_processing import (\n    get_averaged_predictions\n)\n\n\nOUT = Path(r"{npy_out}")\n\nEXPECTED = json.loads(\n    Path(\n        r"{expected_json}"\n    ).read_text()\n)\n\n\nprint(\n    "=== bindNode24 OOF "\n    "fold {k} preflight ==="\n)\n\nvalidate_config()\n\nweights = sorted(\n    Path(\n        get_weight_dir()\n    ).glob(\n        '*.state_dict'\n    )\n)\n\nprint(\n    "weight files:",\n    [p.name for p in weights]\n)\n\nif len(weights) != 1:\n\n    raise RuntimeError(\n        f"OOF fold {k}: "\n        f"expected exactly "\n        f"1 checkpoint, "\n        f"found {{len(weights)}}"\n    )\n\n\nif use_cache():\n    setup_chache()\n\n\nparams = (\n    get_model_parameter_dict(\n        only_first_value=True\n    )\n)\n\nprint(\n    "model parameters:",\n    params\n)\n\n\npredictions = run_prediction(\n    params\n)\n\navg = get_averaged_predictions(\n    predictions\n)\n\n\nexpected_ids = (\n    EXPECTED['ids']\n)\n\nlengths = (\n    EXPECTED['lengths']\n)\n\n\nmissing = sorted(\n    set(expected_ids)\n    -\n    set(avg)\n)\n\nextra = sorted(\n    set(avg)\n    -\n    set(expected_ids)\n)\n\n\nif missing:\n\n    raise RuntimeError(\n        f"OOF fold {k}: "\n        f"missing predictions "\n        f"for {{len(missing)}} "\n        f"proteins: "\n        f"{{missing[:10]}}"\n    )\n\n\nif extra:\n\n    raise RuntimeError(\n        f"OOF fold {k}: "\n        f"unexpected proteins: "\n        f"{{extra[:10]}}"\n    )\n\n\nOUT.mkdir(\n    parents=True,\n    exist_ok=True\n)\n\n\nfor pid in expected_ids:\n\n    arr = np.asarray(\n        avg[pid],\n        dtype=np.float64\n    )\n\n    L = int(\n        lengths[pid]\n    )\n\n    if arr.shape != (\n        L,\n        3\n    ):\n\n        raise RuntimeError(\n            f"{{pid}}: "\n            f"prediction shape "\n            f"{{arr.shape}}, "\n            f"expected "\n            f"{{(L,3)}}"\n        )\n\n    if not np.all(\n        np.isfinite(arr)\n    ):\n\n        raise RuntimeError(\n            f"{{pid}}: "\n            f"non-finite "\n            f"OOF probabilities"\n        )\n\n    np.save(\n        OUT /\n        f"{{pid}}.npy",\n\n        arr.astype(\n            np.float32\n        )\n    )\n\n\nprint(\n    "saved",\n    len(expected_ids),\n    "OOF proteins"\n)\n\nprint(\n    "=== bindNode24 OOF "\n    "fold {k} PASS ==="\n)\n''')
            r = mex('bindnode24', f'python "{script.name}" -f "{cfg}"', cwd=S.BN24, check=False, capture=True)
            print(r.stdout)
            print(f'fold {k} return code:', r.returncode)
            if r.returncode != 0:
                raise RuntimeError(f'bindNode24 OOF fold {k} failed. The complete inner traceback is printed above.')
            missing = []
            bad = []
            for pid, L in expected.items():
                fp = npy_out / f'{pid}.npy'
                if not fp.exists():
                    missing.append(pid)
                    continue
                a = np.load(fp, mmap_mode='r')
                if a.shape != (L, 3):
                    bad.append((pid, a.shape, L))
            if missing or bad:
                raise RuntimeError(f'Fold {k}: incomplete OOF output; missing={missing[:10]}, bad={bad[:10]}')
            print(f'Fold {k}: OOF output audit PASS ({len(expected)} proteins)')
            return npy_out
        S.BN24_OOF_DIRS = {}
        for S.k in range(1, 6):
            print('\n' + '=' * 72)
            print(f'BINDNODE24 OOF FOLD {S.k}/5')
            print('=' * 72)
            S.BN24_OOF_DIRS[S.k] = run_bn24_oof_fold(S.k)
        S.metal = {}
        S.nuc = {}
        S.small = {}
        for S.r in S.bn24_dev_df.itertuples():
            S.pid = str(S.r.prot_id)
            S.fold = int(S.r.fold)
            S.fp = S.BN24_OOF_DIRS[S.fold] / f'{S.pid}.npy'
            if not S.fp.exists():
                raise FileNotFoundError(f'Missing OOF probability for {S.pid}: {S.fp}')
            S.arr = np.load(S.fp)
            if S.arr.shape != (len(S.r.sequence), 3):
                raise RuntimeError(f'{S.pid}: OOF shape {S.arr.shape}; expected {(len(S.r.sequence), 3)}')
            S.metal[S.pid] = S.arr[:, 0].astype(np.float64)
            S.nuc[S.pid] = S.arr[:, 1].astype(np.float64)
            S.small[S.pid] = S.arr[:, 2].astype(np.float64)
        if len(S.metal) != len(S.bn24_dev_df):
            raise RuntimeError(f'Expected OOF predictions for {len(S.bn24_dev_df)} Dev proteins; got {len(S.metal)}')
        S.anyp = {pid: np.maximum.reduce([S.metal[pid], S.nuc[pid], S.small[pid]]) for pid in S.metal}
        S.oofprobs = {'metal_binding': S.metal, 'nuclear_binding': S.nuc, 'small_binding': S.small, 'any_ligand_binding_sites': S.anyp}
        S.y_by = {lab: np.concatenate([np.asarray(x, dtype=np.int8) for x in S.bn24_dev_df[lab]]) for lab in S.LABELS}
        S.p_by = {lab: np.concatenate([np.asarray(S.oofprobs[lab][str(pid)], dtype=np.float64) for pid in S.bn24_dev_df.prot_id]) for lab in S.LABELS}
        for S.lab in S.LABELS:
            if len(S.y_by[S.lab]) != len(S.p_by[S.lab]):
                raise RuntimeError(f'{S.lab}: OOF label/probability length mismatch {len(S.y_by[S.lab])} != {len(S.p_by[S.lab])}')
        S.bind24_thresholds, S.bn24_thr_df = thresholds_from_oof(S.y_by, S.p_by)
        display(S.bn24_thr_df)
        S.bn24_thr_df.to_csv(S.RESULTS / 'bindNode24_OOF_thresholds.csv', index=False)
        print('\nbindNode24 OOF THRESHOLD RECONSTRUCTION: COMPLETE')

# 12A. SCORE bindEmbed21DL + bindNode24
def stage_44():
    S.all_rows = []
    S.per_protein_frames = []
    if S.RUN_BIND21:
        add_method('bindEmbed21DL', S.bind21_probs, S.bind21_thresholds, {'TestSet300': 'A_direct_native', 'TestSetNew46': 'A_direct_native'})
    if S.RUN_BINDNODE24:
        add_method('bindNode24', S.bind24_probs, S.bind24_thresholds, {'TestSet300': 'A_direct_native', 'TestSetNew46': 'C_external_application_project_structures'})
    S.full_direct = pd.DataFrame(S.all_rows)
    S.full_direct.to_csv(S.RESULTS / 'FULL_DIRECT_BASELINE_METRICS_300_46.csv', index=False)
    if S.per_protein_frames:
        pd.concat(S.per_protein_frames, ignore_index=True).to_csv(S.RESULTS / 'FULL_DIRECT_PER_PROTEIN_300_46.csv', index=False)
    display(S.full_direct.sort_values(['dataset', 'label', 'method', 'operating_point']))

STAGES = [stage_34, stage_35, stage_36, stage_38, stage_39, stage_41, stage_42, stage_44]

