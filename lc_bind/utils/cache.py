"""Feature-store helpers shared by Nuclear and Small."""

from __future__ import annotations

import numpy as np


def get_record(store, pid):
    pid = str(pid)
    if pid in store:
        return store[pid]
    raise KeyError(pid)


def get_block(store, pid, block):
    rec = get_record(store, pid)
    if isinstance(rec, dict):
        if block in rec:
            return np.asarray(rec[block])
        if isinstance(rec.get("features"), dict) and block in rec["features"]:
            return np.asarray(rec["features"][block])
    if hasattr(rec, block):
        return np.asarray(getattr(rec, block))
    for attr in ("features", "feature_dict", "blocks"):
        bucket = getattr(rec, attr, None)
        if isinstance(bucket, dict) and block in bucket:
            return np.asarray(bucket[block])
    raise KeyError(f"{pid}/{block}")


def set_block(store, pid, block, arr):
    rec = get_record(store, pid)
    arr = np.asarray(arr, dtype=np.float32)
    if isinstance(rec, dict):
        if isinstance(rec.get("features"), dict):
            rec["features"][block] = arr
        else:
            rec[block] = arr
        return
    for attr in ("features", "feature_dict", "blocks"):
        bucket = getattr(rec, attr, None)
        if isinstance(bucket, dict):
            bucket[block] = arr
            return
    setattr(rec, block, arr)


def load_cached(path):
    path = str(path)
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as blob:
            return {key: blob[key] for key in blob.files}
    return np.load(path, allow_pickle=True)


def save_cached(path, array):
    path = str(path)
    if path.endswith(".npz"):
        np.savez_compressed(path, **array)
    else:
        np.save(path, array)
