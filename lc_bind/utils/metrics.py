"""Residue-level metrics shared by the Nuclear and Small pipelines."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    return {
        "AUPRC": float(average_precision_score(y, p)),
        "AUROC": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred)),
    }


def best_mcc_threshold(y, p):
    best = (-999, None)
    y = np.asarray(y)
    p = np.asarray(p)
    for t in np.linspace(0.01, 0.99, 199):
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > best[0]:
            best = (m, float(t))
    return best[1], best[0]
