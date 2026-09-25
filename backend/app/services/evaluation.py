"""Metrics, curves and split helpers shared by all pipelines."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as M

from app import config as C


def downsample(a, n=500, seed=C.RANDOM_STATE):
    a = np.asarray(a)
    if len(a) <= n:
        return np.arange(len(a))
    return np.sort(np.random.default_rng(seed).choice(len(a), n, replace=False))


# ---------------------------------------------------------------- classification
def classification_metrics(y_true, y_pred, y_proba, n_classes: int) -> dict:
    avg = "binary" if n_classes == 2 else "macro"
    out = {
        "accuracy": float(M.accuracy_score(y_true, y_pred)),
        "precision": float(M.precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(M.recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(M.f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "roc_auc": None, "pr_auc": None,
    }
    try:
        if n_classes == 2:
            out["roc_auc"] = float(M.roc_auc_score(y_true, y_proba[:, 1]))
            out["pr_auc"] = float(M.average_precision_score(y_true, y_proba[:, 1]))
        else:
            out["roc_auc"] = float(M.roc_auc_score(y_true, y_proba, multi_class="ovr", labels=list(range(n_classes))))
    except ValueError:
        pass
    return out


def classification_curves(y_true, y_pred, y_proba, class_names: list[str]) -> dict:
    n = len(class_names)
    cm = M.confusion_matrix(y_true, y_pred, labels=list(range(n)))
    rep = M.classification_report(y_true, y_pred, labels=list(range(n)), target_names=class_names, output_dict=True, zero_division=0)
    out = {"labels": class_names, "confusion_matrix": cm.tolist(), "report": rep}
    if n == 2 and len(set(np.asarray(y_true).tolist())) == 2:
        fpr, tpr, _ = M.roc_curve(y_true, y_proba[:, 1])
        pr, rc, _ = M.precision_recall_curve(y_true, y_proba[:, 1])
        i, j = downsample(fpr, 120), downsample(pr, 120)
        out["roc"] = {"fpr": fpr[i].tolist(), "tpr": tpr[i].tolist()}
        out["pr"] = {"precision": pr[j].tolist(), "recall": rc[j].tolist(), "baseline": float(np.mean(y_true))}
    return out


# ---------------------------------------------------------------- regression
def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0 or np.min(np.abs(y)) < 1e-8:
        return None
    return float(np.mean(np.abs((y - p) / y)) * 100)


def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = np.abs(y) + np.abs(p)
    mask = d > 1e-12
    if not mask.any():
        return None
    return float(np.mean(2 * np.abs(y - p)[mask] / d[mask]) * 100)


def regression_metrics(y_true, y_pred) -> dict:
    mse = float(M.mean_squared_error(y_true, y_pred))
    return {"mae": float(M.mean_absolute_error(y_true, y_pred)), "mse": mse, "rmse": float(np.sqrt(mse)),
            "r2": float(M.r2_score(y_true, y_pred)) if len(y_true) > 1 else None, "mape": mape(y_true, y_pred)}


def regression_diagnostics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    idx = downsample(y_true)
    res = y_true - y_pred
    counts, edges = np.histogram(res, bins=25)
    return {"actual": y_true[idx].tolist(), "predicted": y_pred[idx].tolist(), "residuals": res[idx].tolist(),
            "error_hist": {"counts": counts.tolist(), "edges": edges.tolist()}, "sampled": len(y_true) > len(idx)}


# ---------------------------------------------------------------- forecasting
def forecast_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"mae": float(np.mean(np.abs(y_true - y_pred))), "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "mape": mape(y_true, y_pred), "smape": smape(y_true, y_pred)}


def chronological_split(n: int, train_frac=C.FORECAST_TRAIN_FRAC, val_frac=C.FORECAST_VAL_FRAC):
    """Index boundaries (train_end, val_end). Never shuffles: train < val < test in time."""
    a = int(n * train_frac)
    b = int(n * (train_frac + val_frac))
    if not (0 < a < b < n):
        raise ValueError("Series too short to split chronologically.")
    return a, b
