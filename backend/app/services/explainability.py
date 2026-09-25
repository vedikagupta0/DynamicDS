"""Model explanations. Feature importance (how much a model relies on a feature) is not feature effect
(direction of influence), and neither implies causality."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from app import config as C


def to_original(name: str, originals: list[str]) -> str:
    n = name.split("__", 1)[1] if "__" in name and name.split("__", 1)[0] in ("num", "numlog", "cat", "freq") else name
    n = n.replace("missingindicator_", "")
    best = ""
    for o in originals:
        if (n == o or n.startswith(o + "_")) and len(o) > len(best):
            best = o
    return best or n


def permutation_importances(pipe, X: pd.DataFrame, y, scorer, n_repeats=5, max_rows=500):
    if len(X) > max_rows:
        idx = np.random.default_rng(C.RANDOM_STATE).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    r = permutation_importance(pipe, X, y, scoring=scorer, n_repeats=n_repeats, random_state=C.RANDOM_STATE, n_jobs=1)
    rows = [{"feature": c, "importance": float(m), "std": float(s)} for c, m, s in zip(X.columns, r.importances_mean, r.importances_std)]
    return sorted(rows, key=lambda d: -d["importance"])


def linear_effects(pipe, top=20):
    model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"):
        return None
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = np.asarray(model.coef_)
    note = None
    if coef.ndim == 2 and coef.shape[0] > 1:
        coef, note = np.abs(coef).mean(axis=0), "Multiclass: mean absolute coefficient across classes (direction omitted)."
    else:
        coef = coef.ravel()
    rows = sorted(({"feature": str(n), "coefficient": float(c)} for n, c in zip(names, coef)), key=lambda d: -abs(d["coefficient"]))[:top]
    return {"effects": rows, "note": note or "Coefficients are on standardised features; sign shows direction of association, not causation."}


def shap_summary(pipe, X_ref: pd.DataFrame, problem_type: str, n_rows=100, top=10):
    """Mean |SHAP| aggregated to original columns plus beeswarm points. Tree models only."""
    import shap
    model, prep = pipe.named_steps["model"], pipe.named_steps["prep"]
    if not hasattr(model, "feature_importances_"):
        return None
    Xs = X_ref.sample(min(n_rows, len(X_ref)), random_state=C.RANDOM_STATE)
    Z = np.asarray(prep.transform(Xs), dtype=float)
    names = list(prep.get_feature_names_out())
    sv = np.asarray(shap.TreeExplainer(model).shap_values(Z))
    if sv.ndim == 3:  # (rows, features, classes) or (classes, rows, features)
        sv = sv[:, :, -1] if sv.shape[0] == Z.shape[0] else sv[-1]
    orig = [to_original(n, list(X_ref.columns)) for n in names]
    per = pd.DataFrame(sv, columns=orig).T.groupby(level=0).sum().T
    order = per.abs().mean().sort_values(ascending=False)
    pts = []
    for f in order.index[:top]:
        v = Xs[f] if f in Xs.columns else None
        norm = np.full(len(Xs), 0.5)
        if v is not None and v.dtype.kind == "f" and v.nunique() > 1:
            r = v.rank(pct=True).fillna(0.5).to_numpy()
            norm = r
        pts.append({"feature": f, "points": [{"shap": float(s), "value": float(n)} for s, n in zip(per[f].to_numpy(), norm)]})
    return {"importance": [{"feature": f, "mean_abs_shap": float(v)} for f, v in order.head(20).items()], "beeswarm": pts,
            "note": "SHAP values describe how the model uses features, not causal effects. Computed on a sample of test rows."}


def typical_row(X_train: pd.DataFrame) -> dict:
    base = {}
    for c in X_train:
        s = X_train[c].dropna()
        base[c] = (float(s.median()) if X_train[c].dtype.kind == "f" else (s.mode().iloc[0] if len(s) else np.nan)) if len(s) else np.nan
    return base


def local_explanation(pipe, row: pd.DataFrame, baseline: dict, problem_type: str, class_index: int | None, top=8):
    """Occlusion: change in output when each feature is replaced by its typical training value."""
    cols = list(row.columns)
    variants = pd.concat([row] * (len(cols) + 1), ignore_index=True)
    for i, c in enumerate(cols):
        variants.loc[i + 1, c] = baseline[c]
    variants = variants.astype({c: row[c].dtype for c in cols}, errors="ignore")
    if problem_type == "classification":
        out = pipe.predict_proba(variants)[:, class_index]
    else:
        out = pipe.predict(variants)
    ref = out[0]
    rows = [{"feature": c, "value": None if pd.isna(row.iloc[0][c]) else str(row.iloc[0][c]), "contribution": float(ref - out[i + 1])}
            for i, c in enumerate(cols)]
    rows.sort(key=lambda d: -abs(d["contribution"]))
    return {"method": "occlusion", "top_features": rows[:top],
            "note": "Contribution = change in the prediction when this feature is replaced by a typical training value. Not causal."}
