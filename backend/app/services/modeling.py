"""Baseline classification / regression pipelines with leak-free preprocessing and CV comparison."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from app import config as C
from app.services import evaluation as ev
from app.services import explainability as ex
from app.services.data_quality import normalize_placeholders
from app.services.preprocessing import build_preprocessor, expand_features, make_spec, select_features
from app.services.type_inference import coerce_types

CLS_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
REG_METRICS = ["rmse", "mae", "r2"]
LOWER_BETTER = {"rmse", "mae", "mse", "smape", "mape"}
LABELS = {"logreg": "Logistic Regression", "linear": "Linear Regression", "rf": "Random Forest", "xgb": "XGBoost", "dummy": "Baseline (Dummy)"}


def default_primary(problem_type: str, n_classes: int, imbalanced: bool) -> str:
    if problem_type == "regression":
        return "rmse"
    return "pr_auc" if (n_classes == 2 and imbalanced) else "f1"


def _cls_scorers(n_classes):
    avg = {} if n_classes == 2 else {"average": "macro"}
    s = {"accuracy": "accuracy",
         "precision": make_scorer(precision_score, zero_division=0, **avg),
         "recall": make_scorer(recall_score, zero_division=0, **avg),
         "f1": make_scorer(f1_score, zero_division=0, **avg),
         "roc_auc": "roc_auc" if n_classes == 2 else "roc_auc_ovr"}
    if n_classes == 2:
        s["pr_auc"] = "average_precision"
    return s


_REG_SCORERS = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}


def _estimators(pt, n_classes, imbalanced, pos_weight, model_params=None):
    rs, cw = C.RANDOM_STATE, ("balanced" if imbalanced else None)
    mp = model_params or {}
    logreg_p = mp.get("logreg", {})
    rf_p = mp.get("rf", {})
    xgb_p = mp.get("xgb", {})

    logreg_c = float(logreg_p.get("C", 1.0))

    rf_n_estimators = int(rf_p.get("n_estimators", 200))
    rf_max_depth = int(rf_p["max_depth"]) if rf_p.get("max_depth") is not None else 14
    rf_min_samples_leaf = int(rf_p.get("min_samples_leaf", 2))

    xgb_n_estimators = int(xgb_p.get("n_estimators", 250))
    xgb_max_depth = int(xgb_p.get("max_depth", 5))
    xgb_lr = float(xgb_p.get("learning_rate", 0.08))

    if pt == "classification":
        xgb_kw = {"scale_pos_weight": pos_weight} if (n_classes == 2 and imbalanced) else {}
        return [("dummy", DummyClassifier(strategy="prior"), False),
                ("logreg", LogisticRegression(C=logreg_c, max_iter=1000, class_weight=cw), True),
                ("rf", RandomForestClassifier(n_estimators=rf_n_estimators, min_samples_leaf=rf_min_samples_leaf, max_depth=rf_max_depth, class_weight=cw, n_jobs=-1, random_state=rs), False),
                ("xgb", XGBClassifier(n_estimators=xgb_n_estimators, learning_rate=xgb_lr, max_depth=xgb_max_depth, subsample=0.9, colsample_bytree=0.9,
                                      n_jobs=-1, random_state=rs, verbosity=0, **xgb_kw), False)]
    return [("dummy", DummyRegressor(strategy="mean"), False),
            ("linear", LinearRegression(), True),
            ("rf", RandomForestRegressor(n_estimators=rf_n_estimators, min_samples_leaf=rf_min_samples_leaf, max_depth=rf_max_depth, n_jobs=-1, random_state=rs), False),
            ("xgb", XGBRegressor(n_estimators=xgb_n_estimators, learning_rate=xgb_lr, max_depth=xgb_max_depth, subsample=0.9, colsample_bytree=0.9,
                                 n_jobs=-1, random_state=rs, verbosity=0), False)]





def _input_schema(spec, d):
    out = []
    for f in spec:
        e = {"name": f["name"], "kind": f["kind"]}
        s = d[f["name"]].dropna()
        if f["kind"] == "category":
            e["options"] = [str(v) for v in s.astype(str).value_counts().head(30).index]
        elif f["kind"] == "numeric" and len(s):
            e.update(min=float(s.min()), median=float(s.median()), max=float(s.max()))
        elif f["kind"] == "bool":
            e["options"] = ["true", "false"]
        elif f["kind"] == "datetime" and len(s):
            e["example"] = s.iloc[0].isoformat()
        out.append(e)
    return out


def train_tabular(df: pd.DataFrame, profile: dict, cfg: dict):
    target, pt = cfg["target"], cfg["problem_type"]
    meta = {m["name"]: m for m in profile["columns"]}
    if target not in meta:
        raise ValueError(f"Target '{target}' not found.")
    tm = meta[target]
    if pt == "regression" and tm["semantic_type"] != "numerical":
        raise ValueError("Regression requires a numerical target.")
    if tm["semantic_type"] in ("identifier", "text", "datetime", "empty"):
        raise ValueError(f"A {tm['semantic_type']} column cannot be used as a target.")

    feats, decisions = select_features(profile, target, cfg.get("keep_columns", []), cfg.get("drop_columns", []))
    if not feats:
        raise ValueError("No usable feature columns remain.")
    spec = make_spec(feats, df)
    ok = df[target].notna()
    if (~ok).any():
        decisions.append({"column": target, "action": "rows_removed", "count": int((~ok).sum()),
                          "reason": f"{int((~ok).sum())} rows removed because the target is missing."})
    d = df[ok]
    X = expand_features(d, spec)
    if len(X) < C.MIN_ROWS_MODEL:
        raise ValueError(f"Need at least {C.MIN_ROWS_MODEL} rows with a target; found {len(X)}.")

    classes = None
    if pt == "classification":
        yr = d[target]
        if tm["semantic_type"] == "boolean":
            lb = tm.get("bool_labels", {"true": "True", "false": "False"})
            yr = yr.astype(object).map({True: lb["true"], False: lb["false"]})
        elif tm["semantic_type"] == "numerical" and tm["is_integer"]:
            yr = yr.astype("int64").astype(str)
        else:
            yr = yr.astype(str)
        le = LabelEncoder()
        y = le.fit_transform(yr)
        classes = [str(c) for c in le.classes_]
        counts = np.bincount(y)
        if len(classes) < 2:
            raise ValueError("Target has only one class.")
        if counts.min() < 2:
            raise ValueError(f"Class '{classes[int(counts.argmin())]}' has fewer than 2 rows; stratified splitting is impossible.")
        stratify = y
    else:
        y, stratify = d[target].astype(float).to_numpy(), None
    n_classes = len(classes) if classes else 0

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg.get("test_size", 0.2), random_state=C.RANDOM_STATE, stratify=stratify)
    minority = float(np.bincount(y_tr).min() / len(y_tr)) if classes else None
    imbalanced = bool(classes and minority < C.IMBALANCE_MINORITY_SHARE)
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)) if n_classes == 2 else 1.0

    primary = cfg.get("primary_metric") or default_primary(pt, n_classes, imbalanced)
    allowed = (CLS_METRICS if n_classes == 2 else [m for m in CLS_METRICS if m != "pr_auc"]) if classes else REG_METRICS
    if primary not in allowed:
        raise ValueError(f"Primary metric '{primary}' is not available for this task. Choose one of {allowed}.")
    lower = primary in LOWER_BETTER
    scorers = _cls_scorers(n_classes) if classes else _REG_SCORERS
    k = cfg.get("cv_folds", 5 if len(X_tr) <= 50_000 else 3)
    if classes:
        k = max(2, min(k, int(np.bincount(y_tr).min())))
        cv = StratifiedKFold(k, shuffle=True, random_state=C.RANDOM_STATE)
    else:
        cv = KFold(max(2, k), shuffle=True, random_state=C.RANDOM_STATE)

    models, fitted, prep_info = [], {}, None
    for key, est, linear in _estimators(pt, n_classes, imbalanced, pos_weight, cfg.get("model_params")):
        if cfg.get("models") and key not in cfg["models"] and key != "dummy":
            continue
        row = {"key": key, "name": LABELS[key], "status": "ok", "is_baseline": key == "dummy"}
        try:
            prep, info = build_preprocessor(X_tr, linear)
            pipe = Pipeline([("prep", prep), ("model", est)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cvr = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scorers, error_score="raise")
                t0 = time.perf_counter()
                pipe.fit(X_tr, y_tr)
                row["train_time_s"] = round(time.perf_counter() - t0, 3)
            sign = -1 if not classes else 1
            row["cv"] = {m: {"mean": float(sign * cvr[f"test_{m}"].mean() if (m in LOWER_BETTER and not classes) else cvr[f"test_{m}"].mean()),
                             "std": float(cvr[f"test_{m}"].std())} for m in scorers}
            if key != "dummy" and info and prep_info is None:
                prep_info = info
            pred = pipe.predict(X_te)
            if classes:
                proba = pipe.predict_proba(X_te)
                row["test"] = ev.classification_metrics(y_te, pred, proba, n_classes)
                row["diagnostics"] = ev.classification_curves(y_te, pred, proba, classes)
            else:
                row["test"] = ev.regression_metrics(y_te, pred)
                row["diagnostics"] = ev.regression_diagnostics(y_te, pred)
            row["primary_cv"] = row["cv"][primary]["mean"]
            row["primary_cv_std"] = row["cv"][primary]["std"]
            row["primary_test"] = row["test"].get(primary)
            fitted[key] = pipe
        except Exception as e:  # keep going; report the failure honestly
            row.update(status="failed", error=f"{type(e).__name__}: {e}")
        models.append(row)

    base = next((m for m in models if m["is_baseline"] and m["status"] == "ok"), None)
    cands = [m for m in models if not m["is_baseline"] and m["status"] == "ok"]
    if not cands:
        raise ValueError("All candidate models failed: " + "; ".join(m.get("error", "") for m in models if m["status"] == "failed"))
    leader = (min if lower else max)(cands, key=lambda m: m["primary_cv"])
    for m in models:
        if base and m["status"] == "ok":
            delta = m["primary_cv"] - base["primary_cv"]
            m["delta_vs_baseline"] = float(-delta if lower else delta)
        m["leader"] = m is leader
    beats = bool(base and leader["delta_vs_baseline"] > 0)

    # explainability for the leading model only
    pipe = fitted[leader["key"]]
    xai = {"model": leader["name"]}
    try:
        xai["permutation_importance"] = ex.permutation_importances(pipe, X_te, y_te, scorers[primary])
    except Exception as e:
        xai["permutation_error"] = str(e)
    try:
        xai["linear"] = ex.linear_effects(pipe)
        if leader["key"] in ("rf", "xgb"):
            xai["shap"] = ex.shap_summary(pipe, X_te, pt)
    except Exception as e:
        xai["shap_error"] = f"{type(e).__name__}: {e}"

    results = {
        "mode": "tabular", "problem_type": pt, "target": target, "classes": classes, "primary_metric": primary,
        "lower_is_better": lower, "available_metrics": allowed, "metric_note": (
            "Primary metric defaults depend on the task (PR-AUC for imbalanced binary targets, macro/binary F1 otherwise, RMSE for regression). "
            "'Leader' means highest score on this one metric only; check the other columns before choosing."),
        "split": {"method": "random hold-out" + (" (stratified)" if classes else ""), "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
                  "cv_folds": k, "test_size": cfg.get("test_size", 0.2)},
        "imbalance": {"imbalanced": imbalanced, "minority_share": minority,
                      "handling": "class_weight='balanced' (linear/RF) and scale_pos_weight (binary XGBoost). No oversampling/SMOTE."} if classes else None,
        "decisions": decisions, "features_used": [f["name"] for f in feats], "model_input_columns": list(X.columns),
        "preprocessing": prep_info, "models": models, "leader": leader["key"], "beats_baseline": beats,
        "baseline_note": ("The leading model beats the baseline on the primary CV metric." if beats else
                          "The leading model does NOT clearly beat the naive baseline. Treat these results with caution."),
        "explainability": xai,
        "notes": ["Models are evaluated with cross-validation on the training split and once on an untouched hold-out test split.",
                  "Column groups (e.g., which columns are high-cardinality) are chosen from the training split; all learned statistics are fitted inside training folds only."],
    }
    artifact = {"kind": "tabular", "problem_type": pt, "target": target, "classes": classes, "spec": spec,
                "model_columns": list(X.columns), "pipelines": fitted, "leader": leader["key"], "primary_metric": primary,
                "baseline_row": ex.typical_row(X_tr), "input_schema": _input_schema(spec, d)}
    return results, artifact


# ---------------------------------------------------------------- prediction
def _prepare_input(art: dict, records: pd.DataFrame):
    names = [f["name"] for f in art["spec"]]
    missing_cols = [c for c in names if c not in records.columns]
    raw = pd.DataFrame({c: records[c] if c in records.columns else np.nan for c in names}, index=records.index).astype(object)
    raw = raw.map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == "") else str(v))
    norm, _ = normalize_placeholders(raw)
    typed = coerce_types(norm, [f["meta"] for f in art["spec"]])
    bad = [c for c in names if norm[c].notna().any() and typed[c].isna().sum() > norm[c].isna().sum()]
    return expand_features(typed, art["spec"]), missing_cols, bad


def predict_tabular(art: dict, records: pd.DataFrame, model_key: str | None = None, explain: bool = False) -> dict:
    key = model_key or art["leader"]
    if key not in art["pipelines"]:
        raise ValueError(f"Unknown model '{key}'. Available: {list(art['pipelines'])}")
    pipe = art["pipelines"][key]
    X, missing_cols, bad = _prepare_input(art, records)
    out = {"model": key, "missing_columns": missing_cols, "unparseable_columns": bad, "predictions": []}
    if art["problem_type"] == "classification":
        proba = pipe.predict_proba(X)
        idx = proba.argmax(axis=1)
        for i in range(len(X)):
            out["predictions"].append({"prediction": art["classes"][idx[i]], "confidence": float(proba[i, idx[i]]),
                                       "probabilities": {c: float(p) for c, p in zip(art["classes"], proba[i])}})
        cls_i = int(idx[0]) if len(X) else None
    else:
        pred = pipe.predict(X)
        out["predictions"] = [{"prediction": float(p)} for p in pred]
        cls_i = None
    if explain and len(X) == 1:
        out["explanation"] = ex.local_explanation(pipe, X, art["baseline_row"], art["problem_type"], cls_i)
    return out
