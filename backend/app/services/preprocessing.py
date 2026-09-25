"""Stateless feature preparation + leak-free (fit-on-train-only) sklearn preprocessors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from app import config as C


# ---------------------------------------------------------------- column selection (visible decisions)
def select_features(profile: dict, target: str | None, keep=(), drop=()):
    """Return (feature meta list, decisions). Every exclusion is recorded with a reason."""
    feats, decisions = [], []
    for m in profile["columns"]:
        c = m["name"]
        if c == target:
            continue
        reason = None
        if c in drop:
            reason = "Excluded by user."
        elif m["semantic_type"] == "empty":
            reason = "Column is entirely missing."
        elif m["constant"]:
            reason = "Constant column (single unique value)."
        elif m["semantic_type"] == "identifier" and c not in keep:
            reason = f"{m['identifier_level'].capitalize()} identifier ({'; '.join(m['evidence'])}). Keep it explicitly to override."
        elif m["semantic_type"] == "text":
            reason = "Free text is not modelled by the baseline pipeline."
        if reason:
            decisions.append({"column": c, "action": "excluded", "reason": reason})
        else:
            feats.append(m)
    return feats, decisions


def make_spec(feats: list[dict], df: pd.DataFrame) -> list[dict]:
    spec = []
    for m in feats:
        st = m["storage_type"]
        kind = {"numeric": "numeric", "bool": "bool", "datetime": "datetime"}.get(st, "category")
        e = {"name": m["name"], "kind": kind, "meta": {k: m[k] for k in ("name", "semantic_type", "storage_type", "datetime_format", "formatted_numeric")}}
        if kind == "datetime":
            x = df[m["name"]].dropna()
            e["has_time"] = bool((x != x.dt.normalize()).any()) if len(x) else False
        spec.append(e)
    return spec


def expand_features(df: pd.DataFrame, spec: list[dict]) -> pd.DataFrame:
    """Deterministic, stateless transformation of a typed frame into model-ready columns."""
    out = {}
    for f in spec:
        c, kind = f["name"], f["kind"]
        s = df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
        if kind == "numeric":
            out[c] = pd.to_numeric(s, errors="coerce").astype(float)
        elif kind == "bool":
            out[c] = pd.Series(s.astype("Float64").to_numpy(dtype="float64", na_value=np.nan), index=df.index)
        elif kind == "datetime":
            dt = pd.to_datetime(s, errors="coerce")
            parts = {"year": dt.dt.year, "month": dt.dt.month, "day": dt.dt.day, "weekday": dt.dt.weekday}
            if f.get("has_time"):
                parts["hour"] = dt.dt.hour
            for k, v in parts.items():
                out[f"{c}__{k}"] = v.astype(float)
            for k, period in (("month", 12), ("weekday", 7), ("hour", 24)):
                if k in parts:
                    ang = 2 * np.pi * parts[k].astype(float) / period
                    out[f"{c}__{k}_sin"], out[f"{c}__{k}_cos"] = np.sin(ang), np.cos(ang)
        else:
            out[c] = s.astype(object).where(s.notna(), np.nan)
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------- transformers
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by its training-set relative frequency (unseen -> 0)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.maps_ = [X[c].value_counts(normalize=True).to_dict() for c in X.columns]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        cols = [X.iloc[:, i].map(self.maps_[i]).fillna(0.0).astype(float).to_numpy() for i in range(X.shape[1])]
        return np.column_stack(cols) if cols else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in input_features])


def build_preprocessor(X_train: pd.DataFrame, linear: bool) -> tuple[ColumnTransformer, dict]:
    """Choose column groups from TRAIN data only. All learned statistics are fitted later, inside the CV/fit call."""
    num = [c for c in X_train if X_train[c].dtype.kind == "f"]
    cat = [c for c in X_train if c not in num]
    log_cols = []
    if linear:
        for c in num:
            v = X_train[c].dropna()
            if len(v) > 2 and v.min() >= 0 and v.std() > 0 and v.skew() > C.SKEW_STRONG:
                log_cols.append(c)
    plain = [c for c in num if c not in log_cols]
    low = [c for c in cat if X_train[c].nunique() <= C.OHE_MAX_CATEGORIES]
    high = [c for c in cat if c not in low]

    def numeric_steps(with_log):
        steps = []
        if with_log:
            steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
        steps.append(("impute", SimpleImputer(strategy="median", add_indicator=True)))
        if linear:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    parts = []
    if plain:
        parts.append(("num", numeric_steps(False), plain))
    if log_cols:
        parts.append(("numlog", numeric_steps(True), log_cols))
    if low:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                                       ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                                             sparse_output=False, drop="if_binary"))]), low))
    if high:
        steps = [("freq", FrequencyEncoder())]
        if linear:
            steps.append(("scale", StandardScaler()))
        parts.append(("freq", Pipeline(steps), high))
    if not parts:
        raise ValueError("No usable feature columns remain after exclusions.")
    ct = ColumnTransformer(parts, remainder="drop")
    info = {"numeric_imputation": "median (+ missing indicators for columns with gaps in training data)",
            "scaling": "StandardScaler" if linear else "none (tree-based model)",
            "log1p_columns": log_cols, "one_hot_columns": low,
            "frequency_encoded_columns": high, "numeric_columns": plain + log_cols,
            "fit_scope": "All imputers, scalers and encoders are fitted on training data only (inside each CV fold)."}
    return ct, info
