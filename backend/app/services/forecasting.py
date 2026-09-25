"""Time-series forecasting with strictly chronological validation and mandatory baselines."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf
from xgboost import XGBRegressor

from app import config as C
from app.services import evaluation as ev
from app.utils.timeutils import infer_frequency, regularize_index

_SEASON_CANDIDATES = {"D": [7], "h": [24], "MS": [12], "QS": [4], "W": [52], "B": [5], "min": [60]}
LABELS = {"naive": "Naive (last value)", "snaive": "Seasonal naive", "ets": "Exponential smoothing (Holt-Winters)", "lag_xgb": "Lag-feature XGBoost"}
BASELINES = {"naive", "snaive"}
METRICS = ["mae", "rmse", "smape", "mape"]


# ---------------------------------------------------------------- series preparation
def prepare_series(df: pd.DataFrame, dt_col: str, target: str):
    if dt_col == target:
        raise ValueError("Datetime column and target column must be different.")
    ts, y = pd.to_datetime(df[dt_col], errors="coerce"), pd.to_numeric(df[target], errors="coerce")
    decisions = []
    bad = ts.isna()
    if bad.any():
        decisions.append({"column": dt_col, "action": "rows_removed", "count": int(bad.sum()), "reason": f"{int(bad.sum())} rows removed: missing timestamp."})
    ts, y = ts[~bad], y[~bad]
    fi = infer_frequency(ts)
    if fi is None:
        raise ValueError("Could not infer a regular frequency from the datetime column (need at least 3 distinct timestamps).")
    freq = fi["freq"]
    snapped = regularize_index(ts, freq)
    s = pd.Series(y.to_numpy(), index=pd.DatetimeIndex(snapped)).sort_index()
    dup = int(s.index.duplicated().sum())
    if dup:
        s = s.groupby(level=0).mean()
        decisions.append({"column": dt_col, "action": "aggregated", "count": dup, "reason": f"{dup} rows shared a timestamp; their target values were averaged."})
    full = pd.date_range(s.index.min(), s.index.max(), freq=freq)
    if len(full) > 200_000:
        raise ValueError("Series is too long after regularisation. Aggregate the data to a coarser frequency first.")
    s = s.reindex(full)
    n_nan = int(s.isna().sum())
    if n_nan:
        s = s.ffill().dropna()
        decisions.append({"column": target, "action": "filled", "count": n_nan,
                          "reason": f"{n_nan} missing periods/values were forward-filled (uses past values only, no leakage from the future)."})
    if len(s) < C.MIN_ROWS_FORECAST:
        raise ValueError(f"Need at least {C.MIN_ROWS_FORECAST} regular periods; found {len(s)}.")
    s.index.freq = freq
    return s, decisions, fi


def detect_season(y: np.ndarray, freq: str):
    key = "W" if freq.startswith("W") else freq
    best, best_acf = None, 0.3
    for m in _SEASON_CANDIDATES.get(key, []):
        if len(y) < 2 * m + 4:
            continue
        x = y - np.polyval(np.polyfit(np.arange(len(y)), y, 1), np.arange(len(y)))
        if x.std() == 0:
            continue
        a = acf(x, nlags=m, fft=True)[m]
        if a >= best_acf:
            best, best_acf = m, float(a)
    return best, best_acf if best else None


# ---------------------------------------------------------------- models
def _calendar(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.column_stack([idx.month, idx.dayofweek, idx.hour, idx.dayofyear]).astype(float)


class LagModel:
    """Tree model on lagged values, shifted rolling means and calendar features. Forecasts recursively."""

    def __init__(self, m):
        self.lags = sorted({1, 2, 3} | ({m, 2 * m} if m else {7}))
        self.windows = sorted({3} | ({m} if m else {7}))
        self.maxh = max(self.lags + self.windows)
        self.model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9, n_jobs=-1, random_state=C.RANDOM_STATE, verbosity=0)

    @property
    def feature_names(self):
        return [f"lag_{l}" for l in self.lags] + [f"roll_mean_{w}" for w in self.windows] + ["month", "weekday", "hour", "dayofyear"]

    def fit(self, y: np.ndarray, idx: pd.DatetimeIndex):
        s = pd.Series(y)
        cols = [s.shift(l) for l in self.lags] + [s.shift(1).rolling(w).mean() for w in self.windows]
        X = np.column_stack([c.to_numpy() for c in cols] + [_calendar(idx)])
        keep = ~np.isnan(X[:, :len(cols)]).any(axis=1)
        if keep.sum() < 20:
            raise ValueError("Not enough history for lag features.")
        self.model.fit(X[keep], y[keep])
        self.y_ = y
        return self

    def forecast(self, future_idx: pd.DatetimeIndex):
        hist = list(self.y_)
        cal = _calendar(future_idx)
        out = []
        for i in range(len(future_idx)):
            h = np.asarray(hist)
            row = [h[-l] for l in self.lags] + [h[-w:].mean() for w in self.windows] + list(cal[i])
            p = float(self.model.predict(np.asarray([row]))[0])
            out.append(p)
            hist.append(p)
        return np.asarray(out)


def run_model(key: str, y: np.ndarray, idx: pd.DatetimeIndex, future_idx: pd.DatetimeIndex, m):
    h = len(future_idx)
    if key == "naive":
        return np.repeat(y[-1], h), None
    if key == "snaive":
        return np.array([y[-m + (i % m)] for i in range(h)]), None
    if key == "ets":
        seasonal = m if (m and len(y) >= 2 * m + 2) else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add" if seasonal else None,
                                       seasonal_periods=seasonal, initialization_method="estimated").fit()
            return np.asarray(fit.forecast(h)), None
    if key == "lag_xgb":
        lm = LagModel(m).fit(y, idx)
        return lm.forecast(future_idx), lm
    raise ValueError(key)


# ---------------------------------------------------------------- training
def train_forecast(df: pd.DataFrame, profile: dict, cfg: dict):
    meta = {m["name"]: m for m in profile["columns"]}
    dt_col, target = cfg["datetime_column"], cfg["target"]
    if dt_col == target:
        raise ValueError("Datetime column and target column must be different.")
    if dt_col not in meta or meta[dt_col]["semantic_type"] != "datetime":
        raise ValueError(f"'{dt_col}' is not a datetime column.")
    if target not in meta or meta[target]["semantic_type"] != "numerical":
        raise ValueError("The forecasting target must be a numerical column.")
    series, decisions, fi = prepare_series(df, dt_col, target)
    y, idx, n = series.to_numpy(float), series.index, len(series)
    a, b = ev.chronological_split(n)
    primary = cfg.get("primary_metric") or "mae"
    if primary not in METRICS:
        raise ValueError(f"Primary metric must be one of {METRICS}.")
    m, m_acf = detect_season(y[:a], fi["freq"]) if not cfg.get("season_period") else (int(cfg["season_period"]), None)
    keys = ["naive"] + (["snaive"] if m else []) + ["ets", "lag_xgb"]
    rows, test_preds, lag_obj = [], {}, None
    for key in keys:
        r = {"key": key, "name": LABELS[key], "is_baseline": key in BASELINES, "status": "ok"}
        try:
            t0 = time.perf_counter()
            pv, _ = run_model(key, y[:a], idx[:a], idx[a:b], m)
            r["train_time_s"] = round(time.perf_counter() - t0, 3)
            r["validation"] = ev.forecast_metrics(y[a:b], pv)
            pt, obj = run_model(key, y[:b], idx[:b], idx[b:], m)
            r["test"] = ev.forecast_metrics(y[b:], pt)
            test_preds[key] = pt
            if key == "lag_xgb":
                lag_obj = obj
            r["primary_val"], r["primary_test"] = r["validation"][primary], r["test"][primary]
        except Exception as e:
            r.update(status="failed", error=f"{type(e).__name__}: {e}")
        rows.append(r)
    ok = [r for r in rows if r["status"] == "ok" and r["primary_val"] is not None]
    cands = [r for r in ok if not r["is_baseline"]]
    if not cands:
        raise ValueError("No forecasting model could be fitted: " + "; ".join(r.get("error", "") for r in rows if r["status"] == "failed"))
    leader = min(cands, key=lambda r: r["primary_val"])
    base_best = min((r for r in ok if r["is_baseline"]), key=lambda r: r["primary_test"])
    for r in rows:
        r["leader"] = r is leader
        if r["status"] == "ok" and r["primary_test"] is not None and base_best["primary_test"]:
            r["improvement_vs_best_baseline_pct"] = float((base_best["primary_test"] - r["primary_test"]) / base_best["primary_test"] * 100)
    beats = leader["primary_test"] < base_best["primary_test"]
    horizon = int(cfg.get("horizon", 12))
    fut_idx = pd.date_range(idx[-1], periods=horizon + 1, freq=fi["freq"])[1:]
    fut, _ = run_model(leader["key"], y, idx, fut_idx, m)

    stride = max(1, n // 1500)
    tstride = max(1, (n - b) // 500)
    tt = idx[b:][::tstride]
    resid = (y[b:] - test_preds[leader["key"]])[::tstride]
    xai = None
    if lag_obj is not None:
        imp = lag_obj.model.feature_importances_
        xai = {"model": LABELS["lag_xgb"], "feature_importance": sorted(({"feature": f, "importance": float(v)} for f, v in zip(lag_obj.feature_names, imp)), key=lambda d: -d["importance"]),
               "note": "Importance of lag/calendar features inside the tree model. Not a causal statement."}
    results = {
        "mode": "forecast", "datetime_column": dt_col, "target": target, "frequency": fi, "season_period": m, "season_acf": m_acf,
        "primary_metric": primary, "lower_is_better": True, "available_metrics": METRICS,
        "split": {"method": "chronological hold-out (no shuffling)", "n": n,
                  "train": {"start": idx[0].isoformat(), "end": idx[a - 1].isoformat(), "n": a},
                  "validation": {"start": idx[a].isoformat(), "end": idx[b - 1].isoformat(), "n": b - a},
                  "test": {"start": idx[b].isoformat(), "end": idx[-1].isoformat(), "n": n - b},
                  "explanation": "Time-aware validation is used to prevent future observations from leaking into model training. "
                                 "Models are compared on the validation window, then refit on train+validation and scored once on the test window."},
        "decisions": decisions, "models": rows, "leader": leader["key"], "beats_baseline": bool(beats),
        "baseline_note": ("The leading model beats the best naive baseline on the test window." if beats else
                          "The leading model does NOT beat the best naive baseline on the test window. Do not trust it over the baseline."),
        "charts": {"history": {"t": [t.isoformat() for t in idx[::stride]], "y": y[::stride].tolist(), "stride": stride},
                   "test": {"t": [t.isoformat() for t in tt], "actual": y[b:][::tstride].tolist(),
                            "pred": {k: v[::tstride].tolist() for k, v in test_preds.items()}},
                   "residuals": resid.tolist(), "future": {"t": [t.isoformat() for t in fut_idx], "yhat": fut.tolist(), "model": leader["name"]},
                   "split_points": {"train_end": idx[a - 1].isoformat(), "val_end": idx[b - 1].isoformat()}},
        "explainability": xai,
        "notes": ["Multi-step forecasts are scored over the whole window from the end of the training data (no rolling one-step look-ahead).",
                  "Tree models cannot extrapolate trends beyond the range seen in training.",
                  "MAPE/sMAPE are shown only where mathematically defined; no prediction intervals are provided in V1."]
                 + (["Test window is very short; treat the ranking as indicative only."] if n - b < 12 else []),
    }
    artifact = {"kind": "forecast", "values": y.tolist(), "index": [t.isoformat() for t in idx], "freq": fi["freq"], "season_period": m,
                "leader": leader["key"], "models": [r["key"] for r in rows if r["status"] == "ok"], "target": target}
    return results, artifact


def forecast_future(art: dict, periods: int, model_key: str | None = None) -> dict:
    key = model_key or art["leader"]
    if key not in art["models"]:
        raise ValueError(f"Unknown model '{key}'. Available: {art['models']}")
    if not 1 <= periods <= 1000:
        raise ValueError("periods must be between 1 and 1000.")
    idx = pd.DatetimeIndex(pd.to_datetime(art["index"]), freq=art["freq"])
    y = np.asarray(art["values"], float)
    fut = pd.date_range(idx[-1], periods=periods + 1, freq=art["freq"])[1:]
    yhat, _ = run_model(key, y, idx, fut, art["season_period"])
    return {"model": key, "t": [t.isoformat() for t in fut], "yhat": yhat.tolist(),
            "note": "Point forecast from a model refit on all available history. No prediction interval."}
