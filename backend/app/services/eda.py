"""Automatic EDA: numeric/categorical/boolean/text/datetime summaries, correlations, target analysis."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats as sps

from app import config as C
from app.utils.timeutils import infer_frequency

_STOP = set("the and for with that this from are was were you your not have has had but all can will out our their its into than then them they there what when which who how one two".split())


# ------------------------------------------------------------------ helpers
def skew_label(skew):
    if skew is None or not np.isfinite(skew):
        return None
    a = abs(skew)
    if a < C.SKEW_SYMMETRIC:
        return "approximately symmetric"
    side = "right-skewed" if skew > 0 else "left-skewed"
    return ("moderately " if a < C.SKEW_STRONG else "strongly ") + side


def corr_strength(r):
    if r is None or not np.isfinite(r):
        return None
    a = abs(r)
    return "low" if a < C.CORR_LOW else ("moderate" if a < C.CORR_HIGH else "high")


def cramers_v(a: pd.Series, b: pd.Series):
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None
    chi2 = sps.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    if n < 2:
        return None
    r, k = ct.shape
    phi2c = max(0.0, chi2 / n - ((k - 1) * (r - 1)) / (n - 1))
    rc, kc = r - ((r - 1) ** 2) / (n - 1), k - ((k - 1) ** 2) / (n - 1)
    d = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / d)) if d > 0 else None


def correlation_ratio(cat: pd.Series, x: pd.Series):
    d = pd.DataFrame({"c": cat, "x": x}).dropna()
    if len(d) < 3 or d["c"].nunique() < 2:
        return None
    mu = d["x"].mean()
    g = d.groupby("c")["x"].agg(["count", "mean"])
    ssb = float((g["count"] * (g["mean"] - mu) ** 2).sum())
    sst = float(((d["x"] - mu) ** 2).sum())
    return float(np.sqrt(ssb / sst)) if sst > 0 else None


def _cap_categories(s: pd.Series, top=50):
    vc = s.value_counts()
    if len(vc) <= top:
        return s
    keep = set(vc.index[:top])
    return s.where(s.isin(keep), "__other__")


# ------------------------------------------------------------------ numeric
def numeric_stats(s: pd.Series, n_rows: int) -> dict:
    x = s.dropna().astype(float)
    cnt = len(x)
    d = {"column": s.name, "count": cnt, "missing": n_rows - cnt, "missing_pct": round((n_rows - cnt) / max(n_rows, 1) * 100, 3)}
    if cnt == 0:
        return d
    q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - C.OUTLIER_IQR_K * iqr, q3 + C.OUTLIER_IQR_K * iqr
    n_out = int(((x < lo) | (x > hi)).sum())
    std = float(x.std()) if cnt > 1 else 0.0
    skew = float(x.skew()) if cnt > 2 and std > 0 else None
    kurt = float(x.kurt()) if cnt > 3 and std > 0 else None
    d.update(mean=float(x.mean()), median=float(med), std=std, min=float(x.min()), q1=float(q1), q3=float(q3),
             max=float(x.max()), iqr=float(iqr), skewness=skew, kurtosis=kurt, skew_class=skew_label(skew),
             outliers={"count": n_out, "pct": round(n_out / cnt * 100, 3), "lower_bound": float(lo), "upper_bound": float(hi)})
    inside = x[(x >= lo) & (x <= hi)]
    d["boxplot"] = {"whisker_low": float(inside.min()) if len(inside) else float(x.min()), "q1": float(q1), "median": float(med),
                    "q3": float(q3), "whisker_high": float(inside.max()) if len(inside) else float(x.max())}
    bins = int(min(30, max(5, np.sqrt(cnt))))
    counts, edges = np.histogram(x, bins=bins)
    d["histogram"] = {"counts": counts.tolist(), "edges": edges.tolist()}
    sample = x if cnt <= C.VIS_SAMPLE_ROWS else x.sample(C.VIS_SAMPLE_ROWS, random_state=C.RANDOM_STATE)
    d["sampled_for_plots"] = cnt > C.VIS_SAMPLE_ROWS
    if std > 0 and cnt >= 30 and x.nunique() > 10:
        try:
            grid = np.linspace(x.min(), x.max(), 80)
            d["kde"] = {"x": grid.tolist(), "y": sps.gaussian_kde(sample)(grid).tolist()}
        except Exception:  # singular data
            pass
    if 8 <= cnt and x.nunique() >= 3 and std > 0:
        smp = x if cnt <= C.SHAPIRO_MAX else x.sample(C.SHAPIRO_MAX, random_state=C.RANDOM_STATE)
        p = float(sps.shapiro(smp).pvalue)
        d["normality"] = {"test": "Shapiro-Wilk", "n": int(len(smp)), "p_value": p, "sampled": cnt > C.SHAPIRO_MAX,
                          "conclusion": "Normality rejected at 5% level" if p < 0.05 else "Normality not rejected at 5% level"}
    return d


# ------------------------------------------------------------------ categorical / boolean / text / datetime
def categorical_stats(s: pd.Series, n_rows: int) -> dict:
    nn = s.dropna().astype(str)
    vc = nn.value_counts()
    cnt = len(nn)
    d = {"column": s.name, "count": cnt, "missing": n_rows - cnt, "missing_pct": round((n_rows - cnt) / max(n_rows, 1) * 100, 3),
         "unique": int(len(vc)), "cardinality_ratio": round(len(vc) / cnt, 4) if cnt else 0}
    if cnt == 0:
        return d
    cls = "low" if len(vc) <= C.CARD_LOW else ("medium" if len(vc) <= C.CARD_HIGH else "high")
    share = vc / cnt
    d.update(most_frequent=str(vc.index[0]), frequency=int(vc.iloc[0]), frequency_pct=round(float(share.iloc[0]) * 100, 3),
             cardinality_class=cls, rare_categories=int((share < C.RARE_CATEGORY_SHARE).sum()),
             top5_concentration_pct=round(float(share.iloc[:5].sum()) * 100, 3),
             top_values=[{"value": str(k), "count": int(v), "pct": round(float(v / cnt) * 100, 3)} for k, v in vc.head(20).items()],
             plottable=len(vc) <= 30, showing_top_only=len(vc) > 20)
    if cls == "high":
        d["warning"] = ("High-cardinality categorical feature. Consider target encoding, frequency encoding, hashing, "
                        "or domain-specific aggregation.")
    return d


def boolean_stats(s: pd.Series, n_rows: int) -> dict:
    v = s.dropna().astype(bool)
    t, f = int(v.sum()), int((~v).sum())
    tot = max(t + f, 1)
    return {"column": s.name, "true": t, "false": f, "missing": n_rows - (t + f),
            "true_pct": round(t / tot * 100, 3), "false_pct": round(f / tot * 100, 3)}


def text_stats(s: pd.Series, n_rows: int) -> dict:
    nn = s.dropna().astype(str)
    lens = nn.str.len()
    smp = nn if len(nn) <= 20000 else nn.sample(20000, random_state=C.RANDOM_STATE)
    toks = Counter(t for row in smp.str.lower() for t in re.findall(r"[a-z']{3,}", row) if t not in _STOP)
    return {"column": s.name, "avg_length": float(lens.mean()) if len(nn) else None, "median_length": float(lens.median()) if len(nn) else None,
            "empty_or_missing": n_rows - len(nn), "unique": int(nn.nunique()),
            "common_tokens": [{"token": k, "count": v} for k, v in toks.most_common(15)], "sampled": len(nn) > 20000}


def datetime_stats(s: pd.Series, n_rows: int) -> dict:
    x = s.dropna().sort_values()
    d = {"column": s.name, "missing": n_rows - len(x), "forecast_candidate": True}
    if x.empty:
        return d
    fi = infer_frequency(x) if len(x) >= 3 else None
    has_time = bool((x != x.dt.normalize()).any())
    d.update(min=x.min().isoformat(), max=x.max().isoformat(), duration_days=float((x.max() - x.min()) / pd.Timedelta(days=1)),
             frequency=fi, has_time_component=has_time,
             by_year=x.dt.year.value_counts().sort_index().to_dict(), by_month=x.dt.month.value_counts().sort_index().to_dict(),
             by_weekday=x.dt.weekday.value_counts().sort_index().to_dict())
    if has_time:
        d["by_hour"] = x.dt.hour.value_counts().sort_index().to_dict()
    return d


# ------------------------------------------------------------------ correlations
def correlation_analysis(df: pd.DataFrame, num_cols: list[str], meta_by_name: dict) -> dict:
    usable = [c for c in num_cols if df[c].nunique(dropna=True) > 1]
    truncated = len(usable) > C.CORR_MAX_COLS
    cols = usable[:C.CORR_MAX_COLS]
    out = {"columns": cols, "truncated": truncated, "pearson": None, "spearman": None, "high_pairs": [], "categorical_pairs": []}
    if len(cols) >= 2:
        sub = df[cols]
        pear = sub.corr()
        sp_sub = sub if len(sub) <= 200_000 else sub.sample(200_000, random_state=C.RANDOM_STATE)
        spear = sp_sub.corr(method="spearman")
        out["pearson"], out["spearman"] = pear.round(3).values.tolist(), spear.round(3).values.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r, rs = pear.iat[i, j], spear.iat[i, j]
                if np.isfinite(r) and abs(r) >= C.CORR_HIGH:
                    out["high_pairs"].append({"a": cols[i], "b": cols[j], "pearson": float(r),
                                              "spearman": float(rs) if np.isfinite(rs) else None, "strength": corr_strength(r),
                                              "note": "Potential multicollinearity."})
        out["high_pairs"].sort(key=lambda p: -abs(p["pearson"]))
    cats = [m["name"] for m in meta_by_name.values()
            if m["semantic_type"] == "categorical" and 2 <= m["n_unique"] <= 30][:15]
    if len(cats) >= 2:
        sub = df[cats].astype(object)
        sub = sub if len(sub) <= 50_000 else sub.sample(50_000, random_state=C.RANDOM_STATE)
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                v = cramers_v(sub[cats[i]], sub[cats[j]])
                if v is not None and v >= 0.5:
                    out["categorical_pairs"].append({"a": cats[i], "b": cats[j], "cramers_v": v})
    out["note"] = ("Pearson/Spearman are computed on numerical columns only. Categorical pairs use bias-corrected Cramér's V; "
                   "categorical columns are never label-encoded and correlated.")
    return out


# ------------------------------------------------------------------ target analysis
def recommend_problem_type(s: pd.Series, m: dict) -> dict:
    t = m["semantic_type"]
    if t in ("boolean", "categorical"):
        return {"problem_type": "classification", "reason": f"Target is {t} with {m['n_unique']} classes."}
    if t == "numerical":
        if m["n_unique"] == 2:
            return {"problem_type": "classification", "reason": "Numerical target with exactly two distinct values."}
        if m["is_integer"] and m["n_unique"] <= 10:
            return {"problem_type": "classification",
                    "reason": f"Integer target with only {m['n_unique']} distinct values. Confirm whether these are classes or counts."}
        return {"problem_type": "regression", "reason": f"Numerical target with {m['n_unique']} distinct values."}
    return {"problem_type": None, "reason": f"A {t} column cannot be used as a standard ML target."}


def analyze_target(df: pd.DataFrame, meta_by_name: dict, target: str, problem_type: str | None = None) -> dict:
    m = meta_by_name[target]
    s = df[target]
    rec = recommend_problem_type(s, m)
    pt = problem_type or rec["problem_type"]
    out = {"target": target, "semantic_type": m["semantic_type"], "recommendation": rec, "problem_type": pt,
           "missing": int(s.isna().sum()), "missing_pct": round(float(s.isna().mean()) * 100, 3), "associations": [], "leakage_suspects": []}
    if pt is None:
        return out
    ok = s.notna()
    d = df[ok]
    if len(d) > C.ASSOC_SAMPLE_ROWS:
        d = d.sample(C.ASSOC_SAMPLE_ROWS, random_state=C.RANDOM_STATE)
    y = d[target]
    if pt == "classification":
        vc = s.dropna().astype(str).value_counts()
        out["distribution"] = {"classes": [{"value": str(k), "count": int(v), "pct": round(v / vc.sum() * 100, 3)} for k, v in vc.head(30).items()],
                               "n_classes": int(len(vc)), "minority_pct": round(float(vc.min() / vc.sum()) * 100, 3),
                               "imbalance_ratio": round(float(vc.max() / vc.min()), 3)}
        out["imbalanced"] = bool(vc.min() / vc.sum() < C.IMBALANCE_MINORITY_SHARE)
        yc = y.astype(str)
    else:
        if m["semantic_type"] != "numerical":
            out["error"] = "Regression needs a numerical target."
            return out
        out["distribution"] = numeric_stats(s, len(df))
        yc = None
    feats = [k for k, v in meta_by_name.items() if k != target and v["semantic_type"] in ("numerical", "categorical", "boolean")]
    for f in feats:
        fm = meta_by_name[f]
        x = d[f]
        if fm["semantic_type"] == "numerical":
            if pt == "regression":
                r = x.corr(y.astype(float))
                rs = x.corr(y.astype(float), method="spearman")
                val, measure, extra = (abs(r) if np.isfinite(r) else None), "pearson", {"signed": None if not np.isfinite(r) else float(r),
                                                                                       "spearman": None if not np.isfinite(rs) else float(rs)}
            else:
                val, measure, extra = correlation_ratio(yc, x), "correlation_ratio", {}
        else:
            xc = _cap_categories(x.astype(object).where(x.notna(), np.nan).astype(str) if fm["semantic_type"] != "boolean" else x.astype(str))
            if pt == "regression":
                val, measure, extra = correlation_ratio(xc, y.astype(float)), "correlation_ratio", {}
            else:
                val, measure, extra = cramers_v(xc, yc), "cramers_v", {}
        if val is not None and np.isfinite(val):
            out["associations"].append({"feature": f, "strength": float(val), "measure": measure, "band": corr_strength(val), **extra})
        same = (df[f].astype(str) == df[target].astype(str))[df[f].notna() & df[target].notna()]
        if len(same) and same.mean() > 0.99:
            out["leakage_suspects"].append({"feature": f, "reason": "Feature values are identical to the target.", "strength": 1.0})
    out["associations"].sort(key=lambda a: -a["strength"])
    for a in out["associations"]:
        if a["strength"] >= C.LEAKAGE_ASSOC and not any(l["feature"] == a["feature"] for l in out["leakage_suspects"]):
            out["leakage_suspects"].append({"feature": a["feature"], "strength": a["strength"],
                                            "reason": f"Very strong association with the target ({a['measure']} = {a['strength']:.3f})."})
    out["association_note"] = ("Pearson for numerical-numerical; correlation ratio (eta) for numerical vs categorical; Cramér's V for "
                               "categorical-categorical. Associations are not causal and leakage detection is heuristic.")
    return out


# ------------------------------------------------------------------ orchestrator
def build_eda(df: pd.DataFrame, profile: dict, target: str | None = None, problem_type: str | None = None) -> dict:
    meta = {m["name"]: m for m in profile["columns"]}
    n = len(df)
    by = lambda t: [c for c, m in meta.items() if m["semantic_type"] == t]
    num_cols = [c for c in by("numerical")]
    eda = {
        "numeric": [numeric_stats(df[c], n) for c in num_cols],
        "categorical": [categorical_stats(df[c], n) for c in by("categorical")],
        "boolean": [boolean_stats(df[c], n) for c in by("boolean")],
        "text": [text_stats(df[c], n) for c in by("text")],
        "datetime": [datetime_stats(df[c], n) for c in by("datetime")],
        "correlation": correlation_analysis(df, num_cols, meta),
        "sampling": {"plots_sampled_above_rows": C.VIS_SAMPLE_ROWS, "rows": n,
                     "note": "Statistics use the full dataset. KDE curves and normality tests use a random sample when the column has more rows than the limit."},
        "target": analyze_target(df, meta, target, problem_type) if target and target in meta else None,
        "thresholds": {"skew_symmetric": C.SKEW_SYMMETRIC, "skew_strong": C.SKEW_STRONG, "corr_low": C.CORR_LOW,
                       "corr_high": C.CORR_HIGH, "outlier_iqr_k": C.OUTLIER_IQR_K},
    }
    return eda
