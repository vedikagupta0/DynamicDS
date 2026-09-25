"""Data-quality engine: placeholders, missingness, duplicates, constants, formatting, identifiers.

Every function only *detects* and *recommends*. Nothing here mutates the user's dataset,
except `normalize_placeholders`, whose changes are reported in full.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app import config as C


# ---------------------------------------------------------------- placeholders
def normalize_placeholders(df: pd.DataFrame):
    """Replace placeholder tokens / blank strings by NaN. Returns (frame, report)."""
    out = df.copy()
    report = []
    for c in df.columns:
        nn = out[c].dropna()
        if nn.empty:
            continue
        nn = nn.astype(str)
        key = nn.str.strip().str.lower()
        mask = key.isin(C.PLACEHOLDER_TOKENS) | (key == "")
        if not mask.any():
            continue
        shown = nn[mask].str.strip().replace("", "<blank>")
        report.append({
            "column": c, "count": int(mask.sum()),
            "values": {str(k): int(v) for k, v in shown.value_counts().head(10).items()},
            "action": "Converted to missing (NaN)",
            "note": "Values such as 'none' or 'unknown' can be genuine categories in some domains; review before relying on this.",
        })
        out.loc[mask[mask].index, c] = np.nan
    return out, report


# ---------------------------------------------------------------- missingness
def missing_level(pct: float) -> str:
    if pct <= 0:
        return "none"
    if pct < C.MISSING_LOW:
        return "low"
    if pct < C.MISSING_MODERATE:
        return "moderate"
    if pct < C.MISSING_HIGH:
        return "high"
    return "very_high"


_MISSING_TEXT = {
    "none": ("No missing values", None),
    "low": ("Low missingness", "Most values are present. Imputation may be preferable to dropping rows."),
    "moderate": ("Moderate missingness", "Impute with an appropriate strategy and consider adding a missing-indicator."),
    "high": ("High missingness", "Investigate why values are missing; imputation quality may be limited."),
    "very_high": ("Very high missingness",
                  "Column contains a large proportion of missing values. Consider dropping it unless domain knowledge indicates otherwise."),
}


def analyze_missing(df: pd.DataFrame) -> dict:
    n, k = len(df), df.shape[1]
    cols = []
    for c in df.columns:
        m = int(df[c].isna().sum())
        pct = m / n * 100 if n else 0.0
        lvl = missing_level(pct)
        label, rec = _MISSING_TEXT[lvl]
        cols.append({"column": c, "missing": m, "non_null": n - m, "missing_pct": round(pct, 3),
                     "level": lvl, "label": label, "recommendation": rec})
    row_counts = df.isna().sum(axis=1)
    flagged = row_counts[row_counts / max(k, 1) >= C.ROW_MISSING_FLAG].sort_values(ascending=False)
    rows = [{"row": int(i) + 1, "missing": int(v), "total": k,
             "recommendation": "Consider dropping this row."} for i, v in flagged.head(20).items()]
    return {
        "columns": cols, "total_missing_cells": int(df.isna().sum().sum()),
        "missing_cell_pct": round(float(df.isna().sum().sum()) / max(n * k, 1) * 100, 3),
        "rows_with_any_missing": int((row_counts > 0).sum()),
        "flagged_row_count": int(len(flagged)), "flagged_row_threshold": C.ROW_MISSING_FLAG, "flagged_rows": rows,
    }


# ---------------------------------------------------------------- duplicates
def analyze_duplicates(df: pd.DataFrame, meta: list[dict]) -> dict:
    strong = [m["name"] for m in meta if m["identifier_level"] == "strong"]
    exact = int(df.duplicated().sum())
    rest = [c for c in df.columns if c not in strong]
    excl = int(df[rest].duplicated().sum()) if strong and rest else (exact if not strong else None)
    if strong and excl is not None and excl > exact:
        interp = "The apparent uniqueness may be caused by an identifier column."
    elif exact > 0:
        interp = "Exact duplicate rows exist. Confirm they are not legitimate repeated records before removing them."
    else:
        interp = "No duplicate rows detected."
    return {
        "exact_duplicates": exact, "excluding_strong_identifiers": excl, "strong_identifiers_removed": strong,
        "example_rows": [int(i) + 1 for i in df.index[df.duplicated(keep="first")][:10]],
        "interpretation": interp, "action": "None. Rows are not removed automatically.",
    }


# ---------------------------------------------------------------- constants
def analyze_constants(df: pd.DataFrame) -> list[dict]:
    out = []
    for c in df.columns:
        nn = df[c].dropna()
        if nn.empty:
            out.append({"column": c, "kind": "all_missing", "unique": 0, "action": "Drop",
                        "message": "Column has no values."})
            continue
        vc = nn.value_counts()
        if len(vc) == 1:
            out.append({"column": c, "kind": "constant", "unique": 1, "top_value": str(vc.index[0]), "action": "Drop",
                        "message": "Column contains a single unique value and carries no information."})
        else:
            share = float(vc.iloc[0] / len(nn))
            if share >= C.NEAR_CONSTANT_SHARE:
                out.append({"column": c, "kind": "near_constant", "unique": int(len(vc)), "top_value": str(vc.index[0]),
                            "top_share": round(share, 4), "action": "Review",
                            "message": "Near-constant column. Consider removing because it contains very little information."})
    return out


# ---------------------------------------------------------------- formatting
_EMAIL = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
_URL = r"^(https?://|www\.)\S+$"
_PHONE = r"^\+?[\d\s\-().]{7,20}$"


def analyze_formatting(df_raw: pd.DataFrame, meta: list[dict]) -> list[dict]:
    """Inspect raw string values for formatted numbers, emails, URLs, phones, punctuation-rich text."""
    out = []
    for m in meta:
        c, t = m["name"], m["semantic_type"]
        nn = df_raw[c].dropna().astype(str)
        if nn.empty or m["storage_type"] in ("datetime", "bool"):
            continue
        s = nn if len(nn) <= C.VIS_SAMPLE_ROWS else nn.sample(C.VIS_SAMPLE_ROWS, random_state=0)
        ex = [str(v) for v in s.iloc[:3]]
        if m["formatted_numeric"]:
            out.append({"column": c, "kind": "formatted_numeric", "share": 1.0, "examples": ex,
                        "message": "Column contains formatted numerical values. Consider normalization."})
            continue
        if m["storage_type"] == "numeric":
            continue
        for kind, pat, msg in (("email", _EMAIL, "Column appears to contain email addresses."),
                               ("url", _URL, "Column appears to contain URLs."),
                               ("phone", _PHONE, "Column appears to contain phone-like values.")):
            share = float(s.str.match(pat).mean())
            if share >= 0.5 and not (kind == "phone" and not s.str.contains(r"[\s\-()+]").any()):
                out.append({"column": c, "kind": kind + "_like", "share": round(share, 3), "examples": ex, "message": msg})
                break
        else:
            share = float(s.str.contains(r"[^\w\s]").mean())
            if share >= 0.3:
                if t == "text":
                    out.append({"column": c, "kind": "punctuation_rich_text", "share": round(share, 3), "examples": ex,
                                "message": "Column contains punctuation-rich text and may require text preprocessing."})
                else:
                    out.append({"column": c, "kind": "categorical_with_punctuation", "share": round(share, 3), "examples": ex,
                                "message": "Categorical strings contain punctuation. Check for inconsistent spellings of the same category."})
    return out


# ---------------------------------------------------------------- aggregate
def build_quality(df: pd.DataFrame, df_raw: pd.DataFrame, meta: list[dict], placeholders: list[dict]) -> dict:
    ids = [{"column": m["name"], "level": m["identifier_level"], "unique_ratio": round(m["unique_ratio"], 4),
            "evidence": m["evidence"],
            "label": {"strong": "Strong identifier", "possible": "Possible identifier"}[m["identifier_level"]]}
           for m in meta if m["identifier_level"]]
    high_card = [{"column": m["name"], "unique": m["n_unique"], "unique_ratio": round(m["unique_ratio"], 4),
                  "label": "High-cardinality categorical"} for m in meta if m["high_cardinality"]]
    return {
        "placeholders": placeholders,
        "missing": analyze_missing(df),
        "duplicates": analyze_duplicates(df, meta),
        "constants": analyze_constants(df),
        "identifiers": ids,
        "high_cardinality": high_card,
        "formatting": analyze_formatting(df_raw, meta),
    }
