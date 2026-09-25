"""Semantic type inference on raw string columns (placeholders already normalised to NaN)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from app import config as C

TRUE_VOCAB = {"true", "t", "yes", "y"}
FALSE_VOCAB = {"false", "f", "no", "n"}
_FORMATTED = r"^[-+]?[$€£₹¥]?\s?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?\s?%?$"
_STRIP = r"[,$€£₹¥%\s]"
DT_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d",
    "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y", "%d %b %Y", "%b %d, %Y",
]
_AMBIGUOUS = [{"%d/%m/%Y", "%m/%d/%Y"}, {"%d-%m-%Y", "%m-%d-%Y"}]


def name_looks_like_id(name: str) -> bool:
    n = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name)).lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", n) if t]
    return any(t in tokens for t in ("id", "uuid", "guid")) or "identifier" in n


def parse_numeric(nn: pd.Series):
    """Return (numeric_series, formatted_flag, parse_ratio) for a string Series."""
    direct = pd.to_numeric(nn, errors="coerce")
    ratio = float(direct.notna().mean()) if len(nn) else 0.0
    if ratio >= 0.95:
        return direct, False, ratio
    st = nn.astype(str).str.strip()
    if st.str.match(_FORMATTED).mean() >= 0.95:
        cleaned = pd.to_numeric(st.str.replace(_STRIP, "", regex=True), errors="coerce")
        return cleaned, True, float(cleaned.notna().mean())
    return direct, False, ratio


def detect_datetime_format(nn: pd.Series):
    """Return (format|'mixed', ambiguous) or None when the column is not datetime-like."""
    sample = nn if len(nn) <= 1000 else nn.sample(1000, random_state=0)
    sample = sample.astype(str)
    if sample.str.contains(r"\d").mean() < 0.95 or sample.str.len().median() < 6:
        return None
    passing = []
    for fmt in DT_FORMATS:
        if pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean() >= 0.95:
            passing.append(fmt)
    if passing:
        ambiguous = any(a <= set(passing) for a in _AMBIGUOUS)
        return passing[0], ambiguous
    if sample.str.contains(r"[-/:]|[A-Za-z]{3}").mean() >= 0.95:
        if pd.to_datetime(sample, format="mixed", errors="coerce").notna().mean() >= 0.95:
            return "mixed", False
    return None


def parse_datetime(s: pd.Series, fmt: str | None) -> pd.Series:
    if fmt in (None, "mixed"):
        return pd.to_datetime(s, format="mixed", errors="coerce")
    return pd.to_datetime(s, format=fmt, errors="coerce")


def cardinality_class(n_unique: int) -> str:
    if n_unique <= C.CARD_LOW:
        return "low"
    return "medium" if n_unique <= C.CARD_HIGH else "high"


def _base(name, n_unique, ratio, n_non_null):
    return {
        "name": name, "semantic_type": "categorical", "storage_type": "string",
        "identifier_level": None, "n_unique": int(n_unique), "unique_ratio": float(ratio),
        "n_non_null": int(n_non_null), "formatted_numeric": False, "datetime_format": None,
        "is_integer": False, "cardinality_class": None, "high_cardinality": False,
        "evidence": [], "constant": n_unique <= 1,
    }


def infer_column(name: str, s: pd.Series) -> dict:
    nn = s.dropna()
    n = len(nn)
    n_unique = int(nn.nunique()) if n else 0
    ratio = n_unique / n if n else 0.0
    m = _base(name, n_unique, ratio, n)
    if n == 0:
        m["semantic_type"] = "empty"
        m["evidence"].append("all values are missing")
        return m
    nn = nn.astype(str)
    name_id = name_looks_like_id(name)
    lower = nn.str.strip().str.lower()

    # boolean: exactly one true-token and one false-token (0/1 integers stay numerical)
    vals = set(lower.unique())
    if vals <= (TRUE_VOCAB | FALSE_VOCAB) and len(vals & TRUE_VOCAB) == 1 and len(vals & FALSE_VOCAB) == 1:
        m.update(semantic_type="boolean", storage_type="bool")
        orig = nn.str.strip()
        m["bool_labels"] = {"true": str(orig[lower.isin(TRUE_VOCAB)].iloc[0]), "false": str(orig[lower.isin(FALSE_VOCAB)].iloc[0])}
        m["evidence"].append(f"values {sorted(vals)} form a true/false vocabulary")
        return m

    # numerical
    leading_zero = nn.str.match(r"^0\d+$").mean() > 0.05
    num, formatted, pr = parse_numeric(nn) if not leading_zero else (None, False, 0.0)
    if num is not None and pr >= 0.95:
        vals_n = num.dropna()
        is_int = bool((vals_n % 1 == 0).all())
        m.update(semantic_type="numerical", storage_type="numeric", formatted_numeric=formatted, is_integer=is_int)
        if formatted:
            m["evidence"].append("values contain currency/percent/thousands formatting")
        if not is_int and not name_id:
            return m
        return _identifier_check(m, name_id, ratio, n, integer=is_int, numeric_values=vals_n)

    # datetime
    dt = detect_datetime_format(nn)
    if dt is not None:
        fmt, ambiguous = dt
        m.update(semantic_type="datetime", storage_type="datetime", datetime_format=fmt)
        m["evidence"].append(f"parses as datetime (format {fmt})")
        if ambiguous:
            m["evidence"].append("day/month order is ambiguous; day-first was assumed")
        return m

    # text
    tokens = nn.str.split().str.len()
    if (nn.str.len().mean() >= C.TEXT_MIN_AVG_LEN or tokens.median() >= C.TEXT_MIN_MEDIAN_TOKENS) and ratio >= C.TEXT_MIN_UNIQUE_RATIO:
        m["semantic_type"] = "text"
        m["evidence"].append(f"avg length {nn.str.len().mean():.0f} chars, median {tokens.median():.0f} tokens")
        return m

    # identifier / categorical
    m = _identifier_check(m, name_id, ratio, n, integer=False)
    if m["semantic_type"] == "categorical":
        m["cardinality_class"] = cardinality_class(n_unique)
        m["high_cardinality"] = m["cardinality_class"] == "high"
        if m["high_cardinality"]:
            m["evidence"].append(f"{n_unique} unique values ({ratio:.1%} of rows): high-cardinality categorical")
    return m


def _identifier_check(m, name_id, ratio, n, integer, numeric_values=None):
    level, why = None, None
    sequential = False
    if integer and numeric_values is not None and n >= C.ID_MIN_ROWS and ratio == 1.0:
        v = np.sort(numeric_values.to_numpy())
        sequential = bool(np.all(np.diff(v) == 1))
    if name_id and ratio >= C.ID_STRONG_UNIQUE:
        level, why = "strong", f"id-like name and {ratio:.1%} unique"
    elif sequential and name_id is False:
        level, why = "possible", "sequential integers, all unique (row-index-like)"
    elif name_id and ratio >= C.ID_NAME_POSSIBLE_UNIQUE:
        level, why = "possible", f"id-like name and {ratio:.1%} unique"
    elif not name_id and n >= C.ID_MIN_ROWS and ratio >= C.ID_POSSIBLE_UNIQUE and m["storage_type"] == "string":
        level, why = "possible", f"{ratio:.1%} of values are unique"
    if level:
        m["semantic_type"] = "identifier"
        m["identifier_level"] = level
        m["evidence"].append(why)
    return m


def infer_types(df: pd.DataFrame) -> list[dict]:
    return [infer_column(c, df[c]) for c in df.columns]


def coerce_types(df: pd.DataFrame, meta: list[dict]) -> pd.DataFrame:
    """Convert a raw string frame to typed columns according to inferred metadata."""
    out = {}
    for m in meta:
        c = m["name"]
        if c not in df.columns:
            continue
        s = df[c]
        st = m["storage_type"]
        if st == "numeric":
            nn = s.dropna().astype(str)
            num, _, _ = parse_numeric(nn) if len(nn) else (nn, False, 0)
            out[c] = num.reindex(s.index).astype("float64")
        elif st == "bool":
            low = s.astype("object").map(lambda v: v.strip().lower() if isinstance(v, str) else v)
            mp = low.map(lambda v: True if v in TRUE_VOCAB else (False if v in FALSE_VOCAB else pd.NA))
            out[c] = mp.astype("boolean")
        elif st == "datetime":
            out[c] = parse_datetime(s, m["datetime_format"])
        else:
            out[c] = s
    return pd.DataFrame(out, index=df.index)
