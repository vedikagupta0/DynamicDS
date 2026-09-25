"""Frequency inference shared by EDA and forecasting."""
from __future__ import annotations

import pandas as pd

_LABELS = {"MS": "monthly", "QS": "quarterly", "YS": "yearly", "D": "daily", "h": "hourly", "B": "business-daily",
           "min": "minutely", "s": "secondly"}


def normalise_alias(alias: str) -> str:
    a = alias.upper() if alias[:1] in "mqyaMQYA" and not alias.startswith("min") else alias
    if a.startswith(("M", "ME")):
        return "MS"
    if a.startswith("Q"):
        return "QS"
    if a.startswith(("Y", "A")):
        return "YS"
    if alias.startswith("W"):
        return "W"
    return alias


def infer_frequency(ts: pd.Series):
    """Return dict(freq, label, gaps, regular) or None when it cannot be inferred."""
    u = pd.DatetimeIndex(pd.to_datetime(ts).dropna().sort_values().unique())
    if len(u) < 3:
        return None
    freq = None
    try:
        freq = pd.infer_freq(u)
    except (ValueError, TypeError):
        freq = None
    if freq is None:
        days = pd.Series(u[1:] - u[:-1]).median() / pd.Timedelta(days=1)
        if 27 <= days <= 32:
            freq = "MS"
        elif 88 <= days <= 93:
            freq = "QS"
        elif 360 <= days <= 370:
            freq = "YS"
        else:
            freq = pd.tseries.frequencies.to_offset(pd.Series(u[1:] - u[:-1]).median()).freqstr
    freq = normalise_alias(freq)
    lo = u.min()
    idx = regularize_index(pd.Series(u), freq)
    full = pd.date_range(idx.min(), idx.max(), freq=freq)
    gaps = max(len(full) - idx.nunique(), 0)
    label = _LABELS.get(freq, "weekly" if freq.startswith("W") else f"custom ({freq})")
    return {"freq": freq, "label": label, "gaps": int(gaps), "regular": gaps == 0}


def regularize_index(ts: pd.Series, freq: str) -> pd.Series:
    """Snap month/quarter/year timestamps to period starts so they align on a regular grid."""
    ts = pd.to_datetime(ts)
    if freq in ("MS", "QS", "YS"):
        p = {"MS": "M", "QS": "Q", "YS": "Y"}[freq]
        return ts.dt.to_period(p).dt.to_timestamp()
    return ts
