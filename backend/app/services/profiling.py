"""Dataset processing service: raw strings -> normalised -> typed frame + profile."""
from __future__ import annotations

import pandas as pd

from app.services import data_quality as dq
from app.services import type_inference as ti


def build_profile(raw: pd.DataFrame, filename: str, file_size: int):
    norm, placeholders = dq.normalize_placeholders(raw)
    meta = ti.infer_types(norm)
    typed = ti.coerce_types(norm, meta)
    counts = {"numerical": 0, "categorical": 0, "boolean": 0, "datetime": 0, "identifier": 0, "text": 0, "empty": 0}
    for m in meta:
        counts[m["semantic_type"]] += 1
    quality = dq.build_quality(typed, norm, meta, placeholders)
    overview = {
        "filename": filename, "file_size_bytes": int(file_size), "rows": int(len(typed)), "columns": int(typed.shape[1]),
        "memory_mb": round(float(typed.memory_usage(deep=True).sum()) / 1e6, 3),
        "type_counts": counts,
        "missing_cell_pct": quality["missing"]["missing_cell_pct"],
        "missing_cells": quality["missing"]["total_missing_cells"],
        "duplicate_rows": quality["duplicates"]["exact_duplicates"],
    }
    profile = {"overview": overview, "columns": meta}
    return typed, norm, profile, quality
