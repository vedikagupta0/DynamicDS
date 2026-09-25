import math
from datetime import date, datetime
import numpy as np
import pandas as pd


def to_jsonable(o):
    """Recursively convert numpy/pandas objects to JSON-safe python (NaN/inf -> None)."""
    if o is None or isinstance(o, (str, bool)):
        return o
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [to_jsonable(v) for v in o.tolist()]
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return str(o)
