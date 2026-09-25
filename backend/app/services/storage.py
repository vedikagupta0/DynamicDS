"""File-based dataset storage with strict validation. Uploaded content is only ever parsed as data, never executed."""
from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app import config as C
from app.services.type_inference import coerce_types
from app.services.data_quality import normalize_placeholders

_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_lock = threading.Lock()
_cache: "OrderedDict[str, pd.DataFrame]" = OrderedDict()


class UploadError(ValueError):
    pass


def dataset_dir(dataset_id: str) -> Path:
    if not _ID_RE.match(dataset_id):
        raise UploadError("Invalid dataset id.")
    return C.DATA_DIR / "datasets" / dataset_id


def safe_filename(name: str) -> str:
    base = Path(name or "upload").name
    return re.sub(r"[^\w.\- ]", "_", base)[:120] or "upload"


def parse_upload(filename: str, data: bytes) -> pd.DataFrame:
    """Validate and parse bytes into an all-string DataFrame (types are inferred later)."""
    ext = Path(filename).suffix.lower()
    if ext not in C.ALLOWED_EXTENSIONS:
        raise UploadError(f"Unsupported file type '{ext}'. Upload a .csv or .xlsx file.")
    if len(data) == 0:
        raise UploadError("The file is empty.")
    if len(data) > C.MAX_UPLOAD_MB * 1024 * 1024:
        raise UploadError(f"File exceeds the {C.MAX_UPLOAD_MB} MB limit.")
    try:
        if ext == ".csv":
            head = data[:20000]
            enc = "utf-8-sig"
            try:
                head.decode(enc)
            except UnicodeDecodeError:
                enc = "latin-1"
            line = head.decode(enc, errors="ignore").splitlines()[0] if head else ""
            sep = max([",", ";", "\t", "|"], key=line.count)
            df = pd.read_csv(io.BytesIO(data), sep=sep, encoding=enc, dtype=str, keep_default_na=False, na_values=[""], nrows=C.MAX_ROWS + 1)
        else:
            if not data.startswith(b"PK"):
                raise UploadError("File content is not a valid .xlsx workbook.")
            df = pd.read_excel(io.BytesIO(data), dtype=str, nrows=C.MAX_ROWS + 1, engine="openpyxl")
    except UploadError:
        raise
    except Exception as e:
        raise UploadError(f"Could not parse the file: {type(e).__name__}.") from e
    if df.empty or df.shape[1] == 0:
        raise UploadError("The file contains no data rows.")
    if len(df) > C.MAX_ROWS:
        raise UploadError(f"Dataset has more than {C.MAX_ROWS:,} rows (limit).")
    if df.shape[1] > C.MAX_COLS:
        raise UploadError(f"Dataset has {df.shape[1]} columns; the limit is {C.MAX_COLS}.")
    df.columns = [str(c).strip() or f"unnamed_{i}" for i, c in enumerate(df.columns)]
    return df


def save_dataset(filename: str, data: bytes, raw: pd.DataFrame, profile: dict, quality: dict) -> dict:
    did = uuid.uuid4().hex[:12]
    d = dataset_dir(did)
    d.mkdir(parents=True, exist_ok=True)
    raw.to_csv(d / "data.csv", index=False)
    meta = {"id": did, "filename": safe_filename(filename), "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "profile": profile, "quality": quality}
    (d / "meta.json").write_text(json.dumps(meta))
    return meta


def load_meta(dataset_id: str) -> dict:
    p = dataset_dir(dataset_id) / "meta.json"
    if not p.exists():
        raise FileNotFoundError("Dataset not found.")
    return json.loads(p.read_text())


def list_datasets() -> list[dict]:
    base = C.DATA_DIR / "datasets"
    out = []
    if base.exists():
        for p in base.glob("*/meta.json"):
            m = json.loads(p.read_text())
            ov = m["profile"]["overview"]
            out.append({"id": m["id"], "filename": m["filename"], "uploaded_at": m["uploaded_at"], "rows": ov["rows"], "columns": ov["columns"]})
    return sorted(out, key=lambda d: d["uploaded_at"], reverse=True)


def load_typed(dataset_id: str) -> pd.DataFrame:
    """Typed frame using the *stored* inference (so training and reports never disagree with what the user saw)."""
    with _lock:
        if dataset_id in _cache:
            _cache.move_to_end(dataset_id)
            return _cache[dataset_id]
    meta = load_meta(dataset_id)
    raw = pd.read_csv(dataset_dir(dataset_id) / "data.csv", dtype=str, keep_default_na=False, na_values=[""])
    norm, _ = normalize_placeholders(raw)
    typed = coerce_types(norm, meta["profile"]["columns"])
    with _lock:
        _cache[dataset_id] = typed
        while len(_cache) > 4:
            _cache.popitem(last=False)
    return typed
