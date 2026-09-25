"""Experiment tracking + model registry (JSON files + joblib artifacts). Optional MLflow mirroring."""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone

import joblib

from app import config as C
from app.utils.jsonable import to_jsonable

_lock = threading.Lock()
_ID = re.compile(r"^[a-f0-9]{10}$")


def _dir(kind):
    p = C.DATA_DIR / kind
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now():
    return datetime.now(timezone.utc).isoformat()


def _check(exp_id):
    if not _ID.match(exp_id):
        raise FileNotFoundError("Experiment not found.")


def create(dataset_meta: dict, mode: str, config: dict) -> dict:
    with _lock:
        exp_id = uuid.uuid4().hex[:10]
        target = config["target"]
        kind = config.get("problem_type") if mode == "standard" else "forecast"
        name = f"{dataset_meta['filename'].rsplit('.', 1)[0]}-{target}-{kind}"
        version = 1 + sum(1 for e in list_all() if e["model"]["name"] == name)
        exp = {"id": exp_id, "created_at": _now(), "updated_at": _now(), "status": "created", "error": None,
               "dataset_id": dataset_meta["id"], "dataset_name": dataset_meta["filename"], "dataset_hash": dataset_meta["sha256"],
               "mode": mode, "task": kind, "config": config, "results": None,
               "model": {"name": name, "version": f"v{version}", "registry_status": "Development"}}
        save(exp)
        return exp


def save(exp: dict):
    exp["updated_at"] = _now()
    p = _dir("experiments") / f"{exp['id']}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(to_jsonable(exp)))
    os.replace(tmp, p)


def get(exp_id: str) -> dict:
    _check(exp_id)
    p = _dir("experiments") / f"{exp_id}.json"
    if not p.exists():
        raise FileNotFoundError("Experiment not found.")
    return json.loads(p.read_text())


def list_all() -> list[dict]:
    out = [json.loads(p.read_text()) for p in _dir("experiments").glob("*.json")]
    return sorted(out, key=lambda e: e["created_at"], reverse=True)


def save_artifact(exp_id: str, art: dict):
    joblib.dump(art, _dir("models") / f"{exp_id}.joblib")


def load_artifact(exp_id: str) -> dict:
    """Artifacts are pickled by this application only. Never place untrusted files in the models directory."""
    _check(exp_id)
    p = _dir("models") / f"{exp_id}.joblib"
    if not p.exists():
        raise FileNotFoundError("Model artifact not found (training may not be finished).")
    return joblib.load(p)


def set_registry_status(exp_id: str, status: str) -> dict:
    if status not in C.MODEL_STATUSES:
        raise ValueError(f"Status must be one of {C.MODEL_STATUSES}.")
    with _lock:
        exp = get(exp_id)
        if exp["status"] != "completed":
            raise ValueError("Only completed experiments can be registered.")
        if status == "Production":  # a single Production version per model name
            for e in list_all():
                if e["model"]["name"] == exp["model"]["name"] and e["id"] != exp_id and e["model"]["registry_status"] == "Production":
                    e["model"]["registry_status"] = "Archived"
                    save(e)
        exp["model"]["registry_status"] = status
        save(exp)
        return exp


def headline(exp: dict) -> dict | None:
    r = exp.get("results")
    if not r:
        return None
    row = next((m for m in r["models"] if m["key"] == r["leader"]), None)
    return {"leader": row["name"] if row else None, "primary_metric": r["primary_metric"], "test": row.get("test") if row else None,
            "beats_baseline": r["beats_baseline"]}


def log_mlflow(exp: dict):
    """Mirror a finished run to MLflow when DYNAMICDS_MLFLOW=1 or AUTODS_MLFLOW=1 and mlflow is installed. Failures never break training."""
    if os.getenv("DYNAMICDS_MLFLOW") != "1" and os.getenv("AUTODS_MLFLOW") != "1":
        return
    try:
        import mlflow
        mlflow.set_experiment("dynamicds")
        with mlflow.start_run(run_name=exp["id"]):
            mlflow.log_params({"dataset_id": exp["dataset_id"], "dataset_hash": exp["dataset_hash"][:32], "mode": exp["mode"],
                               "target": exp["config"]["target"], "model_version": exp["model"]["version"]})
            h = headline(exp)
            for k, v in ((h or {}).get("test") or {}).items():
                if v is not None:
                    mlflow.log_metric(k, v)
    except Exception:
        pass
