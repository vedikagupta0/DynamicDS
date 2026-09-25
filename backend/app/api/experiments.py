import io

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from app import config as C
from app.schemas.requests import ExperimentCreate, ForecastRequest, PredictRequest, StatusUpdate
from app.services import eda as eda_svc
from app.services import forecasting, modeling, report, storage, tracker
from app.services.warnings_engine import build_warnings
from app.utils.jsonable import to_jsonable

router = APIRouter(prefix="/api", tags=["experiments"])


def _exp(exp_id):
    try:
        return tracker.get(exp_id)
    except FileNotFoundError:
        raise HTTPException(404, "Experiment not found.")


def _art(exp_id):
    exp = _exp(exp_id)
    if exp["status"] != "completed":
        raise HTTPException(409, f"Experiment is {exp['status']}; predictions need a completed run.")
    return exp, tracker.load_artifact(exp_id)


def run_training(exp_id: str):
    exp = tracker.get(exp_id)
    exp["status"], exp["error"] = "running", None
    tracker.save(exp)
    try:
        df = storage.load_typed(exp["dataset_id"])
        profile = storage.load_meta(exp["dataset_id"])["profile"]
        cfg = exp["config"]
        results, art = (forecasting.train_forecast if exp["mode"] == "forecast" else modeling.train_tabular)(df, profile, cfg)
        tracker.save_artifact(exp_id, art)
        exp["results"], exp["status"] = results, "completed"
        tracker.log_mlflow(exp)
    except Exception as e:
        exp["status"], exp["error"] = "failed", f"{type(e).__name__}: {e}"
    tracker.save(exp)


@router.post("/experiments")
def create_experiment(body: ExperimentCreate):
    try:
        meta = storage.load_meta(body.dataset_id)
    except (FileNotFoundError, storage.UploadError):
        raise HTTPException(404, "Dataset not found.")
    cols = {c["name"]: c for c in meta["profile"]["columns"]}
    if body.target not in cols:
        raise HTTPException(422, f"Unknown target column '{body.target}'.")
    cfg = body.model_dump()
    cfg.pop("dataset_id")
    mode = cfg.pop("mode")
    for c in body.keep_columns + body.drop_columns:
        if c not in cols:
            raise HTTPException(422, f"Unknown column '{c}'.")
    if mode == "forecast":
        if not body.datetime_column or body.datetime_column not in cols or cols[body.datetime_column]["semantic_type"] != "datetime":
            raise HTTPException(422, "Choose a datetime column for forecasting.")
        if body.datetime_column == body.target:
            raise HTTPException(422, "Datetime column and target must be different.")
        if cols[body.target]["semantic_type"] != "numerical":
            raise HTTPException(422, "The forecasting target must be numerical.")
    else:
        if body.problem_type is None:
            raise HTTPException(422, "problem_type is required for standard ML.")
        cfg.pop("datetime_column", None)
    return to_jsonable(tracker.create(meta, mode, cfg))


@router.post("/experiments/{exp_id}/train", status_code=202)
def train(exp_id: str, background: BackgroundTasks):
    exp = _exp(exp_id)
    if exp["status"] in ("queued", "running"):
        raise HTTPException(409, "Training already in progress.")
    exp["status"] = "queued"
    tracker.save(exp)
    background.add_task(run_training, exp_id)
    return {"id": exp_id, "status": "queued"}


@router.get("/experiments")
def list_experiments():
    return to_jsonable([{k: e[k] for k in ("id", "created_at", "status", "dataset_id", "dataset_name", "mode", "task", "model")} |
                        {"target": e["config"]["target"], "headline": tracker.headline(e)} for e in tracker.list_all()])


@router.get("/experiments/{exp_id}")
def get_experiment(exp_id: str):
    return to_jsonable(_exp(exp_id))


@router.get("/experiments/{exp_id}/metrics")
def metrics(exp_id: str):
    exp = _exp(exp_id)
    r = exp.get("results")
    if not r:
        raise HTTPException(409, f"Experiment is {exp['status']}.")
    keep = ("key", "name", "status", "leader", "is_baseline", "primary_cv", "primary_cv_std", "primary_test", "primary_val", "test", "validation",
            "train_time_s", "delta_vs_baseline", "improvement_vs_best_baseline_pct", "error")
    return to_jsonable({"primary_metric": r["primary_metric"], "lower_is_better": r["lower_is_better"], "leader": r["leader"], "baseline_note": r["baseline_note"],
                        "models": [{k: m[k] for k in keep if k in m} for m in r["models"]]})


@router.get("/experiments/{exp_id}/explainability")
def explainability(exp_id: str):
    exp = _exp(exp_id)
    if not exp.get("results"):
        raise HTTPException(409, f"Experiment is {exp['status']}.")
    return to_jsonable(exp["results"].get("explainability"))


@router.get("/experiments/{exp_id}/report", response_class=HTMLResponse)
def experiment_report(exp_id: str):
    exp = _exp(exp_id)
    meta = storage.load_meta(exp["dataset_id"])
    target = exp["config"]["target"]
    e = to_jsonable(eda_svc.build_eda(storage.load_typed(exp["dataset_id"]), meta["profile"], target if exp["mode"] == "standard" else None,
                                      exp["config"].get("problem_type")))
    html = report.build_report(meta, build_warnings(meta["profile"], meta["quality"], e, target if exp["mode"] == "standard" else None), e, exp)
    return HTMLResponse(html, headers={"Content-Disposition": f'attachment; filename="autods-model-{exp_id}.html"'})


# ---------------------------------------------------------------- registry
@router.get("/models")
def models():
    return to_jsonable([{"id": e["id"], "name": e["model"]["name"], "version": e["model"]["version"], "status": e["model"]["registry_status"],
                         "task": e["task"], "dataset": e["dataset_name"], "created_at": e["created_at"], "headline": tracker.headline(e)}
                        for e in tracker.list_all() if e["status"] == "completed"])


@router.patch("/models/{exp_id}/status")
def set_status(exp_id: str, body: StatusUpdate):
    _exp(exp_id)
    try:
        return to_jsonable(tracker.set_registry_status(exp_id, body.status)["model"])
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.get("/models/{exp_id}/schema")
def schema(exp_id: str):
    exp, art = _art(exp_id)
    if art["kind"] == "forecast":
        return {"kind": "forecast", "target": art["target"], "frequency": art["freq"], "models": art["models"], "leader": art["leader"]}
    return to_jsonable({"kind": "tabular", "problem_type": art["problem_type"], "classes": art["classes"], "inputs": art["input_schema"],
                        "models": list(art["pipelines"]), "leader": art["leader"]})


# ---------------------------------------------------------------- predictions
@router.post("/models/{exp_id}/predict")
def predict(exp_id: str, body: PredictRequest):
    _, art = _art(exp_id)
    if art["kind"] != "tabular":
        raise HTTPException(422, "Use /api/forecast for forecasting models.")
    try:
        return to_jsonable(modeling.predict_tabular(art, pd.DataFrame(body.records), body.model, explain=body.explain))
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.post("/models/{exp_id}/batch-predict")
async def batch_predict(exp_id: str, file: UploadFile = File(...), model: str | None = None):
    _, art = _art(exp_id)
    if art["kind"] != "tabular":
        raise HTTPException(422, "Batch prediction is available for standard ML models.")
    data = await file.read(C.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    try:
        df = storage.parse_upload(file.filename or "batch.csv", data)
        res = modeling.predict_tabular(art, df, model)
    except (storage.UploadError, ValueError) as e:
        raise HTTPException(422, str(e))
    if len(res["missing_columns"]) == len(art["spec"]):
        raise HTTPException(422, "None of the model's input columns were found in the file.")
    out = df.copy()
    out["prediction"] = [p["prediction"] for p in res["predictions"]]
    if art["problem_type"] == "classification":
        out["confidence"] = [p["confidence"] for p in res["predictions"]]
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="predictions-{exp_id}.csv"',
                                                                    "X-Missing-Columns": ",".join(res["missing_columns"])})


@router.post("/forecast")
def forecast(body: ForecastRequest):
    _, art = _art(body.experiment_id)
    if art["kind"] != "forecast":
        raise HTTPException(422, "This experiment is not a forecasting model.")
    try:
        return to_jsonable(forecasting.forecast_future(art, body.periods, body.model))
    except ValueError as e:
        raise HTTPException(422, str(e))
