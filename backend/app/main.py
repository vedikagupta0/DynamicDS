from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config as C
from app.api import datasets, experiments

app = FastAPI(title="Dynamic DS", version="1.0.0", description="EDA provider and baseline model tester.")
app.include_router(datasets.router)
app.include_router(experiments.router)

FRONTEND = C.ROOT / "frontend" / "dist"
if (FRONTEND / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")


@app.get("/api/health")
def health():
    return {"status": "ok", "max_upload_mb": C.MAX_UPLOAD_MB, "max_rows": C.MAX_ROWS, "max_cols": C.MAX_COLS}


@app.get("/api/config")
def config():
    return JSONResponse({"model_statuses": C.MODEL_STATUSES, "thresholds": {
        "skew_symmetric": C.SKEW_SYMMETRIC, "skew_strong": C.SKEW_STRONG, "corr_low": C.CORR_LOW, "corr_high": C.CORR_HIGH,
        "missing_low": C.MISSING_LOW, "missing_moderate": C.MISSING_MODERATE, "missing_high": C.MISSING_HIGH,
        "near_constant_share": C.NEAR_CONSTANT_SHARE, "id_strong_unique": C.ID_STRONG_UNIQUE}})


@app.get("/")
def index():
    p = FRONTEND / "index.html"
    return FileResponse(p) if p.exists() else JSONResponse({"message": "Dynamic DS API. See /docs. (Frontend not built: run `npm run build` in frontend/.)"})
