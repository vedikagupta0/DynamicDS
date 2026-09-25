from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from app import config as C
from app.services import eda as eda_svc
from app.services import report, storage
from app.services.profiling import build_profile
from app.services.warnings_engine import build_warnings
from app.utils.jsonable import to_jsonable

router = APIRouter(prefix="/api/datasets", tags=["datasets"])
_eda_cache: dict = {}


def _meta(dataset_id):
    try:
        return storage.load_meta(dataset_id)
    except (FileNotFoundError, storage.UploadError):
        raise HTTPException(404, "Dataset not found.")


def _eda(dataset_id, target, problem_type):
    key = (dataset_id, target, problem_type)
    if key not in _eda_cache:
        meta = _meta(dataset_id)
        if target and target not in [c["name"] for c in meta["profile"]["columns"]]:
            raise HTTPException(422, f"Unknown column '{target}'.")
        if len(_eda_cache) > 20:
            _eda_cache.pop(next(iter(_eda_cache)))
        _eda_cache[key] = to_jsonable(eda_svc.build_eda(storage.load_typed(dataset_id), meta["profile"], target, problem_type))
    return _eda_cache[key]


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    limit = C.MAX_UPLOAD_MB * 1024 * 1024
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(413, f"File exceeds the {C.MAX_UPLOAD_MB} MB limit.")
    try:
        raw = storage.parse_upload(file.filename or "upload", bytes(data))
        _, _, profile, quality = build_profile(raw, storage.safe_filename(file.filename or "upload"), len(data))
        meta = storage.save_dataset(file.filename or "upload", bytes(data), raw, profile, quality)
    except storage.UploadError as e:
        raise HTTPException(422, str(e))
    return to_jsonable({"id": meta["id"], "filename": meta["filename"], "profile": profile})


@router.get("")
def list_datasets():
    return storage.list_datasets()


@router.get("/{dataset_id}/profile")
def profile(dataset_id: str):
    m = _meta(dataset_id)
    return to_jsonable({"id": m["id"], "filename": m["filename"], "uploaded_at": m["uploaded_at"], "sha256": m["sha256"], **m["profile"]})


@router.get("/{dataset_id}/quality")
def quality(dataset_id: str):
    return to_jsonable(_meta(dataset_id)["quality"])


@router.get("/{dataset_id}/eda")
def eda(dataset_id: str, target: str | None = None, problem_type: str | None = None):
    return _eda(dataset_id, target, problem_type)


@router.get("/{dataset_id}/warnings")
def warnings(dataset_id: str, target: str | None = None):
    m = _meta(dataset_id)
    return to_jsonable(build_warnings(m["profile"], m["quality"], _eda(dataset_id, target, None), target))


@router.get("/{dataset_id}/report", response_class=HTMLResponse)
def dataset_report(dataset_id: str, target: str | None = None):
    m = _meta(dataset_id)
    e = _eda(dataset_id, target, None)
    html = report.build_report(m, build_warnings(m["profile"], m["quality"], e, target), e)
    return HTMLResponse(html, headers={"Content-Disposition": f'attachment; filename="autods-eda-{dataset_id}.html"'})
