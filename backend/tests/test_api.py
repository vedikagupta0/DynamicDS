import io

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _csv(df):
    return {"file": ("data.csv", df.to_csv(index=False).encode(), "text/csv")}


@pytest.fixture(scope="module")
def dataset(churn_raw):
    r = client.post("/api/datasets/upload", files=_csv(churn_raw))
    assert r.status_code == 200, r.text
    return r.json()


def test_upload_validation():
    assert client.post("/api/datasets/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 422
    assert client.post("/api/datasets/upload", files={"file": ("x.csv", b"", "text/csv")}).status_code == 422
    assert client.post("/api/datasets/upload", files={"file": ("x.xlsx", b"not a workbook", "application/octet-stream")}).status_code == 422
    r = client.post("/api/datasets/upload", files={"file": ("../../evil<script>.csv", b"a,b\n1,2\n3,4\n", "text/csv")})
    assert r.status_code == 200 and "/" not in r.json()["filename"] and "<" not in r.json()["filename"]


def test_row_and_column_limits(monkeypatch):
    from app import config as C
    monkeypatch.setattr(C, "MAX_ROWS", 5)
    body = "a,b\n" + "\n".join("1,2" for _ in range(10))
    assert client.post("/api/datasets/upload", files={"file": ("x.csv", body.encode(), "text/csv")}).status_code == 422
    monkeypatch.setattr(C, "MAX_ROWS", 1000); monkeypatch.setattr(C, "MAX_COLS", 2)
    assert client.post("/api/datasets/upload", files={"file": ("x.csv", b"a,b,c\n1,2,3\n", "text/csv")}).status_code == 422


def test_xlsx_upload(churn_raw):
    buf = io.BytesIO()
    churn_raw.head(50).to_excel(buf, index=False)
    r = client.post("/api/datasets/upload", files={"file": ("d.xlsx", buf.getvalue(), "application/vnd.ms-excel")})
    assert r.status_code == 200 and r.json()["profile"]["overview"]["rows"] == 50


def test_profile_quality_eda_warnings(dataset):
    did = dataset["id"]
    ov = client.get(f"/api/datasets/{did}/profile").json()["overview"]
    assert ov["rows"] == 600 and ov["type_counts"]["identifier"] >= 1
    q = client.get(f"/api/datasets/{did}/quality").json()
    assert q["duplicates"]["exact_duplicates"] == 0
    e = client.get(f"/api/datasets/{did}/eda", params={"target": "churn"}).json()
    assert e["target"]["recommendation"]["problem_type"] == "classification"
    w = client.get(f"/api/datasets/{did}/warnings", params={"target": "churn"}).json()
    assert w and w[0]["severity"] in ("HIGH", "WARNING", "INFO")
    assert client.get(f"/api/datasets/{did}/eda", params={"target": "nope"}).status_code == 422
    assert client.get("/api/datasets/../etc/profile").status_code in (404, 422)
    assert client.get("/api/datasets/aaaaaaaaaaaa/profile").status_code == 404


def test_full_standard_workflow(dataset):
    did = dataset["id"]
    r = client.post("/api/experiments", json={"dataset_id": did, "mode": "standard", "target": "churn", "problem_type": "classification", "models": ["logreg", "xgb"], "cv_folds": 3})
    assert r.status_code == 200, r.text
    exp = r.json()
    assert exp["status"] == "created" and exp["model"]["registry_status"] == "Development"
    assert client.post(f"/api/experiments/{exp['id']}/predict").status_code in (404, 405)
    assert client.post(f"/api/models/{exp['id']}/predict", json={"records": [{"age": "30"}]}).status_code == 409  # not trained yet
    assert client.post(f"/api/experiments/{exp['id']}/train").status_code == 202
    done = client.get(f"/api/experiments/{exp['id']}").json()
    assert done["status"] == "completed", done.get("error")
    m = client.get(f"/api/experiments/{exp['id']}/metrics").json()
    assert m["models"][0]["is_baseline"] and m["leader"] in ("logreg", "xgb")
    assert "permutation_importance" in client.get(f"/api/experiments/{exp['id']}/explainability").json()
    p = client.post(f"/api/models/{exp['id']}/predict", json={"records": [{"age": "41", "income": "30000", "plan": "basic", "city": "city2", "signup": "2023-05-01", "spend": "10"}]}).json()
    assert p["predictions"][0]["prediction"] in ("yes", "no") and p["explanation"]["top_features"]
    batch = client.post(f"/api/models/{exp['id']}/batch-predict", files=_csv(__import__("pandas").DataFrame({"age": [30, 50], "plan": ["pro", "basic"]})))
    assert batch.status_code == 200 and "prediction" in batch.text.splitlines()[0]
    # registry
    assert client.patch(f"/api/models/{exp['id']}/status", json={"status": "Production"}).json()["registry_status"] == "Production"
    assert client.patch(f"/api/models/{exp['id']}/status", json={"status": "Bogus"}).status_code == 422
    assert any(x["id"] == exp["id"] for x in client.get("/api/models").json())
    assert any(x["id"] == exp["id"] for x in client.get("/api/experiments").json())
    rep = client.get(f"/api/experiments/{exp['id']}/report")
    assert rep.status_code == 200 and "Model comparison" in rep.text and "<script" not in rep.text.lower()


def test_forecast_workflow(ts_raw):
    did = client.post("/api/datasets/upload", files=_csv(ts_raw)).json()["id"]
    bad = client.post("/api/experiments", json={"dataset_id": did, "mode": "forecast", "target": "sales", "datetime_column": "sales"})
    assert bad.status_code == 422
    exp = client.post("/api/experiments", json={"dataset_id": did, "mode": "forecast", "target": "sales", "datetime_column": "date", "horizon": 7}).json()
    client.post(f"/api/experiments/{exp['id']}/train")
    done = client.get(f"/api/experiments/{exp['id']}").json()
    assert done["status"] == "completed", done.get("error")
    f = client.post("/api/forecast", json={"experiment_id": exp["id"], "periods": 5}).json()
    assert len(f["yhat"]) == 5
    assert client.post(f"/api/models/{exp['id']}/predict", json={"records": [{"a": 1}]}).status_code == 422


def test_failed_training_is_reported(dataset):
    exp = client.post("/api/experiments", json={"dataset_id": dataset["id"], "mode": "standard", "target": "customer_id", "problem_type": "classification"}).json()
    client.post(f"/api/experiments/{exp['id']}/train")
    done = client.get(f"/api/experiments/{exp['id']}").json()
    assert done["status"] == "failed" and "identifier" in done["error"]


def test_report_escapes_html(dataset):
    r = client.get(f"/api/datasets/{dataset['id']}/report", params={"target": "churn"})
    assert r.status_code == 200 and "Dataset summary" in r.text
