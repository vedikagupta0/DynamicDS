import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from app.services import evaluation as ev
from app.services import modeling
from app.services.preprocessing import build_preprocessor, expand_features, select_features, make_spec


@pytest.fixture(scope="module")
def cls(churn):
    return modeling.train_tabular(churn["typed"], churn["profile"], {"target": "churn", "problem_type": "classification", "cv_folds": 3})


@pytest.fixture(scope="module")
def cls_multiclass(churn):
    # "plan" has 3 classes (basic/pro/team) - exercises the multiclass code paths.
    return modeling.train_tabular(churn["typed"], churn["profile"], {"target": "plan", "problem_type": "classification", "cv_folds": 3})


@pytest.fixture(scope="module")
def cls_tuned(churn):
    return modeling.train_tabular(churn["typed"], churn["profile"],
                                   {"target": "churn", "problem_type": "classification", "cv_folds": 3, "tune": True, "tune_iter": 5})


def test_multiclass_pipeline(cls_multiclass):
    res, art = cls_multiclass
    assert len(res["classes"]) == 3
    assert all(m["status"] == "ok" for m in res["models"])
    assert res["primary_metric"] in res["available_metrics"]


def test_hyperparameter_tuning(cls_tuned):
    res, art = cls_tuned
    tuned_models = {m["key"]: m for m in res["models"] if m["status"] == "ok" and m["key"] in ("rf", "xgb")}
    assert tuned_models, "rf/xgb should have trained successfully"
    for key, m in tuned_models.items():
        assert m["tuned"] is True
        assert "best_params" in m
    # untunable models are left alone
    dummy = next(m for m in res["models"] if m["key"] == "dummy")
    assert dummy.get("tuned", False) is False


def test_tuning_failure_falls_back_gracefully(churn, monkeypatch):
    # If the search itself blows up, training must not fail - it should fall back
    # to the default, untuned pipeline for that model.
    def boom(*a, **kw):
        raise RuntimeError("search exploded")
    monkeypatch.setattr(modeling, "_tune", boom)
    res, art = modeling.train_tabular(churn["typed"], churn["profile"],
                                       {"target": "churn", "problem_type": "classification", "cv_folds": 3, "tune": True})
    rf = next(m for m in res["models"] if m["key"] == "rf")
    assert rf["status"] == "ok"
    assert rf["tuned"] is False
    assert "tune_error" in rf


def test_classification_pipeline(cls):
    res, art = cls
    keys = [m["key"] for m in res["models"]]
    assert keys == ["dummy", "logreg", "rf", "xgb"] and all(m["status"] == "ok" for m in res["models"])
    assert res["classes"] == ["no", "yes"]
    lead = next(m for m in res["models"] if m["leader"])
    assert "confusion_matrix" in lead["diagnostics"] and "roc" in lead["diagnostics"] and "pr" in lead["diagnostics"]
    assert res["models"][0]["is_baseline"] and "delta_vs_baseline" in lead


def test_decisions_are_visible(cls):
    res, _ = cls
    reasons = {d["column"]: d["reason"] for d in res["decisions"]}
    assert "identifier" in reasons["customer_id"].lower() and "Constant" in reasons["const"]
    assert "customer_id" not in res["features_used"] and "const" not in res["features_used"]


def test_imbalance_uses_class_weights_not_smote(churn):
    d = churn["typed"].copy()
    d["churn"] = pd.array(np.where(np.arange(len(d)) % 20 == 0, True, False), dtype="boolean")
    res, _ = modeling.train_tabular(d, churn["profile"], {"target": "churn", "problem_type": "classification", "cv_folds": 3, "models": ["logreg"]})
    assert res["imbalance"]["imbalanced"] and res["primary_metric"] == "pr_auc"
    assert "SMOTE" in res["imbalance"]["handling"] and "No oversampling" in res["imbalance"]["handling"]


def test_split_is_disjoint_and_stratified():
    y = np.array([0] * 90 + [1] * 10)
    X = pd.DataFrame({"a": range(100)})
    a, b, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    assert set(a.a).isdisjoint(set(b.a)) and yb.sum() == 2


def test_preprocessing_fit_only_on_train():
    """Imputer statistics must come from training data, never from test data."""
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    X_tr = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan, 5.0] * 10})
    X_te = pd.DataFrame({"a": [1000.0, np.nan, 2000.0]})
    prep, _ = build_preprocessor(X_tr, linear=True)
    pipe = Pipeline([("prep", prep), ("m", LinearRegression())]).fit(X_tr, np.arange(50, dtype=float))
    imp = pipe.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    assert imp.statistics_[0] == X_tr["a"].median()
    scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    assert scaler.mean_[0] < 10  # not influenced by the 1000/2000 test values
    pipe.predict(X_te)  # transform-only on test works with unseen values


def test_unseen_categories_and_frequency_encoding():
    X_tr = pd.DataFrame({"c": [f"v{i % 50}" for i in range(200)]}, dtype=object)
    prep, info = build_preprocessor(X_tr, linear=False)
    assert info["frequency_encoded_columns"] == ["c"] and not info["one_hot_columns"]  # >30 categories: no one-hot explosion
    prep.fit(X_tr)
    assert prep.transform(pd.DataFrame({"c": ["never_seen"]}, dtype=object))[0, 0] == 0.0


def test_classification_metrics_known_values():
    y = [0, 0, 1, 1]
    p = [0, 1, 1, 1]
    proba = np.array([[.9, .1], [.4, .6], [.2, .8], [.1, .9]])
    m = ev.classification_metrics(y, p, proba, 2)
    assert m["accuracy"] == 0.75 and m["precision"] == pytest.approx(2 / 3) and m["recall"] == 1.0
    assert m["f1"] == pytest.approx(0.8) and m["roc_auc"] == 1.0 and m["pr_auc"] == pytest.approx(1.0)


def test_regression_metrics_known_values():
    m = ev.regression_metrics([1, 2, 3, 4], [1, 2, 3, 6])
    assert m["mae"] == 0.5 and m["mse"] == 1.0 and m["rmse"] == 1.0 and m["r2"] == pytest.approx(1 - 4 / 5)
    assert ev.mape([0, 1], [1, 1]) is None  # undefined with zeros
    assert ev.smape([0, 0], [0, 0]) is None


def test_regression_pipeline(churn):
    res, art = modeling.train_tabular(churn["typed"], churn["profile"], {"target": "spend", "problem_type": "regression", "cv_folds": 3, "models": ["linear", "xgb"]})
    assert res["primary_metric"] == "rmse" and res["lower_is_better"]
    lead = next(m for m in res["models"] if m["leader"])
    assert {"actual", "predicted", "residuals", "error_hist"} <= set(lead["diagnostics"])
    assert set(lead["test"]) >= {"mae", "mse", "rmse", "r2"}


def test_invalid_configs_raise(churn):
    with pytest.raises(ValueError, match="Regression requires"):
        modeling.train_tabular(churn["typed"], churn["profile"], {"target": "churn", "problem_type": "regression"})
    with pytest.raises(ValueError, match="identifier"):
        modeling.train_tabular(churn["typed"], churn["profile"], {"target": "customer_id", "problem_type": "classification"})
    with pytest.raises(ValueError, match="Primary metric"):
        modeling.train_tabular(churn["typed"], churn["profile"], {"target": "churn", "problem_type": "classification", "primary_metric": "rmse"})


def test_missing_target_rows_removed_visibly(churn):
    d = churn["typed"].copy()
    d.loc[d.index[:10], "spend"] = np.nan
    res, _ = modeling.train_tabular(d, churn["profile"], {"target": "spend", "problem_type": "regression", "cv_folds": 3, "models": ["linear"]})
    assert any(x["action"] == "rows_removed" and x["count"] == 10 for x in res["decisions"])


def test_prediction_with_explanation(cls, churn):
    _, art = cls
    out = modeling.predict_tabular(art, pd.DataFrame([{"age": "30", "income": "5000", "city": "city1", "plan": "basic", "signup": "2023-01-01", "spend": "12"}]), explain=True)
    p = out["predictions"][0]
    assert p["prediction"] in ("yes", "no") and abs(sum(p["probabilities"].values()) - 1) < 1e-6
    assert out["explanation"]["top_features"] and out["unparseable_columns"] == []
    bad = modeling.predict_tabular(art, pd.DataFrame([{"age": "abc"}]))
    assert "age" in bad["unparseable_columns"]
