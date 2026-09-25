import numpy as np
import pandas as pd

from app import config as C
from app.services import eda
from app.services.profiling import build_profile
from app.services.warnings_engine import build_warnings


def test_outlier_detection_iqr():
    x = pd.Series(list(range(1, 101)) + [1000.0], name="v")
    s = eda.numeric_stats(x, len(x))
    q1, q3 = x.quantile(.25), x.quantile(.75)
    assert s["outliers"]["count"] == 1
    assert np.isclose(s["outliers"]["upper_bound"], q3 + 1.5 * (q3 - q1))
    assert s["outliers"]["pct"] == round(1 / 101 * 100, 3)


def test_skewness_classification_uses_config():
    assert eda.skew_label(0.1) == "approximately symmetric"
    assert eda.skew_label(0.7) == "moderately right-skewed"
    assert eda.skew_label(-0.7) == "moderately left-skewed"
    assert eda.skew_label(2.5) == "strongly right-skewed"
    assert C.SKEW_SYMMETRIC == 0.5 and C.SKEW_STRONG == 1.0


def test_no_blind_normality_claim():
    rng = np.random.default_rng(0)
    s = eda.numeric_stats(pd.Series(rng.exponential(size=500), name="e"), 500)
    assert s["normality"]["conclusion"].startswith("Normality rejected")
    assert "normally distributed" not in str(s["skew_class"])


def test_correlation_bands_and_pairs():
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    df = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(0, .1, 300), "c": rng.normal(size=300)})
    meta = {c: {"semantic_type": "numerical", "n_unique": 300} for c in df}
    r = eda.correlation_analysis(df, list(df), meta)
    assert len(r["high_pairs"]) == 1 and {r["high_pairs"][0]["a"], r["high_pairs"][0]["b"]} == {"a", "b"}
    assert eda.corr_strength(0.2) == "low" and eda.corr_strength(-0.5) == "moderate" and eda.corr_strength(0.8) == "high"


def test_cramers_v_perfect_and_independent():
    x = pd.Series(["a", "b"] * 100)
    assert eda.cramers_v(x, x) > 0.95
    rng = np.random.default_rng(0)
    assert eda.cramers_v(pd.Series(rng.choice(list("ab"), 400)), pd.Series(rng.choice(list("xy"), 400))) < 0.15


def test_problem_type_recommendation(churn):
    p = churn["profile"]
    t = eda.analyze_target(churn["typed"], {m["name"]: m for m in p["columns"]}, "churn")
    assert t["recommendation"]["problem_type"] == "classification"
    t2 = eda.analyze_target(churn["typed"], {m["name"]: m for m in p["columns"]}, "spend")
    assert t2["recommendation"]["problem_type"] == "regression"
    t3 = eda.analyze_target(churn["typed"], {m["name"]: m for m in p["columns"]}, "customer_id")
    assert t3["problem_type"] is None


def test_leakage_suspect_flagged_and_warned():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.normal(size=n)
    raw = pd.DataFrame({"x": rng.normal(size=n), "leak": y * 3 + rng.normal(0, .01, n), "y": y}).round(5).astype(str)
    typed, _, p, q = build_profile(raw, "l.csv", 1)
    e = eda.build_eda(typed, p, "y")
    assert [l["feature"] for l in e["target"]["leakage_suspects"]] == ["leak"]
    w = build_warnings(p, q, e, "y")
    assert any(x["severity"] == "HIGH" and x["category"] == "leakage" for x in w)


def test_warning_schema_and_ordering(churn):
    e = eda.build_eda(churn["typed"], churn["profile"], "churn")
    w = build_warnings(churn["profile"], churn["quality"], e, "churn")
    assert w and all({"severity", "column", "issue", "evidence", "recommendation"} <= set(x) for x in w)
    order = [{"HIGH": 0, "WARNING": 1, "INFO": 2}[x["severity"]] for x in w]
    assert order == sorted(order)
    assert any("strong identifier" in x["issue"] for x in w)


def test_visual_sampling_flag():
    x = pd.Series(np.random.default_rng(0).normal(size=C.VIS_SAMPLE_ROWS + 100), name="big")
    assert eda.numeric_stats(x, len(x))["sampled_for_plots"] is True
    assert eda.numeric_stats(x.iloc[:100], 100)["sampled_for_plots"] is False
