import numpy as np
import pandas as pd

from app.services import data_quality as dq
from app.services import type_inference as ti
from app.services.profiling import build_profile
from tests.conftest import col


def test_semantic_types(churn):
    p = churn["profile"]
    assert col(p, "age")["semantic_type"] == "numerical"
    assert col(p, "plan")["semantic_type"] == "categorical"
    assert col(p, "signup")["semantic_type"] == "datetime"
    assert col(p, "churn")["semantic_type"] == "boolean"  # yes/no vocabulary
    assert col(p, "churn")["bool_labels"] == {"true": "yes", "false": "no"}


def test_semantic_not_dtype_based():
    s = pd.Series(["$1,200", "$3,400", "$50"] * 20)
    m = ti.infer_column("price", s)
    assert m["semantic_type"] == "numerical" and m["formatted_numeric"]
    assert ti.infer_column("x", pd.Series(["12.4", "3.1"] * 20))["semantic_type"] == "numerical"
    assert ti.infer_column("f", pd.Series(["true", "false"] * 20))["semantic_type"] == "boolean"
    assert ti.infer_column("g", pd.Series(["male", "female"] * 20))["semantic_type"] == "categorical"


def test_identifier_tiers():
    n = 1000
    strong = pd.Series([f"C{i}" for i in range(n)])
    assert ti.infer_column("customer_id", strong)["identifier_level"] == "strong"
    possible = pd.Series([f"C{i % 978}" for i in range(n)])  # 97.8% unique, id-like name
    m = ti.infer_column("customer_id", possible)
    assert m["identifier_level"] == "possible" and m["semantic_type"] == "identifier"
    rng = np.random.default_rng(0)
    city = pd.Series(rng.choice([f"city{i}" for i in range(312)], n))  # ~31% unique, not an id
    m = ti.infer_column("city", city)
    assert m["semantic_type"] == "categorical" and m["high_cardinality"] and m["identifier_level"] is None


def test_id_name_matching_is_token_based():
    assert ti.name_looks_like_id("customer_id") and ti.name_looks_like_id("userId") and ti.name_looks_like_id("ID")
    assert not ti.name_looks_like_id("paid") and not ti.name_looks_like_id("valid_flag")


def test_continuous_integers_not_identifiers():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.integers(1000, 10_000_000, 500).astype(str))
    assert ti.infer_column("income", s)["semantic_type"] == "numerical"


def test_missing_value_calculation_and_levels():
    df = pd.DataFrame({"a": [1, np.nan, np.nan, 4] * 25, "b": range(100), "c": [np.nan] * 100})
    r = dq.analyze_missing(df)
    a = next(c for c in r["columns"] if c["column"] == "a")
    assert a["missing"] == 50 and a["non_null"] == 50 and a["missing_pct"] == 50.0 and a["level"] == "very_high"
    assert next(c for c in r["columns"] if c["column"] == "b")["level"] == "none"
    assert dq.missing_level(3) == "low" and dq.missing_level(10) == "moderate" and dq.missing_level(30) == "high"


def test_row_missingness_flagged():
    df = pd.DataFrame(np.ones((10, 6)))
    df.iloc[3, :5] = np.nan
    r = dq.analyze_missing(df)
    assert r["flagged_rows"][0]["row"] == 4 and r["flagged_rows"][0]["missing"] == 5


def test_duplicates_with_and_without_identifier():
    df = pd.DataFrame({"customer_id": [f"C{i}" for i in range(100)], "x": [1, 2] * 50})
    meta = ti.infer_types(df.astype(str))
    r = dq.analyze_duplicates(df, meta)
    assert r["exact_duplicates"] == 0 and r["excluding_strong_identifiers"] == 98
    assert "identifier" in r["interpretation"]


def test_constant_and_near_constant():
    df = pd.DataFrame({"c": ["a"] * 100, "n": ["No"] * 99 + ["Yes"], "ok": ["a", "b"] * 50, "e": [np.nan] * 100})
    kinds = {k["column"]: k["kind"] for k in dq.analyze_constants(df)}
    assert kinds == {"c": "constant", "n": "near_constant", "e": "all_missing"}


def test_placeholders_but_not_negative_numbers():
    raw = pd.DataFrame({"a": ["1", "-5", "N/A", "unknown", "-", "?", "3"] * 10})
    out, rep = dq.normalize_placeholders(raw)
    assert out["a"].isna().sum() == 40  # N/A, unknown, -, ?
    assert out["a"].dropna().isin(["1", "-5", "3"]).all()
    assert rep[0]["count"] == 40


def test_formatting_detection():
    raw = pd.DataFrame({"m": ["$1,200"] * 40, "e": ["a@b.com"] * 40})
    _, norm, p, q = build_profile(raw, "f.csv", 1)
    kinds = {f["column"]: f["kind"] for f in q["formatting"]}
    assert kinds == {"m": "formatted_numeric", "e": "email_like"}
