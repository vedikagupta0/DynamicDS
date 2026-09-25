os.environ["DYNAMICDS_DATA_DIR"] = tempfile.mkdtemp(prefix="dynamicds_test_")
os.environ["AUTODS_DATA_DIR"] = os.environ["DYNAMICDS_DATA_DIR"]

import numpy as np
import pandas as pd
import pytest

from app.services.profiling import build_profile


@pytest.fixture(scope="session")
def churn_raw():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "age": rng.integers(18, 70, n).astype(float),
        "income": rng.lognormal(10, 1, n).round(2),
        "city": rng.choice([f"city{i}" for i in range(40)], n),
        "plan": rng.choice(["basic", "pro", "team"], n),
        "signup": pd.date_range("2022-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "const": "x",
    })
    z = (df.age - 40) / 15 + (df.income > df.income.median()) * 1.2 + (df.plan == "basic") * 0.8 + rng.normal(0, 1, n)
    df["churn"] = np.where(z > 1.4, "yes", "no")
    df["spend"] = (df.income * 0.01 + rng.normal(0, 5, n)).round(2)
    df.loc[rng.choice(n, 40, replace=False), "age"] = np.nan
    return df.astype(str).replace("nan", "")


@pytest.fixture(scope="session")
def churn(churn_raw):
    typed, norm, profile, quality = build_profile(churn_raw, "churn.csv", 1000)
    return {"raw": churn_raw, "typed": typed, "profile": profile, "quality": quality}


@pytest.fixture(scope="session")
def ts_raw():
    rng = np.random.default_rng(1)
    n = 300
    d = pd.date_range("2023-01-01", periods=n, freq="D")
    y = 100 + 0.1 * np.arange(n) + 10 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 2, n)
    return pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "sales": y.round(2).astype(str), "label": "a"})


def col(profile, name):
    return next(c for c in profile["columns"] if c["name"] == name)
