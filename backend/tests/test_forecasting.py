import numpy as np
import pandas as pd
import pytest

from app.services import evaluation as ev
from app.services import forecasting as fc
from app.services.profiling import build_profile


@pytest.fixture(scope="module")
def fitted(ts_raw):
    typed, _, p, _ = build_profile(ts_raw, "s.csv", 1)
    res, art = fc.train_forecast(typed, p, {"datetime_column": "date", "target": "sales", "horizon": 5})
    return typed, p, res, art


def test_chronological_split_no_shuffle():
    a, b = ev.chronological_split(100)
    assert (a, b) == (70, 85)
    with pytest.raises(ValueError):
        ev.chronological_split(3)


def test_windows_do_not_overlap_and_are_ordered(fitted):
    _, _, res, _ = fitted
    s = res["split"]
    assert s["train"]["end"] < s["validation"]["start"] and s["validation"]["end"] < s["test"]["start"]
    assert s["train"]["n"] + s["validation"]["n"] + s["test"]["n"] == s["n"]
    assert "leaking" in s["explanation"]


def test_baselines_always_present_and_compared(fitted):
    _, _, res, art = fitted
    keys = [m["key"] for m in res["models"]]
    assert "naive" in keys and "snaive" in keys and res["season_period"] == 7
    lead = next(m for m in res["models"] if m["leader"])
    assert not lead["is_baseline"] and "improvement_vs_best_baseline_pct" in lead
    assert isinstance(res["beats_baseline"], bool) and res["baseline_note"]
    assert set(lead["test"]) == {"mae", "rmse", "mape", "smape"}


def test_shuffling_input_does_not_change_result(ts_raw):
    typed, _, p, _ = build_profile(ts_raw, "s.csv", 1)
    a, _ = fc.train_forecast(typed, p, {"datetime_column": "date", "target": "sales", "models": None})
    b, _ = fc.train_forecast(typed.sample(frac=1, random_state=3), p, {"datetime_column": "date", "target": "sales"})
    assert a["models"][0]["test"] == b["models"][0]["test"]  # data is sorted by time internally


def test_lag_model_uses_no_future_information():
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    y = np.arange(80, dtype=float)
    lm = fc.LagModel(None).fit(y[:60], idx[:60])
    # changing "future" actuals must not change the forecast: the model only sees history
    y2 = y.copy(); y2[60:] = -999
    lm2 = fc.LagModel(None).fit(y2[:60], idx[:60])
    assert np.allclose(lm.forecast(idx[60:]), lm2.forecast(idx[60:]))


def test_validation_rules(ts_raw):
    typed, _, p, _ = build_profile(ts_raw, "s.csv", 1)
    with pytest.raises(ValueError, match="different"):
        fc.train_forecast(typed, p, {"datetime_column": "date", "target": "date"})
    with pytest.raises(ValueError, match="numerical"):
        fc.train_forecast(typed, p, {"datetime_column": "date", "target": "label"})
    with pytest.raises(ValueError, match="not a datetime"):
        fc.train_forecast(typed, p, {"datetime_column": "sales", "target": "label"})


def test_gaps_and_duplicates_are_reported(ts_raw):
    raw = ts_raw.drop(index=[10, 11, 50]).copy()
    raw = pd.concat([raw, raw.iloc[[5]]])
    typed, _, p, _ = build_profile(raw, "s.csv", 1)
    res, _ = fc.train_forecast(typed, p, {"datetime_column": "date", "target": "sales"})
    acts = {d["action"]: d["count"] for d in res["decisions"]}
    assert acts["filled"] == 3 and acts["aggregated"] == 1


def test_future_forecast(fitted):
    _, _, _, art = fitted
    out = fc.forecast_future(art, 4)
    assert len(out["yhat"]) == 4 and out["t"][0] > art["index"][-1]
    with pytest.raises(ValueError):
        fc.forecast_future(art, 0)


def test_monthly_frequency_inferred():
    d = pd.date_range("2015-01-31", periods=48, freq="ME")
    raw = pd.DataFrame({"d": d.strftime("%d/%m/%Y"), "v": (50 + np.arange(48)).astype(str)})
    typed, _, p, _ = build_profile(raw, "m.csv", 1)
    s, _, fi = fc.prepare_series(typed, "d", "v")
    assert fi["freq"] == "MS" and len(s) == 48
