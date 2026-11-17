import numpy as np
import pandas as pd
import pytest

from src import forecasting as fc, time_series as ts


@pytest.fixture
def series():
    idx = pd.date_range("2015-01-01", periods=84, freq="MS")
    rng = np.random.default_rng(7)
    return pd.Series(100 + 0.5 * np.arange(84) + 15 * np.sin(2 * np.pi * np.arange(84) / 12) + rng.normal(0, 1.5, 84), index=idx, name="price")


def test_metrics_exact():
    m = fc.metrics(np.array([100, 200]), np.array([110, 180]))
    assert m["MAE"] == 15 and m["RMSE"] == pytest.approx(np.sqrt(250)) and m["MAPE (%)"] == pytest.approx(10)
    assert np.isnan(fc.metrics(np.array([0.0]), np.array([1.0]))["MAPE (%)"])


def test_run_forecast_all_models(series):
    res = fc.run_forecast(series, 6, period=12)
    assert set(res.metrics["model"]) == set(fc.MODEL_NAMES)
    assert res.metrics["RMSE"].is_monotonic_increasing and res.best_model == res.metrics.iloc[0]["model"]
    for name, f in res.future.items():
        assert len(f) == 6 and (f["lower"] <= f["forecast"]).all() and (f["forecast"] <= f["upper"]).all()
        assert f.index[0] == series.index[-1] + pd.DateOffset(months=1)


def test_split_is_chronological_and_no_leakage(series):
    res = fc.run_forecast(series, 6, period=12)
    assert res.train.index.max() < res.test.index.min()
    assert len(res.train) + len(res.test) == len(series)
    assert res.test.index.equals(series.index[-len(res.test):])


def test_seasonal_models_beat_naive_on_seasonal_data(series):
    m = fc.run_forecast(series, 12, period=12).metrics.set_index("model")
    assert m.loc["Seasonal naive", "RMSE"] < m.loc["Naive", "RMSE"]
    assert m.loc["Exponential smoothing", "RMSE"] < m.loc["Naive", "RMSE"]


def test_arima_and_ets_skip_gracefully_when_requested_subset(series):
    res = fc.run_forecast(series, 3, ["Naive", "ARIMA"], period=12)
    assert list(sorted(res.metrics["model"])) == ["ARIMA", "Naive"]


def test_refuses_short_constant_and_long_horizon(series):
    with pytest.raises(fc.ForecastError, match="at least"):
        fc.run_forecast(series.iloc[:10], 3)
    with pytest.raises(fc.ForecastError, match="constant"):
        fc.run_forecast(pd.Series(5.0, index=series.index), 3)
    with pytest.raises(fc.ForecastError, match="horizon"):
        fc.run_forecast(series, 60)


def test_gaps_are_interpolated_and_reported(series):
    gappy = series.copy()
    gappy.iloc[20:23] = np.nan
    res = fc.run_forecast(gappy, 3, ["Naive"])
    assert any("interpolation" in n for n in res.notes)


def test_end_to_end_from_dataframe(series):
    df = pd.DataFrame({"when": series.index, "price": series.values})
    s = ts.prepare_series(df, "when", "price", "MS")
    assert fc.run_forecast(s, 6, ["Naive", "Exponential smoothing"], 12).best_model in {"Naive", "Exponential smoothing"}
