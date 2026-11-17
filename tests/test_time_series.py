import numpy as np
import pandas as pd
import pytest

from src import time_series as ts


@pytest.fixture
def monthly():
    idx = pd.date_range("2016-01-01", periods=72, freq="MS")
    vals = 100 + np.arange(72) * 1.0 + 20 * np.cos(2 * np.pi * (idx.month - 6) / 12)  # peak June
    return pd.DataFrame({"d": idx, "v": vals})


def test_prepare_series_frequencies(monthly):
    assert len(ts.prepare_series(monthly, "d", "v", "MS")) == 72
    assert len(ts.prepare_series(monthly, "d", "v", "YS")) == 6
    assert ts.prepare_series(monthly, "d", "v", "YS", "sum").iloc[0] == pytest.approx(monthly["v"].iloc[:12].sum())


def test_gaps_stay_visible():
    d = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-04-01", "2020-05-01"]), "v": [1, 2, 4, 5]})
    s = ts.prepare_series(d, "d", "v", "MS")
    assert len(s) == 5 and s.isna().sum() == 1


def test_unsuitable_data_errors():
    with pytest.raises(ts.TimeSeriesError):
        ts.prepare_series(pd.DataFrame({"d": ["x", "y"], "v": [1, 2]}), "d", "v")
    with pytest.raises(ts.TimeSeriesError):
        ts.prepare_series(pd.DataFrame({"d": pd.to_datetime(["2020-01-01"]), "v": [1]}), "d", "v")


def test_moving_average_and_pct_change():
    s = pd.Series([10.0, 20, 30, 40], index=pd.date_range("2020-01-01", periods=4, freq="MS"))
    assert ts.moving_average(s, 2).iloc[-1] == 35
    assert ts.percent_change(s).iloc[1] == pytest.approx(100)
    assert ts.percent_change(s, 2).iloc[2] == pytest.approx(200)


def test_growth_summary(monthly):
    s = ts.prepare_series(monthly, "d", "v")
    g = ts.growth_summary(s)
    assert g["total_change_pct"] == pytest.approx((s.iloc[-1] / s.iloc[0] - 1) * 100)
    assert g["cagr_pct"] > 0 and g["peak_value"] == s.max()


def test_seasonal_profile_finds_peak(monthly):
    s = ts.prepare_series(monthly, "d", "v")
    # remove trend crudely to keep the test about seasonality
    detr = s - np.arange(len(s))
    p = ts.seasonal_profile(detr)
    assert int(p.loc[p["index"].idxmax(), "month"]) == 6 and len(p) == 12


def test_seasonality_needs_two_years():
    s = pd.Series(range(10), index=pd.date_range("2020-01-01", periods=10, freq="MS"), dtype=float)
    with pytest.raises(ts.TimeSeriesError):
        ts.seasonal_profile(s)


def test_decompose(monthly):
    s = ts.prepare_series(monthly, "d", "v")
    dec = ts.decompose(s, 12)
    assert set(dec.columns) == {"observed", "trend", "seasonal", "residual"}
    assert dec["seasonal"].iloc[:12].abs().max() > 10          # recovers the ~20 amplitude
    with pytest.raises(ts.TimeSeriesError):
        ts.decompose(s.iloc[:15], 12)
    with pytest.raises(ts.TimeSeriesError):
        ts.decompose(s, None)


def test_year_tables(monthly):
    s = ts.prepare_series(monthly, "d", "v")
    yoy = ts.year_over_year(s)
    assert len(yoy) == 6 and pd.isna(yoy["yoy_change_pct"].iloc[0]) and yoy["yoy_change_pct"].iloc[1] > 0
    assert ts.year_month_table(s).shape == (6, 12)
