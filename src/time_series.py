"""Time-series preparation and descriptive analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

FREQUENCIES = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS", "Annual": "YS"}
SEASONAL_PERIODS = {"D": 7, "W-MON": 52, "MS": 12, "YS": None}


class TimeSeriesError(ValueError):
    """Raised when data is unsuitable for the requested time-series operation."""


def prepare_series(df: pd.DataFrame, date_col: str, value_col: str, freq: str = "MS", agg: str = "mean") -> pd.Series:
    """Aggregate a value column to a regular time grid. Gaps remain NaN (never silently filled)."""
    if agg not in {"mean", "median", "sum", "min", "max"}:
        raise TimeSeriesError(f"Unsupported aggregation '{agg}'.")
    work = pd.DataFrame({
        "d": pd.to_datetime(df[date_col], errors="coerce", format="mixed"),
        "v": pd.to_numeric(df[value_col], errors="coerce"),
    }).dropna()
    if work.empty:
        raise TimeSeriesError("No rows have both a valid date and a numeric value.")
    series = work.set_index("d")["v"].sort_index().resample(freq).agg(agg)
    if series.notna().sum() < 3:
        raise TimeSeriesError("Fewer than 3 time periods contain data at this frequency.")
    series.name = value_col
    return series


def moving_average(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window=window, min_periods=max(1, window // 2)).mean()


def percent_change(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.pct_change(periods=periods, fill_method=None) * 100


def growth_summary(s: pd.Series) -> dict:
    clean = s.dropna()
    if len(clean) < 2:
        raise TimeSeriesError("At least two data points are needed to measure growth.")
    first, last = float(clean.iloc[0]), float(clean.iloc[-1])
    years = (clean.index[-1] - clean.index[0]).days / 365.25
    total = (last / first - 1) * 100 if first != 0 else np.nan
    cagr = ((last / first) ** (1 / years) - 1) * 100 if first > 0 and last > 0 and years >= 1 else np.nan
    return {"start_date": clean.index[0], "end_date": clean.index[-1], "start_value": first, "end_value": last,
            "total_change_pct": total, "cagr_pct": cagr, "peak_date": clean.idxmax(), "peak_value": float(clean.max()),
            "trough_date": clean.idxmin(), "trough_value": float(clean.min())}


def seasonal_profile(s: pd.Series) -> pd.DataFrame:
    """Average value by calendar month, expressed as an index (100 = overall average)."""
    clean = s.dropna()
    if clean.index.to_series().dt.year.nunique() < 2:
        raise TimeSeriesError("Seasonal patterns need at least 2 calendar years of data.")
    by_month = clean.groupby(clean.index.month).agg(["mean", "count"])
    overall = clean.mean()
    by_month["index"] = by_month["mean"] / overall * 100 if overall else np.nan
    by_month.index.name = "month"
    return by_month.reset_index()


def year_month_table(s: pd.Series) -> pd.DataFrame:
    clean = s.dropna()
    return clean.groupby([clean.index.year.rename("year"), clean.index.month.rename("month")]).mean().unstack("month")


def year_over_year(s: pd.Series) -> pd.DataFrame:
    clean = s.dropna()
    yearly = clean.groupby(clean.index.year).agg(["mean", "min", "max", "count"])
    yearly["yoy_change_pct"] = yearly["mean"].pct_change() * 100
    yearly.index.name = "year"
    return yearly.reset_index()


def decompose(s: pd.Series, period: int | None) -> pd.DataFrame:
    from statsmodels.tsa.seasonal import seasonal_decompose
    if not period:
        raise TimeSeriesError("Decomposition needs a seasonal period (not available for annual data).")
    if len(s) < config.MIN_DECOMPOSE_PERIODS * period:
        raise TimeSeriesError(f"Decomposition needs at least {config.MIN_DECOMPOSE_PERIODS * period} periods ({config.MIN_DECOMPOSE_PERIODS} full cycles); found {len(s)}.")
    filled = s.interpolate(limit_direction="both")
    res = seasonal_decompose(filled, model="additive", period=period)
    return pd.DataFrame({"observed": res.observed, "trend": res.trend, "seasonal": res.seasonal, "residual": res.resid})
