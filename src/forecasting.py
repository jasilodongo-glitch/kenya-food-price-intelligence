"""Forecasting with chronological evaluation and model comparison."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config

MODEL_NAMES = ["Naive", "Seasonal naive", "Moving average", "Exponential smoothing", "ARIMA"]
Z95 = 1.96


class ForecastError(ValueError):
    """Raised when the data cannot be forecast responsibly."""


@dataclass
class ForecastResult:
    metrics: pd.DataFrame
    test_forecasts: dict[str, pd.DataFrame]
    future: dict[str, pd.DataFrame]
    train: pd.Series
    test: pd.Series
    best_model: str
    notes: list[str] = field(default_factory=list)


def check_suitability(s: pd.Series, horizon: int) -> list[str]:
    """Raise ForecastError if unsuitable; otherwise return warnings."""
    clean = s.dropna()
    if len(clean) < config.MIN_FORECAST_POINTS:
        raise ForecastError(f"Forecasting needs at least {config.MIN_FORECAST_POINTS} time periods; this series has {len(clean)}.")
    if clean.nunique() < 3 or clean.std() == 0:
        raise ForecastError("The series is constant or nearly constant, so a forecast would not be informative.")
    if horizon < 1 or horizon > len(clean) // 2:
        raise ForecastError(f"Forecast horizon must be between 1 and {len(clean) // 2} for a series of {len(clean)} periods.")
    warns = []
    gaps = s.isna().sum()
    if gaps:
        warns.append(f"{gaps} missing period(s) were filled by linear interpolation before modelling.")
    if len(clean) < 3 * 12:
        warns.append("Short history: intervals are wide and seasonal estimates are unreliable.")
    return warns


def regularise(s: pd.Series) -> pd.Series:
    s = s.copy()
    if s.index.freq is None:
        s = s.asfreq(pd.infer_freq(s.index) or "MS")
    return s.interpolate(limit_direction="both")


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    err = actual - pred
    nz = actual != 0
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAPE (%)": float(np.mean(np.abs(err[nz] / actual[nz])) * 100) if nz.any() else np.nan}


def _future_index(train: pd.Series, steps: int) -> pd.DatetimeIndex:
    return pd.date_range(start=train.index[-1] + train.index.freq, periods=steps, freq=train.index.freq)


def _band(point: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    h = np.arange(1, len(point) + 1)
    w = Z95 * sigma * np.sqrt(h)
    return point - w, point + w


def _frame(idx, point, lower, upper) -> pd.DataFrame:
    return pd.DataFrame({"forecast": point, "lower": lower, "upper": upper}, index=idx)


def fit_forecast(name: str, train: pd.Series, steps: int, period: int | None) -> pd.DataFrame:
    idx = _future_index(train, steps)
    y = train.to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "Naive":
            point = np.repeat(y[-1], steps)
            sigma = np.std(np.diff(y), ddof=1)
            return _frame(idx, point, *_band(point, sigma))
        if name == "Seasonal naive":
            if not period or len(y) < 2 * period:
                raise ForecastError("Seasonal naive needs at least two full seasonal cycles.")
            point = np.array([y[-period + (i % period)] for i in range(steps)])
            sigma = np.std(y[period:] - y[:-period], ddof=1)
            return _frame(idx, point, *_band(point, sigma))
        if name == "Moving average":
            window = max(3, min(period or 3, len(y) // 4))
            point = np.repeat(y[-window:].mean(), steps)
            roll = pd.Series(y).rolling(window).mean().shift(1)
            sigma = float(np.nanstd(y - roll.to_numpy(), ddof=1))
            return _frame(idx, point, *_band(point, sigma))
        if name == "Exponential smoothing":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            seasonal = "add" if period and len(y) >= 2 * period else None
            fit = ExponentialSmoothing(train, trend="add", damped_trend=True, seasonal=seasonal,
                                       seasonal_periods=period if seasonal else None,
                                       initialization_method="estimated").fit()
            point = np.asarray(fit.forecast(steps), float)
            sigma = float(np.std(np.asarray(fit.resid), ddof=1))
            return _frame(idx, point, *_band(point, sigma))
        if name == "ARIMA":
            from statsmodels.tsa.arima.model import ARIMA
            best, best_aic = None, np.inf
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            fit = ARIMA(train, order=(p, d, q)).fit()
                        except Exception:  # noqa: BLE001 - skip non-converging orders
                            continue
                        if np.isfinite(fit.aic) and fit.aic < best_aic:
                            best, best_aic = fit, fit.aic
            if best is None:
                raise ForecastError("ARIMA could not be fitted to this series.")
            fc = best.get_forecast(steps)
            ci = np.asarray(fc.conf_int(alpha=0.05))
            return _frame(idx, np.asarray(fc.predicted_mean, float), ci[:, 0], ci[:, 1])
    raise ForecastError(f"Unknown model '{name}'.")


def run_forecast(series: pd.Series, horizon: int, models: list[str] | None = None, period: int | None = None) -> ForecastResult:
    notes = check_suitability(series, horizon)
    s = regularise(series)
    models = models or MODEL_NAMES
    test_size = max(3, min(horizon, len(s) // 4))
    train, test = s.iloc[:-test_size], s.iloc[-test_size:]
    rows, test_fc, future = [], {}, {}
    for name in models:
        try:
            tf = fit_forecast(name, train, test_size, period)
            ff = fit_forecast(name, s, horizon, period)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{name} skipped: {exc}")
            continue
        m = metrics(test.to_numpy(), tf["forecast"].to_numpy())
        coverage = float(((test.to_numpy() >= tf["lower"].to_numpy()) & (test.to_numpy() <= tf["upper"].to_numpy())).mean() * 100)
        rows.append({"model": name, **m, "95% interval coverage (%)": coverage})
        test_fc[name], future[name] = tf, ff
    if not rows:
        raise ForecastError("No model could be fitted to this series.")
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    notes.append(f"Models were trained on the first {len(train)} periods and scored on the last {len(test)} (chronological split, no shuffling).")
    return ForecastResult(metrics=table, test_forecasts=test_fc, future=future, train=train, test=test,
                          best_model=str(table.iloc[0]["model"]), notes=notes)
