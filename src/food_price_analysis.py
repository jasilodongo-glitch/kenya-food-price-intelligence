"""Food-price specific analytics. Only analyses supported by the detected columns are offered."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import time_series as ts
from src.data_profiler import detect_column_kinds

ROLE_KEYWORDS = {
    "commodity": ["commodity", "item", "product", "crop", "food", "good"],
    "market": ["market", "town", "location", "site"],
    "county": ["county", "admin1", "region", "province", "state"],
    "country": ["country", "nation", "admin0"],
    "date": ["date", "month", "period", "time", "year"],
    "price": ["price", "cost", "value", "amount"],
    "currency": ["currency", "curr"],
    "unit": ["unit", "uom", "measure", "weight"],
    "price_type": ["pricetype", "price_type", "price type", "observation", "obs_type", "type"],
}
EXPECTED_KIND = {"commodity": {"categorical", "text"}, "market": {"categorical", "text"}, "county": {"categorical", "text"},
                 "country": {"categorical", "text"}, "date": {"datetime"}, "price": {"numeric"}, "currency": {"categorical", "text"},
                 "unit": {"categorical", "text"}, "price_type": {"categorical", "text"}}
ROLE_LABELS = {"commodity": "Commodity", "market": "Market", "county": "County / region", "country": "Country", "date": "Date",
               "price": "Price", "currency": "Currency", "unit": "Measurement unit", "price_type": "Observation type"}

AFFORDABILITY_NOTE = ("Price trends are not household affordability. Affordability depends on incomes, expenditure and "
                      "household size, which are not in a price dataset. Treat these results as price signals only.")


class FoodPriceError(ValueError):
    """Raised when requested analysis is not supported by the data."""


def _score(col: str, keywords: list[str]) -> int:
    c = col.lower().replace("-", "_").strip()
    best = 0
    for k in keywords:
        if c == k:
            best = max(best, 10)
        elif c.startswith(k) or c.endswith(k):
            best = max(best, 6)
        elif k in c:
            best = max(best, 3)
    return best


def detect_roles(df: pd.DataFrame, kinds: dict[str, str] | None = None) -> dict[str, str | None]:
    """Map food-price roles to columns using names AND detected data types. No fixed schema is assumed."""
    kinds = kinds or detect_column_kinds(df)
    roles: dict[str, str | None] = {}
    used: set[str] = set()
    # Resolve roles with most specific keywords first so 'price_type' does not steal 'price'.
    order = ["price_type", "currency", "unit", "commodity", "market", "county", "country", "date", "price"]
    for role in order:
        cands = []
        for col in df.columns:
            if col in used or kinds.get(col) not in EXPECTED_KIND[role]:
                continue
            sc = _score(col, ROLE_KEYWORDS[role])
            if role == "price" and "usd" in col.lower():
                sc = max(sc - 4, 1)  # prefer local-currency price over usdprice
            if sc > 0:
                cands.append((sc, col))
        if cands:
            cands.sort(key=lambda x: (-x[0], list(df.columns).index(x[1])))
            roles[role] = cands[0][1]
            used.add(cands[0][1])
        else:
            roles[role] = None
    return roles


def available_analyses(roles: dict[str, str | None]) -> dict[str, bool]:
    r = {k for k, v in roles.items() if v}
    return {
        "market": {"price", "market"} <= r,
        "geography": {"price", "county"} <= r or {"price", "country"} <= r,
        "commodity": {"price", "commodity"} <= r,
        "volatility": {"price", "commodity", "date"} <= r,
        "trends": {"price", "date"} <= r,
        "seasonality": {"price", "date"} <= r,
        "forecasting": {"price", "date"} <= r,
    }


def standardise(df: pd.DataFrame, roles: dict[str, str | None]) -> pd.DataFrame:
    """Return a frame with canonical role names, typed price/date, rows lacking price dropped."""
    cols = {role: col for role, col in roles.items() if col}
    if "price" not in cols:
        raise FoodPriceError("No price column was detected. Map one in the column selector.")
    out = df[list(cols.values())].copy()
    out.columns = list(cols.keys())
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    if "date" in out:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", format="mixed")
    out = out.dropna(subset=["price"])
    for c in out.columns:
        if c not in ("price", "date"):
            out[c] = out[c].astype("string").str.strip()
    return out[out["price"] > 0] if (out["price"] > 0).any() else out


def dominant_filter(f: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Restrict to the most common unit / currency / observation type so prices are comparable."""
    notes = []
    for role in ("unit", "currency", "price_type"):
        if role in f and f[role].nunique() > 1:
            top = f[role].value_counts().index[0]
            notes.append(f"Several {ROLE_LABELS[role].lower()} values were present; analysis uses the most common ('{top}').")
            f = f[f[role] == top]
    return f, notes


def commodity_subset(f: pd.DataFrame, commodity: str | None) -> tuple[pd.DataFrame, list[str]]:
    if commodity is not None and "commodity" in f:
        f = f[f["commodity"] == commodity]
    if f.empty:
        raise FoodPriceError("No records match the selected commodity.")
    return dominant_filter(f)


def unit_label(f: pd.DataFrame) -> str:
    parts = [str(f[r].iloc[0]) for r in ("currency", "unit") if r in f and not f[r].dropna().empty]
    return " per ".join(parts) if len(parts) == 2 else (parts[0] if parts else "price")


def location_ranking(f: pd.DataFrame, by: str, commodity: str | None = None, min_obs: int = 3) -> tuple[pd.DataFrame, list[str]]:
    """Rank locations (market / county / country) by average price for ONE commodity."""
    if by not in f:
        raise FoodPriceError(f"No '{by}' column is available.")
    if commodity is None and "commodity" in f and f["commodity"].nunique() > 1:
        raise FoodPriceError("Select a single commodity: prices of different foods are not comparable.")
    sub, notes = commodity_subset(f, commodity)
    g = sub.groupby(by)["price"].agg(["count", "mean", "median", "min", "max"]).reset_index()
    g = g[g["count"] >= min_obs]
    if len(g) < 2:
        raise FoodPriceError(f"Fewer than 2 locations have at least {min_obs} observations.")
    overall = sub["price"].mean()
    g["vs_average_pct"] = (g["mean"] / overall - 1) * 100
    g = g.sort_values("mean", ascending=False).reset_index(drop=True)
    g["rank"] = np.arange(1, len(g) + 1)
    return g, notes


def monthly_series(f: pd.DataFrame, commodity: str | None = None, market: str | None = None) -> tuple[pd.Series, list[str]]:
    sub, notes = commodity_subset(f, commodity)
    if market and "market" in sub:
        sub = sub[sub["market"] == market]
    if "date" not in sub:
        raise FoodPriceError("No date column was detected.")
    return ts.prepare_series(sub, "date", "price", "MS", "mean"), notes


def volatility_table(f: pd.DataFrame, min_months: int = 12) -> pd.DataFrame:
    """Volatility = standard deviation of month-on-month % change; CV = std / mean of price level."""
    if not {"commodity", "date"} <= set(f.columns):
        raise FoodPriceError("Volatility needs commodity and date columns.")
    rows = []
    for com in f["commodity"].dropna().unique():
        try:
            s, _ = monthly_series(f, com)
        except (FoodPriceError, ts.TimeSeriesError):
            continue
        s = s.dropna()
        if len(s) < min_months:
            continue
        ch = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna() * 100
        rows.append({"commodity": com, "months": len(s), "mean_price": s.mean(), "cv_pct": s.std() / s.mean() * 100,
                     "monthly_volatility_pct": ch.std(), "max_monthly_rise_pct": ch.max(), "max_monthly_fall_pct": ch.min()})
    if not rows:
        raise FoodPriceError(f"No commodity has at least {min_months} months of data.")
    return pd.DataFrame(rows).sort_values("monthly_volatility_pct", ascending=False).reset_index(drop=True)


def price_index(f: pd.DataFrame, commodities: list[str]) -> pd.DataFrame:
    """Rebase each commodity's monthly average price to 100 at the first month all share, enabling fair comparison."""
    frames = {}
    for c in commodities:
        s, _ = monthly_series(f, c)
        frames[c] = s
    wide = pd.DataFrame(frames).dropna()
    if wide.empty:
        raise FoodPriceError("The selected commodities share no common months.")
    return wide / wide.iloc[0] * 100


def latest_change_table(f: pd.DataFrame) -> pd.DataFrame:
    """Latest month vs 3 and 12 months earlier, per commodity (same unit basis)."""
    rows = []
    for com in f["commodity"].dropna().unique():
        try:
            s, _ = monthly_series(f, com)
        except (FoodPriceError, ts.TimeSeriesError):
            continue
        s = s.dropna()
        if len(s) < 2:
            continue
        last = s.iloc[-1]
        def back(m):
            t = s.index[-1] - pd.DateOffset(months=m)
            return s.get(t, np.nan)
        rows.append({"commodity": com, "latest_month": s.index[-1], "latest_price": last,
                     "change_3m_pct": (last / back(3) - 1) * 100 if pd.notna(back(3)) else np.nan,
                     "change_12m_pct": (last / back(12) - 1) * 100 if pd.notna(back(12)) else np.nan})
    return pd.DataFrame(rows)
