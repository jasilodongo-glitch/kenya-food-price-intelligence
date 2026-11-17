"""Synthetic fixtures. These exist ONLY for automated tests and are never shipped to the app."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def food_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", periods=60, freq="MS")
    rows = []
    for market, mkt_factor, county in [("Alpha", 1.00, "North"), ("Beta", 1.15, "South"), ("Gamma", 0.90, "North")]:
        for com, base, amp in [("Maize", 50, 0.20), ("Beans", 120, 0.08)]:
            for i, d in enumerate(dates):
                seasonal = 1 + amp * np.cos(2 * np.pi * (d.month - 3) / 12)  # peak in March
                trend = 1 + 0.004 * i
                price = base * mkt_factor * seasonal * trend * (1 + rng.normal(0, 0.02))
                rows.append({"date": d, "admin1": county, "market": market, "commodity": com, "unit": "KG",
                             "pricetype": "Retail", "currency": "KES", "price": round(price, 2)})
    return pd.DataFrame(rows)


@pytest.fixture
def messy_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Name ": [" a", "b ", "b ", "c", None, None],
        "Amount": ["1,000", "2,500", "2,500", "300", "4,000", None],
        "When": ["2021-01-05", "2021-02-10", "2021-02-10", "2021-03-15", "2021-04-20", None],
        "Empty": [None] * 6,
        "Score": [1.0, 2.0, 2.0, 3.0, 100.0, None],
    })


@pytest.fixture
def csv_bytes(food_df) -> bytes:
    return food_df.to_csv(index=False).encode()


@pytest.fixture
def xlsx_bytes(food_df) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame({"x": [1]}).to_excel(w, sheet_name="Notes", index=False)
        food_df.to_excel(w, sheet_name="Prices", index=False)
    return buf.getvalue()
