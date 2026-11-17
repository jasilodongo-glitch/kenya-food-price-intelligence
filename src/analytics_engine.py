"""High-level dataset insights built only from the active data."""
from __future__ import annotations

import pandas as pd

from src.data_profiler import DatasetProfile, columns_by_kind, outlier_summary


def key_indicators(df: pd.DataFrame, numeric_cols: list[str], limit: int = 4) -> list[dict]:
    """Choose the numeric columns with the most data and report mean/median."""
    ranked = sorted(numeric_cols, key=lambda c: df[c].notna().sum(), reverse=True)[:limit]
    out = []
    for c in ranked:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if not s.empty:
            out.append({"column": c, "mean": float(s.mean()), "median": float(s.median()), "min": float(s.min()), "max": float(s.max())})
    return out


def auto_observations(df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    """Plain-language observations; every figure is computed from the data."""
    obs: list[str] = []
    obs.append(f"The dataset has {profile.rows:,} records and {profile.columns} variables.")
    if profile.missing_pct == 0:
        obs.append("No values are missing.")
    else:
        obs.append(f"{profile.missing_pct:.1f}% of all cells are missing.")
    if profile.duplicates:
        obs.append(f"{profile.duplicates:,} rows are exact duplicates ({profile.duplicates / max(profile.rows, 1) * 100:.1f}% of records).")
    if profile.date_range and profile.date_column:
        a, b = profile.date_range
        obs.append(f"Dates in '{profile.date_column}' run from {a:%d %b %Y} to {b:%d %b %Y}.")
    cols = columns_by_kind(df, profile.kinds)
    nums = cols.get("numeric", [])
    if nums:
        out = outlier_summary(df, nums)
        if not out.empty and out["outliers"].sum() > 0:
            worst = out.sort_values("outlier_pct", ascending=False).iloc[0]
            obs.append(f"'{worst['column']}' has the most statistical outliers: {int(worst['outliers']):,} values ({worst['outlier_pct']:.1f}%) fall outside the 1.5 x IQR range.")
    for c in cols.get("categorical", [])[:2]:
        vc = df[c].value_counts()
        if len(vc):
            obs.append(f"'{c}' has {len(vc)} distinct values; the most common is '{vc.index[0]}' ({vc.iloc[0] / len(df) * 100:.1f}% of records).")
    return obs
