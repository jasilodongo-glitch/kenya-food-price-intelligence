"""Dataset profiling: column types, missing values, duplicates, outliers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def detect_column_kinds(df: pd.DataFrame) -> dict[str, str]:
    """Classify each column as numeric / datetime / categorical / text / boolean / empty."""
    kinds: dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        if non_null.empty:
            kinds[col] = "empty"
        elif pd.api.types.is_bool_dtype(s):
            kinds[col] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kinds[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            kinds[col] = "numeric"
        else:
            kinds[col] = _classify_object(non_null)
    return kinds


def _classify_object(non_null: pd.Series) -> str:
    sample = non_null.astype(str).head(2000)
    numeric = pd.to_numeric(sample.str.replace(",", "", regex=False), errors="coerce")
    if numeric.notna().mean() >= 0.95:
        return "numeric"
    parsed = _try_parse_dates(sample)
    if parsed is not None and parsed.notna().mean() >= 0.9:
        return "datetime"
    ratio = non_null.nunique() / max(len(non_null), 1)
    if non_null.nunique() <= 50 or ratio < 0.5:
        return "categorical"
    return "text"


def _try_parse_dates(sample: pd.Series) -> pd.Series | None:
    """Parse strings as dates, but only if they look date-like (avoids '1'->date)."""
    looks_dated = sample.str.contains(r"\d{1,4}[-/.\s][A-Za-z0-9]{1,9}[-/.\s]\d{1,4}", regex=True).mean()
    if looks_dated < 0.8:
        return None
    try:
        return pd.to_datetime(sample, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return None


def columns_by_kind(df: pd.DataFrame, kinds: dict[str, str] | None = None) -> dict[str, list[str]]:
    kinds = kinds or detect_column_kinds(df)
    out: dict[str, list[str]] = {}
    for col, kind in kinds.items():
        out.setdefault(kind, []).append(col)
    return out


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    miss = df.isna().sum()
    out = pd.DataFrame({"column": miss.index, "missing": miss.values})
    out["missing_pct"] = (out["missing"] / n * 100).round(2) if n else 0.0
    return out.sort_values("missing", ascending=False).reset_index(drop=True)


def missing_pct_overall(df: pd.DataFrame) -> float:
    total = df.shape[0] * df.shape[1]
    return float(df.isna().sum().sum() / total * 100) if total else 0.0


def duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def outlier_summary(df: pd.DataFrame, numeric_cols: list[str], k: float = 1.5) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 4:
            continue
        lo, hi = iqr_bounds(s, k)
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({"column": col, "lower_bound": lo, "upper_bound": hi,
                     "outliers": n_out, "outlier_pct": round(n_out / len(s) * 100, 2)})
    return pd.DataFrame(rows, columns=["column", "lower_bound", "upper_bound", "outliers", "outlier_pct"])


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    kinds: dict[str, str]
    missing_pct: float
    duplicates: int
    memory_mb: float
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None
    date_column: str | None


def date_coverage(df: pd.DataFrame, kinds: dict[str, str]) -> tuple[str | None, tuple | None]:
    for col, kind in kinds.items():
        if kind == "datetime":
            s = pd.to_datetime(df[col], errors="coerce", format="mixed").dropna()
            if not s.empty:
                return col, (s.min(), s.max())
    return None, None


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    kinds = detect_column_kinds(df)
    date_col, rng = date_coverage(df, kinds)
    return DatasetProfile(
        rows=len(df), columns=df.shape[1], kinds=kinds,
        missing_pct=missing_pct_overall(df), duplicates=duplicate_count(df),
        memory_mb=float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        date_range=rng, date_column=date_col,
    )


def column_profile(df: pd.DataFrame, kinds: dict[str, str]) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({"column": col, "detected_type": kinds.get(col, "unknown"), "dtype": str(s.dtype),
                     "unique": int(s.nunique(dropna=True)), "missing": int(s.isna().sum()),
                     "example": "" if s.dropna().empty else str(s.dropna().iloc[0])[:40]})
    return pd.DataFrame(rows)


def recommendations(df: pd.DataFrame, kinds: dict[str, str]) -> list[dict]:
    """Cleaning recommendations derived from the data (each maps to a cleaner operation key)."""
    recs: list[dict] = []
    dups = duplicate_count(df)
    if dups:
        recs.append({"key": "drop_duplicates", "title": f"Remove {dups:,} duplicate rows",
                     "detail": "Exact duplicate rows can inflate counts and bias averages."})
    empty_cols = [c for c, k in kinds.items() if k == "empty"]
    if empty_cols:
        recs.append({"key": "drop_empty_columns", "title": f"Drop {len(empty_cols)} empty column(s)",
                     "detail": ", ".join(empty_cols[:8])})
    if df.isna().all(axis=1).any():
        recs.append({"key": "drop_empty_rows", "title": "Drop fully empty rows", "detail": "Rows with no values at all."})
    str_cols = [c for c in df.columns if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]))]
    padded = [c for c in str_cols if df[c].dropna().astype(str).map(lambda v: v != v.strip()).any()]
    if padded:
        recs.append({"key": "strip_whitespace", "title": f"Trim stray spaces in {len(padded)} column(s)",
                     "detail": ", ".join(padded[:8])})
    to_numeric = [c for c, k in kinds.items() if k == "numeric" and not pd.api.types.is_numeric_dtype(df[c])]
    if to_numeric:
        recs.append({"key": "convert_numeric", "title": f"Convert {len(to_numeric)} text column(s) to numbers",
                     "detail": ", ".join(to_numeric[:8])})
    to_date = [c for c, k in kinds.items() if k == "datetime" and not pd.api.types.is_datetime64_any_dtype(df[c])]
    if to_date:
        recs.append({"key": "convert_dates", "title": f"Convert {len(to_date)} text column(s) to dates",
                     "detail": ", ".join(to_date[:8])})
    num_cols = [c for c, k in kinds.items() if k == "numeric"]
    miss_num = [c for c in num_cols if df[c].isna().any()]
    if miss_num:
        recs.append({"key": "fill_numeric_median", "title": f"Fill missing numbers (median) in {len(miss_num)} column(s)",
                     "detail": "Optional. Imputation changes the data; review before applying."})
    return recs
