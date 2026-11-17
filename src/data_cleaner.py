"""User-selected cleaning operations. Always returns a new DataFrame plus a log."""
from __future__ import annotations

import pandas as pd

from src.data_profiler import detect_column_kinds, iqr_bounds

OPERATIONS = {
    "drop_duplicates": "Remove duplicate rows",
    "drop_empty_rows": "Drop fully empty rows",
    "drop_empty_columns": "Drop fully empty columns",
    "strip_whitespace": "Trim spaces in text columns",
    "standardize_columns": "Standardise column names (lower_snake_case)",
    "convert_numeric": "Convert numeric-looking text columns to numbers",
    "convert_dates": "Convert date-looking text columns to dates",
    "fill_numeric_median": "Fill missing numeric values with the median",
    "drop_missing_rows": "Drop rows that contain any missing value",
    "remove_outliers_iqr": "Remove numeric outlier rows (1.5 x IQR rule)",
}


def _is_text(s: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s))


def _standardize(name: str) -> str:
    import re
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower() or "column"


def apply_operations(df: pd.DataFrame, operations: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Apply the chosen operations in a safe, fixed order. The input frame is never mutated."""
    out = df.copy()
    log: list[str] = []
    order = list(OPERATIONS)
    for op in sorted(set(operations), key=lambda o: order.index(o) if o in order else 99):
        if op not in OPERATIONS:
            raise ValueError(f"Unknown cleaning operation: {op}")
        before = out.shape
        if op == "drop_duplicates":
            out = out.drop_duplicates()
        elif op == "drop_empty_rows":
            out = out.dropna(how="all")
        elif op == "drop_empty_columns":
            out = out.dropna(axis=1, how="all")
        elif op == "strip_whitespace":
            for c in [c for c in out.columns if _is_text(out[c])]:
                mask = out[c].notna()
                out.loc[mask, c] = out.loc[mask, c].astype(str).str.strip()
        elif op == "standardize_columns":
            cols, seen = [], {}
            for c in out.columns:
                n = _standardize(c)
                seen[n] = seen.get(n, -1) + 1
                cols.append(n if seen[n] == 0 else f"{n}_{seen[n]}")
            out.columns = cols
        elif op == "convert_numeric":
            kinds = detect_column_kinds(out)
            for c in [c for c, k in kinds.items() if k == "numeric" and not pd.api.types.is_numeric_dtype(out[c])]:
                out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
        elif op == "convert_dates":
            kinds = detect_column_kinds(out)
            for c in [c for c, k in kinds.items() if k == "datetime" and not pd.api.types.is_datetime64_any_dtype(out[c])]:
                out[c] = pd.to_datetime(out[c], errors="coerce", format="mixed")
        elif op == "fill_numeric_median":
            for c in out.select_dtypes("number").columns:
                if out[c].isna().any():
                    out[c] = out[c].fillna(out[c].median())
        elif op == "drop_missing_rows":
            out = out.dropna()
        elif op == "remove_outliers_iqr":
            keep = pd.Series(True, index=out.index)
            for c in out.select_dtypes("number").columns:
                s = out[c].dropna()
                if len(s) >= 4:
                    lo, hi = iqr_bounds(s)
                    keep &= out[c].isna() | out[c].between(lo, hi)
            out = out[keep]
        log.append(f"{OPERATIONS[op]}: {before[0]:,}x{before[1]} -> {out.shape[0]:,}x{out.shape[1]}")
    return out.reset_index(drop=True), log
