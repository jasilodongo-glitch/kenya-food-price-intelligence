import numpy as np
import pandas as pd
import pytest

from src import data_cleaner as cl, data_profiler as dp


def test_kind_detection(messy_df, food_df):
    k = dp.detect_column_kinds(messy_df)
    assert k["Amount"] == "numeric" and k["When"] == "datetime" and k["Empty"] == "empty"
    kf = dp.detect_column_kinds(food_df)
    assert kf["date"] == "datetime" and kf["price"] == "numeric" and kf["market"] == "categorical"


def test_mostly_text_column_is_not_forced_numeric(messy_df):
    junk = pd.DataFrame({"v": ["1", "2", "x", "y", "z", "w"]})
    assert dp.detect_column_kinds(junk)["v"] != "numeric"


def test_missing_summary(messy_df):
    m = dp.missing_summary(messy_df).set_index("column")
    assert m.loc["Empty", "missing"] == 6 and m.loc["Empty", "missing_pct"] == 100.0
    assert dp.missing_pct_overall(messy_df) == pytest.approx(messy_df.isna().sum().sum() / messy_df.size * 100)


def test_duplicate_detection(messy_df):
    assert dp.duplicate_count(messy_df) == 1


def test_outlier_detection():
    df = pd.DataFrame({"v": [10, 11, 12, 10, 11, 12, 10, 500]})
    out = dp.outlier_summary(df, ["v"])
    assert out.iloc[0]["outliers"] == 1


def test_recommendations_reflect_data(messy_df):
    keys = {r["key"] for r in dp.recommendations(messy_df, dp.detect_column_kinds(messy_df))}
    assert {"drop_duplicates", "drop_empty_columns", "strip_whitespace", "convert_numeric", "convert_dates"} <= keys
    clean = pd.DataFrame({"a": [1, 2, 3]})
    assert dp.recommendations(clean, dp.detect_column_kinds(clean)) == []


def test_apply_ops_never_mutates_input(messy_df):
    before = messy_df.copy(deep=True)
    out, log = cl.apply_operations(messy_df, ["drop_duplicates", "drop_empty_columns", "strip_whitespace"])
    pd.testing.assert_frame_equal(messy_df, before)
    assert len(out) == 5 and "Empty" not in out.columns and out["Name "].dropna().iloc[0] == "a" and len(log) == 3


def test_convert_numeric_and_dates(messy_df):
    out, _ = cl.apply_operations(messy_df, ["convert_numeric", "convert_dates"])
    assert out["Amount"].iloc[0] == 1000 and out["Amount"].iloc[4] == 4000 and pd.isna(out["Amount"].iloc[5])
    assert pd.api.types.is_datetime64_any_dtype(out["When"])


def test_missing_handling_and_outliers(messy_df):
    filled, _ = cl.apply_operations(messy_df, ["fill_numeric_median"])
    assert not filled["Score"].isna().any()
    dropped, _ = cl.apply_operations(messy_df, ["drop_empty_columns", "drop_missing_rows"])
    assert not dropped.isna().any().any()
    no_out, _ = cl.apply_operations(messy_df, ["remove_outliers_iqr"])
    assert 100.0 not in no_out["Score"].values


def test_standardize_columns_and_unknown_op(messy_df):
    out, _ = cl.apply_operations(messy_df, ["standardize_columns"])
    assert list(out.columns)[:2] == ["name", "amount"]
    with pytest.raises(ValueError):
        cl.apply_operations(messy_df, ["nuke"])
