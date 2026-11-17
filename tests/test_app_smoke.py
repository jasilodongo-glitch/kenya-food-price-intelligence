"""Render every page headlessly, with and without a dataset, and fail on any uncaught exception."""
import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["overview", "data_upload", "data_explorer", "analytics", "trends", "food_price_intelligence", "forecasting", "reports", "settings"]


def _run(page, df=None):
    at = AppTest.from_string(f"from views import {page} as p\np.render()", default_timeout=60)
    if df is not None:
        at.session_state["df"] = df
        at.session_state["original"] = df.copy()
        at.session_state["dataset_name"] = "synthetic.csv"
        at.session_state["clean_log"] = []
    return at.run()


@pytest.mark.parametrize("page", PAGES)
def test_page_without_data(page):
    at = _run(page)
    assert not at.exception, at.exception


@pytest.mark.parametrize("page", PAGES)
def test_page_with_data(page, food_df):
    at = _run(page, food_df)
    assert not at.exception, at.exception


def test_forecast_button_runs(food_df):
    at = _run("forecasting", food_df)
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert not at.warning, [w.value for w in at.warning]
    res = at.session_state["forecast"]
    assert res is not None and len(res.metrics) >= 3


def test_page_with_non_timeseries_data():
    import pandas as pd
    at = _run("trends", pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]}))
    assert not at.exception and not at.error
