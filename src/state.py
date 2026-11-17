"""Session-state helpers: the active dataset and its derived (cached) profile."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import data_profiler as dp


def has_data() -> bool:
    return st.session_state.get("df") is not None


def active_df() -> pd.DataFrame:
    return st.session_state["df"]


def set_dataset(df: pd.DataFrame, original: pd.DataFrame, name: str, meta: dict) -> None:
    st.session_state.update(df=df, original=original, dataset_name=name, meta=meta, clean_log=[], forecast=None, forecast_label="")


def clear_dataset() -> None:
    for k in ("df", "original", "dataset_name", "meta", "clean_log", "forecast", "forecast_label", "pending_upload"):
        st.session_state.pop(k, None)


@st.cache_data(show_spinner=False)
def cached_profile(df: pd.DataFrame):
    return dp.profile_dataset(df)


@st.cache_data(show_spinner=False)
def cached_kinds(df: pd.DataFrame) -> dict[str, str]:
    return dp.detect_column_kinds(df)
