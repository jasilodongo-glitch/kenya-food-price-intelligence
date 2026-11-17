import pandas as pd
import streamlit as st

import config
from src import export_manager as em, state, ui
from src import data_profiler as dp


def render() -> None:
    ui.page_header("Data explorer", "Search, filter, sort and group your records.")
    if not ui.require_data():
        return
    df = state.active_df()
    kinds = state.cached_kinds(df)
    t1, t2, t3 = st.tabs(["Browse", "Group and compare", "Column types"])
    with t1:
        _browse(df, kinds)
    with t2:
        _group(df, kinds)
    with t3:
        st.dataframe(dp.column_profile(df, kinds), use_container_width=True, hide_index=True)


def _browse(df: pd.DataFrame, kinds: dict) -> None:
    cols = st.multiselect("Columns to show", list(df.columns), default=list(df.columns))
    view = df[cols] if cols else df
    q = st.text_input("Search text", placeholder="Search all text columns")
    if q:
        text_cols = [c for c in view.columns if kinds.get(c) in ("categorical", "text")]
        mask = pd.Series(False, index=view.index)
        for c in text_cols:
            mask |= view[c].astype("string").str.contains(q, case=False, regex=False, na=False)
        view = view[mask]
    with st.expander("Filters"):
        fcol = st.selectbox("Filter column", ["(none)"] + list(view.columns))
        if fcol != "(none)":
            if kinds.get(fcol) == "numeric" or pd.api.types.is_numeric_dtype(view[fcol]):
                s = pd.to_numeric(view[fcol], errors="coerce").dropna()
                if not s.empty and s.min() < s.max():
                    lo, hi = st.slider("Range", float(s.min()), float(s.max()), (float(s.min()), float(s.max())))
                    view = view[pd.to_numeric(view[fcol], errors="coerce").between(lo, hi)]
            elif pd.api.types.is_datetime64_any_dtype(view[fcol]) or kinds.get(fcol) == "datetime":
                d = pd.to_datetime(view[fcol], errors="coerce", format="mixed")
                if d.notna().any():
                    rng = st.date_input("Date range", (d.min().date(), d.max().date()))
                    if len(rng) == 2:
                        view = view[(d >= pd.Timestamp(rng[0])) & (d <= pd.Timestamp(rng[1]))]
            else:
                opts = sorted(view[fcol].dropna().astype(str).unique())[:500]
                pick = st.multiselect("Values", opts)
                if pick:
                    view = view[view[fcol].astype(str).isin(pick)]
    c1, c2 = st.columns([3, 1])
    sort_col = c1.selectbox("Sort by", ["(none)"] + list(view.columns))
    asc = c2.radio("Order", ["Ascending", "Descending"], horizontal=True) == "Ascending"
    if sort_col != "(none)":
        view = view.sort_values(sort_col, ascending=asc)
    size = st.selectbox("Rows per page", config.PAGE_SIZE_OPTIONS, index=1)
    pages = max(1, -(-len(view) // size))
    page = st.number_input("Page", 1, pages, 1)
    st.caption(f"{len(view):,} matching records. Page {page} of {pages}.")
    st.dataframe(view.iloc[(page - 1) * size: page * size], use_container_width=True, hide_index=True)
    st.download_button("Download filtered data (CSV)", em.to_csv_bytes(view), file_name="filtered_data.csv", mime="text/csv")


def _group(df: pd.DataFrame, kinds: dict) -> None:
    cats = [c for c, k in kinds.items() if k in ("categorical", "text")]
    nums = [c for c, k in kinds.items() if k == "numeric"]
    if not cats or not nums:
        st.info("Grouping needs at least one categorical and one numeric column.")
        return
    g = st.selectbox("Group by", cats)
    v = st.multiselect("Values", nums, default=nums[:1])
    agg = st.multiselect("Statistics", ["count", "mean", "median", "min", "max", "std", "sum"], default=["count", "mean"])
    if v and agg:
        out = df.assign(**{c: pd.to_numeric(df[c], errors="coerce") for c in v}).groupby(g)[v].agg(agg)
        out.columns = [f"{a}_{b}" for a, b in out.columns]
        out = out.reset_index().sort_values(out.columns[1], ascending=False)
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button("Download grouped table (CSV)", em.to_csv_bytes(out), file_name="grouped.csv", mime="text/csv")
