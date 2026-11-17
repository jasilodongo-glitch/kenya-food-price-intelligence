import streamlit as st

from src import analytics_engine, state, ui, visualization as viz
from src import data_profiler as dp


def render() -> None:
    ui.page_header("Executive overview", "A live summary of the active dataset. Every figure is calculated from your data.")
    if not state.has_data():
        ui.empty_state("Welcome. Start by loading a dataset.",
                       ["Open Data Upload in the sidebar", "Drop in a CSV or Excel file of food prices, or any other table",
                        "Come back here for an instant summary, then explore statistics, trends and forecasts"],
                       "The platform is specialised for Kenyan food-market prices but works with any tabular data. It ships with no sample data, so nothing is shown until you upload yours.")
        return
    df = state.active_df()
    p = state.cached_profile(df)
    cols = dp.columns_by_kind(df, p.kinds)
    cov = f"{p.date_range[0]:%b %Y} to {p.date_range[1]:%b %Y}" if p.date_range else "No date column"
    ui.kpi_strip([("Records", f"{p.rows:,}", ""), ("Variables", f"{p.columns}", ""),
                  ("Numeric columns", str(len(cols.get("numeric", []))), ""), ("Categorical columns", str(len(cols.get("categorical", []))), ""),
                  ("Missing values", f"{p.missing_pct:.1f}%", "of all cells"), ("Date coverage", cov, p.date_column or "")])
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("What the data shows")
        obs = analytics_engine.auto_observations(df, p)
        st.markdown("<ul class='obs'>" + "".join(f"<li>{o}</li>" for o in obs) + "</ul>", unsafe_allow_html=True)
        miss = dp.missing_summary(df)
        if (miss["missing"] > 0).any():
            st.plotly_chart(viz.missing_bar(miss), use_container_width=True)
    with right:
        st.subheader("Key numeric indicators")
        inds = analytics_engine.key_indicators(df, cols.get("numeric", []))
        if not inds:
            st.caption("No numeric columns detected.")
        for i in inds:
            st.markdown(f"**{i['column']}**")
            st.caption(f"Mean {ui.fmt_num(i['mean'])} | Median {ui.fmt_num(i['median'])} | Range {ui.fmt_num(i['min'])} to {ui.fmt_num(i['max'])}")
