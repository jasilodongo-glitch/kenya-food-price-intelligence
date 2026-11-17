import pandas as pd
import streamlit as st

from src import state, time_series as ts, ui, visualization as viz


def date_value_controls(df, kinds, key=""):
    """Shared selectors; returns (date_col, value_col) or None."""
    dates = [c for c, k in kinds.items() if k == "datetime"]
    nums = [c for c, k in kinds.items() if k == "numeric"]
    if not dates or not nums:
        ui.empty_state("This dataset is not suited to time-series analysis",
                       body="Trend analysis needs at least one date column and one numeric column. Other pages still work with this dataset.")
        return None
    a, b = st.columns(2)
    return a.selectbox("Date column", dates, key=f"{key}d"), b.selectbox("Value column", nums, key=f"{key}v")


def render() -> None:
    ui.page_header("Trend analysis", "Trends, growth, seasonality and decomposition over time.")
    if not ui.require_data():
        return
    df = state.active_df()
    sel = date_value_controls(df, state.cached_kinds(df), "tr")
    if not sel:
        return
    dcol, vcol = sel
    a, b, c = st.columns(3)
    fname = a.selectbox("Frequency", list(ts.FREQUENCIES), index=2)
    agg = b.selectbox("Aggregation", ["mean", "median", "sum", "min", "max"])
    window = c.slider("Moving-average window (periods)", 2, 24, 3)
    freq = ts.FREQUENCIES[fname]
    try:
        s = ts.prepare_series(df, dcol, vcol, freq, agg)
    except ts.TimeSeriesError as exc:
        st.warning(str(exc))
        return
    missing = int(s.isna().sum())
    if missing:
        st.caption(f"{missing} period(s) have no data and appear as gaps.")
    t = st.tabs(["Trend", "Change and growth", "Seasonality", "Decomposition", "Year comparison"])
    with t[0]:
        st.plotly_chart(viz.line_series(s, f"{agg.title()} of {vcol} ({fname.lower()})", ts.moving_average(s, window), f"{window}-period moving average", vcol), use_container_width=True)
    with t[1]:
        pc = ts.percent_change(s)
        pcd = pd.DataFrame({"date": pc.dropna().index, "pct_change": pc.dropna().values})
        st.plotly_chart(viz.bar(pcd, "date", "pct_change", "Period-on-period change (%)"), use_container_width=True)
        try:
            g = ts.growth_summary(s)
            ui.kpi_strip([("Total change", f"{g['total_change_pct']:+.1f}%", f"{g['start_date']:%b %Y} to {g['end_date']:%b %Y}"),
                          ("Annual growth (CAGR)", "n/a" if pd.isna(g["cagr_pct"]) else f"{g['cagr_pct']:+.1f}%", "needs 1+ year of data"),
                          ("Peak", ui.fmt_num(g["peak_value"]), f"{g['peak_date']:%b %Y}"), ("Low", ui.fmt_num(g["trough_value"]), f"{g['trough_date']:%b %Y}")])
        except ts.TimeSeriesError as exc:
            st.caption(str(exc))
    with t[2]:
        try:
            prof = ts.seasonal_profile(s)
            st.plotly_chart(viz.seasonal_bar(prof), use_container_width=True)
            hi, lo = prof.loc[prof["index"].idxmax()], prof.loc[prof["index"].idxmin()]
            st.write(f"Month {int(hi['month'])} is on average {hi['index'] - 100:+.1f}% versus the overall mean; month {int(lo['month'])} is {lo['index'] - 100:+.1f}%.")
            st.plotly_chart(viz.month_year_heatmap(ts.year_month_table(s)), use_container_width=True)
        except ts.TimeSeriesError as exc:
            st.info(str(exc))
    with t[3]:
        try:
            st.plotly_chart(viz.decomposition(ts.decompose(s, ts.SEASONAL_PERIODS.get(freq))), use_container_width=True)
        except ts.TimeSeriesError as exc:
            st.info(str(exc))
    with t[4]:
        try:
            st.dataframe(ts.year_over_year(s), use_container_width=True, hide_index=True)
        except Exception as exc:  # noqa: BLE001
            st.info(str(exc))
