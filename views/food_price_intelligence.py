import pandas as pd
import streamlit as st

from src import food_price_analysis as fpa, state, time_series as ts, ui, visualization as viz


def render() -> None:
    ui.page_header("Food price intelligence", "Markets, commodities, volatility and seasonality. Only analyses your data supports are shown.")
    if not ui.require_data():
        return
    df = state.active_df()
    kinds = state.cached_kinds(df)
    roles = fpa.detect_roles(df, kinds)
    with st.expander("Detected columns (adjust if needed)", expanded=not roles.get("price")):
        opts = ["(none)"] + list(df.columns)
        cols = st.columns(3)
        for i, role in enumerate(fpa.ROLE_LABELS):
            cur = roles.get(role)
            pick = cols[i % 3].selectbox(fpa.ROLE_LABELS[role], opts, index=opts.index(cur) if cur in opts else 0, key=f"role_{role}")
            roles[role] = None if pick == "(none)" else pick
    avail = fpa.available_analyses(roles)
    if not roles.get("price"):
        ui.empty_state("No price column found", body="Select the column that holds prices in the panel above. Without it, food-price analysis cannot run.")
        return
    try:
        f = fpa.standardise(df, roles)
    except fpa.FoodPriceError as exc:
        st.error(str(exc))
        return
    if f.empty:
        st.warning("No valid price values remain after removing missing and non-numeric prices.")
        return
    st.session_state["food_roles"] = roles
    ui.note(fpa.AFFORDABILITY_NOTE)

    commodity = None
    if "commodity" in f:
        counts = f["commodity"].value_counts()
        commodity = st.selectbox("Commodity", list(counts.index), format_func=lambda c: f"{c} ({counts[c]:,} records)")

    labels = [("Markets", "market"), ("Counties and regions", "geography"), ("Commodities", "commodity"), ("Volatility", "volatility"), ("Seasonality", "seasonality")]
    live = [(n, k) for n, k in labels if avail[k]]
    if not live:
        st.info("The detected columns do not support market, commodity or seasonal analysis. Map more columns above.")
        return
    tabs = st.tabs([n for n, _ in live])
    for tab, (_, key) in zip(tabs, live):
        with tab:
            {"market": lambda: _location(f, "market", commodity), "geography": lambda: _geography(f, commodity),
             "commodity": lambda: _commodity(f), "volatility": lambda: _volatility(f), "seasonality": lambda: _seasonality(f, commodity)}[key]()


def _location(f, by, commodity):
    try:
        rk, notes = fpa.location_ranking(f, by, commodity)
    except fpa.FoodPriceError as exc:
        st.info(str(exc))
        return
    for n in notes:
        st.caption(n)
    unit = fpa.unit_label(fpa.commodity_subset(f, commodity)[0])
    hi, lo = rk.iloc[0], rk.iloc[-1]
    st.write(f"Average price ranges from **{lo['mean']:.2f}** in {lo[by]} to **{hi['mean']:.2f}** in {hi[by]} ({unit}), a gap of {(hi['mean'] / lo['mean'] - 1) * 100:.0f}%. "
             "Differences may reflect transport, quality, unit sizes or the time periods each location was observed.")
    st.plotly_chart(viz.bar(rk.head(25), by, "mean", f"Average price by {by}, most expensive first", horizontal=True), use_container_width=True)
    st.dataframe(rk, use_container_width=True, hide_index=True)


def _geography(f, commodity):
    by = "county" if "county" in f else "country"
    _location(f, by, commodity)


def _commodity(f):
    if f["commodity"].nunique() < 2:
        st.info("Only one commodity is present, so there is nothing to compare.")
        return
    if "date" in f:
        try:
            chg = fpa.latest_change_table(f)
            if not chg.empty:
                st.subheader("Latest price movement")
                st.dataframe(chg.sort_values("change_12m_pct", ascending=False), use_container_width=True, hide_index=True)
        except (fpa.FoodPriceError, ts.TimeSeriesError) as exc:
            st.caption(str(exc))
        top = list(f["commodity"].value_counts().head(5).index)
        pick = st.multiselect("Compare price movement (rebased to 100)", list(f["commodity"].unique()), default=top[:3])
        if len(pick) >= 2:
            try:
                st.plotly_chart(viz.multi_line(fpa.price_index(f, pick), "Price index, first common month = 100", "Index"), use_container_width=True)
                st.caption("Indexing removes unit differences so commodities can be compared by relative movement, not by price level.")
            except (fpa.FoodPriceError, ts.TimeSeriesError) as exc:
                st.info(str(exc))
    else:
        st.dataframe(f.groupby("commodity")["price"].agg(["count", "mean", "median"]).reset_index(), use_container_width=True, hide_index=True)


def _volatility(f):
    try:
        vol = fpa.volatility_table(f)
    except (fpa.FoodPriceError, ts.TimeSeriesError) as exc:
        st.info(str(exc))
        return
    st.write("Volatility is the standard deviation of month-on-month price changes. Higher means less predictable prices.")
    st.plotly_chart(viz.bar(vol, "commodity", "monthly_volatility_pct", "Monthly price volatility (%)", horizontal=True, color="#B5533C"), use_container_width=True)
    st.dataframe(vol, use_container_width=True, hide_index=True)


def _seasonality(f, commodity):
    try:
        s, notes = fpa.monthly_series(f, commodity)
        prof = ts.seasonal_profile(s)
    except (fpa.FoodPriceError, ts.TimeSeriesError) as exc:
        st.info(str(exc))
        return
    st.plotly_chart(viz.seasonal_bar(prof), use_container_width=True)
    hi, lo = prof.loc[prof["index"].idxmax()], prof.loc[prof["index"].idxmin()]
    st.write(f"Prices are typically highest in month {int(hi['month'])} ({hi['index'] - 100:+.1f}% vs average) and lowest in month {int(lo['month'])} ({lo['index'] - 100:+.1f}%). "
             "This is the average pattern across years; individual years can differ.")
    st.plotly_chart(viz.month_year_heatmap(ts.year_month_table(s)), use_container_width=True)
    st.dataframe(ts.year_over_year(s), use_container_width=True, hide_index=True)
