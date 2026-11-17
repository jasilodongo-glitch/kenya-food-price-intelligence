import streamlit as st

from src import forecasting as fc, state, time_series as ts, ui, visualization as viz
from views.trends import date_value_controls


def render() -> None:
    ui.page_header("Forecasting", "Compare models on a held-out period, then project forward with 95% intervals.")
    if not ui.require_data():
        return
    df = state.active_df()
    kinds = state.cached_kinds(df)
    sel = date_value_controls(df, kinds, "fc")
    if not sel:
        return
    dcol, vcol = sel
    work = df
    cats = [c for c, k in kinds.items() if k == "categorical"]
    label = vcol
    if cats:
        f1, f2 = st.columns(2)
        fcol = f1.selectbox("Filter to one group (recommended for price data)", ["(none)"] + cats)
        if fcol != "(none)":
            val = f2.selectbox("Value", sorted(df[fcol].dropna().astype(str).unique()))
            work = df[df[fcol].astype(str) == val]
            label = f"{vcol} ({val})"
    a, b, c = st.columns(3)
    fname = a.selectbox("Frequency", list(ts.FREQUENCIES), index=2, key="ffreq")
    agg = b.selectbox("Aggregation", ["mean", "median", "sum"], key="fagg")
    horizon = c.slider("Forecast horizon (periods)", 1, 36, 6)
    models = st.multiselect("Models", fc.MODEL_NAMES, default=fc.MODEL_NAMES)
    freq = ts.FREQUENCIES[fname]
    if st.button("Run forecast", type="primary"):
        try:
            s = ts.prepare_series(work, dcol, vcol, freq, agg)
            with st.spinner("Fitting models"):
                res = fc.run_forecast(s, horizon, models, ts.SEASONAL_PERIODS.get(freq))
            st.session_state.update(forecast=res, forecast_label=label)
        except (ts.TimeSeriesError, fc.ForecastError) as exc:
            st.session_state["forecast"] = None
            st.warning(f"This series cannot be forecast. {exc}")
    res = st.session_state.get("forecast")
    if res is None:
        return
    st.subheader(f"Results for {st.session_state.get('forecast_label', '')}")
    st.dataframe(res.metrics.style.format(precision=3), use_container_width=True, hide_index=True)
    st.caption("Lower is better for MAE, RMSE and MAPE. Interval coverage near 95% means the uncertainty bands were well calibrated on the test period.")
    st.plotly_chart(viz.backtest_chart(res.train, res.test, res.test_forecasts), use_container_width=True)
    model = st.selectbox("Show forecast for", list(res.future), index=list(res.future).index(res.best_model))
    st.plotly_chart(viz.forecast_chart(res.train, res.test, res.future[model], model, vcol), use_container_width=True)
    st.dataframe(res.future[model].round(3).reset_index(names="period"), use_container_width=True, hide_index=True)
    for n in res.notes:
        st.caption(n)
    ui.note("Forecasts extrapolate past patterns. They cannot anticipate weather, policy or supply shocks, and wide intervals signal low confidence.")
