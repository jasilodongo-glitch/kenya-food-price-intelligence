import pandas as pd
import streamlit as st

from src import data_profiler as dp, state, statistical_analysis as sa, ui, visualization as viz


def render() -> None:
    ui.page_header("Statistical analysis", "Summaries, distributions, relationships and tests, explained in plain language.")
    if not ui.require_data():
        return
    df = state.active_df()
    kinds = state.cached_kinds(df)
    cols = dp.columns_by_kind(df, kinds)
    nums = cols.get("numeric", [])
    cats = cols.get("categorical", [])
    # Work on numeric-coerced copies so text-formatted numbers are handled consistently.
    work = df.copy()
    for c in nums:
        work[c] = pd.to_numeric(work[c].astype(str).str.replace(",", "", regex=False), errors="coerce") if not pd.api.types.is_numeric_dtype(work[c]) else work[c]
    t = st.tabs(["Summary", "Distributions", "Relationships", "Outliers", "Tests and regression"])
    with t[0]:
        if nums:
            st.dataframe(sa.describe_numeric(work, nums), use_container_width=True, hide_index=True)
        else:
            st.info("No numeric columns detected.")
        if cats:
            c = st.selectbox("Frequency table for", cats)
            st.dataframe(sa.frequency_table(df[c]), use_container_width=True, hide_index=True)
    with t[1]:
        if not nums:
            st.info("Distributions need a numeric column.")
        else:
            v = st.selectbox("Variable", nums, key="dist_v")
            g = st.selectbox("Box plot grouped by (optional)", ["(none)"] + cats, key="dist_g")
            a, b = st.columns(2)
            a.plotly_chart(viz.histogram(work[v], v), use_container_width=True)
            b.plotly_chart(viz.boxplot(work, v, None if g == "(none)" else g), use_container_width=True)
            try:
                nt = sa.normality_test(work[v])
                st.write(f"**Normality ({nt['test']}, n={nt['n']:,}):** p = {nt['p_value']:.4g}. {nt['interpretation']}")
            except ValueError as exc:
                st.caption(str(exc))
    with t[2]:
        if len(nums) < 2:
            st.info("Relationships need at least two numeric columns.")
        else:
            method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
            corr = sa.correlation_matrix(work, nums, method)
            st.plotly_chart(viz.corr_heatmap(corr), use_container_width=True)
            top = sa.top_correlations(corr, 8)
            for _, r in top.iterrows():
                st.write(f"- **{r.variable_a}** and **{r.variable_b}**: {sa.describe_strength(r.correlation)} association (r = {r.correlation:.2f}).")
            ui.note("Correlation measures how two variables move together. It does not show that one causes the other.")
            x, y = st.columns(2)
            xv, yv = x.selectbox("X axis", nums, key="sx"), y.selectbox("Y axis", nums, index=1, key="sy")
            st.plotly_chart(viz.scatter(work, xv, yv, None), use_container_width=True)
    with t[3]:
        out = dp.outlier_summary(work, nums)
        st.caption("Outliers are values outside 1.5 x IQR from the quartiles. They may be errors or genuine extremes.")
        st.dataframe(out, use_container_width=True, hide_index=True) if not out.empty else st.info("Not enough numeric data.")
    with t[4]:
        _inference(work, nums, cats)


def _inference(work, nums, cats) -> None:
    if not nums:
        st.info("Inference needs a numeric column.")
        return
    st.subheader("Confidence interval for a mean")
    v = st.selectbox("Variable", nums, key="ci_v")
    lvl = st.select_slider("Confidence level", [0.90, 0.95, 0.99], value=0.95)
    try:
        ci = sa.mean_confidence_interval(work[v], lvl)
        st.write(f"The mean of **{v}** is {ci['mean']:.4g}. We are {int(lvl * 100)}% confident the true mean lies between {ci['lower']:.4g} and {ci['upper']:.4g} (n = {ci['n']:,}).")
    except ValueError as exc:
        st.caption(str(exc))
    if cats:
        st.subheader("Compare group averages")
        gcol = st.selectbox("Group column", cats, key="cg")
        try:
            res = sa.compare_groups(work, v, gcol)
            st.write(f"**{res['test']}**: statistic = {res['statistic']:.3f}, p = {res['p_value']:.4g}. {res['interpretation']}")
            st.dataframe(res["summary"], use_container_width=True, hide_index=True)
        except ValueError as exc:
            st.caption(str(exc))
    if len(nums) >= 2:
        st.subheader("Linear regression")
        target = st.selectbox("Target", nums, key="rt")
        preds = st.multiselect("Predictors", [c for c in nums if c != target], key="rp")
        if preds:
            try:
                r = sa.ols_regression(work, target, preds)
                st.write(f"R-squared {r['r_squared']:.3f}, adjusted {r['adj_r_squared']:.3f}, n = {r['n']:,}, model p = {r['f_p_value']:.4g}.")
                st.dataframe(r["table"], use_container_width=True, hide_index=True)
                ui.note(r["interpretation"])
            except ValueError as exc:
                st.caption(str(exc))
