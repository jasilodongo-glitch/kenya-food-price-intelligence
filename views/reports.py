import streamlit as st

from src import food_price_analysis as fpa, report_generator as rg, state, ui


def render() -> None:
    ui.page_header("Reports", "Generate a complete analytical report from the active dataset. Every statement is computed from your data.")
    if not ui.require_data():
        return
    df = state.active_df()
    name = st.session_state.get("dataset_name", "dataset")
    title = st.text_input("Report title", f"Analysis report: {name}")
    roles = fpa.detect_roles(df, state.cached_kinds(df)) if st.checkbox("Include food price intelligence section when columns are detected", value=True) else None
    has_fc = st.session_state.get("forecast") is not None
    use_fc = st.checkbox("Include the latest forecast", value=has_fc, disabled=not has_fc,
                         help="Run a forecast on the Forecasting page first." if not has_fc else None)
    if st.button("Generate report", type="primary"):
        with st.spinner("Building report"):
            st.session_state["report"] = rg.build_report(df, name, title, roles if roles and roles.get("price") else None,
                                                         st.session_state.get("forecast") if use_fc else None, st.session_state.get("forecast_label", ""))
    rep = st.session_state.get("report")
    if rep is None:
        return
    c1, c2 = st.columns(2)
    c1.download_button("Download HTML report (print to PDF)", rg.to_html(rep), file_name="report.html", mime="text/html", type="primary")
    c2.download_button("Download Excel tables", rg.to_excel(rep), file_name="report_tables.xlsx")
    st.divider()
    st.header(rep.title)
    st.caption(f"Generated {rep.generated_at}")
    for sec in rep.sections:
        st.subheader(sec.title)
        for p in sec.paragraphs:
            st.markdown(f"- {p}" if sec.title in {"Major findings", "Business implications", "Recommendations", "Limitations"} else p)
        for tname, t in sec.tables.items():
            st.caption(tname)
            st.dataframe(t.head(100), use_container_width=True, hide_index=True)
        for fig in sec.figures:
            st.plotly_chart(fig, use_container_width=True)
