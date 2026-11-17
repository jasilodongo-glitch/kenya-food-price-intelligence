import streamlit as st

import config
from src import data_cleaner, data_loader as dl, data_profiler as dp, export_manager as em, metadata_store, state, ui, visualization as viz


@st.cache_data(show_spinner=False, max_entries=3)
def _sheets(data: bytes, ext: str) -> list[str]:
    return dl.excel_sheet_names(data, ext)


def render() -> None:
    ui.page_header("Data upload", f"CSV, XLSX or XLS up to {config.MAX_UPLOAD_MB} MB. Your original file is kept untouched.")
    up = st.file_uploader("Drag a file here or browse", type=["csv", "xlsx", "xls"], accept_multiple_files=False)
    if up is not None:
        _handle_upload(up)
    if state.has_data():
        st.divider()
        _profile_and_clean()


def _handle_upload(up) -> None:
    data = up.getvalue()
    try:
        ext = dl.validate_file(up.name, len(data))
        sheet = None
        if ext in (".xlsx", ".xls"):
            names = _sheets(data, ext)
            sheet = st.selectbox("Worksheet", names) if len(names) > 1 else names[0]
        res = dl.load_dataset(data, up.name, sheet)
    except dl.DataLoadError as exc:
        st.error(f"Could not process this file. {exc}")
        return
    for n in res.notes:
        st.info(n)
    st.success(f"Read {len(res.df):,} rows and {res.df.shape[1]} columns from {res.filename}.")
    st.dataframe(res.df.head(config.PREVIEW_ROWS), use_container_width=True, hide_index=True)
    replacing = state.has_data()
    if st.button("Replace active dataset" if replacing else "Use this dataset", type="primary"):
        saved = dl.save_original(data, up.name)
        metadata_store.record_dataset(res.filename, res.file_hash, res.sheet, len(res.df), res.df.shape[1], res.size_bytes)
        state.set_dataset(res.df.copy(), res.df, res.filename, {"sheet": res.sheet, "saved_as": saved.name, "sha256": res.file_hash})
        st.rerun()


def _profile_and_clean() -> None:
    df, original = state.active_df(), st.session_state["original"]
    p = state.cached_profile(df)
    st.subheader("Dataset profile")
    ui.kpi_strip([("Rows", f"{p.rows:,}", ""), ("Columns", str(p.columns), ""), ("Missing", f"{p.missing_pct:.1f}%", ""),
                  ("Duplicate rows", f"{p.duplicates:,}", ""), ("Memory", f"{p.memory_mb:.1f} MB", "")])
    t1, t2, t3, t4 = st.tabs(["Columns", "Missing values", "Outliers", "Cleaning"])
    with t1:
        st.dataframe(dp.column_profile(df, p.kinds), use_container_width=True, hide_index=True)
    with t2:
        miss = dp.missing_summary(df)
        if (miss["missing"] > 0).any():
            st.dataframe(miss[miss["missing"] > 0], use_container_width=True, hide_index=True)
            st.plotly_chart(viz.missing_bar(miss), use_container_width=True)
        else:
            st.success("No missing values detected.")
    with t3:
        nums = [c for c, k in p.kinds.items() if k == "numeric"]
        out = dp.outlier_summary(df, nums)
        st.caption("Outliers are values outside 1.5 x IQR of the quartiles. They may be genuine, so review before removing.")
        st.dataframe(out, use_container_width=True, hide_index=True) if not out.empty else st.info("No numeric columns with enough data.")
    with t4:
        _cleaning(original, df)


def _cleaning(original, df) -> None:
    kinds_orig = dp.detect_column_kinds(original)
    recs = dp.recommendations(original, kinds_orig)
    st.caption("Choose what to apply. Operations always start from your original upload, so you can change your mind.")
    if recs:
        st.markdown("**Recommended for this dataset**")
    chosen = []
    for r in recs:
        if st.checkbox(r["title"], key=f"rec_{r['key']}", help=r["detail"]):
            chosen.append(r["key"])
    rec_keys = {r["key"] for r in recs}
    extra = st.multiselect("Other operations", [k for k in data_cleaner.OPERATIONS if k not in rec_keys],
                           format_func=lambda k: data_cleaner.OPERATIONS[k])
    c1, c2 = st.columns(2)
    if c1.button("Apply cleaning", type="primary", disabled=not (chosen or extra)):
        new, log = data_cleaner.apply_operations(original, chosen + extra)
        st.session_state.update(df=new, clean_log=log, forecast=None)
        st.rerun()
    if c2.button("Reset to original"):
        st.session_state.update(df=original.copy(), clean_log=[], forecast=None)
        st.rerun()
    for line in st.session_state.get("clean_log", []):
        st.write("- " + line)
    d1, d2 = st.columns(2)
    d1.download_button("Download cleaned CSV", em.to_csv_bytes(df), file_name=f"{st.session_state['dataset_name']}_cleaned.csv", mime="text/csv")
    d2.download_button("Download cleaned Excel", em.to_excel_bytes({"Data": df}), file_name=f"{st.session_state['dataset_name']}_cleaned.xlsx")
