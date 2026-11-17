import streamlit as st

import config
from src import export_manager as em, metadata_store as ms, state, ui


def render() -> None:
    ui.page_header("Settings", "Limits, stored files and data deletion. Files stay on this machine and are never exposed through URLs.")
    st.subheader("Limits")
    st.write(f"Maximum upload size: **{config.MAX_UPLOAD_MB} MB**. Maximum rows: **{config.MAX_ROWS:,}**. Accepted formats: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}.")
    st.caption("Raise the upload limit by editing config.py and .streamlit/config.toml.")

    st.subheader("Active dataset")
    if state.has_data():
        st.write(f"**{st.session_state.get('dataset_name')}** ({len(state.active_df()):,} rows).")
        if st.button("Unload dataset from this session"):
            state.clear_dataset()
            st.rerun()
    else:
        st.caption("No dataset loaded.")

    st.subheader("Stored files")
    files = em.stored_files()
    if files:
        st.dataframe([{k: v for k, v in f.items() if k != "path"} for f in files], use_container_width=True, hide_index=True)
        st.caption("Originals are saved under data/uploads (unmodified). Cleaned exports saved under data/processed.")
    else:
        st.caption("No files are stored.")

    st.subheader("Upload history")
    hist = ms.history()
    st.dataframe(hist, use_container_width=True, hide_index=True) if hist else st.caption("No uploads recorded. Only file names and sizes are logged, never the data.")

    st.subheader("Delete data")
    confirm = st.checkbox("I understand this permanently deletes stored files and upload history")
    if st.button("Delete all stored data", disabled=not confirm):
        n = em.delete_stored_files()
        ms.clear_history()
        state.clear_dataset()
        st.success(f"Deleted {n} file(s), cleared history and unloaded the dataset.")
