
"""Shared UI components and styling."""
from __future__ import annotations

import html

import streamlit as st

import config

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}

.stApp {
    background: #F6F8F5;
}

/* Hide unnecessary Streamlit elements without hiding the toolbar */
#MainMenu,
footer,
[data-testid="stDecoration"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Keep the header and sidebar controls accessible */
header[data-testid="stHeader"],
[data-testid="stToolbar"] {
    visibility: visible !important;
    opacity: 1 !important;
}

[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 999999 !important;
}

[data-testid="stSidebarCollapsedControl"] button {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}

/* Sidebar toggle button when sidebar is expanded */
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    opacity: 1 !important;
}

/* Main layout */
.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1240px;
}

h1, h2, h3 {
    font-family: 'Fraunces', Georgia, serif !important;
    color: #17211C;
    letter-spacing: -0.01em;
    font-weight: 600;
}

/* Sidebar: deep field green */
[data-testid="stSidebar"] {
    background: #14392B;
    border-right: none;
}

[data-testid="stSidebar"] * {
    color: #DCE8DF;
}

[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {
    border-radius: 6px;
    padding: .45rem .75rem;
    margin: 1px 0;
    border-left: 3px solid transparent;
}

[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {
    background: rgba(255,255,255,.07);
}

[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {
    background: rgba(255,255,255,.10);
    border-left-color: #D9A21B;
}

[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] span {
    color: #FFFFFF;
    font-weight: 600;
}

.brand {
    padding: 4px 8px 18px 8px;
}

.brand .name {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.28rem;
    line-height: 1.15;
    color: #FFFFFF;
    font-weight: 600;
}

.brand .tag {
    font-size: .78rem;
    color: #9DB8A8;
    margin-top: 6px;
    line-height: 1.35;
}

.side-status {
    margin: 18px 8px 0 8px;
    padding-top: 14px;
    border-top: 1px solid rgba(255,255,255,.15);
    font-size: .82rem;
    color: #B7CDBF;
}

.side-status b {
    color: #FFFFFF;
    font-weight: 600;
}

/* Page header */
.page-head {
    margin-bottom: 1.4rem;
}

.page-head h1 {
    font-size: 2.1rem;
    margin: 0 0 .25rem 0;
}

.page-head p {
    color: #66736B;
    margin: 0;
    max-width: 62ch;
    font-size: 1.02rem;
}

/* KPI strip */
.kpi-strip {
    display: flex;
    flex-wrap: wrap;
    border-top: 2px solid #1F5F46;
    border-bottom: 1px solid #DCE3DB;
    margin: .5rem 0 1.6rem 0;
    background: #FFFFFF;
}

.kpi {
    flex: 1 1 150px;
    padding: 16px 20px;
    border-right: 1px solid #DCE3DB;
    min-width: 140px;
}

.kpi:last-child {
    border-right: none;
}

.kpi .v {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.9rem;
    line-height: 1.1;
    color: #17211C;
    font-weight: 600;
}

.kpi .l {
    font-size: .84rem;
    color: #66736B;
    margin-top: 4px;
}

.kpi .s {
    font-size: .76rem;
    color: #8A958E;
    margin-top: 2px;
}

/* Empty state and notes */
.empty {
    border: 1px dashed #B9C7BC;
    background: #FFFFFF;
    padding: 34px 36px;
    border-radius: 8px;
    max-width: 760px;
}

.empty h3 {
    margin-top: 0;
}

.empty ol {
    color: #3B4A42;
    line-height: 1.8;
    padding-left: 1.2rem;
    margin-bottom: 0;
}

.note {
    border-left: 3px solid #D9A21B;
    background: #FFF9E8;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    font-size: .92rem;
    color: #4A4326;
    margin: .6rem 0 1rem 0;
}

.obs li {
    margin-bottom: .35rem;
}

/* Controls */
.stButton > button,
.stDownloadButton > button {
    border-radius: 6px;
    font-weight: 500;
    border: 1px solid #1F5F46;
}

.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    background: #1F5F46;
    color: #FFFFFF;
}

.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible {
    outline: 3px solid #D9A21B;
    outline-offset: 2px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #DCE3DB;
}

.stTabs [aria-selected="true"] {
    color: #1F5F46;
    font-weight: 600;
}

[data-testid="stFileUploader"] section {
    background: #FFFFFF;
    border: 1px dashed #9FB5A7;
    border-radius: 8px;
}

@media (max-width: 640px) {
    .kpi {
        flex-basis: 45%;
    }

    .page-head h1 {
        font-size: 1.6rem;
    }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def sidebar_brand() -> None:
    with st.sidebar:
        st.markdown(
            f"<div class='brand'><div class='name'>{html.escape(config.APP_NAME)}</div>"
            f"<div class='tag'>{html.escape(config.TAGLINE)}</div></div>",
            unsafe_allow_html=True
        )


def sidebar_status() -> None:
    with st.sidebar:
        if st.session_state.get("df") is not None:
            df = st.session_state["df"]
            st.markdown(
                f"<div class='side-status'>Active dataset<br>"
                f"<b>{html.escape(str(st.session_state.get('dataset_name', '')))}</b><br>"
                f"{len(df):,} rows, {df.shape[1]} columns</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='side-status'>No dataset loaded.<br>"
                "Go to Data Upload to begin.</div>",
                unsafe_allow_html=True
            )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"<div class='page-head'><h1>{html.escape(title)}</h1>"
        f"<p>{html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True
    )


def kpi_strip(items: list[tuple[str, str, str]]) -> None:
    """items: (label, value, small caption)."""
    cells = "".join(
        f"<div class='kpi'><div class='v'>{html.escape(str(v))}</div>"
        f"<div class='l'>{html.escape(l)}</div>"
        + (f"<div class='s'>{html.escape(s)}</div>" if s else "")
        + "</div>"
        for l, v, s in items
    )
    st.markdown(
        f"<div class='kpi-strip'>{cells}</div>",
        unsafe_allow_html=True
    )


def note(text: str) -> None:
    st.markdown(
        f"<div class='note'>{html.escape(text)}</div>",
        unsafe_allow_html=True
    )


def empty_state(title: str, steps: list[str] | None = None, body: str = "") -> None:
    li = "".join(
        f"<li>{html.escape(s)}</li>"
        for s in (steps or [])
    )

    st.markdown(
        f"<div class='empty'><h3>{html.escape(title)}</h3>"
        f"<p>{html.escape(body)}</p>"
        + (f"<ol>{li}</ol>" if li else "")
        + "</div>",
        unsafe_allow_html=True
    )


def require_data() -> bool:
    """Render an empty state and return False when no dataset is loaded."""
    if st.session_state.get("df") is None:
        empty_state(
            "Load a dataset to use this page",
            [
                "Open Data Upload from the sidebar",
                "Drop in a CSV or Excel file",
                "Select Use this dataset"
            ],
            "This page analyses the active dataset. Nothing is shown until you provide one."
        )
        return False

    return True


def fmt_num(v: float) -> str:
    if v != v:  # NaN
        return "n/a"

    a = abs(v)

    return (
        f"{v:,.0f}" if a >= 1000
        else f"{v:,.2f}" if a >= 1
        else f"{v:.3g}"
    )
