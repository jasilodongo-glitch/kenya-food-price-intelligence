"""Kenya Food Price Intelligence: Streamlit entry point."""
import streamlit as st

import config
from src import ui
from views import analytics, data_explorer, data_upload, food_price_intelligence, forecasting, overview, reports, settings, trends

st.set_page_config(page_title=config.APP_NAME, page_icon="🌽", layout="wide", initial_sidebar_state="expanded")
ui.inject_css()
ui.sidebar_brand()

pages = [
    st.Page(overview.render, title="Executive Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page(data_upload.render, title="Data Upload", icon=":material/upload_file:", url_path="upload"),
    st.Page(data_explorer.render, title="Data Explorer", icon=":material/table_view:", url_path="explorer"),
    st.Page(analytics.render, title="Statistical Analysis", icon=":material/analytics:", url_path="statistics"),
    st.Page(trends.render, title="Trend Analysis", icon=":material/show_chart:", url_path="trends"),
    st.Page(food_price_intelligence.render, title="Food Price Intelligence", icon=":material/grass:", url_path="food-prices"),
    st.Page(forecasting.render, title="Forecasting", icon=":material/timeline:", url_path="forecasting"),
    st.Page(reports.render, title="Reports", icon=":material/description:", url_path="reports"),
    st.Page(settings.render, title="Settings", icon=":material/settings:", url_path="settings"),
]
nav = st.navigation(pages)
ui.sidebar_status()
nav.run()
