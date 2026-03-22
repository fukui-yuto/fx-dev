"""
app.py

FX Dashboardのエントリポイント。
Streamlitのナビゲーションで3画面を管理する。

起動:
    pipenv run streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="FX Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "トレード": [
        st.Page("dashboard/pages/chart.py",   title="チャート",     icon="📈"),
        st.Page("dashboard/pages/backtest.py", title="バックテスト", icon="📊"),
    ],
    "情報": [
        st.Page("dashboard/pages/news.py",        title="ファンダメンタル", icon="📰"),
        st.Page("dashboard/pages/calendar.py",    title="経済指標",         icon="📅"),
        st.Page("dashboard/pages/data_viewer.py", title="データ確認",       icon="🗂️"),
    ],
}

pg = st.navigation(pages)
pg.run()
