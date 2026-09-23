"""Phase 8: Investment Research Dashboard.

    streamlit run app.py

The sidebar (assets, period, analysis mode) is rendered here, so it is shared
by every page. Only the open page runs; its calculations come from the
Phase 2-7 modules through src/dashboard/compute.py and ml_compute.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src" / "dashboard") not in sys.path:
    sys.path.insert(0, str(ROOT / "src" / "dashboard"))

import streamlit as st  # noqa: E402

import common  # noqa: E402
import sidebar  # noqa: E402

st.set_page_config(page_title="Investment Research Dashboard", page_icon=":material/query_stats:", layout="wide")
# smaller metric values so that seven KPIs fit on one row without being cut off
st.html("<style>[data-testid='stMetricValue'] { font-size: 1.45rem; }</style>")

PAGES = [
    st.Page("src/dashboard/tab_overview.py", title="① Overview", url_path="overview", default=True),
    st.Page("src/dashboard/tab_asset.py", title="② Asset Analysis", url_path="assets"),
    st.Page("src/dashboard/tab_portfolio.py", title="③ Portfolio", url_path="portfolio"),
    st.Page("src/dashboard/tab_backtest.py", title="④ Backtest", url_path="backtest"),
    st.Page("src/dashboard/tab_stats.py", title="⑤ Statistics", url_path="statistics"),
    st.Page("src/dashboard/tab_ml.py", title="⑥ Machine Learning", url_path="ml"),
]

page = st.navigation(PAGES, position="top")
sidebar.render(common.prices())
page.run()
