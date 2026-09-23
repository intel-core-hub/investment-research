"""Common sidebar: assets, period and analysis mode, shared by every page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from common import SETTINGS_KEY, Settings

DEFAULT_ASSETS = ["VOO", "VT", "EWJ", "BND", "BIL"]
PERIOD_PRESETS = {"全期間": None, "直近10年": 10, "直近5年": 5, "直近3年": 3, "直近1年": 1}
MODES = {"標準": False, "詳細（データ表・CSV も表示）": True}


def _apply_preset(first, last) -> None:
    years = PERIOD_PRESETS[st.session_state["period_preset"]]
    start = first if years is None else max(first, (pd.Timestamp(last) - pd.DateOffset(years=years)).date())
    st.session_state["period"] = (start, last)


def render(prices: pd.DataFrame) -> Settings:
    tickers = list(prices.columns)
    first, last = prices.index[0].date(), prices.index[-1].date()
    state = st.session_state
    state.setdefault("assets", [t for t in DEFAULT_ASSETS if t in tickers])
    state.setdefault("period", (first, last))
    state.setdefault("period_preset", "全期間")

    sb = st.sidebar
    sb.header("共通設定")
    assets = sb.multiselect("資産", tickers, key="assets",
                            help="① Overview と ② Asset Analysis の対象。⑤ では比較対象の候補になります。")
    sb.selectbox("期間の選び方", list(PERIOD_PRESETS), key="period_preset", on_change=_apply_preset,
                 args=(first, last))
    period = sb.date_input("期間", min_value=first, max_value=last, key="period", format="YYYY-MM-DD",
                           help="①〜⑤ に適用。⑥ Machine Learning は Phase 7 の固定の期間分割を使います。")
    mode = sb.radio("分析モード", list(MODES), key="mode")

    if isinstance(period, (tuple, list)) and len(period) == 2:
        start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    else:
        sb.warning("期間の終了日を選んでください。選び終わるまでは全期間を使います。")
        start, end = pd.Timestamp(first), pd.Timestamp(last)

    sb.divider()
    sb.caption(f"データ：USD 建ての配当込み調整後終値（{first} 〜 {last}）。無リスク金利は BIL。"
               "金額はすべて正規化した値で、実際の資産額ではありません。")

    settings = Settings(assets=tuple(assets), start=start, end=end, detailed=MODES[mode])
    state[SETTINGS_KEY] = settings
    return settings
