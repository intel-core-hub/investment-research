"""Streamlit helpers shared by every page: cached data, common settings, persistent widgets, tables."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

import compute as cp
from charts import SERIES_COLORS

SETTINGS_KEY = "settings"


@st.cache_data(show_spinner=False)
def prices() -> pd.DataFrame:
    return cp.load_prices()


@dataclass(frozen=True)
class Settings:
    """Chosen in the sidebar (app.py) and shared by every page through st.session_state."""
    assets: tuple[str, ...]
    start: pd.Timestamp
    end: pd.Timestamp
    detailed: bool


def settings() -> Settings:
    return st.session_state[SETTINGS_KEY]


def period_prices(columns: list[str] | None = None, min_days: int = 2) -> pd.DataFrame | None:
    """Prices of the chosen period (and columns); warns and returns None if the period is too short."""
    s = settings()
    p = cp.trading_days(prices(), s.start, s.end)
    if columns is not None:
        p = p[list(columns)]
    if len(p) < min_days:
        st.warning(f"選んだ期間の取引日は {len(p)} 日です。この分析には {min_days} 日以上必要です。"
                   "サイドバーで期間を広げてください。")
        return None
    return p


BACKTEST_SETTINGS_KEY = "backtest_settings"


def backtest_settings() -> cp.BacktestSettings:
    """Capital, costs and strategy parameters last set on ④ Backtest (config/backtest.toml by default)."""
    return st.session_state.get(BACKTEST_SETTINGS_KEY) or cp.default_backtest_settings()


def strategy_names() -> list[str]:
    return list(cp.strategy_specs(cp.default_backtest_settings()))


def strategy_colors() -> dict[str, str]:
    """Fixed color per strategy, in config order (Buy & Hold VOO shares VOO's color)."""
    return dict(zip(strategy_names(), SERIES_COLORS))


@st.cache_data(show_spinner="バックテストを実行中…")
def run_strategies(start, end, names: tuple[str, ...], settings: cp.BacktestSettings):
    return cp.run_strategies(prices(), list(names), settings, start, end)


def need_assets(minimum: int = 1) -> list[str] | None:
    assets = list(settings().assets)
    if len(assets) < minimum:
        st.warning(f"サイドバーで資産を {minimum} つ以上選んでください。")
        return None
    return assets


def persistent(key: str, default) -> str:
    """Widget key whose value survives page switches.

    Streamlit drops the state of widgets that are not on the current page, so the
    value is copied to `key` (see `remember`) and restored from there.
    """
    widget_key = f"_{key}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(key, default)
    return widget_key


def remember(*keys: str) -> None:
    for key in keys:
        st.session_state[key] = st.session_state[f"_{key}"]


def table(frame: pd.DataFrame, percent: tuple[str, ...] = (), decimals: dict[str, int] | None = None,
          labels: dict[str, str] | None = None, money: tuple[str, ...] = (), height: int | None = None,
          hide_index: bool = False) -> None:
    """Sortable table; fractions shown as percentages with 2 decimals, dates without time,
    labels translated for display. Undefined values (NaN) appear as "None".

    Without a height, the table is tall enough to show every row (up to 25) without scrolling.
    """
    labels = labels or {}
    frame = frame.copy()
    config = {}
    for col in frame.columns:
        label = labels.get(col, col)
        if col in percent:
            frame[col] = pd.to_numeric(frame[col]) * 100
            config[col] = st.column_config.NumberColumn(label, format="%.2f%%")
        elif col in money:
            config[col] = st.column_config.NumberColumn(label, format="%,.0f")
        elif decimals and col in decimals:
            frame[col] = pd.to_numeric(frame[col])
            config[col] = st.column_config.NumberColumn(label, format=f"%.{decimals[col]}f")
        elif pd.api.types.is_datetime64_any_dtype(frame[col]):
            config[col] = st.column_config.DateColumn(label, format="YYYY-MM-DD")
        else:
            config[col] = st.column_config.Column(label)
    # header + rows + room for a horizontal scrollbar on wide tables
    st.dataframe(frame, column_config=config, height=height or 50 + 35 * min(len(frame), 25),
                 hide_index=hide_index)


def data_view(frame: pd.DataFrame, title: str, file_name: str) -> None:
    """In detailed mode, the chart's underlying numbers as a table with a CSV download."""
    if not settings().detailed:
        return
    with st.expander(f"データ表：{title}"):
        st.dataframe(frame)
        st.download_button("CSV をダウンロード", frame.to_csv().encode("utf-8"), file_name=file_name,
                           mime="text/csv", key=f"download_{file_name}")


def fmt_pct(value: float, digits: int = 2) -> str:
    return "−" if pd.isna(value) else f"{value:.{digits}%}".replace("-", "−")


def fmt_num(value: float, digits: int = 2) -> str:
    return "−" if pd.isna(value) else f"{value:.{digits}f}".replace("-", "−")
