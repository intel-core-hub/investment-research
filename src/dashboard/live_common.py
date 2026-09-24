"""Shared by the Phase 9 pages: which account to show (demo or real), its settings and market data.

Demo: a synthetic account built by src/live/demo.py in a temporary folder for this
browser session (approvals and fills made there are thrown away). Real: data/live/,
which git ignores.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from compute import ROOT

if str(ROOT / "src" / "live") not in sys.path:
    sys.path.insert(0, str(ROOT / "src" / "live"))

import demo  # noqa: E402
import market as mk  # noqa: E402
from settings import LiveStore, default_store, load_settings  # noqa: E402

import common  # noqa: E402

DEMO, REAL = "デモ（合成データ）", "実運用（data/live）"


@dataclass(frozen=True)
class Account:
    mode: str
    store: LiveStore
    settings: dict
    market: mk.MarketData | None
    today: pd.Timestamp

    @property
    def is_demo(self) -> bool:
        return self.mode == DEMO


@st.cache_resource(show_spinner="デモ用の口座を作成中（合成データで過去 1 年分の運用を再現）…")
def demo_template() -> Path:
    """Built once per server: a year of the real workflow on past data (never real money)."""
    fx = None
    live = mk.load(default_store(load_settings()))
    if live is not None:
        fx = live.fx
    root = Path(tempfile.mkdtemp(prefix="live_demo_template_")) / "account"
    demo.build(root, demo.demo_settings(load_settings()), demo.research_market(fx), start="2025-09-01",
               reject_every=5)
    return root


def demo_store() -> LiveStore:
    """This session's own copy of the demo account."""
    if "demo_root" not in st.session_state:
        root = Path(tempfile.mkdtemp(prefix="live_demo_")) / "account"
        shutil.copytree(demo_template(), root)
        st.session_state["demo_root"] = str(root)
    return LiveStore(Path(st.session_state["demo_root"]))


def reset_demo() -> None:
    st.session_state.pop("demo_root", None)


def account() -> Account:
    """Mode switch at the top of the page, then the matching store, settings and market data."""
    real_store = default_store(load_settings())
    default = REAL if real_store.transactions.exists() else DEMO
    mode = st.radio("口座", [DEMO, REAL], horizontal=True, key=common.persistent("live_mode", default),
                    help="デモは合成データで、このブラウザのセッションだけの一時フォルダに書き込みます。"
                         "実運用は data/live/（git の管理外）を読み書きします。")
    common.remember("live_mode")
    if mode == DEMO:
        st.info("デモ表示です。研究データ（配当込み価格）で過去 1 年の運用を再現した合成の口座で、実際の取引ではありません。"
                "60/40・毎月 5 万円・整数株・5 回に 1 回は却下、の設定です。", icon=":material/science:")
        store = demo_store()
        live = mk.load(real_store)
        market = demo.research_market(live.fx if live is not None else None)
        return Account(mode, store, demo.demo_settings(load_settings()), market, market.as_of)
    settings = load_settings()
    return Account(mode, real_store, settings, mk.load(real_store), pd.Timestamp.now().normalize())


def need_market(acc: Account) -> bool:
    if acc.market is None:
        st.warning("実運用の市場データがまだありません。先に `python src/live/run_live.py fetch` を実行してください"
                   "（data/live/market/ に保存され、研究データは変わりません）。")
        return False
    return True
