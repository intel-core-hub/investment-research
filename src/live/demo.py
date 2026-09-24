"""A synthetic demo account for the dashboard and the tests; never real money.

It runs the real workflow month by month on past data: deposit -> propose ->
(approve) -> fill on the next trading day at a slightly different price and at the
broker's FX rate -> recalculate. Everything is written to a folder the caller gives
(a temporary folder on the dashboard), never to data/live/.

Market data: the research prices (data/processed, dividend-adjusted, so the demo
has no separate dividend rows) and, for USD/JPY, the live FX file if it was
downloaded, otherwise a constant 150 JPY per USD.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import audit
import bridge
import fx_and_taxes as fxt
from market import MarketData
from settings import ROOT, LiveStore, merge

if str(ROOT / "src" / "data") not in sys.path:
    sys.path.insert(0, str(ROOT / "src" / "data"))

DEMO_SETTINGS = {"signal": {"strategy": "60/40 annual rebalance", "mode": "buy_only", "fractional_shares": False},
                 "account": {"type": "nisa"}, "limits": {"max_order_jpy": 150_000}}
CONSTANT_FX = 150.0


def research_market(fx: pd.Series | None = None) -> MarketData:
    import prices as rawdata

    close = pd.read_csv(ROOT / "data" / "processed" / "prices.csv", index_col=0, parse_dates=True)
    volume = pd.DataFrame({"VOO": rawdata.load_raw("VOO")["Volume"]}).reindex(close.index)
    if fx is None or fx.dropna().empty:
        fx = pd.Series(CONSTANT_FX, index=close.index, name="usdjpy")
    return MarketData(close=close, adjusted=close, volume=volume, fx=fx)


def truncate(market: MarketData, day: pd.Timestamp) -> MarketData:
    return MarketData(market.close.loc[:day], market.adjusted.loc[:day], market.volume.loc[:day],
                      market.fx.loc[:day])


def build(root: Path, settings: dict, market: MarketData, start: str = "2024-10-01", monthly_jpy: float = 50_000,
          seed: int = 0, reject_every: int = 0) -> LiveStore:
    """Create a demo history in `root` (must be empty or missing). reject_every=n rejects every n-th proposal."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"{root} is not empty")
    store = LiveStore(root).ensure()
    rng = np.random.default_rng(seed)
    costs = fxt.CostSettings.from_config(settings)
    account = settings["account"]["type"]
    days = market.close.index[market.close.index >= pd.Timestamp(start)]
    month_starts = days.to_series().groupby(days.to_period("M")).first()
    for n, day in enumerate(month_starts, start=1):
        bridge.record_cash(store, "DEPOSIT", day, monthly_jpy, account, note="demo")
        proposal, _ = bridge.propose(store, settings, truncate(market, day), today=day)
        if proposal.status != "PASS":
            continue
        approve = not (reject_every and n % reject_every == 0)
        audit.decide(store, proposal.decision_id, approve, note="demo")
        if not approve:
            continue
        later = days[days > day]
        if len(later) == 0:
            continue
        fill_day = later[0]
        fx_mid = float(market.fx_on(pd.DatetimeIndex([fill_day])).iloc[0])
        for o in sorted(proposal.orders, key=lambda o: o.side != "sell"):
            close = float(market.close.loc[fill_day, o.asset])
            price = close * (1 + rng.normal(0, 0.001))    # a real fill is rarely exactly the close
            rate = fxt.conversion_rate(fx_mid, costs.fx_spread_jpy, o.side)
            fees = fxt.commission_jpy(o.quantity * price * rate, costs)
            bridge.record_fill(store, o.order_id, fill_day, o.quantity, price, rate, account, fees_jpy=fees)
    bridge.propose(store, settings, market, today=market.as_of)   # fills the last paper orders
    return store


def demo_settings(settings: dict) -> dict:
    return merge(copy.deepcopy(settings), DEMO_SETTINGS)
