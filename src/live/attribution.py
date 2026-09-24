"""9.9 Performance attribution: where every yen of profit or loss came from.

For each market day t, with q the shares held after day t-1, P the USD close,
X the USD/JPY mid rate and V_inv = sum q P_{t-1} X_{t-1} the invested value:

  price move = sum q (P_t - P_{t-1}) X_{t-1} = V_inv x sum w_held r     (r = P_t / P_{t-1} - 1)
    split by three sets of weights
      w_target  the latest approved target (from a signal before day t)
      w_trade   the weights just after the latest trade
      w_held    the weights at the start of day t (after drift)
    asset effect        V_inv x sum w_target r               what the research target earned
    allocation effect   V_inv x sum (w_trade - w_target) r   holdings chosen differently from the target
                                                              (whole shares, buy-only, late or rejected trades)
    rebalancing effect  V_inv x sum (w_held - w_trade) r     drift since the last trade, not rebalanced away
  FX effect         sum q P_t (X_t - X_{t-1})
  dividend effect   gross dividends
  fee / tax effect  -(fees + taxes), including taxes withheld from dividends and sales
  execution effect  sum over fills of s x quantity x (P_t X_t - P_fill X_fill), s = +1 buy / -1 sell:
                    trading at the actual price and FX rate instead of the close and the mid rate

The effects add up exactly to the profit V_t - V_{t-1} - net deposits (tested).
Transactions on a non-market day count on the next market day, as in the valuation.
Before any approved target exists, the target is the post-trade weights (allocation 0).
Prices are actual closes, so dividends appear once, as the dividend effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import real_ledger as rl
from market import MarketData

EFFECTS = ["asset", "allocation", "rebalancing", "fx", "dividend", "fee_tax", "execution"]


def _latest_target(targets: dict[pd.Timestamp, dict], before: pd.Timestamp) -> dict | None:
    known = [d for d in targets if d < before]
    if not known:
        return None
    weights = targets[max(known)]
    total = sum(weights.values())
    return {a: w / total for a, w in weights.items()} if total > 0 else None


def daily_attribution(transactions: pd.DataFrame, market: MarketData,
                      targets: dict[pd.Timestamp, dict] | None = None) -> pd.DataFrame:
    """One row per market day: each effect in JPY, the profit (value change minus deposits) and the value."""
    tx = rl.normalize(transactions).sort_values("date", kind="stable")
    columns = EFFECTS + ["profit", "value"]
    if tx.empty:
        return pd.DataFrame(columns=columns)
    targets = targets or {}
    dates = market.close.index[market.close.index >= tx["date"].iloc[0]]
    if len(dates) == 0 or tx["date"].iloc[-1] > dates[-1]:
        raise ValueError("market data must cover every transaction date")
    tx["market_day"] = dates[dates.searchsorted(tx["date"])]
    close = market.close.loc[dates]
    fx = market.fx_on(dates)

    q: dict[str, float] = {}
    cash = 0.0
    w_trade: dict[str, float] | None = None
    value_prev = 0.0
    rows = []
    for i, day in enumerate(dates):
        e = dict.fromkeys(EFFECTS, 0.0)
        p, x = close.loc[day], fx.loc[day]
        if i > 0 and q:
            p0, x0 = close.loc[dates[i - 1]], fx.loc[dates[i - 1]]
            held_value = {a: n * p0[a] * x0 for a, n in q.items()}
            invested = sum(held_value.values())
            if invested > 0:
                w_held = {a: v / invested for a, v in held_value.items()}
                trade = w_trade or w_held
                target = _latest_target(targets, day) or trade
                assets = set(w_held) | set(trade) | set(target)
                r = {a: (p[a] / p0[a] - 1) if a in close.columns else 0.0 for a in assets}
                e["asset"] = invested * sum(target.get(a, 0.0) * r[a] for a in assets)
                e["allocation"] = invested * sum((trade.get(a, 0.0) - target.get(a, 0.0)) * r[a] for a in assets)
                e["rebalancing"] = invested * sum((w_held.get(a, 0.0) - trade.get(a, 0.0)) * r[a] for a in assets)
            e["fx"] = sum(n * p[a] * (x - x0) for a, n in q.items())

        flow, traded = 0.0, False
        for r_ in tx[tx["market_day"] == day].itertuples():
            cash += rl.cash_change(r_)
            if r_.action == "DEPOSIT":
                flow += r_.amount_jpy
            elif r_.action == "WITHDRAW":
                flow -= r_.amount_jpy
            elif r_.action == "DIVIDEND":
                e["dividend"] += r_.amount_jpy
            e["fee_tax"] -= r_.fees_jpy + r_.tax_jpy + (r_.amount_jpy if r_.action in ("FEE", "TAX") else 0.0)
            if r_.action in rl.TRADES:
                s = 1 if r_.action == "BUY" else -1
                e["execution"] += s * r_.quantity * (p[r_.asset] * x - r_.price * r_.fx_rate)
                q[r_.asset] = q.get(r_.asset, 0.0) + s * r_.quantity
                if q[r_.asset] <= rl.TOLERANCE:
                    del q[r_.asset]
                traded = True
        invested_now = {a: n * p[a] * x for a, n in q.items()}
        if traded:
            total = sum(invested_now.values())
            w_trade = {a: v / total for a, v in invested_now.items()} if total > 0 else None
        value = cash + sum(invested_now.values())
        rows.append({"date": day, **e, "profit": value - value_prev - flow, "value": value})
        value_prev = value
    return pd.DataFrame(rows, columns=["date", *columns]).set_index("date")


def totals(daily: pd.DataFrame) -> pd.Series:
    """Sum of each effect, the profit, and what the effects leave unexplained (≈ 0)."""
    if daily.empty:
        return pd.Series(dtype=float)
    out = daily[EFFECTS + ["profit"]].sum()
    out["unexplained"] = out["profit"] - out[EFFECTS].sum()
    return out


def check_identity(daily: pd.DataFrame, tolerance: float = 1e-6) -> bool:
    return bool(np.allclose(daily[EFFECTS].sum(axis=1), daily["profit"], atol=tolerance, rtol=0))
