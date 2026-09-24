"""9.8 Research vs paper vs real: execution costs and performance side by side.

Execution cost of one fill, in JPY:
    |P_actual - P_backtest| x quantity x FX_mid  +  fees  +  FX cost
  P_backtest  the close on the fill date. The Phase 5 backtest fills at the close of
              the trading day after the signal, so a fill on day t is compared with t's close.
  FX cost     quantity x P_actual x |FX_applied - FX_mid|: the spread paid to the broker.
`execution_effect_jpy` is the signed price-and-FX part (positive = better than the
backtest assumption); attribution.py uses the same definition. `delay_days` counts
business days after the backtest's fill day (0 = filled on schedule).

Performance is compared with time-weighted returns in JPY, so deposits do not count
as gains. The backtest runs the same Phase 5 strategy on dividend-adjusted prices and
is converted to JPY with the same FX series. Sharpe uses a risk-free rate of 0
(JPY cash earns about nothing).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import real_ledger as rl
from market import MarketData
from risk_controller import business_days_between
from settings import ROOT

sys.path[:0] = [str(ROOT / "src" / d) for d in ("backtest", "statistics") if str(ROOT / "src" / d) not in sys.path]

import performance as pf  # noqa: E402
from engine import CostModel, run_backtest  # noqa: E402
from strategies import build_strategy  # noqa: E402


def execution_costs(transactions: pd.DataFrame, market: MarketData, signal_dates: dict[str, str]) -> pd.DataFrame:
    """One row per BUY/SELL: prices and FX actually paid vs the backtest assumption, and the costs."""
    trades = rl.normalize(transactions)
    trades = trades[trades["action"].isin(rl.TRADES)]
    rows = []
    for r in trades.itertuples():
        on_market_day = r.date in market.close.index and r.asset in market.close.columns
        close = float(market.close.loc[r.date, r.asset]) if on_market_day else np.nan
        fx_mid = float(market.fx_on(pd.DatetimeIndex([r.date])).iloc[0])
        sign = 1 if r.action == "BUY" else -1
        signal = signal_dates.get(r.order_id)
        delay = business_days_between(pd.Timestamp(signal), r.date) - 1 if signal else np.nan
        rows.append({
            "date": r.date, "asset": r.asset, "side": r.action.lower(), "quantity": r.quantity,
            "price_actual": r.price, "price_backtest": close, "fx_applied": r.fx_rate, "fx_mid": fx_mid,
            "price_cost_jpy": abs(r.price - close) * r.quantity * fx_mid,
            "fees_jpy": r.fees_jpy,
            "fx_cost_jpy": r.quantity * r.price * abs(r.fx_rate - fx_mid),
            "execution_effect_jpy": sign * (close * fx_mid - r.price * r.fx_rate) * r.quantity,
            "order_id": r.order_id, "signal_date": signal, "delay_days": delay,
        })
    table = pd.DataFrame(rows, columns=["date", "asset", "side", "quantity", "price_actual", "price_backtest",
                                        "fx_applied", "fx_mid", "price_cost_jpy", "fees_jpy", "fx_cost_jpy",
                                        "execution_effect_jpy", "order_id", "signal_date", "delay_days"])
    table["execution_cost_jpy"] = table["price_cost_jpy"] + table["fees_jpy"] + table["fx_cost_jpy"]
    return table


def backtest_returns(market: MarketData, spec: dict, start, end, costs: CostModel) -> tuple[pd.Series, float]:
    """Daily JPY returns of the Phase 5 strategy from `start`, and its trading costs / average equity."""
    strategy = build_strategy(spec)
    prices = market.adjusted.loc[:pd.Timestamp(end), strategy.tickers].dropna()
    result = run_backtest(prices, strategy, 10_000.0, costs, start=pd.Timestamp(start))
    equity_jpy = result.equity * market.fx_on(result.equity.index)
    tx = result.transactions
    cost_usd = float(tx["commission"].sum() + tx["slippage_cost"].sum()) if len(tx) else 0.0
    return (equity_jpy / equity_jpy.shift(1) - 1).iloc[1:], cost_usd / float(result.equity.mean())


def cost_ratio(execution: pd.DataFrame, valuations: pd.DataFrame) -> float:
    """Execution costs (fees + FX spread + price difference) / average total value."""
    if execution.empty or valuations.empty:
        return np.nan
    return float(execution["execution_cost_jpy"].sum() / valuations["total_jpy"].mean())


def metrics(daily_returns: pd.Series) -> dict:
    """Metrics of a time-weighted return series (growth of 1 JPY)."""
    r = daily_returns.dropna()
    if len(r) < 2:
        return {"start": None, "end": None, "days": len(r), "total_return": np.nan, "cagr": np.nan,
                "volatility": np.nan, "max_drawdown": np.nan, "sharpe_ratio": np.nan}
    growth = pd.concat([pd.Series([1.0], index=[r.index[0] - pd.Timedelta(days=1)]), (1 + r).cumprod()])
    return {"start": r.index[0].date(), "end": r.index[-1].date(), "days": len(r),
            "total_return": pf.total_return(growth), "cagr": pf.cagr(growth), "volatility": pf.volatility(r),
            "max_drawdown": pf.drawdown_stats(growth)["max_drawdown"],
            "sharpe_ratio": pf.sharpe_ratio(r, pd.Series(0.0, index=r.index))}


def comparison(series: dict[str, pd.Series], cost_ratios: dict[str, float]) -> pd.DataFrame:
    """Backtest / paper / real on their common dates (the latest start of the three).

    cost_ratio: trading costs over the whole history / average value (not only the common dates).
    """
    series = {k: v.dropna() for k, v in series.items() if v is not None and len(v.dropna())}
    if not series:
        return pd.DataFrame()
    start = max(v.index[0] for v in series.values())
    rows = [{"series": k, **metrics(v.loc[start:]), "cost_ratio": cost_ratios.get(k, np.nan)}
            for k, v in series.items()]
    return pd.DataFrame(rows).set_index("series")
