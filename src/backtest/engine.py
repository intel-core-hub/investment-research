"""Daily backtest engine: positions, cash, orders, costs and equity.

Timeline of each trading day t:
  1. orders decided at the close of the previous trading day are filled at t's close
  2. equity at t's close is recorded (cash + shares x close)
  3. the strategy sees prices up to t and may emit new target weights,
     which are filled on the next trading day

Fills: sells first, then buys. A buy fills at close x (1 + slippage_rate) and a
sell at close x (1 - slippage_rate); commission is max(rate x traded value,
minimum) per trade. If cash cannot cover the buys plus costs, the buys are
scaled down, so cash never goes negative. Fractional shares are allowed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies import Strategy

WEIGHT_TOLERANCE = 1e-9
MIN_TRADE_VALUE = 1e-6


@dataclass(frozen=True)
class CostModel:
    commission_rate: float = 0.0
    commission_min: float = 0.0
    slippage_rate: float = 0.0

    def __post_init__(self):
        if self.commission_rate < 0 or self.commission_min < 0:
            raise ValueError("commission must not be negative")
        if not 0 <= self.slippage_rate < 1:
            raise ValueError("slippage_rate must be in [0, 1)")

    def commission(self, value: float) -> float:
        return max(self.commission_rate * value, self.commission_min) if value > 0 else 0.0


@dataclass(frozen=True)
class BacktestResult:
    strategy: str
    equity: pd.Series
    cash: pd.Series
    positions: pd.DataFrame
    transactions: pd.DataFrame
    signals: pd.DataFrame

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        holdings = self.positions * prices.loc[self.positions.index, self.positions.columns]
        return holdings.div(self.equity, axis=0)


def validate_target(target: dict, tickers: list[str], strategy: str) -> np.ndarray:
    unknown = set(target) - set(tickers)
    if unknown:
        raise ValueError(f"{strategy}: signal uses unknown ticker(s) {sorted(unknown)}")
    weights = np.array([float(target.get(t, 0.0)) for t in tickers])
    if (weights < -WEIGHT_TOLERANCE).any():
        raise ValueError(f"{strategy}: negative target weight in {target}")
    if weights.sum() > 1 + WEIGHT_TOLERANCE:
        raise ValueError(f"{strategy}: target weights sum to {weights.sum():.6f} (> 1) in {target}")
    return np.clip(weights, 0, None)


def _fill(date, signal_date, ticker, side, qty, close, costs, cash):
    fill = close * (1 + costs.slippage_rate if side == "buy" else 1 - costs.slippage_rate)
    gross = qty * fill
    fee = costs.commission(gross)
    cash = cash - gross - fee if side == "buy" else cash + gross - fee
    record = {
        "date": date, "signal_date": signal_date, "ticker": ticker, "side": side,
        "quantity": qty, "close": close, "fill_price": fill, "gross_value": gross,
        "commission": fee, "slippage_cost": qty * abs(fill - close), "cash_after": cash,
    }
    return cash, record


def _execute(target, signal_date, date, tickers, close, shares, cash, costs):
    """Trade toward target weights; returns new shares, cash and the fill records."""
    equity = cash + shares @ close
    delta = target * equity / close - shares
    records = []

    for j in np.flatnonzero(delta * close < -MIN_TRADE_VALUE):
        qty = -delta[j]
        cash, record = _fill(date, signal_date, tickers[j], "sell", qty, close[j], costs, cash)
        shares[j] -= qty
        records.append(record)

    buys = np.flatnonzero(delta * close > MIN_TRADE_VALUE)
    buy_price = close * (1 + costs.slippage_rate)

    def needed(scale):
        return sum(scale * delta[j] * buy_price[j] + costs.commission(scale * delta[j] * buy_price[j])
                   for j in buys)

    scale = 1.0
    for _ in range(50):
        required = needed(scale)
        if required <= cash + 1e-9:
            break
        scale *= max(cash, 0.0) / required * (1 - 1e-12)
    for j in buys:
        qty = scale * delta[j]
        if qty * close[j] <= MIN_TRADE_VALUE:
            continue
        cash, record = _fill(date, signal_date, tickers[j], "buy", qty, close[j], costs, cash)
        shares[j] += qty
        records.append(record)
    return shares, cash, records


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    initial_cash: float,
    costs: CostModel = CostModel(),
    start: pd.Timestamp | None = None,
) -> BacktestResult:
    """Run `strategy` day by day from `start` (default: first date).

    Prices before `start` are visible to the strategy as history (warm-up) but
    no trading happens and no equity is recorded before `start`.
    """
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    missing = set(strategy.tickers) - set(prices.columns)
    if missing:
        raise ValueError(f"{strategy.name}: prices have no column(s) {sorted(missing)}")

    tickers = list(prices.columns)
    values = prices.to_numpy(dtype=float)
    first = 0 if start is None else int(prices.index.searchsorted(pd.Timestamp(start)))
    if first >= len(prices):
        raise ValueError(f"start {start} is after the last price date")

    shares = np.zeros(len(tickers))
    cash = float(initial_cash)
    pending = None
    equity, cash_path, positions, transactions, signals = [], [], [], [], []

    for i in range(first, len(prices)):
        date, close = prices.index[i], values[i]
        if pending is not None:
            target, signal_date = pending
            shares, cash, records = _execute(target, signal_date, date, tickers, close, shares, cash, costs)
            transactions.extend(records)
            pending = None

        equity.append(cash + shares @ close)
        cash_path.append(cash)
        positions.append(shares.copy())

        decision = strategy.decide(prices.iloc[: i + 1], first_day=(i == first))
        if decision is not None:
            pending = (validate_target(decision, tickers, strategy.name), date)
            signals.append({"date": date, **{t: w for t, w in zip(tickers, pending[0]) if w > 0}})

    index = prices.index[first:]
    columns = ["date", "signal_date", "ticker", "side", "quantity", "close", "fill_price",
               "gross_value", "commission", "slippage_cost", "cash_after"]
    return BacktestResult(
        strategy=strategy.name,
        equity=pd.Series(equity, index=index, name=strategy.name),
        cash=pd.Series(cash_path, index=index, name="cash"),
        positions=pd.DataFrame(positions, index=index, columns=tickers),
        transactions=pd.DataFrame(transactions, columns=columns),
        signals=pd.DataFrame(signals).fillna(0.0) if signals else pd.DataFrame(columns=["date"]),
    )
