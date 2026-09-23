"""Turn stored predictions into a Phase 5 strategy.

The prediction made with features at day t's close is looked up only when the
engine asks the strategy to decide at t; the engine then fills the order at the
next trading day's close. A missing prediction on a decision day is an error,
so a backtest can only run on dates that were actually predicted (e.g. only
out-of-sample dates).
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy, is_period_start


class PredictionStrategy(Strategy):
    def __init__(self, name: str, invest: pd.Series, asset: str, risk_off: str, frequency: str = "monthly"):
        self.name = name
        self.invest = invest.astype(bool)
        self.asset = asset
        self.risk_off = risk_off
        self.frequency = frequency

    @property
    def tickers(self) -> list[str]:
        return [self.asset, self.risk_off]

    def decide(self, history, first_day):
        if not (first_day or is_period_start(history, self.frequency)):
            return None
        today = history.index[-1]
        if today not in self.invest.index:
            raise KeyError(f"{self.name}: no prediction for decision day {today.date()}")
        return {self.asset: 1.0} if self.invest[today] else {self.risk_off: 1.0}
