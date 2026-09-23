"""Trading strategies with a common interface.

A strategy only sees `history`: prices up to and including the decision day.
It returns target weights (ticker -> fraction of equity) or None to keep the
current holdings. The engine executes the targets on the next trading day, so
a strategy can never trade on the price it used to decide.

Strategies are stateless: the same history always produces the same signal.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

PERIODS = {"annual": "Y", "quarterly": "Q", "monthly": "M"}


def is_period_start(history: pd.DataFrame, frequency: str) -> bool:
    """True if the last day of history is the first trading day of a new period.

    Only compares the last two dates, so no future date is needed.
    """
    if len(history) < 2:
        return False
    period = PERIODS[frequency]
    return history.index[-1].to_period(period) != history.index[-2].to_period(period)


class Strategy(ABC):
    name: str

    @property
    @abstractmethod
    def tickers(self) -> list[str]:
        """Every ticker the strategy may hold or read."""

    @abstractmethod
    def decide(self, history: pd.DataFrame, first_day: bool) -> dict[str, float] | None:
        """Target weights at the close of history's last day, or None for no change."""


class BuyAndHold(Strategy):
    def __init__(self, name: str, weights: dict[str, float]):
        self.name = name
        self.weights = dict(weights)

    @property
    def tickers(self) -> list[str]:
        return list(self.weights)

    def decide(self, history, first_day):
        return dict(self.weights) if first_day else None


class PeriodicRebalance(Strategy):
    def __init__(self, name: str, weights: dict[str, float], frequency: str = "annual"):
        if frequency not in PERIODS:
            raise ValueError(f"{name}: unknown frequency {frequency!r}; use one of {list(PERIODS)}")
        self.name = name
        self.weights = dict(weights)
        self.frequency = frequency

    @property
    def tickers(self) -> list[str]:
        return list(self.weights)

    def decide(self, history, first_day):
        if first_day or is_period_start(history, self.frequency):
            return dict(self.weights)
        return None


class MovingAverageTrend(Strategy):
    """Hold `asset` while its close is above its N-day average, otherwise hold `risk_off`."""

    def __init__(self, name: str, asset: str, risk_off: str, window: int = 200, frequency: str = "monthly"):
        if window < 2:
            raise ValueError(f"{name}: window must be at least 2")
        self.name = name
        self.asset = asset
        self.risk_off = risk_off
        self.window = window
        self.frequency = frequency

    @property
    def tickers(self) -> list[str]:
        return [self.asset, self.risk_off]

    def decide(self, history, first_day):
        if not (first_day or is_period_start(history, self.frequency)):
            return None
        closes = history[self.asset]
        if len(closes) < self.window:
            return None
        above = closes.iloc[-1] > closes.iloc[-self.window:].mean()
        return {self.asset: 1.0} if above else {self.risk_off: 1.0}


class Momentum(Strategy):
    """Hold the `top_n` assets with the best trailing return, equally weighted.

    Absolute momentum filter: an asset is only held if its trailing return beats
    `risk_off`; the unused share goes to `risk_off`.
    """

    def __init__(self, name: str, assets: list[str], risk_off: str, lookback_days: int = 252,
                 top_n: int = 1, frequency: str = "monthly"):
        if not 1 <= top_n <= len(assets):
            raise ValueError(f"{name}: top_n must be between 1 and {len(assets)}")
        self.name = name
        self.assets = list(assets)
        self.risk_off = risk_off
        self.lookback_days = lookback_days
        self.top_n = top_n
        self.frequency = frequency

    @property
    def tickers(self) -> list[str]:
        return list(dict.fromkeys(self.assets + [self.risk_off]))

    def decide(self, history, first_day):
        if not (first_day or is_period_start(history, self.frequency)):
            return None
        if len(history) <= self.lookback_days:
            return None
        trailing = history.iloc[-1] / history.iloc[-1 - self.lookback_days] - 1
        hurdle = trailing[self.risk_off]
        chosen = trailing[self.assets].sort_values(ascending=False).head(self.top_n)
        chosen = chosen[chosen > hurdle]
        weights = {t: 1 / self.top_n for t in chosen.index}
        leftover = 1 - sum(weights.values())
        if leftover > 1e-12:
            weights[self.risk_off] = weights.get(self.risk_off, 0) + leftover
        return weights


STRATEGY_TYPES = {
    "buy_and_hold": (BuyAndHold, {"weights"}),
    "periodic_rebalance": (PeriodicRebalance, {"weights", "frequency"}),
    "moving_average": (MovingAverageTrend, {"asset", "risk_off", "window", "frequency"}),
    "momentum": (Momentum, {"assets", "risk_off", "lookback_days", "top_n", "frequency"}),
}


def build_strategy(spec: dict) -> Strategy:
    """Create a strategy from a config table: name, type and the type's parameters."""
    spec = dict(spec)
    name = spec.pop("name", None)
    kind = spec.pop("type", None)
    if not name:
        raise ValueError("strategy has no name")
    if kind not in STRATEGY_TYPES:
        raise ValueError(f"{name}: unknown strategy type {kind!r}; use one of {list(STRATEGY_TYPES)}")
    cls, allowed = STRATEGY_TYPES[kind]
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"{name}: unknown parameter(s) {sorted(unknown)} for {kind}")
    try:
        return cls(name, **spec)
    except TypeError as e:
        raise ValueError(f"{name}: {e}") from None
