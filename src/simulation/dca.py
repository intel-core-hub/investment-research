"""Fixed monthly contribution and lump-sum simulations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = ROOT / "data" / "processed" / "prices.csv"


@dataclass(frozen=True)
class SimulationResult:
    ticker: str
    monthly_contribution: float
    contributions: pd.Series
    units: pd.Series
    portfolio_value: pd.Series

    @property
    def invested(self) -> float:
        return float(self.contributions.iloc[-1])

    @property
    def final_value(self) -> float:
        return float(self.portfolio_value.iloc[-1])

    @property
    def gain(self) -> float:
        return self.final_value - self.invested

    @property
    def return_on_contributions(self) -> float:
        return self.final_value / self.invested - 1


def load_prices(path: Path = PRICES_PATH) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def contribution_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First available trading day of each calendar month."""
    frame = pd.DataFrame(index=index)
    dates = frame.groupby(index.to_period("M")).apply(
        lambda group: group.index[0], include_groups=False
    )
    return pd.DatetimeIndex(dates.to_numpy())


def simulate_dca(
    prices: pd.Series, monthly_contribution: float = 10_000
) -> SimulationResult:
    """Invest a fixed amount on the first available trading day each month."""
    if monthly_contribution <= 0:
        raise ValueError("monthly_contribution must be positive")
    if prices.empty:
        raise ValueError("prices must not be empty")

    dates = contribution_dates(prices.index)
    contribution = pd.Series(0.0, index=prices.index)
    contribution.loc[dates] = monthly_contribution
    cumulative_contribution = contribution.cumsum()

    units = (contribution / prices).cumsum()
    value = units * prices

    return SimulationResult(
        ticker=prices.name or "asset",
        monthly_contribution=monthly_contribution,
        contributions=cumulative_contribution,
        units=units,
        portfolio_value=value,
    )


def simulate_lump_sum(
    prices: pd.Series, monthly_contribution: float = 10_000
) -> pd.Series:
    """Invest the same total amount as DCA on the first available day."""
    if monthly_contribution <= 0:
        raise ValueError("monthly_contribution must be positive")
    if prices.empty:
        raise ValueError("prices must not be empty")

    months = prices.index.to_period("M").nunique()
    total = monthly_contribution * months
    return (total / prices.iloc[0]) * prices


def summarize(result: SimulationResult, lump_sum: pd.Series) -> dict:
    """Return a compact summary for one DCA run."""
    return {
        "ticker": result.ticker,
        "start": result.portfolio_value.index[0].date(),
        "end": result.portfolio_value.index[-1].date(),
        "months": int(result.invested / result.monthly_contribution),
        "monthly_contribution": result.monthly_contribution,
        "total_contributed": result.invested,
        "final_value": result.final_value,
        "gain": result.gain,
        "return_on_contributions": result.return_on_contributions,
        "lump_sum_final_value": float(lump_sum.iloc[-1]),
        "dca_vs_lump_sum": result.final_value / lump_sum.iloc[-1] - 1,
    }
