"""Fixed monthly contribution and lump-sum simulations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = ROOT / "data" / "processed" / "prices.csv"
EPSILON = 1e-9


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


def contribution_adjusted_drawdown(value: pd.Series, contributions: pd.Series) -> pd.Series:
    """Drawdown of a portfolio that receives contributions.

    The high-water mark is the previous peak value plus everything contributed
    since then, so new money neither hides a loss nor counts as a recovery.
    With constant contributions (a lump sum) this is the ordinary drawdown.
    """
    high_water = contributions + (value - contributions).cummax()
    return value / high_water - 1


def drawdown_stats(value: pd.Series, contributions: pd.Series) -> dict:
    """Largest contribution-adjusted drawdown and how long it took to recover.

    days_to_recover counts calendar days from the trough back to the high-water
    mark; it is NaN when the portfolio has not recovered by the last date.
    """
    dd = contribution_adjusted_drawdown(value, contributions)
    trough = dd.idxmin()
    if dd[trough] >= -EPSILON:
        return {"max_drawdown": 0.0, "drawdown_peak": pd.NaT, "drawdown_trough": pd.NaT,
                "recovery_date": pd.NaT, "days_to_recover": 0.0}

    before = dd.loc[:trough]
    peak = before[before >= -EPSILON].index[-1]
    after = dd.loc[trough:]
    recovered = after[after >= -EPSILON]
    recovery = recovered.index[0] if len(recovered) else pd.NaT
    return {
        "max_drawdown": float(dd[trough]),
        "drawdown_peak": peak,
        "drawdown_trough": trough,
        "recovery_date": recovery,
        "days_to_recover": float((recovery - trough).days) if len(recovered) else float("nan"),
    }


def cost_basis_stats(value: pd.Series, contributions: pd.Series) -> dict:
    """How far and how often the portfolio fell below the money put in."""
    vs_contributed = value / contributions - 1
    return {
        "worst_vs_contributed": float(vs_contributed.min()),
        "share_days_below_contributed": float((vs_contributed < 0).mean()),
    }


def _date(ts: pd.Timestamp):
    return ts.date() if pd.notna(ts) else None


def lump_sum_contributions(lump_sum: pd.Series) -> pd.Series:
    return pd.Series(float(lump_sum.iloc[0]), index=lump_sum.index)


def summarize(result: SimulationResult, lump_sum: pd.Series) -> dict:
    """Return a compact summary for one DCA run."""
    dca_dd = drawdown_stats(result.portfolio_value, result.contributions)
    lump_dd = drawdown_stats(lump_sum, lump_sum_contributions(lump_sum))
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
        "dca_max_drawdown": dca_dd["max_drawdown"],
        "dca_drawdown_trough": _date(dca_dd["drawdown_trough"]),
        "dca_days_to_recover": dca_dd["days_to_recover"],
        "lump_sum_max_drawdown": lump_dd["max_drawdown"],
        "lump_sum_drawdown_trough": _date(lump_dd["drawdown_trough"]),
        "lump_sum_days_to_recover": lump_dd["days_to_recover"],
        **{f"dca_{k}": v for k, v in cost_basis_stats(result.portfolio_value, result.contributions).items()},
    }
