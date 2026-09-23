"""Multi-asset portfolio simulation: validation, buy & hold, rebalancing, DCA and metrics.

Prices are adjusted closes (total return). Trades happen at the close of the
event day: money is split by the target weights, and on a rebalance day the
whole portfolio is reset to the target weights. There are no costs or taxes.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

import metrics as m
from dca import drawdown_stats

REBALANCE_PERIODS = {"none": None, "annual": "Y", "quarterly": "Q", "monthly": "M"}
WEIGHT_TOLERANCE = 1e-6


class PortfolioConfigError(ValueError):
    pass


def validate_config(config: dict, available_tickers) -> None:
    """Raise PortfolioConfigError listing every problem found in the portfolio config."""
    available = set(available_tickers)
    errors = []

    for key in ("initial_investment", "monthly_contribution"):
        value = config.get(key)
        if not isinstance(value, Real) or isinstance(value, bool) or value <= 0:
            errors.append(f"{key} must be a positive number, got {value!r}")

    frequencies = config.get("rebalance_frequencies", [])
    if not frequencies:
        errors.append("rebalance_frequencies must list at least one frequency")
    for freq in frequencies:
        if freq not in REBALANCE_PERIODS:
            errors.append(f"unknown rebalance frequency {freq!r}; use one of {list(REBALANCE_PERIODS)}")
    if config.get("default_rebalance") not in frequencies:
        errors.append(f"default_rebalance {config.get('default_rebalance')!r} is not in rebalance_frequencies")

    if config.get("risk_free_ticker") not in available:
        errors.append(f"risk_free_ticker {config.get('risk_free_ticker')!r} is not in the price data")

    portfolios = config.get("portfolios", [])
    if not portfolios:
        errors.append("no portfolios defined")
    names = [p.get("name") for p in portfolios]
    for name in {n for n in names if names.count(n) > 1}:
        errors.append(f"portfolio name {name!r} is used more than once")

    for p in portfolios:
        label = p.get("name") or "<unnamed>"
        if not p.get("name"):
            errors.append("a portfolio has no name")
        weights = p.get("weights")
        if not isinstance(weights, dict) or not weights:
            errors.append(f"{label}: weights must be a non-empty table of ticker = weight")
            continue
        for ticker, w in weights.items():
            if ticker not in available:
                errors.append(f"{label}: {ticker} is not in the price data")
            if not isinstance(w, Real) or isinstance(w, bool) or not 0 < w <= 1:
                errors.append(f"{label}: weight of {ticker} must be > 0 and <= 1, got {w!r}")
        numeric = [w for w in weights.values() if isinstance(w, Real) and not isinstance(w, bool)]
        if len(numeric) == len(weights) and abs(sum(numeric) - 1) > WEIGHT_TOLERANCE:
            errors.append(f"{label}: weights sum to {sum(numeric):.6f}, not 1")

    if errors:
        raise PortfolioConfigError("invalid portfolio config:\n  - " + "\n  - ".join(errors))


def period_starts(index: pd.DatetimeIndex, period: str) -> pd.DatetimeIndex:
    """First trading day of each calendar period ("M", "Q" or "Y") in the index."""
    firsts = index.to_series().groupby(index.to_period(period)).first()
    return pd.DatetimeIndex(firsts.to_numpy())


def rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Rebalance days: the start of every period after the first (day 1 already buys at target)."""
    period = REBALANCE_PERIODS[frequency]
    if period is None:
        return pd.DatetimeIndex([])
    return period_starts(index, period)[1:]


@dataclass(frozen=True)
class PortfolioResult:
    value: pd.Series
    cash_flows: pd.Series
    units: pd.DataFrame
    traded_value: float
    rebalances: int

    @property
    def contributions(self) -> pd.Series:
        return self.cash_flows.cumsum()

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        holdings = self.units * prices[self.units.columns]
        return holdings.div(holdings.sum(axis=1), axis=0)


def simulate_portfolio(
    prices: pd.DataFrame,
    weights: dict,
    rebalance: str = "none",
    initial_investment: float = 0.0,
    monthly_contribution: float = 0.0,
) -> PortfolioResult:
    """Invest by target weights and optionally rebalance back to them.

    initial_investment is invested on the first day (lump sum); monthly_contribution
    on the first trading day of every month (DCA). Both can be combined.
    """
    if initial_investment <= 0 and monthly_contribution <= 0:
        raise ValueError("need a positive initial_investment or monthly_contribution")
    target = pd.Series(weights, dtype=float)
    prices = prices[target.index]
    index = prices.index

    flows = pd.Series(0.0, index=index)
    flows.iloc[0] += initial_investment
    if monthly_contribution > 0:
        flows.loc[period_starts(index, "M")] += monthly_contribution
    rebalance_days = set(rebalance_dates(index, rebalance))

    events = sorted(set(flows.index[flows > 0]) | rebalance_days)
    units = pd.DataFrame(np.nan, index=index, columns=target.index)
    held = pd.Series(0.0, index=target.index)
    traded = 0.0
    for day in events:
        price = prices.loc[day]
        if flows[day] > 0:
            held = held + flows[day] * target / price
        if day in rebalance_days:
            goal = (held * price).sum() * target / price
            traded += float(((goal - held).abs() * price).sum())
            held = goal
        units.loc[day] = held

    units = units.ffill()
    value = (units * prices).sum(axis=1)
    return PortfolioResult(value, flows, units, traded, len(rebalance_days))


def daily_returns(result: PortfolioResult) -> pd.Series:
    """Time-weighted daily returns: the day's contribution is removed before measuring the gain."""
    return ((result.value - result.cash_flows) / result.value.shift(1) - 1).iloc[1:]


def nav(result: PortfolioResult) -> pd.Series:
    """Growth of 1 unit of currency, excluding the effect of contributions."""
    growth = (1 + daily_returns(result)).cumprod()
    return pd.concat([pd.Series([1.0], index=result.value.index[:1]), growth])


def summarize(result: PortfolioResult, risk_free_prices: pd.Series) -> dict:
    growth = nav(result)
    drawdown = m.drawdowns(growth)
    years = m.years(growth)
    money_dd = drawdown_stats(result.value, result.contributions)
    return {
        "start": growth.index[0].date(),
        "end": growth.index[-1].date(),
        "total_return": float(m.total_return(growth)),
        "cagr": float(m.cagr(growth)),
        "annual_volatility": float(m.annual_volatility(growth)),
        "max_drawdown": float(drawdown.min()),
        "max_drawdown_date": drawdown.idxmin().date(),
        "sharpe_ratio": float(m.sharpe_ratio(growth, risk_free_prices.loc[growth.index])),
        "rebalances": result.rebalances,
        # one-way turnover: half of everything bought and sold, per year, relative to average value
        "annual_turnover": result.traded_value / 2 / float(result.value.mean()) / years,
        "total_contributed": float(result.contributions.iloc[-1]),
        "final_value": float(result.value.iloc[-1]),
        "return_on_contributions": float(result.value.iloc[-1] / result.contributions.iloc[-1] - 1),
        "money_max_drawdown": money_dd["max_drawdown"],
        "money_days_to_recover": money_dd["days_to_recover"],
    }


def diversification(weights: dict, annual_covariance: pd.DataFrame) -> dict:
    """Volatility of the weighted mix versus the weighted average of each asset's volatility.

    If every pair were perfectly correlated the two would be equal; the gap is
    the risk removed by imperfect correlation (for weights held constant).
    """
    w = pd.Series(weights, dtype=float)
    cov = annual_covariance.loc[w.index, w.index]
    asset_vols = np.sqrt(np.diag(cov))
    weighted_average = float((w.to_numpy() * asset_vols).sum())
    portfolio = float(np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy()))
    return {
        "weighted_average_volatility": weighted_average,
        "portfolio_volatility": portfolio,
        "diversification_ratio": weighted_average / portfolio,
        "volatility_reduction": 1 - portfolio / weighted_average,
    }
