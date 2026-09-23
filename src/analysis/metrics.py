"""Return and risk metrics shared by every analysis.

Each function takes adjusted close prices indexed by date, either a Series
(one asset) or a DataFrame (one column per asset), so all assets are measured
with exactly the same definitions.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(prices):
    return prices.pct_change().iloc[1:]


def cumulative_returns(prices):
    return prices / prices.iloc[0] - 1


def total_return(prices):
    return prices.iloc[-1] / prices.iloc[0] - 1


def years(prices) -> float:
    return (prices.index[-1] - prices.index[0]).days / 365.25


def cagr(prices):
    return (1 + total_return(prices)) ** (1 / years(prices)) - 1


def annual_volatility(prices):
    return daily_returns(prices).std() * np.sqrt(TRADING_DAYS)


def drawdowns(prices):
    return prices / prices.cummax() - 1


def max_drawdown(prices):
    return drawdowns(prices).min()


def rolling_return(prices, window: int):
    """Trailing return over `window` trading days, annualized (window=252 is the plain 1y return)."""
    return ((prices / prices.shift(window)) ** (TRADING_DAYS / window) - 1).iloc[window:]


def rolling_volatility(prices, window: int):
    return (daily_returns(prices).rolling(window).std() * np.sqrt(TRADING_DAYS)).iloc[window - 1 :]


def period_returns(prices, freq: str):
    """Returns per calendar period ("YE" yearly, "ME" monthly); the first period starts at the first price."""
    period_end = prices.resample(freq).last()
    returns = period_end.pct_change()
    returns.iloc[0] = period_end.iloc[0] / prices.iloc[0] - 1
    return returns


def correlation(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()
