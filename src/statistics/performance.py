"""Performance, risk, distribution and rolling statistics for equity curves.

Conventions (identical for every strategy):
- Input is a daily equity (or price) series. Missing values are dropped first,
  so returns are measured between consecutive valid observations.
- Returns are simple returns in decimals (0.01 = 1%).
- Annualization: mean x 252, standard deviation x sqrt(252), CAGR by calendar
  years (days / 365.25).
- The risk-free rate is the daily return of a T-bill series (BIL) on the same days.
- When a statistic is undefined (too few observations, zero volatility, zero
  downside deviation, zero drawdown in a ratio) it is NaN, never +/-inf.
- VaR and CVaR are reported as positive numbers for losses (0.02 = a 2% loss).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EPSILON = 1e-12
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def clean(equity: pd.Series) -> pd.Series:
    return equity.dropna()


def daily_returns(equity: pd.Series) -> pd.Series:
    e = clean(equity)
    return (e / e.shift(1) - 1).iloc[1:]


def monthly_returns(equity: pd.Series) -> pd.Series:
    """Returns of complete calendar months.

    The first month has no prior month-end and is skipped. The last month is kept
    only if the data reaches that month's last business day.
    """
    e = clean(equity)
    if e.empty:
        return pd.Series(dtype=float)
    month_end = e.resample("ME").last()
    returns = (month_end / month_end.shift(1) - 1).iloc[1:]
    last = e.index[-1]
    if last < last + pd.offsets.BMonthEnd(0):
        returns = returns.iloc[:-1]
    return returns


def excess_returns(returns: pd.Series, risk_free_returns: pd.Series) -> pd.Series:
    rf = risk_free_returns.reindex(returns.index)
    return (returns - rf).dropna()


def years(equity: pd.Series) -> float:
    e = clean(equity)
    return (e.index[-1] - e.index[0]).days / 365.25 if len(e) >= 2 else float("nan")


# --- A. basic performance ------------------------------------------------------------

def total_return(equity: pd.Series) -> float:
    e = clean(equity)
    return float(e.iloc[-1] / e.iloc[0] - 1) if len(e) >= 2 else float("nan")


def cagr(equity: pd.Series) -> float:
    e = clean(equity)
    span = years(e)
    if len(e) < 2 or not span > 0:
        return float("nan")
    growth = e.iloc[-1] / e.iloc[0]
    return float(growth ** (1 / span) - 1) if growth > 0 else -1.0


def volatility(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(r) >= 2 else float("nan")


def drawdowns(equity: pd.Series) -> pd.Series:
    e = clean(equity)
    return e / e.cummax() - 1


def drawdown_stats(equity: pd.Series) -> dict:
    """Largest peak-to-trough decline and its recovery.

    recovery_days: calendar days from trough back to the previous peak value.
    underwater_days: calendar days from peak to recovery. Both NaN if not recovered.
    """
    e = clean(equity)
    empty = {"max_drawdown": float("nan"), "peak_date": pd.NaT, "trough_date": pd.NaT,
             "recovery_date": pd.NaT, "recovery_days": float("nan"), "underwater_days": float("nan")}
    if e.empty:
        return empty
    dd = drawdowns(e)
    trough = dd.idxmin()
    if dd[trough] >= -EPSILON:
        return {**empty, "max_drawdown": 0.0, "recovery_days": 0.0, "underwater_days": 0.0}
    peak = e.loc[:trough].idxmax()
    after = e.loc[trough:]
    recovered = after[after >= e[peak]]
    recovery = recovered.index[0] if len(recovered) else pd.NaT
    return {
        "max_drawdown": float(dd[trough]),
        "peak_date": peak,
        "trough_date": trough,
        "recovery_date": recovery,
        "recovery_days": float((recovery - trough).days) if len(recovered) else float("nan"),
        "underwater_days": float((recovery - peak).days) if len(recovered) else float("nan"),
    }


# --- B. risk-adjusted ---------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, risk_free_returns: pd.Series) -> float:
    """Mean excess daily return / its standard deviation x sqrt(252). NaN if that deviation is 0."""
    ex = excess_returns(returns, risk_free_returns)
    if len(ex) < 2:
        return float("nan")
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > EPSILON else float("nan")


def downside_deviation(returns: pd.Series, risk_free_returns: pd.Series) -> float:
    """Annualized sqrt(mean(min(excess, 0)^2)) over all days (target = risk-free rate)."""
    ex = excess_returns(returns, risk_free_returns)
    if ex.empty:
        return float("nan")
    return float(math.sqrt((np.minimum(ex, 0) ** 2).mean()) * math.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free_returns: pd.Series) -> float:
    """Annualized mean excess return / annualized downside deviation. NaN if no downside."""
    ex = excess_returns(returns, risk_free_returns)
    dd = downside_deviation(returns, risk_free_returns)
    if len(ex) < 2 or not dd > EPSILON:
        return float("nan")
    return float(ex.mean() * TRADING_DAYS / dd)


def calmar_ratio(equity: pd.Series) -> float:
    """CAGR / |max drawdown| over the whole period. NaN if there was no drawdown."""
    mdd = drawdown_stats(equity)["max_drawdown"]
    if not abs(mdd) > EPSILON:
        return float("nan")
    return float(cagr(equity) / abs(mdd))


# --- C. return distribution ------------------------------------------------------------

def distribution(returns: pd.Series) -> dict:
    """Moments and percentiles. Skewness needs 3 and excess kurtosis 4 observations."""
    r = returns.dropna()
    n = len(r)
    stats = {
        "observations": n,
        "mean": r.mean() if n else float("nan"),
        "median": r.median() if n else float("nan"),
        "std": r.std(ddof=1) if n >= 2 else float("nan"),
        "skewness": r.skew() if n >= 3 else float("nan"),
        "excess_kurtosis": r.kurt() if n >= 4 else float("nan"),
        "min": r.min() if n else float("nan"),
        "max": r.max() if n else float("nan"),
    }
    for p in PERCENTILES:
        stats[f"p{p:02d}"] = float(np.percentile(r, p)) if n else float("nan")
    return {k: (float(v) if k != "observations" else v) for k, v in stats.items()}


# --- D. VaR / CVaR --------------------------------------------------------------------

def min_observations(confidence: float) -> int:
    """At least one observation must fall in the tail: n >= 1 / (1 - confidence)."""
    return math.ceil(round(1 / (1 - confidence), 9))


def historical_var_cvar(returns: pd.Series, confidence: float) -> dict:
    """Historical VaR and CVaR as positive loss numbers.

    VaR = -(1 - confidence) quantile of returns (linear interpolation).
    CVaR = -mean of returns at or below that quantile.
    NaN when there are fewer than min_observations(confidence) returns.
    """
    r = returns.dropna()
    n = len(r)
    if n < min_observations(confidence):
        return {"observations": n, "tail_observations": 0, "var": float("nan"), "cvar": float("nan")}
    threshold = float(np.quantile(r, 1 - confidence))
    tail = r[r <= threshold]
    return {"observations": n, "tail_observations": len(tail),
            "var": -threshold, "cvar": float(-tail.mean())}


# --- E. rolling ----------------------------------------------------------------------

def _window_max_drawdown(values: np.ndarray) -> float:
    return float((values / np.maximum.accumulate(values) - 1).min())


def rolling_metrics(equity: pd.Series, risk_free_returns: pd.Series, window: int,
                    benchmark_equity: pd.Series | None = None) -> pd.DataFrame:
    """Trailing-window statistics; each date uses only that date and the `window` days before it.

    rolling_return_annualized: (E_t / E_{t-window}) ^ (252 / window) - 1
    rolling_volatility_annualized: std of the last `window` daily returns x sqrt(252)
    rolling_sharpe: Sharpe ratio of the last `window` daily excess returns
    rolling_max_drawdown: largest drawdown inside the last `window` + 1 equity values
    rolling_excess_return_vs_benchmark: rolling return minus the benchmark's rolling return
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    e = clean(equity)
    r = daily_returns(e)
    ex = (r - risk_free_returns.reindex(r.index)).reindex(e.index)
    mean = ex.rolling(window, min_periods=window).mean()
    sd = ex.rolling(window, min_periods=window).std()

    def rolling_return(series):
        return (series / series.shift(window)) ** (TRADING_DAYS / window) - 1

    out = pd.DataFrame(index=e.index)
    out["rolling_return_annualized"] = rolling_return(e)
    out["rolling_volatility_annualized"] = r.reindex(e.index).rolling(window, min_periods=window).std() \
        * math.sqrt(TRADING_DAYS)
    out["rolling_sharpe"] = (mean / sd.where(sd > EPSILON)) * math.sqrt(TRADING_DAYS)
    out["rolling_max_drawdown"] = e.rolling(window + 1, min_periods=window + 1).apply(
        _window_max_drawdown, raw=True)
    if benchmark_equity is not None:
        b = clean(benchmark_equity).reindex(e.index)
        out["rolling_excess_return_vs_benchmark"] = out["rolling_return_annualized"] - rolling_return(b)
    return out.iloc[window:]


# --- G. benchmark comparison ------------------------------------------------------------

def summary(equity: pd.Series, risk_free_returns: pd.Series) -> dict:
    r = daily_returns(equity)
    dd = drawdown_stats(equity)
    return {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "volatility_annualized": volatility(r),
        "max_drawdown": dd["max_drawdown"],
        "sharpe_ratio": sharpe_ratio(r, risk_free_returns),
        "sortino_ratio": sortino_ratio(r, risk_free_returns),
        "calmar_ratio": calmar_ratio(equity),
        "final_equity": float(clean(equity).iloc[-1]) if len(clean(equity)) else float("nan"),
    }


def benchmark_comparison(equities: pd.DataFrame, benchmark: str, risk_free_returns: pd.Series) -> pd.DataFrame:
    """Strategy value, benchmark value and difference (strategy - benchmark) for each metric.

    Every strategy is compared on the dates it shares with the benchmark only.
    """
    rows = []
    for name in equities.columns:
        if name == benchmark:
            continue
        both = equities[[name, benchmark]].dropna()
        s, b = summary(both[name], risk_free_returns), summary(both[benchmark], risk_free_returns)
        for metric in s:
            rows.append({"strategy": name, "benchmark": benchmark, "metric": metric,
                         "strategy_value": s[metric], "benchmark_value": b[metric],
                         "difference": s[metric] - b[metric]})
    return pd.DataFrame(rows)


# --- H. correlation and covariance ------------------------------------------------------

def correlation_tables(returns: pd.DataFrame, monthly: pd.DataFrame, downside_on: str) -> pd.DataFrame:
    """Stacked matrices: daily, monthly and downside correlation; daily and annualized covariance.

    Downside correlation uses only days on which `downside_on` had a negative return.
    """
    down = returns[returns[downside_on] < 0]
    matrices = {
        "correlation_daily": returns.corr(),
        "correlation_monthly": monthly.corr(),
        f"correlation_downside_{downside_on}_negative_days": down.corr(),
        "covariance_daily": returns.cov(),
        "covariance_annualized": returns.cov() * TRADING_DAYS,
    }
    frames = []
    for name, matrix in matrices.items():
        frame = matrix.copy()
        frame.insert(0, "matrix", name)
        frame.index.name = "series"
        frames.append(frame.reset_index())
    return pd.concat(frames, ignore_index=True)
