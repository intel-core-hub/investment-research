"""Market regimes of a reference asset, defined with past data only.

On each day t the label uses prices up to t's close:
  bull              close > its `trend_window`-day simple moving average
  bear              close <= that average
  high_volatility   trailing `volatility_window`-day volatility (annualized) >= high_volatility
  low_volatility    that volatility <= low_volatility
  large_drawdown    close is at least `large_drawdown` below its running all-time high

A label computed at t's close applies to the return of day t + 1, so a return is
never attributed to a regime that was only known at the end of the same day.
Days without enough history (before the moving average or volatility window
fills) have no bull/bear or volatility label.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import performance as pf

REGIMES = ["bull", "bear", "high_volatility", "low_volatility", "large_drawdown"]


@dataclass(frozen=True)
class RegimeSettings:
    trend_window: int = 200
    volatility_window: int = 63
    high_volatility: float = 0.20
    low_volatility: float = 0.12
    large_drawdown: float = 0.15


def classify(prices: pd.Series, settings: RegimeSettings = RegimeSettings()) -> pd.DataFrame:
    """Boolean regime labels at each day's close (see module docstring)."""
    p = prices.dropna()
    sma = p.rolling(settings.trend_window, min_periods=settings.trend_window).mean()
    vol = (p / p.shift(1) - 1).rolling(settings.volatility_window, min_periods=settings.volatility_window) \
        .std() * math.sqrt(pf.TRADING_DAYS)
    return pd.DataFrame({
        "bull": (p > sma) & sma.notna(),
        "bear": (p <= sma) & sma.notna(),
        "high_volatility": (vol >= settings.high_volatility) & vol.notna(),
        "low_volatility": (vol <= settings.low_volatility) & vol.notna(),
        "large_drawdown": p / p.cummax() - 1 <= -settings.large_drawdown,
    })


def return_day_labels(labels: pd.DataFrame, return_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Labels that apply to each return date: the label at the previous trading day's close."""
    return labels.shift(1).reindex(return_dates).fillna(False).astype(bool)


def spells(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(first, last) date of every run of consecutive True values."""
    values = mask.to_numpy()
    runs, start = [], None
    for i, flag in enumerate(values):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(values) - 1):
            end = i if flag else i - 1
            runs.append((mask.index[start], mask.index[end]))
            start = None
    return runs


def regime_periods(day_labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in REGIMES:
        for first, last in spells(day_labels[regime]):
            rows.append({"regime": regime, "start": first.date(), "end": last.date(),
                         "trading_days": int(day_labels.loc[first:last, regime].sum())})
    return pd.DataFrame(rows, columns=["regime", "start", "end", "trading_days"])


def regime_stats(equity: pd.Series, risk_free_returns: pd.Series, day_labels: pd.DataFrame) -> pd.DataFrame:
    """Per regime: days, spells, compounded and annualized return, volatility, Sharpe,
    worst drawdown inside a spell, and recovery after losing spells.

    Recovery: for each spell that ended below its starting equity, calendar days after
    the spell's last day until equity regains the level it had when the spell began.
    """
    e = pf.clean(equity)
    r = pf.daily_returns(e)
    rows = []
    for regime in REGIMES:
        mask = day_labels[regime].reindex(r.index, fill_value=False)
        days = int(mask.sum())
        in_regime = r[mask]
        runs = spells(mask)
        row = {"regime": regime, "trading_days": days, "spells": len(runs)}
        if days == 0:
            rows.append(row)
            continue
        compounded = float(np.prod(1 + in_regime) - 1)
        worst_dd, losing, recovered, recovery_days = 0.0, 0, 0, []
        for first, last in runs:
            spell = r.loc[first:last]
            path = pd.concat([pd.Series([1.0]), (1 + spell).cumprod()], ignore_index=True)
            worst_dd = min(worst_dd, float((path / path.cummax() - 1).min()))
            start_level = e[e.index < first].iloc[-1]
            if e[last] < start_level:
                losing += 1
                later = e.loc[last:]
                back = later[later >= start_level]
                if len(back):
                    recovered += 1
                    recovery_days.append((back.index[0] - last).days)
        row.update({
            "cumulative_return": compounded,
            "return_annualized": (1 + compounded) ** (pf.TRADING_DAYS / days) - 1,
            "volatility_annualized": pf.volatility(in_regime),
            "sharpe_ratio": pf.sharpe_ratio(in_regime, risk_free_returns),
            "max_drawdown_within_spell": worst_dd,
            "losing_spells": losing,
            "recovered_spells": recovered,
            "median_recovery_days": float(np.median(recovery_days)) if recovery_days else float("nan"),
        })
        rows.append(row)
    return pd.DataFrame(rows)
