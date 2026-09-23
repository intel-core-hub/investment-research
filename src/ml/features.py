"""Features and target for predicting the direction of an asset over the next N trading days.

Features at date t use only data up to and including t's close.
The target at t uses the close h trading days later, so it is known only at
that later date (`label_end`); training and evaluation use this to decide
which labels are already known at a given cut-off.
"""
from __future__ import annotations

import math

import pandas as pd

import regimes as rg

TRADING_DAYS = 252
RETURN_WINDOWS = (1, 5, 20, 60, 120, 252)

BASE_FEATURES = [f"ret_{n}d" for n in RETURN_WINDOWS] + ["vol_20d", "vol_60d", "ma200_gap", "volume_change"]
PHASE6_FEATURES = ["drawdown", "rolling_sharpe_252d", "regime_bull", "regime_high_volatility",
                   "regime_large_drawdown"]


def relative_feature(other: str) -> str:
    return f"rel_ret_60d_vs_{other}"


def build_features(prices: pd.DataFrame, volume: pd.Series, asset: str, risk_free: str,
                   relative_to: list[str], regime_settings: rg.RegimeSettings) -> pd.DataFrame:
    """All candidate features, one row per date (NaN until each window has enough history)."""
    close = prices[asset]
    daily = close / close.shift(1) - 1
    rf_daily = prices[risk_free] / prices[risk_free].shift(1) - 1
    f = pd.DataFrame(index=prices.index)

    for n in RETURN_WINDOWS:
        f[f"ret_{n}d"] = close / close.shift(n) - 1
    for n in (20, 60):
        f[f"vol_{n}d"] = daily.rolling(n, min_periods=n).std() * math.sqrt(TRADING_DAYS)
    f["ma200_gap"] = close / close.rolling(200, min_periods=200).mean() - 1
    v = volume.reindex(prices.index)
    f["volume_change"] = v.rolling(20, min_periods=20).mean() / v.rolling(120, min_periods=120).mean() - 1

    f["drawdown"] = close / close.cummax() - 1
    excess = daily - rf_daily
    sd = excess.rolling(TRADING_DAYS, min_periods=TRADING_DAYS).std()
    f["rolling_sharpe_252d"] = excess.rolling(TRADING_DAYS, min_periods=TRADING_DAYS).mean() \
        / sd.where(sd > 0) * math.sqrt(TRADING_DAYS)
    labels = rg.classify(close, regime_settings)
    ready = close.rolling(max(regime_settings.trend_window, regime_settings.volatility_window)).count() \
        >= max(regime_settings.trend_window, regime_settings.volatility_window)
    for regime in ("bull", "high_volatility", "large_drawdown"):
        f[f"regime_{regime}"] = labels[regime].astype(float).where(ready)

    own_60d = f["ret_60d"]
    for other in relative_to:
        f[relative_feature(other)] = own_60d - (prices[other] / prices[other].shift(60) - 1)
    return f


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return from t's close to the close `horizon` trading days later (NaN at the end)."""
    return close.shift(-horizon) / close - 1


def direction_target(close: pd.Series, horizon: int) -> pd.Series:
    """1.0 if the forward return is positive, 0.0 if not, NaN where it is not known yet."""
    fwd = forward_return(close, horizon)
    return (fwd > 0).astype(float).where(fwd.notna())


def label_end_dates(index: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """Date on which each label becomes known (NaT if beyond the data)."""
    ends = pd.Series(index, index=index).shift(-horizon)
    return pd.to_datetime(ends)


def trainable_rows(target: pd.Series, label_end: pd.Series, start, end) -> pd.Index:
    """Rows dated in [start, end] whose label is known by `end` (purges the last `horizon` rows)."""
    in_range = (target.index >= pd.Timestamp(start)) & (target.index <= pd.Timestamp(end))
    known = target.notna() & (label_end <= pd.Timestamp(end))
    return target.index[in_range & known.to_numpy()]


def complete_rows(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Rows where every listed feature is available (drops the warm-up period; no imputation)."""
    return features.dropna(subset=columns)[columns]


def split_segments(index: pd.DatetimeIndex, train_end, validation_end) -> dict[str, pd.DatetimeIndex]:
    """Chronological train (<= train_end), validation (<= validation_end) and test (later) dates."""
    train_end, validation_end = pd.Timestamp(train_end), pd.Timestamp(validation_end)
    if not train_end < validation_end:
        raise ValueError("train_end must be before validation_end")
    return {"train": index[index <= train_end],
            "validation": index[(index > train_end) & (index <= validation_end)],
            "test": index[index > validation_end]}


def feature_sets(relative_to: list[str]) -> dict[str, list[str]]:
    return {
        "base": list(BASE_FEATURES),
        "extended": BASE_FEATURES + PHASE6_FEATURES + [relative_feature(o) for o in relative_to],
    }
