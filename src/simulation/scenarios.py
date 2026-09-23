"""Run the DCA engine over many start dates, horizons and crash scenarios.

Every scenario is a window of complete calendar months: contributions start on
`start` and then on the first trading day of each following month, and the
portfolio is valued at the last trading day of the final month. The lump sum
invests the same total on `start`.
"""
from __future__ import annotations

import pandas as pd

from dca import (
    contribution_dates,
    cost_basis_stats,
    drawdown_stats,
    lump_sum_contributions,
    simulate_dca,
    simulate_lump_sum,
)


def window(prices: pd.Series, start: pd.Timestamp, months: int) -> pd.Series | None:
    """Prices from `start` to the end of the `months`-th month, or None if the data is too short.

    The data must extend into the month after the window, so the final month is
    known to be complete (the latest month in the data may be partial).
    """
    end_month = start.to_period("M") + months - 1
    if prices.index[-1].to_period("M") <= end_month:
        return None
    return prices.loc[start:end_month.end_time]


def evaluate(prices: pd.Series, monthly_contribution: float) -> dict:
    dca = simulate_dca(prices, monthly_contribution)
    lump = simulate_lump_sum(prices, monthly_contribution)
    dca_dd = drawdown_stats(dca.portfolio_value, dca.contributions)
    lump_dd = drawdown_stats(lump, lump_sum_contributions(lump))
    dca_cost = cost_basis_stats(dca.portfolio_value, dca.contributions)
    lump_cost = cost_basis_stats(lump, lump_sum_contributions(lump))
    return {
        "start": prices.index[0].date(),
        "end": prices.index[-1].date(),
        "months": int(round(dca.invested / monthly_contribution)),
        "total_contributed": dca.invested,
        "dca_final_value": dca.final_value,
        "lump_sum_final_value": float(lump.iloc[-1]),
        "dca_return": dca.return_on_contributions,
        "lump_sum_return": float(lump.iloc[-1]) / dca.invested - 1,
        "dca_vs_lump_sum": dca.final_value / float(lump.iloc[-1]) - 1,
        "dca_max_drawdown": dca_dd["max_drawdown"],
        "dca_days_to_recover": dca_dd["days_to_recover"],
        "lump_sum_max_drawdown": lump_dd["max_drawdown"],
        "lump_sum_days_to_recover": lump_dd["days_to_recover"],
        "dca_worst_vs_contributed": dca_cost["worst_vs_contributed"],
        "lump_sum_worst_vs_contributed": lump_cost["worst_vs_contributed"],
    }


def rolling_windows(prices: pd.Series, horizons_years: list[int], monthly_contribution: float) -> pd.DataFrame:
    """One run per (horizon, start month), starting on each month's first trading day."""
    price_drawdown = prices / prices.cummax() - 1
    rows = []
    for years in horizons_years:
        for start in contribution_dates(prices.index):
            prices_window = window(prices, start, 12 * years)
            if prices_window is None:
                break
            rows.append({
                "ticker": prices.name,
                "horizon_years": years,
                "start_price_drawdown": float(price_drawdown[start]),
                **evaluate(prices_window, monthly_contribution),
            })
    return pd.DataFrame(rows)


def _share(condition: pd.Series) -> float:
    return float(condition.mean())


def window_stats(windows: pd.DataFrame) -> pd.DataFrame:
    """Distribution of outcomes across all start dates, per asset and horizon."""
    grouped = windows.groupby(["ticker", "horizon_years"], sort=False)
    return grouped.agg(
        windows=("start", "size"),
        first_start=("start", "min"),
        last_start=("start", "max"),
        dca_return_median=("dca_return", "median"),
        dca_return_min=("dca_return", "min"),
        dca_return_max=("dca_return", "max"),
        lump_sum_return_median=("lump_sum_return", "median"),
        lump_sum_return_min=("lump_sum_return", "min"),
        lump_sum_return_max=("lump_sum_return", "max"),
        lump_sum_win_rate=("dca_vs_lump_sum", lambda s: _share(s < 0)),
        dca_vs_lump_sum_median=("dca_vs_lump_sum", "median"),
        dca_vs_lump_sum_min=("dca_vs_lump_sum", "min"),
        dca_vs_lump_sum_max=("dca_vs_lump_sum", "max"),
        dca_loss_rate=("dca_return", lambda s: _share(s < 0)),
        lump_sum_loss_rate=("lump_sum_return", lambda s: _share(s < 0)),
        dca_max_drawdown_median=("dca_max_drawdown", "median"),
        dca_max_drawdown_worst=("dca_max_drawdown", "min"),
        lump_sum_max_drawdown_median=("lump_sum_max_drawdown", "median"),
        lump_sum_max_drawdown_worst=("lump_sum_max_drawdown", "min"),
        dca_unrecovered_rate=("dca_days_to_recover", lambda s: _share(s.isna())),
        lump_sum_unrecovered_rate=("lump_sum_days_to_recover", lambda s: _share(s.isna())),
    ).reset_index()


def start_year_runs(prices: pd.Series, monthly_contribution: float) -> pd.DataFrame:
    """Start on the first trading day of each year and keep investing until the last complete month."""
    last_complete_month = prices.index[-1].to_period("M") - 1
    firsts = prices.index.to_series().groupby(prices.index.year).first()
    rows = []
    for year, start in firsts.items():
        months = (last_complete_month - start.to_period("M")).n + 1
        if months < 1:
            continue
        rows.append({
            "ticker": prices.name,
            "start_year": year,
            **evaluate(window(prices, start, months), monthly_contribution),
        })
    return pd.DataFrame(rows)


def drawdown_episodes(prices: pd.Series, threshold: float) -> pd.DataFrame:
    """Peak-to-recovery episodes whose price drawdown reached at least `threshold`.

    recovery is the first day back at the previous peak (NaT if not yet recovered).
    """
    drawdown = prices / prices.cummax() - 1
    episode_id = (drawdown == 0).cumsum()
    rows = []
    for _, segment in drawdown.groupby(episode_id):
        depth = segment.min()
        if depth > -threshold:
            continue
        last_pos = drawdown.index.get_loc(segment.index[-1])
        rows.append({
            "ticker": prices.name,
            "peak": segment.index[0],
            "trough": segment.idxmin(),
            "recovery": drawdown.index[last_pos + 1] if last_pos + 1 < len(drawdown) else pd.NaT,
            "depth": float(depth),
        })
    episodes = pd.DataFrame(rows, columns=["ticker", "peak", "trough", "recovery", "depth"])
    for column in ("peak", "trough", "recovery"):
        episodes[column] = pd.to_datetime(episodes[column])
    return episodes


def crash_start_runs(
    prices: pd.Series, episodes: pd.DataFrame, horizons_years: list[int], monthly_contribution: float
) -> pd.DataFrame:
    """Start DCA and lump sum at each episode's peak (before), trough (bottom) and recovery (after)."""
    rows = []
    for episode in episodes.itertuples():
        for point in ("peak", "trough", "recovery"):
            start = getattr(episode, point)
            if pd.isna(start):
                continue
            for years in horizons_years:
                prices_window = window(prices, start, 12 * years)
                if prices_window is None:
                    continue
                rows.append({
                    "ticker": prices.name,
                    "episode_peak": episode.peak.date(),
                    "episode_depth": episode.depth,
                    "start_point": point,
                    "horizon_years": years,
                    **evaluate(prices_window, monthly_contribution),
                })
    return pd.DataFrame(rows)
