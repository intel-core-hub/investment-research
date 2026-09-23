import numpy as np
import pandas as pd
import pytest

from scenarios import (
    crash_start_runs,
    drawdown_episodes,
    evaluate,
    rolling_windows,
    start_year_runs,
    window,
)


def business_days(start: str, end: str, name: str = "TEST") -> pd.Series:
    index = pd.bdate_range(start, end)
    return pd.Series(np.linspace(100, 200, len(index)), index=index, name=name)


def test_window_ends_on_the_last_trading_day_of_the_final_month():
    prices = business_days("2024-01-01", "2024-06-14")
    w = window(prices, pd.Timestamp("2024-01-01"), 3)

    assert w.index[0] == pd.Timestamp("2024-01-01")
    assert w.index[-1] == pd.Timestamp("2024-03-29")


def test_window_requires_data_beyond_the_final_month():
    prices = business_days("2024-01-01", "2024-06-14")

    assert window(prices, pd.Timestamp("2024-01-01"), 5) is not None
    # June is the latest (possibly partial) month, so a window ending in June is refused
    assert window(prices, pd.Timestamp("2024-01-01"), 6) is None


def test_window_can_start_mid_month():
    prices = business_days("2024-01-01", "2024-06-14")
    result = evaluate(window(prices, pd.Timestamp("2024-01-15"), 3), 100)

    assert result["start"] == pd.Timestamp("2024-01-15").date()
    assert result["months"] == 3
    assert result["total_contributed"] == 300


def test_rolling_windows_cover_every_start_month_with_enough_data():
    # 24 months of data; the 24th month counts as partial, so the last complete month is the 23rd
    prices = business_days("2023-01-02", "2024-12-16")
    windows = rolling_windows(prices, [1], 100)

    assert len(windows) == 12
    assert windows["start"].iloc[0] == pd.Timestamp("2023-01-02").date()
    assert windows["start"].iloc[-1] == pd.Timestamp("2023-12-01").date()
    assert (windows["months"] == 12).all()
    assert (windows["total_contributed"] == 1200).all()


def test_constant_prices_give_identical_dca_and_lump_sum():
    prices = pd.Series(50.0, index=pd.bdate_range("2023-01-02", "2024-12-16"), name="FLAT")
    windows = rolling_windows(prices, [1], 100)

    assert windows["dca_return"].abs().max() == pytest.approx(0)
    assert windows["dca_vs_lump_sum"].abs().max() == pytest.approx(0)
    assert (windows["dca_max_drawdown"] == 0).all()


def test_start_year_runs_end_at_the_last_complete_month():
    prices = business_days("2023-03-01", "2025-06-13")
    runs = start_year_runs(prices, 100)

    assert list(runs["start_year"]) == [2023, 2024, 2025]
    assert list(runs["months"]) == [27, 17, 5]
    assert (runs["end"] == pd.Timestamp("2025-05-30").date()).all()


def price_path(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), name="TEST", dtype=float)


def test_drawdown_episodes_keep_only_deep_drawdowns():
    prices = price_path([100, 110, 80, 90, 115, 105, 120])
    episodes = drawdown_episodes(prices, threshold=0.15)

    assert len(episodes) == 1
    ep = episodes.iloc[0]
    assert ep["peak"] == prices.index[1]
    assert ep["trough"] == prices.index[2]
    assert ep["recovery"] == prices.index[4]
    assert ep["depth"] == pytest.approx(80 / 110 - 1)


def test_unrecovered_episode_has_no_recovery_date():
    episodes = drawdown_episodes(price_path([100, 120, 90, 95]), threshold=0.15)

    assert len(episodes) == 1
    assert pd.isna(episodes.iloc[0]["recovery"])


def test_no_episodes_still_returns_date_columns():
    episodes = drawdown_episodes(price_path([100, 101, 102]), threshold=0.15)

    assert episodes.empty
    assert pd.api.types.is_datetime64_any_dtype(episodes["recovery"])


def test_crash_starts_skip_horizons_beyond_the_data():
    prices = business_days("2023-01-02", "2024-12-16")
    episodes = pd.DataFrame({
        "peak": [pd.Timestamp("2023-02-01")],
        "trough": [pd.Timestamp("2023-06-01")],
        "recovery": [pd.NaT],
        "depth": [-0.2],
    })
    runs = crash_start_runs(prices, episodes, [1, 2], 100)

    # 2-year windows from Feb or Jun 2023 would end after Nov 2024; no recovery date to start from
    assert set(zip(runs["start_point"], runs["horizon_years"])) == {("peak", 1), ("trough", 1)}
