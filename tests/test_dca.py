import pandas as pd
import pytest

from dca import (
    contribution_adjusted_drawdown,
    contribution_dates,
    cost_basis_stats,
    drawdown_stats,
    lump_sum_contributions,
    simulate_dca,
    simulate_lump_sum,
    summarize,
)


def series(values: dict) -> pd.Series:
    s = pd.Series(values, dtype=float, name="TEST")
    s.index = pd.to_datetime(s.index)
    return s


# Jan 2 and Jan 3 are in the same month, so only Jan 2 receives a contribution.
CRASH_AND_TOP_UP = series({"2024-01-02": 10, "2024-01-03": 5, "2024-02-01": 5, "2024-02-02": 10})


def test_contribution_dates_are_first_trading_day_of_each_month():
    index = pd.to_datetime(["2024-01-15", "2024-01-20", "2024-02-01", "2024-02-15", "2024-03-04"])
    assert list(contribution_dates(index)) == list(pd.to_datetime(["2024-01-15", "2024-02-01", "2024-03-04"]))


def test_dca_buys_fractional_units_at_each_contribution():
    prices = series({"2024-01-02": 10, "2024-02-01": 20, "2024-03-01": 10})
    result = simulate_dca(prices, 100)

    assert list(result.units) == pytest.approx([10, 15, 25])
    assert list(result.contributions) == [100, 200, 300]
    assert list(result.portfolio_value) == pytest.approx([100, 300, 250])
    assert result.gain == pytest.approx(-50)
    assert result.return_on_contributions == pytest.approx(-1 / 6)


def test_dca_at_constant_price_neither_gains_nor_loses():
    prices = pd.Series(50.0, index=pd.bdate_range("2024-01-01", "2024-06-28"))
    result = simulate_dca(prices, 100)

    assert result.invested == 600
    assert result.gain == pytest.approx(0)


def test_lump_sum_invests_the_same_total_on_the_first_day():
    prices = series({"2024-01-02": 10, "2024-02-01": 20, "2024-03-01": 10})
    lump = simulate_lump_sum(prices, 100)

    assert list(lump) == pytest.approx([300, 600, 300])


@pytest.mark.parametrize("values, lump_sum_should_win", [([10, 12, 14, 16], True), ([16, 14, 12, 10], False)])
def test_lump_sum_wins_in_rising_markets_and_loses_in_falling_ones(values, lump_sum_should_win):
    prices = pd.Series(values, index=pd.to_datetime(["2024-01-02", "2024-02-01", "2024-03-01", "2024-04-01"]),
                       dtype=float)
    dca = simulate_dca(prices, 100)
    lump = simulate_lump_sum(prices, 100)

    assert bool(lump.iloc[-1] > dca.final_value) == lump_sum_should_win


def test_invalid_inputs_are_rejected():
    prices = series({"2024-01-02": 10})
    with pytest.raises(ValueError):
        simulate_dca(prices, 0)
    with pytest.raises(ValueError):
        simulate_lump_sum(prices, -1)
    with pytest.raises(ValueError):
        simulate_dca(pd.Series(dtype=float), 100)


def test_new_contributions_do_not_hide_a_drawdown():
    result = simulate_dca(CRASH_AND_TOP_UP, 100)
    dd = contribution_adjusted_drawdown(result.portfolio_value, result.contributions)

    # Feb 1: value 150 exceeds the old peak of 100 only because 100 was added,
    # so it is still 25% below the 200 that peak-plus-contributions would be.
    assert list(dd) == pytest.approx([0, -0.5, -0.25, 0])
    naive = result.portfolio_value / result.portfolio_value.cummax() - 1
    assert naive.loc["2024-02-01"] == 0


def test_drawdown_stats_report_depth_dates_and_recovery_days():
    result = simulate_dca(CRASH_AND_TOP_UP, 100)
    stats = drawdown_stats(result.portfolio_value, result.contributions)

    assert stats["max_drawdown"] == pytest.approx(-0.5)
    assert stats["drawdown_peak"] == pd.Timestamp("2024-01-02")
    assert stats["drawdown_trough"] == pd.Timestamp("2024-01-03")
    assert stats["recovery_date"] == pd.Timestamp("2024-02-02")
    assert stats["days_to_recover"] == 30


def test_unrecovered_drawdown_has_no_recovery_date():
    prices = series({"2024-01-02": 10, "2024-01-03": 8, "2024-01-04": 9})
    lump = simulate_lump_sum(prices, 100)
    stats = drawdown_stats(lump, lump_sum_contributions(lump))

    assert stats["max_drawdown"] == pytest.approx(-0.2)
    assert pd.isna(stats["recovery_date"])
    assert pd.isna(stats["days_to_recover"])


def test_lump_sum_drawdown_equals_price_drawdown():
    prices = series({"2024-01-02": 10, "2024-01-03": 12, "2024-01-04": 9, "2024-01-05": 11})
    lump = simulate_lump_sum(prices, 100)
    dd = contribution_adjusted_drawdown(lump, lump_sum_contributions(lump))

    assert list(dd) == pytest.approx(list(prices / prices.cummax() - 1))


def test_rising_path_has_no_drawdown():
    prices = series({"2024-01-02": 10, "2024-02-01": 11, "2024-03-01": 12})
    result = simulate_dca(prices, 100)
    stats = drawdown_stats(result.portfolio_value, result.contributions)

    assert stats["max_drawdown"] == 0
    assert stats["days_to_recover"] == 0


def test_cost_basis_stats():
    result = simulate_dca(CRASH_AND_TOP_UP, 100)
    stats = cost_basis_stats(result.portfolio_value, result.contributions)

    # value vs contributed: 100/100, 50/100, 150/200, 300/200
    assert stats["worst_vs_contributed"] == pytest.approx(-0.5)
    assert stats["share_days_below_contributed"] == pytest.approx(0.5)


def test_summarize_compares_dca_with_lump_sum_of_the_same_total():
    prices = series({"2024-01-02": 10, "2024-02-01": 20, "2024-03-01": 10})
    summary = summarize(simulate_dca(prices, 100), simulate_lump_sum(prices, 100))

    assert summary["months"] == 3
    assert summary["total_contributed"] == 300
    assert summary["lump_sum_final_value"] == pytest.approx(300)
    assert summary["dca_vs_lump_sum"] == pytest.approx(250 / 300 - 1)
