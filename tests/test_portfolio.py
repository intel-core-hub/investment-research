import copy

import numpy as np
import pandas as pd
import pytest

import metrics as m
from dca import simulate_dca
from portfolio import (
    PortfolioConfigError,
    daily_returns,
    diversification,
    nav,
    period_starts,
    rebalance_dates,
    simulate_portfolio,
    summarize,
    validate_config,
)

TICKERS = ["AAA", "BBB", "CASH"]
VALID_CONFIG = {
    "initial_investment": 100,
    "monthly_contribution": 100,
    "rebalance_frequencies": ["none", "annual", "quarterly", "monthly"],
    "default_rebalance": "annual",
    "risk_free_ticker": "CASH",
    "portfolios": [
        {"name": "one", "weights": {"AAA": 1.0}},
        {"name": "mix", "weights": {"AAA": 0.6, "BBB": 0.4}},
    ],
}


def config_with(**changes) -> dict:
    config = copy.deepcopy(VALID_CONFIG)
    config.update(changes)
    return config


def frame(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows, dtype=float).T
    df.index = pd.to_datetime(df.index)
    return df


# --- config validation ---------------------------------------------------------

def test_valid_config_passes():
    validate_config(VALID_CONFIG, TICKERS)


@pytest.mark.parametrize("weights, message", [
    ({"AAA": 0.6, "BBB": 0.3}, "sum to 0.900000"),
    ({"AAA": 1.2, "BBB": -0.2}, "must be > 0 and <= 1"),
    ({"AAA": 1.0, "BBB": 0.0}, "must be > 0 and <= 1"),
    ({"AAA": 0.5, "ZZZ": 0.5}, "ZZZ is not in the price data"),
    ({"AAA": "0.5", "BBB": 0.5}, "must be > 0 and <= 1"),
    ({}, "non-empty table"),
])
def test_invalid_weights_are_rejected(weights, message):
    config = config_with(portfolios=[{"name": "bad", "weights": weights}])
    with pytest.raises(PortfolioConfigError, match=message):
        validate_config(config, TICKERS)


@pytest.mark.parametrize("changes, message", [
    ({"rebalance_frequencies": ["none", "weekly"]}, "unknown rebalance frequency 'weekly'"),
    ({"default_rebalance": "monthly", "rebalance_frequencies": ["none"]}, "default_rebalance"),
    ({"risk_free_ticker": "TBILL"}, "risk_free_ticker"),
    ({"monthly_contribution": 0}, "monthly_contribution must be a positive number"),
    ({"portfolios": []}, "no portfolios defined"),
    ({"portfolios": [{"name": "x", "weights": {"AAA": 1.0}}] * 2}, "used more than once"),
])
def test_invalid_settings_are_rejected(changes, message):
    with pytest.raises(PortfolioConfigError, match=message):
        validate_config(config_with(**changes), TICKERS)


def test_every_problem_is_reported_at_once():
    config = config_with(
        risk_free_ticker="TBILL",
        portfolios=[{"name": "bad", "weights": {"AAA": 0.5, "ZZZ": 0.2}}],
    )
    with pytest.raises(PortfolioConfigError) as error:
        validate_config(config, TICKERS)
    text = str(error.value)
    assert "risk_free_ticker" in text
    assert "ZZZ is not in the price data" in text
    assert "sum to 0.700000" in text


# --- schedule -------------------------------------------------------------------

def test_period_starts_and_rebalance_dates():
    index = pd.bdate_range("2023-11-15", "2024-07-31")

    assert list(period_starts(index, "Q")) == list(pd.to_datetime(
        ["2023-11-15", "2024-01-01", "2024-04-01", "2024-07-01"]))
    assert list(rebalance_dates(index, "annual")) == [pd.Timestamp("2024-01-01")]
    assert len(rebalance_dates(index, "monthly")) == 8
    assert rebalance_dates(index, "none").empty


# --- simulation -----------------------------------------------------------------

TWO_ASSETS = frame({
    "2024-01-02": {"AAA": 10, "BBB": 10},
    "2024-01-03": {"AAA": 20, "BBB": 10},
    "2024-02-01": {"AAA": 20, "BBB": 10},
    "2024-02-02": {"AAA": 10, "BBB": 10},
})


def test_buy_and_hold_lets_weights_drift():
    result = simulate_portfolio(TWO_ASSETS, {"AAA": 0.5, "BBB": 0.5}, "none", initial_investment=100)

    assert list(result.units.iloc[-1]) == pytest.approx([5, 5])
    assert list(result.value) == pytest.approx([100, 150, 150, 100])
    assert result.weights(TWO_ASSETS).loc["2024-01-03", "AAA"] == pytest.approx(2 / 3)
    assert result.traded_value == 0


def test_rebalancing_resets_weights_without_changing_value():
    result = simulate_portfolio(TWO_ASSETS, {"AAA": 0.5, "BBB": 0.5}, "monthly", initial_investment=100)
    weights = result.weights(TWO_ASSETS)

    # Feb 1: 150 split back to 75/75, i.e. sell 25 of AAA and buy 25 of BBB
    assert result.value.loc["2024-02-01"] == pytest.approx(150)
    assert list(weights.loc["2024-02-01"]) == pytest.approx([0.5, 0.5])
    assert result.traded_value == pytest.approx(50)
    assert result.rebalances == 1
    # after AAA halves, the rebalanced portfolio (3.75 AAA + 7.5 BBB) is worth 112.5, not 100
    assert result.value.iloc[-1] == pytest.approx(112.5)


def test_rebalancing_is_a_no_op_when_assets_move_together():
    prices = pd.DataFrame({"AAA": np.linspace(10, 30, 300), "BBB": np.linspace(20, 60, 300)},
                          index=pd.bdate_range("2024-01-01", periods=300))
    hold = simulate_portfolio(prices, {"AAA": 0.3, "BBB": 0.7}, "none", initial_investment=100)
    monthly = simulate_portfolio(prices, {"AAA": 0.3, "BBB": 0.7}, "monthly", initial_investment=100)

    assert list(monthly.value) == pytest.approx(list(hold.value))
    assert monthly.traded_value == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("frequency", ["none", "annual", "quarterly", "monthly"])
def test_single_asset_portfolio_tracks_the_asset(frequency):
    index = pd.bdate_range("2023-01-02", periods=400)
    price = pd.Series(np.linspace(50, 80, 400), index=index)
    result = simulate_portfolio(price.to_frame("AAA"), {"AAA": 1.0}, frequency, initial_investment=100)

    assert list(result.value) == pytest.approx(list(100 * price / price.iloc[0]))


def test_dca_portfolio_of_one_asset_matches_the_phase_3_engine():
    index = pd.bdate_range("2023-01-02", periods=400)
    price = pd.Series(50 + 10 * np.sin(np.arange(400) / 30), index=index, name="AAA")
    portfolio = simulate_portfolio(price.to_frame(), {"AAA": 1.0}, "annual", monthly_contribution=100)
    single = simulate_dca(price, 100)

    assert list(portfolio.value) == pytest.approx(list(single.portfolio_value))
    assert list(portfolio.contributions) == list(single.contributions)


def test_dca_contributions_are_split_by_target_weights():
    result = simulate_portfolio(TWO_ASSETS, {"AAA": 0.25, "BBB": 0.75}, "none", monthly_contribution=100)

    # Jan 2: 25 / 10 = 2.5 AAA, 75 / 10 = 7.5 BBB; Feb 1: 25 / 20 = 1.25 AAA, 75 / 10 = 7.5 BBB
    assert list(result.units.iloc[-1]) == pytest.approx([3.75, 15])
    assert result.contributions.iloc[-1] == 200


def test_needs_some_money():
    with pytest.raises(ValueError):
        simulate_portfolio(TWO_ASSETS, {"AAA": 1.0}, "none")


# --- returns and metrics ----------------------------------------------------------

def test_daily_returns_exclude_contributions():
    prices = pd.DataFrame({"AAA": 10.0}, index=pd.bdate_range("2024-01-01", "2024-06-28"))
    result = simulate_portfolio(prices, {"AAA": 1.0}, "none", monthly_contribution=100)

    assert result.value.iloc[-1] == 600
    assert daily_returns(result).abs().max() == pytest.approx(0)
    assert nav(result).iloc[-1] == pytest.approx(1)


def test_nav_of_a_lump_sum_is_value_over_initial_investment():
    result = simulate_portfolio(TWO_ASSETS, {"AAA": 0.5, "BBB": 0.5}, "none", initial_investment=100)

    assert list(nav(result)) == pytest.approx(list(result.value / 100))
    assert list(daily_returns(result)) == pytest.approx([0.5, 0, -1 / 3])


def test_sharpe_ratio_uses_excess_returns_over_the_risk_free_asset():
    index = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.Series([100, 101, 100.5, 102, 101, 103], index=index, dtype=float)
    risk_free = pd.Series(100 * 1.0001 ** np.arange(6), index=index)

    excess = prices.pct_change().iloc[1:] - 0.0001
    expected = excess.mean() * 252 / (excess.std() * np.sqrt(252))
    assert m.sharpe_ratio(prices, risk_free) == pytest.approx(expected)


@pytest.mark.parametrize("correlation, expected_ratio", [
    (1.0, 1.0),
    (0.5, 0.2 / np.sqrt(0.03)),  # variance 0.25*0.04*2 + 2*0.25*0.5*0.04 = 0.03
    (0.0, np.sqrt(2)),
])
def test_diversification_depends_on_correlation(correlation, expected_ratio):
    vol = 0.2
    cov = pd.DataFrame([[vol**2, correlation * vol**2], [correlation * vol**2, vol**2]],
                       index=["AAA", "BBB"], columns=["AAA", "BBB"])
    result = diversification({"AAA": 0.5, "BBB": 0.5}, cov)

    assert result["weighted_average_volatility"] == pytest.approx(0.2)
    assert result["diversification_ratio"] == pytest.approx(expected_ratio)


def test_summary_reports_metrics_on_the_same_definitions_as_phase_2():
    index = pd.bdate_range("2023-01-02", periods=500)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame({
        "AAA": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 500)),
        "CASH": 100 * 1.0001 ** np.arange(500),
    }, index=index)
    result = simulate_portfolio(prices, {"AAA": 1.0}, "none", initial_investment=100)
    summary = summarize(result, prices["CASH"])

    assert summary["cagr"] == pytest.approx(m.cagr(prices["AAA"]))
    assert summary["annual_volatility"] == pytest.approx(m.annual_volatility(prices["AAA"]))
    assert summary["max_drawdown"] == pytest.approx(m.max_drawdown(prices["AAA"]))
    assert summary["sharpe_ratio"] == pytest.approx(m.sharpe_ratio(prices["AAA"], prices["CASH"]))
    assert summary["annual_turnover"] == 0
