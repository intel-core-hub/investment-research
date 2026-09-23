import math

import numpy as np
import pandas as pd
import pytest

import inference as inf
import metrics as phase2
import performance as pf
import regimes as rg
import trades as tr

SQRT252 = math.sqrt(252)


def equity(values, start="2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def from_returns(returns, start="2024-01-01") -> pd.Series:
    return equity(100 * np.cumprod(np.r_[1.0, 1 + np.asarray(returns, dtype=float)]), start)


def zero_rf(e: pd.Series) -> pd.Series:
    return pd.Series(0.0, index=e.index)


def random_equity(n=800, seed=3, start="2020-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    return from_returns(rng.normal(0.0004, 0.01, n - 1), start)


# --- metric definitions ------------------------------------------------------------------

def test_cagr_uses_calendar_years():
    # 1461 calendar days = 4 x 365.25
    e = pd.Series([100.0, 146.41], index=pd.to_datetime(["2020-01-01", "2024-01-01"]))
    assert pf.cagr(e) == pytest.approx(0.10)
    assert pf.total_return(e) == pytest.approx(0.4641)


def test_volatility_is_sample_std_times_sqrt_252():
    r = pd.Series([0.01, -0.01, 0.02, 0.0])
    assert pf.volatility(r) == pytest.approx(np.std(r, ddof=1) * SQRT252)


def test_max_drawdown_dates_and_recovery():
    e = equity([100, 120, 90, 95, 130])
    stats = pf.drawdown_stats(e)

    assert stats["max_drawdown"] == pytest.approx(90 / 120 - 1)
    assert stats["peak_date"] == e.index[1]
    assert stats["trough_date"] == e.index[2]
    assert stats["recovery_date"] == e.index[4]
    assert stats["recovery_days"] == (e.index[4] - e.index[2]).days
    assert stats["underwater_days"] == (e.index[4] - e.index[1]).days


def test_sharpe_matches_manual_and_phase_2_definition():
    e = random_equity()
    rf_prices = equity(100 * 1.0001 ** np.arange(len(e)), "2020-01-01")
    r, rf = pf.daily_returns(e), pf.daily_returns(rf_prices)
    ex = r - rf
    assert pf.sharpe_ratio(r, rf) == pytest.approx(ex.mean() / ex.std() * SQRT252)
    assert pf.sharpe_ratio(r, rf) == pytest.approx(phase2.sharpe_ratio(e, rf_prices))
    assert pf.cagr(e) == pytest.approx(phase2.cagr(e))
    assert pf.volatility(r) == pytest.approx(phase2.annual_volatility(e))
    assert pf.drawdown_stats(e)["max_drawdown"] == pytest.approx(phase2.max_drawdown(e))


def test_sortino_uses_downside_deviation_over_all_days():
    e = from_returns([0.01, -0.02, 0.03, -0.01])
    r = pf.daily_returns(e)
    downside = math.sqrt((0.02 ** 2 + 0.01 ** 2) / 4) * SQRT252

    assert pf.downside_deviation(r, zero_rf(r)) == pytest.approx(downside)
    assert pf.sortino_ratio(r, zero_rf(r)) == pytest.approx(r.mean() * 252 / downside)


def test_calmar_is_cagr_over_max_drawdown():
    e = random_equity()
    assert pf.calmar_ratio(e) == pytest.approx(pf.cagr(e) / abs(pf.drawdown_stats(e)["max_drawdown"]))


def test_historical_var_and_cvar_are_positive_losses():
    r = pd.Series(np.arange(-50, 50) / 1000)  # -5.0% ... +4.9%
    result = pf.historical_var_cvar(r, 0.95)
    threshold = np.quantile(r, 0.05)

    assert result["var"] == pytest.approx(-threshold)
    assert result["cvar"] == pytest.approx(-r[r <= threshold].mean())
    assert result["cvar"] >= result["var"] > 0
    assert result["tail_observations"] == (r <= threshold).sum()


def test_min_observations_for_var():
    assert pf.min_observations(0.95) == 20
    assert pf.min_observations(0.99) == 100


def test_distribution_percentiles_and_moments():
    r = pd.Series(np.linspace(-0.05, 0.05, 101))
    d = pf.distribution(r)

    assert d["p50"] == pytest.approx(0)
    assert d["p01"] == pytest.approx(np.percentile(r, 1))
    assert d["skewness"] == pytest.approx(0, abs=1e-12)
    assert d["excess_kurtosis"] == pytest.approx(r.kurt())


def test_rolling_metrics_match_manual_window():
    e = random_equity(300)
    rf = zero_rf(e)
    out = pf.rolling_metrics(e, rf, 20)
    last = e.index[-1]
    window = e.iloc[-21:]
    r = pf.daily_returns(window)

    assert out.index[0] == e.index[20]
    assert out.loc[last, "rolling_return_annualized"] == pytest.approx((window.iloc[-1] / window.iloc[0]) ** (252 / 20) - 1)
    assert out.loc[last, "rolling_volatility_annualized"] == pytest.approx(r.std() * SQRT252)
    assert out.loc[last, "rolling_sharpe"] == pytest.approx(r.mean() / r.std() * SQRT252)
    assert out.loc[last, "rolling_max_drawdown"] == pytest.approx(pf.drawdown_stats(window)["max_drawdown"])


def test_rolling_window_can_be_changed_and_must_be_at_least_2():
    e = random_equity(300)
    assert len(pf.rolling_metrics(e, zero_rf(e), 63)) == len(e) - 63
    with pytest.raises(ValueError):
        pf.rolling_metrics(e, zero_rf(e), 1)


def test_monthly_returns_skip_first_and_partial_last_month():
    index = pd.bdate_range("2024-01-15", "2024-04-10")
    e = pd.Series(np.linspace(100, 110, len(index)), index=index)
    months = pf.monthly_returns(e)

    assert [d.strftime("%Y-%m") for d in months.index] == ["2024-02", "2024-03"]


# --- boundary cases ----------------------------------------------------------------------

def test_all_zero_returns():
    e = equity([100.0] * 50)
    r = pf.daily_returns(e)
    assert pf.cagr(e) == 0
    assert pf.volatility(r) == 0
    assert pf.drawdown_stats(e)["max_drawdown"] == 0
    assert math.isnan(pf.sharpe_ratio(r, zero_rf(r)))
    assert math.isnan(pf.sortino_ratio(r, zero_rf(r)))
    assert math.isnan(pf.calmar_ratio(e))


def test_constant_positive_returns_have_no_risk_ratio():
    e = from_returns([0.001] * 100)
    r = pf.daily_returns(e)
    assert pf.volatility(r) == pytest.approx(0, abs=1e-12)
    assert math.isnan(pf.sharpe_ratio(r, zero_rf(r)))      # zero volatility
    assert math.isnan(pf.sortino_ratio(r, zero_rf(r)))     # zero downside deviation
    assert math.isnan(pf.calmar_ratio(e))                  # zero drawdown
    assert pf.drawdown_stats(e)["recovery_days"] == 0


def test_all_negative_returns():
    e = from_returns([-0.01] * 100)
    r = pf.daily_returns(e)
    stats = pf.drawdown_stats(e)
    assert pf.cagr(e) < 0
    assert stats["max_drawdown"] == pytest.approx(e.iloc[-1] / e.iloc[0] - 1)
    assert pd.isna(stats["recovery_date"]) and math.isnan(stats["recovery_days"])
    assert math.isnan(pf.sharpe_ratio(r, zero_rf(r)))      # constant returns: zero volatility
    assert pf.calmar_ratio(e) < 0
    wobbly = from_returns([-0.01, -0.02] * 50)
    assert pf.sortino_ratio(pf.daily_returns(wobbly), zero_rf(pf.daily_returns(wobbly))) < 0


def test_single_observation():
    e = equity([100.0])
    r = pf.daily_returns(e)
    assert r.empty
    assert math.isnan(pf.total_return(e))
    assert math.isnan(pf.cagr(e))
    assert math.isnan(pf.volatility(r))
    assert math.isnan(pf.sharpe_ratio(r, zero_rf(r)))
    assert pf.distribution(r)["observations"] == 0
    assert math.isnan(pf.historical_var_cvar(r, 0.95)["var"])


def test_insufficient_data():
    r = pd.Series([0.01, -0.02, 0.005])
    assert math.isnan(pf.distribution(r)["excess_kurtosis"])          # needs 4
    assert not math.isnan(pf.distribution(r)["skewness"])             # 3 is enough
    assert math.isnan(pf.historical_var_cvar(pd.Series(np.zeros(50)), 0.99)["var"])  # needs 100
    e = random_equity(10)
    assert pf.rolling_metrics(e, zero_rf(e), 20).empty
    assert inf.mean_confidence_intervals(pd.Series([0.01]), 0.95) == []
    assert inf.sharpe_inference(pd.Series([0.01, 0.02]), pd.Series([0.0, 0.0]), 0.95) == []


def test_missing_values_are_dropped_before_computing_returns():
    e = equity([100, 110, np.nan, 121, 133.1])
    without = e.dropna()
    assert list(pf.daily_returns(e)) == pytest.approx([0.1, 0.1, 0.1])
    assert pf.cagr(e) == pytest.approx(pf.cagr(without))
    assert pf.drawdown_stats(e)["max_drawdown"] == 0
    assert pf.distribution(pd.Series([0.01, np.nan, 0.03]))["observations"] == 2


# --- no look-ahead -----------------------------------------------------------------------

def test_rolling_metrics_do_not_use_future_data():
    e = random_equity(500)
    bench = random_equity(500, seed=9)
    rf = zero_rf(e)
    full = pf.rolling_metrics(e, rf, 63, bench)
    cut = 300
    truncated = pf.rolling_metrics(e.iloc[:cut], rf, 63, bench.iloc[:cut])
    shocked_e, shocked_b = e.copy(), bench.copy()
    shocked_e.iloc[cut:] *= 2
    shocked_b.iloc[cut:] *= 0.5
    shocked = pf.rolling_metrics(shocked_e, rf, 63, shocked_b)

    pd.testing.assert_frame_equal(truncated, full.loc[truncated.index])
    pd.testing.assert_frame_equal(shocked.loc[truncated.index], full.loc[truncated.index])


def test_regime_labels_do_not_use_future_data():
    prices = random_equity(600)
    full = rg.classify(prices)
    truncated = rg.classify(prices.iloc[:400])
    shocked = prices.copy()
    shocked.iloc[400:] *= 0.5

    pd.testing.assert_frame_equal(truncated, full.iloc[:400])
    pd.testing.assert_frame_equal(rg.classify(shocked).iloc[:400], full.iloc[:400])


def test_regime_label_applies_to_the_next_days_return():
    labels = pd.DataFrame({r: [False, True, False] for r in rg.REGIMES}, index=pd.bdate_range("2024-01-01", periods=3))
    applied = rg.return_day_labels(labels, labels.index[1:])
    assert list(applied["bull"]) == [False, True]


def test_regime_definitions():
    prices = pd.Series(np.r_[np.linspace(100, 200, 250), np.linspace(200, 150, 50)],
                       index=pd.bdate_range("2023-01-02", periods=300))
    settings = rg.RegimeSettings(trend_window=50, volatility_window=20, large_drawdown=0.15)
    labels = rg.classify(prices, settings)

    assert not labels.iloc[:49][["bull", "bear"]].any().any()   # moving average not ready
    assert labels["bull"].iloc[200] and labels["bear"].iloc[-1]
    assert labels["large_drawdown"].iloc[-1] and not labels["large_drawdown"].iloc[260]


def test_benchmark_comparison_uses_only_shared_dates():
    s = random_equity(300)
    b = random_equity(400, seed=5)
    rf = zero_rf(b)
    table = pf.benchmark_comparison(pd.DataFrame({"s": s, "b": b}), "b", rf).set_index("metric")
    expected = pf.summary(b.loc[s.index], rf)

    assert table.loc["cagr", "benchmark_value"] == pytest.approx(expected["cagr"])
    assert table.loc["cagr", "difference"] == pytest.approx(pf.cagr(s) - expected["cagr"])
    assert set(table.index) == {"total_return", "cagr", "volatility_annualized", "max_drawdown",
                                "sharpe_ratio", "sortino_ratio", "calmar_ratio", "final_equity"}


def test_downside_correlation_uses_negative_reference_days_only():
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0, 0.01, (300, 2)), columns=["VOO", "X"])
    table = pf.correlation_tables(r, r, "VOO")
    down = table[table.matrix == "correlation_downside_VOO_negative_days"].set_index("series")
    assert down.loc["X", "VOO"] == pytest.approx(r[r.VOO < 0].corr().loc["X", "VOO"])


# --- statistical inference ---------------------------------------------------------------

def test_mean_confidence_intervals():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))
    iid, nw = inf.mean_confidence_intervals(r, 0.95)
    se = r.std() / math.sqrt(len(r)) * 252

    assert iid["estimate"] == pytest.approx(r.mean() * 252)
    assert iid["ci_lower"] == pytest.approx(r.mean() * 252 - 1.959964 * se, rel=1e-5)
    assert nw["lag"] == inf.newey_west_lag(2000)


def test_newey_west_grows_with_positive_autocorrelation():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 5000)
    smooth = np.convolve(noise, np.ones(5) / 5, mode="valid")
    assert inf.newey_west_variance(noise, 10) == pytest.approx(noise.var(), rel=0.1)
    assert inf.newey_west_variance(smooth, 10) > 2 * smooth.var()


def test_sharpe_standard_error_follows_lo_2002():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    rows = {row["method"]: row for row in inf.sharpe_inference(r, pd.Series(0.0, index=r.index), 0.95)}
    sr = r.mean() / r.std()

    assert rows["lo_2002_iid"]["standard_error"] == pytest.approx(math.sqrt((1 + sr ** 2 / 2) / 1000) * SQRT252)
    assert 0.5 < rows["bailey_lopez_de_prado_2012"]["estimate"] < 1


def test_holm_adjustment():
    adjusted = inf.holm_adjust(pd.Series([0.01, 0.04, 0.03], index=["a", "b", "c"]))
    assert list(adjusted) == pytest.approx([0.03, 0.06, 0.06])


def test_block_bootstrap_draws_consecutive_days():
    idx = inf.bootstrap_indices(100, 3, np.random.default_rng(0), block_size=10)
    assert idx.shape == (3, 100)
    assert ((np.diff(idx[:, :10], axis=1) % 100) == 1).all()


def _bootstrap_input():
    rng = np.random.default_rng(2)
    index = pd.bdate_range("2020-01-01", periods=500)
    returns = pd.DataFrame(rng.normal(0.0004, 0.01, (500, 2)), index=index, columns=["bench", "s"])
    return returns, pd.Series(0.0001, index=index)


def test_bootstrap_is_reproducible_with_a_seed():
    returns, rf = _bootstrap_input()
    first = pd.DataFrame(inf.bootstrap(returns, rf, "bench", 300, seed=7, confidence=0.95, block_size=5))
    second = pd.DataFrame(inf.bootstrap(returns, rf, "bench", 300, seed=7, confidence=0.95, block_size=5))
    other = pd.DataFrame(inf.bootstrap(returns, rf, "bench", 300, seed=8, confidence=0.95, block_size=5))

    pd.testing.assert_frame_equal(first, second)
    assert not first["ci_lower"].equals(other["ci_lower"])


def test_bootstrap_rows_and_point_estimates():
    returns, rf = _bootstrap_input()
    rows = pd.DataFrame(inf.bootstrap(returns, rf, "bench", 300, seed=1, confidence=0.95))
    sharpe = rows[(rows.strategy == "s") & (rows.statistic == "sharpe_ratio")].iloc[0]

    assert sharpe["estimate"] == pytest.approx(pf.sharpe_ratio(returns["s"], rf))
    assert sharpe["ci_lower"] < sharpe["estimate"] < sharpe["ci_upper"]
    assert sharpe["observations"] == 500 and sharpe["resamples"] == 300
    assert not ((rows.strategy == "bench") & rows.statistic.str.endswith("vs_benchmark")).any()
    assert rows["p_value"].dropna().between(1 / 301, 1).all()


# --- trades ------------------------------------------------------------------------------

def tx(date, ticker, side, qty, price, commission=0.0):
    return {"date": pd.Timestamp(date), "ticker": ticker, "side": side, "quantity": qty,
            "gross_value": qty * price, "commission": commission}


def test_round_trip_profit_includes_commission():
    transactions = pd.DataFrame([
        tx("2024-01-02", "AAA", "buy", 10, 10, 1),
        tx("2024-02-01", "AAA", "sell", 10, 12, 1),
        tx("2024-02-01", "BBB", "buy", 5, 20),
    ])
    closed, still_open = tr.round_trips(transactions)

    assert len(closed) == 1
    assert closed.iloc[0]["profit"] == pytest.approx(119 - 101)
    assert closed.iloc[0]["holding_days"] == 30
    assert list(still_open["ticker"]) == ["BBB"]


def test_trade_stats():
    trips = pd.DataFrame({
        "exit_date": pd.bdate_range("2024-01-01", periods=5),
        "profit": [10.0, 5.0, -3.0, -2.0, 4.0],
        "return": [0.1, 0.05, -0.03, -0.02, 0.04],
        "holding_days": [10, 20, 30, 40, 50],
    })
    stats = tr.trade_stats(trips)

    assert stats["win_rate"] == pytest.approx(0.6)
    assert stats["profit_factor"] == pytest.approx(19 / 5)
    assert stats["max_consecutive_wins"] == 2
    assert stats["max_consecutive_losses"] == 2
    assert stats["average_loss_return"] == pytest.approx(-0.025)
    no_losses = tr.trade_stats(trips[trips.profit > 0])
    assert math.isnan(no_losses["profit_factor"])


def test_trade_statistics_are_not_forced_on_buy_and_hold():
    rows = tr.strategy_trade_table("bh", "buy_and_hold", pd.DataFrame(), None)
    assert rows == [{"strategy": "bh", "strategy_type": "buy_and_hold", "scope": "all", "applicable": False,
                     "reason": tr.NOT_APPLICABLE_REASON["buy_and_hold"]}]
