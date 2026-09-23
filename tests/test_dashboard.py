import json
import math

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import compute as cp
import inference as inf
import performance as pf
from compute import ROOT

REPORTS = ROOT / "reports"
PRICES = cp.load_prices()


def close(a, b, tol=1e-6):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


# --- calculations reuse the Phase 2-7 definitions ---------------------------------------------

def test_asset_kpis_are_the_phase6_summary():
    p = PRICES[["VOO", "BND"]]
    kpis = cp.asset_kpis(p, PRICES["BIL"])
    rf = pf.daily_returns(PRICES["BIL"])
    for t in p:
        expected = pf.summary(p[t], rf)
        for key in ("total_return", "cagr", "volatility_annualized", "max_drawdown", "sharpe_ratio",
                    "sortino_ratio", "calmar_ratio"):
            assert close(kpis.loc[t, key], expected[key])


def test_growth_and_correlation():
    p = PRICES[["VOO", "VT", "BND"]]
    assert (cp.growth(p).iloc[0] == 1).all()
    daily = cp.correlation(p, "daily")
    assert np.allclose(np.diag(daily), 1) and np.allclose(daily, daily.T)
    assert daily.loc["VOO", "VT"] > daily.loc["VOO", "BND"]
    monthly = cp.correlation(p, "monthly")
    assert monthly.shape == (3, 3)
    with pytest.raises(ValueError):
        cp.correlation(p, "weekly")


@pytest.mark.parametrize("weights, ok", [({"VOO": 60, "BND": 40}, True), ({"VOO": 60, "BND": 30}, False),
                                         ({"VOO": 0, "BND": 0}, False), ({"VOO": 110, "BND": -10}, False)])
def test_weight_problems(weights, ok):
    assert (cp.weight_problems(weights) == []) == ok


def test_normalize_percent_sums_to_100_and_keeps_proportions():
    assert cp.normalize_percent({"a": 1, "b": 1, "c": 1}) == {"a": 34, "b": 33, "c": 33}
    assert cp.normalize_percent({"a": 30, "b": 30, "c": 0}) == {"a": 50, "b": 50, "c": 0}
    result = cp.normalize_percent({"a": 7, "b": 13, "c": 29, "d": 0})
    assert sum(result.values()) == 100
    with pytest.raises(ValueError):
        cp.normalize_percent({"a": 0})


def test_portfolio_matches_the_phase4_report():
    report = pd.read_csv(REPORTS / "portfolio" / "portfolio_summary.csv")
    presets = cp.presets()
    checked = 0
    for (name, rebalance, method), row in report.set_index(["portfolio", "rebalance", "mode"]).iterrows():
        if rebalance not in ("none", "annual") or name not in ("US 60/40", "Conservative"):
            continue
        percent = {t: w * 100 for t, w in presets[name].items()}
        _, summary = cp.simulate(PRICES, percent, rebalance, method, 10_000)
        for key in ("cagr", "annual_volatility", "max_drawdown", "sharpe_ratio", "final_value"):
            assert close(summary[key], row[key]), (name, rebalance, method, key)
        checked += 1
    assert checked == 8  # 2 portfolios x 2 rebalancing rules x (lump sum, DCA)


def test_simulate_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        cp.simulate(PRICES, {"VOO": 100}, "none", "weekly", 1000)


def test_trading_start_keeps_the_warm_up():
    p = PRICES
    assert cp.trading_start(p, p.index[0], 252) == p.index[252]
    assert cp.trading_start(p, p.index[1000], 252) == p.index[1000]
    assert cp.trading_start(p, p.index[-1] + pd.Timedelta(days=30), 252) == p.index[-1]
    with pytest.raises(ValueError):
        cp.trading_start(p.iloc[:100], p.index[0], 252)


def test_strategy_parameters_are_applied_without_changing_the_config():
    settings = cp.BacktestSettings(5000, 0.001, 0.002, ma_window=100, momentum_lookback=126, momentum_top_n=2)
    specs = cp.strategy_specs(settings)
    assert specs["VOO 200-day trend"]["window"] == 100
    assert specs["Momentum top 1"]["lookback_days"] == 126 and specs["Momentum top 1"]["top_n"] == 2
    assert cp.strategy_specs(cp.default_backtest_settings())["VOO 200-day trend"]["window"] == 200


def test_default_backtests_match_the_phase5_report():
    settings = cp.default_backtest_settings()
    names = list(cp.strategy_specs(settings))
    results = cp.run_strategies(PRICES, names, settings, PRICES.index[0], PRICES.index[-1])
    report = pd.read_csv(REPORTS / "backtest" / "backtest_summary.csv")
    report = report[report.costs == "with costs"].set_index("strategy")
    table = cp.backtest_table(results, settings, cp.risk_free_returns(PRICES))
    for name in names:
        assert close(table.loc[name, "final_equity"], report.loc[name, "final_equity"])
        assert close(table.loc[name, "sharpe_ratio"], report.loc[name, "sharpe_ratio"])
        assert table.loc[name, "transactions"] == report.loc[name, "trades"]
    assert np.isnan(table.loc["Buy & Hold VOO", "win_rate"])
    assert not table.loc["Buy & Hold VOO", "trade_stats_applicable"]
    trades = pd.read_csv(REPORTS / "statistics" / "trade_statistics.csv")
    trend = trades[(trades.strategy == "VOO 200-day trend") & (trades.scope == "all")].iloc[0]
    assert close(table.loc["VOO 200-day trend", "win_rate"], trend.win_rate)


def test_higher_costs_lower_the_result():
    base = cp.default_backtest_settings()
    costly = cp.BacktestSettings(base.initial_cash, 0.01, 0.01, base.ma_window, base.momentum_lookback,
                                 base.momentum_top_n)
    args = (PRICES, ["VOO 200-day trend"], PRICES.index[0], PRICES.index[-1])
    cheap = cp.run_strategies(args[0], args[1], base, *args[2:])["VOO 200-day trend"]
    dear = cp.run_strategies(args[0], args[1], costly, *args[2:])["VOO 200-day trend"]
    assert dear.equity.iloc[-1] < cheap.equity.iloc[-1]


def test_statistics_tables_use_the_phase6_functions():
    idx = pd.bdate_range("2020-01-01", periods=600)
    rng = np.random.default_rng(1)
    equities = pd.DataFrame({"a": 100 * np.cumprod(1 + rng.normal(4e-4, 0.01, 600)),
                             "b": 100 * np.cumprod(1 + rng.normal(2e-4, 0.01, 600))}, index=idx)
    equities.iloc[:10, 1] = np.nan  # b starts later: every statistic must use the shared dates
    rf = pd.Series(1e-4, index=idx)
    returns = cp.aligned_returns(equities)
    assert returns.index[0] == idx[11] and returns.notna().all().all()

    sharpe = cp.sharpe_table(returns, rf, 0.95)
    lo = inf.sharpe_inference(returns["a"], rf, 0.95)[0]
    assert close(sharpe.loc["a", "sharpe_ratio"], lo["estimate"]) and close(sharpe.loc["a", "se_lo"], lo["standard_error"])
    assert 0 <= sharpe.loc["a", "psr"] <= 1

    risk = cp.risk_table(returns)
    assert close(risk.loc["a", "var_95"], pf.historical_var_cvar(returns["a"], 0.95)["var"])

    boot = cp.bootstrap_table(returns, rf, "a", 200, 7, 0.95, 21)
    assert set(boot.method) == {"bootstrap_iid", "bootstrap_block"}
    diffs = boot[boot.p_value.notna()]
    assert (diffs.p_value_holm >= diffs.p_value - 1e-12).all()
    assert boot.equals(cp.bootstrap_table(returns, rf, "a", 200, 7, 0.95, 21))  # seeded


# --- Phase 7 through the dashboard -------------------------------------------------------------

@pytest.fixture(scope="module")
def ml_setup():
    import ml_compute as mc
    cfg, data = mc.load_setup()
    return mc, cfg, data


def test_ml_page_reproduces_the_phase7_report(ml_setup):
    mc, cfg, data = ml_setup
    search = mc.validation_search(data, cfg, "logistic_regression", "base")
    params = search.loc[search.selected, "hyperparameters"].iloc[0]
    assert params == '{"C": 0.01}'

    _, scores = mc.predict(data, cfg, "logistic_regression", "base", json.loads(params))
    metrics = mc.segment_metrics(data, {"ml:logistic_regression/base": scores}, 0.5).set_index("segment")
    report = pd.read_csv(REPORTS / "ml" / "metrics" / "prediction_metrics.csv")
    report = report[report.predictor == "ml:logistic_regression/base"].set_index("segment")
    for s in ("train", "validation", "test"):
        assert close(metrics.loc[s, "roc_auc"], report.loc[s, "roc_auc"])
        assert close(metrics.loc[s, "accuracy"], report.loc[s, "accuracy"])

    test = data.rows["test"]
    _, invest = mc.backtest(data, cfg, {"ml": scores.loc[test], **mc.rule_scores(data, cfg, test)}, test[0],
                            data.last, 10_000, 0.0005, 0.0005, 0.5)
    inv = pd.read_csv(REPORTS / "ml" / "metrics" / "investment_metrics.csv")
    inv = inv[inv.segment == "test"].set_index("predictor")
    assert close(invest.loc["ml", "cagr"], inv.loc["ml:logistic_regression/base", "cagr"])
    assert close(invest.loc[mc.BUY_AND_HOLD, "sharpe_ratio"], inv.loc["baseline:always_up", "sharpe_ratio"])
    assert close(invest.loc["momentum_252d", "cagr"], inv.loc["baseline:momentum_252d", "cagr"])


def test_walk_forward_uses_only_earlier_labels(ml_setup):
    mc, cfg, data = ml_setup
    scores, yearly = mc.walk_forward(data, cfg, "logistic_regression", "base")
    for row in yearly.itertuples():
        assert pd.Timestamp(row.fit_end) < pd.Timestamp(row.year, 1, 1)
        assert data.label_end.loc[:pd.Timestamp(row.fit_end)].max() < pd.Timestamp(row.year, 1, 1)
    assert scores.index[0].year == cfg["walk_forward"]["first_test_year"]
    report = pd.read_csv(REPORTS / "ml" / "metrics" / "walk_forward_prediction_metrics.csv")
    report = report[report.predictor == "ml:logistic_regression/base"].set_index("year")
    assert np.allclose(yearly.set_index("year")["roc_auc"], report["roc_auc"])


# --- the app itself ----------------------------------------------------------------------------

PAGES = ["tab_overview", "tab_asset", "tab_portfolio", "tab_backtest", "tab_stats", "tab_ml"]


def app() -> AppTest:
    return AppTest.from_file("../app.py", default_timeout=600)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_errors(page):
    at = app().run()
    if page != "tab_overview":
        at.switch_page(f"src/dashboard/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.title[0].value.startswith(("①", "②", "③", "④", "⑤", "⑥"))


def test_sidebar_settings_are_shared_and_validated():
    at = app().run()
    at.sidebar.multiselect[0].set_value([]).run()
    assert any("資産" in w.value for w in at.warning)
    at.sidebar.multiselect[0].set_value(["VOO", "BND"]).run()
    assert not at.warning
    assert [m.label for m in at.metric][:7] == ["累積リターン", "CAGR", "年率ボラ", "最大DD", "Sharpe", "Sortino", "Calmar"]
    first = PRICES.index[0].date()
    at.sidebar.date_input[0].set_value((first, first)).run()
    assert any("取引日" in w.value for w in at.warning)


def test_portfolio_rejects_weights_that_do_not_sum_to_100():
    at = app().run()
    at.switch_page("src/dashboard/tab_portfolio.py").run()
    voo = next(s for s in at.slider if s.label.startswith("VOO"))
    voo.set_value(50).run()
    assert any("合計" in e.value for e in at.error)
    at.button[0].click().run()  # scale to 100%
    assert not at.error
    assert sum(s.value for s in at.slider) == 100


def test_detailed_mode_shows_data_tables():
    at = app().run()
    assert not at.expander
    at.sidebar.radio[0].set_value("詳細（データ表・CSV も表示）").run()
    assert any(e.label.startswith("データ表") for e in at.expander)
