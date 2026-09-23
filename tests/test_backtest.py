import numpy as np
import pandas as pd
import pytest

from engine import CostModel, run_backtest
from portfolio import simulate_portfolio
from strategies import (
    BuyAndHold,
    Momentum,
    MovingAverageTrend,
    PeriodicRebalance,
    Strategy,
    build_strategy,
    is_period_start,
)


def prices_from(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows, dtype=float)
    df.index = pd.bdate_range("2024-01-01", periods=len(df))
    return df


def random_prices(days: int = 600, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=days)
    return pd.DataFrame({
        "AAA": 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, days)),
        "BBB": 50 * np.cumprod(1 + rng.normal(0.0002, 0.005, days)),
        "CASH": 10 * 1.0001 ** np.arange(days),
    }, index=index)


class FixedSignals(Strategy):
    """Emits the given weights on the given dates (by position in the price index)."""

    def __init__(self, signals: dict[int, dict], tickers: list[str]):
        self.name = "fixed"
        self.signals = signals
        self._tickers = tickers

    @property
    def tickers(self):
        return self._tickers

    def decide(self, history, first_day):
        return self.signals.get(len(history) - 1)


class Spy(Strategy):
    """Records what it was shown, and rebalances monthly to 50/50."""

    def __init__(self):
        self.name = "spy"
        self.seen = []

    @property
    def tickers(self):
        return ["AAA", "BBB"]

    def decide(self, history, first_day):
        self.seen.append((history.index[-1], len(history)))
        if first_day or is_period_start(history, "monthly"):
            return {"AAA": 0.5, "BBB": 0.5}
        return None


# --- Strategy interface and signals ------------------------------------------------

def test_strategy_is_an_abstract_interface():
    with pytest.raises(TypeError):
        Strategy()


def test_build_strategy_from_config():
    spec = {"name": "trend", "type": "moving_average", "asset": "AAA", "risk_off": "CASH", "window": 50}
    strategy = build_strategy(spec)

    assert isinstance(strategy, MovingAverageTrend)
    assert strategy.tickers == ["AAA", "CASH"]
    assert strategy.window == 50


@pytest.mark.parametrize("spec, message", [
    ({"name": "x", "type": "magic"}, "unknown strategy type"),
    ({"name": "x", "type": "buy_and_hold", "weights": {"AAA": 1}, "leverage": 2}, "unknown parameter"),
    ({"name": "x", "type": "moving_average", "asset": "AAA"}, "risk_off"),
    ({"type": "buy_and_hold", "weights": {"AAA": 1}}, "no name"),
    ({"name": "x", "type": "periodic_rebalance", "weights": {"AAA": 1}, "frequency": "weekly"}, "unknown frequency"),
    ({"name": "x", "type": "momentum", "assets": ["AAA"], "risk_off": "CASH", "top_n": 2}, "top_n"),
])
def test_invalid_strategy_config_is_rejected(spec, message):
    with pytest.raises(ValueError, match=message):
        build_strategy(spec)


def test_period_start_uses_only_past_dates():
    history = pd.DataFrame({"AAA": [1.0, 1.0, 1.0]}, index=pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"]))

    assert is_period_start(history, "monthly")
    assert not is_period_start(history.iloc[:2], "monthly")
    assert not is_period_start(history.iloc[:1], "monthly")
    assert not is_period_start(history, "annual")


def test_buy_and_hold_signals_once():
    history = random_prices(30)
    strategy = BuyAndHold("bh", {"AAA": 1.0})

    assert strategy.decide(history, first_day=True) == {"AAA": 1.0}
    assert strategy.decide(history, first_day=False) is None


def test_periodic_rebalance_signals_at_each_new_period():
    index = pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02"])
    history = pd.DataFrame({"AAA": 1.0, "BBB": 1.0}, index=index)
    strategy = PeriodicRebalance("pr", {"AAA": 0.6, "BBB": 0.4}, "annual")

    assert strategy.decide(history, first_day=False) == {"AAA": 0.6, "BBB": 0.4}
    assert strategy.decide(history.iloc[:2], first_day=False) is None


def test_moving_average_switches_between_asset_and_risk_off():
    strategy = MovingAverageTrend("trend", "AAA", "CASH", window=3)
    rising = prices_from({"AAA": [10, 11, 12, 13], "CASH": [1, 1, 1, 1]})
    falling = prices_from({"AAA": [13, 12, 11, 10], "CASH": [1, 1, 1, 1]})

    assert strategy.decide(rising, first_day=True) == {"AAA": 1.0}
    assert strategy.decide(falling, first_day=True) == {"CASH": 1.0}
    assert strategy.decide(rising.iloc[:2], first_day=True) is None  # not enough history


def test_momentum_picks_the_best_trailing_return_above_the_hurdle():
    history = prices_from({
        "AAA": [100, 110], "BBB": [100, 120], "CCC": [100, 101], "CASH": [100, 100.5],
    })

    top1 = Momentum("m", ["AAA", "BBB", "CCC"], "CASH", lookback_days=1, top_n=1)
    assert top1.decide(history, first_day=True) == {"BBB": 1.0}

    top2 = Momentum("m", ["AAA", "BBB", "CCC"], "CASH", lookback_days=1, top_n=2)
    assert top2.decide(history, first_day=True) == {"BBB": 0.5, "AAA": 0.5}

    losing = prices_from({"AAA": [100, 99], "BBB": [100, 98], "CCC": [100, 97], "CASH": [100, 100.1]})
    assert top2.decide(losing, first_day=True) == {"CASH": 1.0}

    # only one asset beats the hurdle: its half goes to it, the unused half to CASH
    mixed = prices_from({"AAA": [100, 105], "BBB": [100, 99], "CCC": [100, 98], "CASH": [100, 100.1]})
    assert top2.decide(mixed, first_day=True) == {"AAA": 0.5, "CASH": 0.5}


# --- Engine: execution, positions, cash, transactions ------------------------------

TWO_DAYS = prices_from({"AAA": [10, 20, 20], "BBB": [5, 5, 10]})


def test_orders_fill_on_the_next_day_at_that_close():
    result = run_backtest(TWO_DAYS, FixedSignals({0: {"AAA": 1.0}}, ["AAA", "BBB"]), 1000)
    tx = result.transactions.iloc[0]

    assert tx["signal_date"] == TWO_DAYS.index[0]
    assert tx["date"] == TWO_DAYS.index[1]
    assert tx["fill_price"] == 20
    assert tx["quantity"] == pytest.approx(50)
    assert list(result.equity) == pytest.approx([1000, 1000, 1000])


def test_positions_cash_and_equity_are_tracked_daily():
    signals = {0: {"AAA": 1.0}, 1: {"BBB": 1.0}}
    result = run_backtest(TWO_DAYS, FixedSignals(signals, ["AAA", "BBB"]), 1000)

    # day 1: buy 50 AAA at 20; day 2: sell 50 AAA at 20, buy 100 BBB at 10
    assert list(result.positions["AAA"]) == pytest.approx([0, 50, 0])
    assert list(result.positions["BBB"]) == pytest.approx([0, 0, 100])
    assert list(result.cash) == pytest.approx([1000, 0, 0], abs=1e-6)
    marked = result.cash + (result.positions * TWO_DAYS).sum(axis=1)
    assert list(result.equity) == pytest.approx(list(marked))
    assert list(result.transactions["side"]) == ["buy", "sell", "buy"]


def test_commission_is_charged_and_buys_are_scaled_to_available_cash():
    costs = CostModel(commission_rate=0.01)
    result = run_backtest(TWO_DAYS, FixedSignals({0: {"AAA": 1.0}}, ["AAA", "BBB"]), 1000, costs)
    tx = result.transactions.iloc[0]

    assert tx["gross_value"] == pytest.approx(1000 / 1.01)
    assert tx["commission"] == pytest.approx(0.01 * tx["gross_value"])
    assert result.cash.iloc[-1] == pytest.approx(0, abs=1e-6)


def test_minimum_commission():
    costs = CostModel(commission_rate=0.001, commission_min=5)
    assert costs.commission(1000) == 5
    assert costs.commission(10_000) == 10
    assert costs.commission(0) == 0


def test_slippage_makes_buys_dearer_and_sells_cheaper():
    costs = CostModel(slippage_rate=0.01)
    signals = {0: {"AAA": 1.0}, 1: {"BBB": 1.0}}
    result = run_backtest(TWO_DAYS, FixedSignals(signals, ["AAA", "BBB"]), 1000, costs)
    buy, sell = result.transactions.iloc[0], result.transactions.iloc[1]

    assert buy["fill_price"] == pytest.approx(20.2)
    assert sell["fill_price"] == pytest.approx(19.8)
    assert buy["slippage_cost"] == pytest.approx(buy["quantity"] * 0.2)


def test_cash_accounting_balances_and_never_goes_negative():
    prices = random_prices()
    costs = CostModel(commission_rate=0.002, commission_min=1, slippage_rate=0.001)
    strategy = Momentum("m", ["AAA", "BBB"], "CASH", lookback_days=60, top_n=1)
    result = run_backtest(prices, strategy, 10_000, costs, start=prices.index[60])
    tx = result.transactions

    bought = tx.loc[tx["side"] == "buy", "gross_value"].sum()
    sold = tx.loc[tx["side"] == "sell", "gross_value"].sum()
    assert result.cash.iloc[-1] == pytest.approx(10_000 - bought + sold - tx["commission"].sum(), abs=1e-6)
    assert result.cash.min() >= -1e-6
    assert len(tx) > 5


def test_zero_cost_buy_and_hold_matches_the_phase_4_engine():
    prices = random_prices()
    result = run_backtest(prices, BuyAndHold("bh", {"AAA": 0.7, "BBB": 0.3}), 1000)
    # the backtest fills the day-0 signal on day 1, so compare with a portfolio bought on day 1
    phase4 = simulate_portfolio(prices.iloc[1:], {"AAA": 0.7, "BBB": 0.3}, "none", initial_investment=1000)

    assert list(result.equity.iloc[1:]) == pytest.approx(list(phase4.value))


def test_invalid_signals_are_rejected():
    for bad in [{"AAA": 0.8, "BBB": 0.8}, {"AAA": -0.5, "BBB": 1.0}, {"ZZZ": 1.0}]:
        with pytest.raises(ValueError):
            run_backtest(TWO_DAYS, FixedSignals({0: bad}, ["AAA", "BBB"]), 1000)


def test_invalid_costs_are_rejected():
    with pytest.raises(ValueError):
        CostModel(commission_rate=-0.01)
    with pytest.raises(ValueError):
        CostModel(slippage_rate=1.5)


# --- Look-ahead and reproducibility ------------------------------------------------

def test_strategy_only_sees_history_up_to_the_decision_day():
    prices = random_prices(200)
    spy = Spy()
    run_backtest(prices, spy, 1000, start=prices.index[50])

    assert spy.seen[0] == (prices.index[50], 51)  # warm-up history is visible, the future is not
    for position, (last_seen, length) in enumerate(spy.seen, start=50):
        assert last_seen == prices.index[position]
        assert length == position + 1


def test_results_do_not_depend_on_future_prices():
    prices = random_prices()
    cut = 400
    full = run_backtest(prices, Spy(), 1000)
    truncated = run_backtest(prices.iloc[:cut], Spy(), 1000)
    shocked = prices.copy()
    shocked.iloc[cut:] *= 3.0
    with_shock = run_backtest(shocked, Spy(), 1000)

    assert list(truncated.equity) == pytest.approx(list(full.equity.iloc[:cut]))
    assert list(with_shock.equity.iloc[:cut]) == pytest.approx(list(full.equity.iloc[:cut]))


def test_input_hash_does_not_depend_on_line_endings(tmp_path):
    from run_backtest import content_sha256

    lf, crlf, other = tmp_path / "lf.csv", tmp_path / "crlf.csv", tmp_path / "other.csv"
    lf.write_bytes(b"Date,AAA\n2024-01-02,10\n")
    crlf.write_bytes(b"Date,AAA\r\n2024-01-02,10\r\n")
    other.write_bytes(b"Date,AAA\n2024-01-02,11\n")

    assert content_sha256(lf) == content_sha256(crlf)
    assert content_sha256(lf) != content_sha256(other)


def test_same_input_gives_identical_results():
    prices = random_prices()
    strategy = Momentum("m", ["AAA", "BBB"], "CASH", lookback_days=60)
    costs = CostModel(commission_rate=0.001, slippage_rate=0.001)
    first = run_backtest(prices, strategy, 1000, costs, start=prices.index[60])
    second = run_backtest(prices, strategy, 1000, costs, start=prices.index[60])

    assert first.equity.equals(second.equity)
    assert first.transactions.equals(second.transactions)
    assert first.positions.equals(second.positions)
