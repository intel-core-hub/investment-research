"""Phase 9 tests. Synthetic data and temporary folders only: nothing reads or writes data/live/."""
import subprocess

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import attribution as at
import audit
import bridge
import demo
import execution_analyzer as ea
import fx_and_taxes as fxt
import paper_engine as paper
import real_ledger as rl
import risk_controller as rc
import signal_pipeline as sp
from market import MarketData
from settings import ROOT, LiveStore, load_settings, merge

ASSETS = ["VOO", "VTI", "VT", "EWJ", "BND", "BIL"]


def make_market(n=320, start="2025-01-01", seed=0, fx_vol=0.004) -> MarketData:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=n)
    close = pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(3e-4, 0.01, n)) for t in ASSETS}, index=index)
    volume = pd.DataFrame({t: rng.integers(1e6, 5e6, n).astype(float) for t in ASSETS}, index=index)
    fx = pd.Series(150 * np.cumprod(1 + rng.normal(0, fx_vol, n)), index=index, name="usdjpy")
    return MarketData(close, close, volume, fx)


def make_settings(**overrides) -> dict:
    base = load_settings(local_path=None)   # never the personal config/live.local.toml
    base = merge(base, {"signal": {"strategy": "60/40 annual rebalance", "mode": "buy_only",
                                   "fractional_shares": False},
                        "limits": {"max_orders_per_day": 5, "max_order_jpy": 1_000_000, "max_weight": 1.0,
                                   "max_data_age_days": 3}})
    return merge(base, overrides)


def tx(*rows) -> pd.DataFrame:
    return rl.normalize(pd.DataFrame(rows))


def deposit(date, amount, **kw):
    return {"date": date, "action": "DEPOSIT", "amount_jpy": amount, "account": "nisa", **kw}


def trade(date, action, asset, quantity, price, fx_rate, fees=0.0, tax=0.0, order_id=""):
    return {"date": date, "action": action, "asset": asset, "quantity": quantity, "price": price, "fx_rate": fx_rate,
            "fees_jpy": fees, "tax_jpy": tax, "account": "nisa", "order_id": order_id}


# --- 9.2 ledger ----------------------------------------------------------------------------------

def test_ledger_recalculates_cash_holdings_cost_basis_and_pnl():
    history = tx(deposit("2025-01-06", 100_000),
                 trade("2025-01-07", "BUY", "VOO", 2, 100.0, 150.0, fees=100),      # cost 30,100
                 trade("2025-01-08", "BUY", "VOO", 1, 110.0, 151.0),               # cost 16,610
                 trade("2025-01-09", "SELL", "VOO", 1, 120.0, 152.0, fees=50, tax=500),
                 {"date": "2025-01-10", "action": "DIVIDEND", "asset": "VOO", "amount_jpy": 1_000, "tax_jpy": 203},
                 {"date": "2025-01-10", "action": "FEE", "amount_jpy": 100})
    state = rl.rebuild(history)
    average = (30_100 + 16_610) / 3
    assert state.positions.loc["VOO", "quantity"] == 2
    assert state.positions.loc["VOO", "cost_basis_jpy"] == pytest.approx(average * 2)
    proceeds = 120 * 152 - 50
    assert state.realized_pnl_jpy == pytest.approx(proceeds - average)
    expected_cash = 100_000 - 30_100 - 16_610 + (proceeds - 500) + (1_000 - 203) - 100
    assert state.cash_jpy == pytest.approx(expected_cash)
    assert state.totals["taxes"] == pytest.approx(703) and state.totals["fees"] == pytest.approx(250)
    # the cash history ends at the same balance, date by date
    assert state.history["cash_jpy"].iloc[-1] == pytest.approx(expected_cash)
    assert state.history.loc[pd.Timestamp("2025-01-08"), "VOO"] == 3


def test_ledger_rejects_selling_more_than_held():
    with pytest.raises(rl.LedgerError, match="only 1 are held"):
        rl.rebuild(tx(deposit("2025-01-06", 100_000), trade("2025-01-07", "BUY", "VOO", 1, 100, 150),
                      trade("2025-01-08", "SELL", "VOO", 2, 100, 150)))


def test_ledger_rejects_buying_without_the_cash():
    with pytest.raises(rl.LedgerError, match="negative"):
        rl.rebuild(tx(deposit("2025-01-06", 10_000), trade("2025-01-07", "BUY", "VOO", 1, 100, 150)))


def test_ledger_lists_every_invalid_row():
    with pytest.raises(rl.LedgerError) as e:
        rl.rebuild(tx(deposit("2025-01-06", 10_000), deposit("2025-01-06", 10_000),
                      {"date": "2025-01-07", "action": "TRANSFER", "amount_jpy": 5},
                      trade("2025-01-08", "BUY", "VOO", 1, 0.0, 150)))
    message = str(e.value)
    assert "entered twice" in message and "unknown action" in message and "must be positive" in message


def test_append_is_all_or_nothing(tmp_path):
    path = tmp_path / "transactions.csv"
    rl.append_transactions(path, [deposit("2025-01-06", 50_000)])
    before = path.read_bytes()
    with pytest.raises(rl.LedgerError):
        rl.append_transactions(path, [trade("2025-01-07", "BUY", "VOO", 10, 100, 150)])   # costs 150,000
    assert path.read_bytes() == before
    history = rl.append_transactions(path, [trade("2025-01-07", "BUY", "VOO", 1, 100, 150)])
    assert list(history["transaction_id"]) == ["T00001", "T00002"]


def test_valuation_history_and_derived_files(tmp_path):
    m = make_market(n=30)
    d = m.close.index
    history = tx(deposit(d[0], 100_000), trade(d[1], "BUY", "VOO", 3, float(m.close.loc[d[1], "VOO"]),
                                                float(m.fx.loc[d[1]])),
                 deposit(d[5], 50_000))
    state = rl.rebuild(history)
    val = rl.valuation_history(state, m.close, m.fx)
    day = d[10]
    assert val.loc[day, "total_jpy"] == pytest.approx(state.cash_jpy + 3 * m.close.loc[day, "VOO"] * m.fx.loc[day])
    assert val.loc[d[5], "flow_jpy"] == 50_000
    # a deposit is not a gain: on the deposit day only the price move counts
    price_move = 3 * (m.close.loc[d[5], "VOO"] * m.fx.loc[d[5]] - m.close.loc[d[4], "VOO"] * m.fx.loc[d[4]])
    assert val.loc[d[5], "daily_return"] == pytest.approx(price_move / val.loc[d[4], "total_jpy"])

    store = LiveStore(tmp_path).ensure()
    rl.save_derived(state, val, store.holdings, store.cash, store.valuations)
    holdings = pd.read_csv(store.holdings)
    assert holdings.iloc[-1]["quantity"] == 3
    assert pd.read_csv(store.cash).iloc[-1]["cash_jpy"] == pytest.approx(state.cash_jpy)


# --- 9.1 FX, costs, taxes --------------------------------------------------------------------------

def test_fx_spread_costs_and_return_decomposition():
    assert fxt.conversion_rate(150, 0.25, "buy") == 150.25 and fxt.conversion_rate(150, 0.25, "sell") == 149.75
    with pytest.raises(ValueError):
        fxt.conversion_rate(0, 0.25, "buy")
    costs = fxt.CostSettings(commission_rate=0.001, commission_min_jpy=0, fx_spread_jpy=0.25, slippage_rate=0.0)
    t = fxt.estimate_trade("buy", 2, 100.0, 150.0, costs)
    assert t["gross_jpy"] == pytest.approx(2 * 100 * 150.25)
    assert t["fx_cost_jpy"] == pytest.approx(2 * 100 * 0.25)
    assert t["cash_jpy"] == pytest.approx(-(t["gross_jpy"] * 1.001))
    d = fxt.decompose_jpy_return(100, 110, 150, 140)
    assert d["asset"] + d["fx"] + d["cross"] == pytest.approx(d["total"])


def test_japanese_taxes():
    taxes = fxt.TaxSettings()
    nisa = fxt.dividend_tax(100, 150, "nisa", taxes)
    taxable = fxt.dividend_tax(100, 150, "tokutei", taxes)
    assert nisa["us_withholding_jpy"] == pytest.approx(1_500) and nisa["jp_tax_jpy"] == 0
    assert taxable["jp_tax_jpy"] == pytest.approx(13_500 * 0.20315)
    assert fxt.capital_gains_tax(10_000, "nisa", taxes) == 0
    assert fxt.capital_gains_tax(10_000, "tokutei", taxes) == pytest.approx(2_031.5)
    realized = pd.DataFrame({"date": ["2025-03-01", "2025-09-01", "2026-01-05"], "account": ["tokutei"] * 3,
                             "gain_jpy": [10_000, -4_000, -1_000]})
    table = fxt.annual_capital_gains_tax(realized, taxes).set_index("year")
    assert table.loc[2025, "tax_jpy"] == pytest.approx(6_000 * 0.20315)   # losses offset within the year
    assert table.loc[2026, "tax_jpy"] == 0
    with pytest.raises(ValueError):
        fxt.capital_gains_tax(1, "ideco", taxes)


# --- 9.5 risk controller ---------------------------------------------------------------------------

def order(order_id="d1-VOO-buy", asset="VOO", side="buy", quantity=1.0, price=100.0, fx=150.0):
    return rc.Order(order_id=order_id, decision_id="d1", date="2025-06-02", asset=asset, side=side,
                    quantity=quantity, price=price, fx_rate=fx)


LIMITS = rc.RiskLimits(max_orders_per_day=2, max_order_jpy=50_000, max_weight=0.8, max_data_age_days=3)
COSTS = fxt.CostSettings(fx_spread_jpy=0.25)
PRICES = pd.Series({"VOO": 100.0, "BND": 70.0})


def failed(checks) -> set[str]:
    return {c.name for c in checks if not c.passed}


def state_with_cash(cash, **held):
    rows = [deposit("2025-01-02", cash)] + [trade("2025-01-03", "BUY", a, q, 1.0, 1.0) for a, q in held.items()]
    return rl.rebuild(tx(*rows))


def test_duplicate_orders_are_blocked():
    state = state_with_cash(100_000)
    checks = rc.check_orders([order(), order()], state, PRICES, 150, LIMITS, COSTS, set(), 0)
    assert "no_duplicate_order" in failed(checks)
    checks = rc.check_orders([order()], state, PRICES, 150, LIMITS, COSTS, {"d1-VOO-buy"}, 0)
    assert "no_duplicate_order" in failed(checks)
    assert failed(rc.check_orders([order()], state, PRICES, 150, LIMITS, COSTS, set(), 0)) == set()


def test_buying_more_than_the_cash_is_blocked():
    state = state_with_cash(20_000)
    checks = rc.check_orders([order(quantity=2)], state, PRICES, 150, LIMITS, COSTS, set(), 0)  # about 30,050
    assert "enough_cash" in failed(checks)


def test_order_limits():
    state = state_with_cash(1_000_000, BND=1)
    big = rc.check_orders([order(quantity=4)], state, PRICES, 150, LIMITS, COSTS, set(), 0)
    assert "max_order_amount" in failed(big)
    many = rc.check_orders([order()], state, PRICES, 150, LIMITS, COSTS, set(), orders_already_today=2)
    assert "max_orders_per_day" in failed(many)
    oversell = rc.check_orders([order(asset="BND", side="sell", quantity=5, price=70)], state, PRICES, 150, LIMITS,
                               COSTS, set(), 0)
    assert "sell_within_holdings" in failed(oversell)
    concentrated = rc.check_orders([order(quantity=3)], state_with_cash(46_000), PRICES, 150, LIMITS, COSTS, set(), 0)
    assert "max_weight" in failed(concentrated)
    invalid = rc.check_orders([order(quantity=0), order("x", asset="XYZ")], state, PRICES, 150, LIMITS, COSTS, set(), 0)
    assert {"valid_quantity", "known_asset"} <= failed(invalid)


def test_stale_missing_or_invalid_data_stops():
    m = make_market(n=40)
    last = m.close.index[-1]
    ok = rc.check_data(m.close, m.fx, ["VOO", "BND"], last, 3)
    assert failed(ok) == set()
    stale = rc.check_data(m.close, m.fx, ["VOO"], last + pd.offsets.BDay(4), 3)
    assert {"prices_fresh", "fx_fresh"} <= failed(stale)
    missing = m.close.copy()
    missing.iloc[-1, missing.columns.get_loc("VOO")] = np.nan
    assert "price:VOO" in failed(rc.check_data(missing, m.fx, ["VOO"], last, 3))
    negative = m.close.copy()
    negative.iloc[-1, negative.columns.get_loc("BND")] = -1.0
    assert "price:BND" in failed(rc.check_data(negative, m.fx, ["BND"], last, 3))
    assert "fx_available" in failed(rc.check_data(m.close, pd.Series(dtype=float), ["VOO"], last, 3))
    assert "prices_available" in failed(rc.check_data(m.close.iloc[:0], m.fx, ["VOO"], last, 3))


def test_emergency_stop(tmp_path):
    store = LiveStore(tmp_path).ensure()
    assert rc.check_switch(store).passed
    rc.set_emergency_stop(store, "test")
    assert not rc.check_switch(store).passed
    rc.clear_emergency_stop(store)
    assert rc.check_switch(store).passed


# --- 9.4 pipeline, 9.6 approval, audit ---------------------------------------------------------------

def test_order_sizing_whole_shares_buy_only_and_rebalance():
    state = state_with_cash(100_000)
    close = pd.Series({"VOO": 100.0, "BND": 70.0})
    orders = sp.size_orders({"VOO": 0.6, "BND": 0.4}, state, close, 150, "buy_only", False, COSTS, "d", "2025-06-02")
    assert all(o.side == "buy" and float(o.quantity).is_integer() for o in orders)
    spent = -sum(fxt.estimate_trade("buy", o.quantity, o.price, 150, COSTS)["cash_jpy"] for o in orders)
    assert spent <= 100_000
    assert {o.asset: o.quantity for o in orders} == {"VOO": 3, "BND": 3}
    fractional = sp.size_orders({"VOO": 1.0}, state, close, 150, "buy_only", True, COSTS, "d", "2025-06-02")
    assert fractional[0].quantity == pytest.approx(100_000 / (100 * 150.25 * (1 + COSTS.slippage_rate)), rel=1e-3)

    heavy = rl.rebuild(tx(deposit("2025-01-02", 100_000), trade("2025-01-03", "BUY", "VOO", 6, 100, 150)))
    rebalance = sp.size_orders({"VOO": 0.5, "BND": 0.5}, heavy, close, 150, "rebalance", False, COSTS, "d",
                               "2025-06-02")
    assert [(o.asset, o.side) for o in rebalance][0] == ("VOO", "sell")
    assert not any(o.side == "sell" for o in sp.size_orders({"VOO": 0.5, "BND": 0.5}, heavy, close, 150, "buy_only",
                                                            False, COSTS, "d", "2025-06-02"))


def test_proposal_approval_fill_flow(tmp_path):
    store = LiveStore(tmp_path).ensure()
    settings = make_settings()
    m = make_market()
    today = m.as_of
    bridge.record_cash(store, "DEPOSIT", today, 300_000, "nisa")
    proposal, recorded = bridge.propose(store, settings, m, today)
    assert recorded and proposal.status == "PASS" and proposal.orders
    assert proposal.target_weights == {"VOO": 0.6, "BND": 0.4}
    again, recorded_again = bridge.propose(store, settings, m, today)
    assert again.decision_id == proposal.decision_id and not recorded_again   # never recorded twice

    with pytest.raises(bridge.BridgeError, match="only approved"):
        bridge.record_fill(store, proposal.orders[0].order_id, today, 1, 100, 150, "nisa")
    audit.decide(store, proposal.decision_id, approve=True, note="ok")
    with pytest.raises(audit.AuditError, match="already decided"):
        audit.decide(store, proposal.decision_id, approve=False)

    # approved but not yet filled: a new proposal counts those orders as done and does not repeat them
    after, _ = bridge.propose(store, settings, m, today)
    assert not after.orders or all(o.order_id not in {p.order_id for p in proposal.orders} for o in after.orders)

    first = proposal.orders[0]
    bridge.record_fill(store, first.order_id, today + pd.offsets.BDay(1), first.quantity, first.price * 1.001, 150.3,
                       "nisa", fees_jpy=0)
    with pytest.raises(bridge.BridgeError, match="already filled"):
        bridge.record_fill(store, first.order_id, today + pd.offsets.BDay(1), first.quantity, first.price, 150, "nisa")
    queue = audit.queue_status(store, rl.load_transactions(store.transactions)).set_index("order_id")
    assert queue.loc[first.order_id, "status"] == "FILLED"

    events = [e["event"] for e in audit.read_events(store)]
    assert events.count("APPROVED") == 1 and events[0] == "PROPOSED"
    record = audit.proposals(store)[proposal.decision_id]
    for key in ("decision_id", "timestamp", "data_version", "strategy_version", "git_commit", "signal",
                "target_weights", "orders", "risk"):
        assert key in record


def test_stop_proposals_cannot_be_approved(tmp_path):
    store = LiveStore(tmp_path).ensure()
    settings = make_settings()
    m = make_market()
    bridge.record_cash(store, "DEPOSIT", m.as_of, 300_000, "nisa")
    stale, _ = bridge.propose(store, settings, m, m.as_of + pd.offsets.BDay(10))
    assert stale.status == "STOP" and not stale.orders
    with pytest.raises(audit.AuditError, match="STOP"):
        audit.decide(store, stale.decision_id, approve=True)
    assert store.errors.exists()

    rc.set_emergency_stop(store, "test")
    stopped, _ = bridge.propose(store, settings, m, m.as_of)
    assert stopped.status == "STOP" and "emergency_stop" in {c.name for c in stopped.risk.failures}
    rc.clear_emergency_stop(store)
    fine, _ = bridge.propose(store, settings, m, m.as_of)
    assert fine.status == "PASS"
    rc.set_emergency_stop(store, "set after the proposal")
    with pytest.raises(audit.AuditError, match="emergency stop"):
        audit.decide(store, fine.decision_id, approve=True)
    assert audit.load_queue(store).empty


def test_ml_signal_uses_only_known_labels():
    m = make_market(n=700)
    spec = sp.strategy_spec(make_settings(signal={"strategy": "ml"}))
    target, info = sp.target_from_ml(m.adjusted, m.volume, spec)
    assert list(target) in (["VOO"], ["BIL"]) and 0 <= info["probability_up"] <= 1
    horizon = spec["target"]["horizon_days"]
    assert pd.Timestamp(info["last_known_label"]) == m.adjusted.index[-1 - horizon]


# --- 9.3 paper, 9.8 execution, 9.9 attribution ---------------------------------------------------------

def test_paper_trades_fill_on_the_next_close_with_costs(tmp_path):
    store = LiveStore(tmp_path).ensure()
    settings = make_settings()
    full = make_market()
    signal_day = full.close.index[-5]
    bridge.record_cash(store, "DEPOSIT", signal_day, 300_000, "nisa")
    proposal, _ = bridge.propose(store, settings, demo.truncate(full, signal_day), signal_day)
    assert proposal.paper_orders
    log = paper.load(store)
    assert set(log.loc[log.side != "hold", "status"]) == {"PENDING"}
    costs = fxt.CostSettings.from_config(settings)
    assert paper.fill_pending(store, full, costs) == len(proposal.paper_orders)
    log = paper.load(store).set_index("order_id")
    next_day = full.close.index[-4]
    o = proposal.paper_orders[0]
    assert log.loc[o.order_id, "fill_date"] == str(next_day.date())
    expected = fxt.estimate_trade(o.side, o.quantity, full.close.loc[next_day, o.asset], full.fx.loc[next_day], costs)
    assert log.loc[o.order_id, "fill_price"] == pytest.approx(expected["fill_price"])
    paper_state = rl.rebuild(rl.load_transactions(store.paper_transactions))
    assert paper_state.totals["deposits"] == 300_000   # mirrors the real deposit


def test_execution_cost_formula():
    m = make_market(n=10)
    day = m.close.index[3]
    close, mid = float(m.close.loc[day, "VOO"]), float(m.fx.loc[day])
    fills = tx(deposit(m.close.index[0], 1_000_000),
               trade(day, "BUY", "VOO", 2, close * 1.01, mid + 0.25, fees=300, order_id="o1"))
    table = ea.execution_costs(fills, m, {"o1": str(m.close.index[1].date())})
    row = table.iloc[0]
    assert row["price_cost_jpy"] == pytest.approx(close * 0.01 * 2 * mid)
    assert row["fx_cost_jpy"] == pytest.approx(2 * close * 1.01 * 0.25)
    assert row["execution_cost_jpy"] == pytest.approx(row["price_cost_jpy"] + 300 + row["fx_cost_jpy"])
    assert row["delay_days"] == 1          # the backtest would have filled on index[2]
    assert row["execution_effect_jpy"] < 0


def test_attribution_adds_up_to_the_profit_exactly():
    m = make_market(n=80, fx_vol=0.006)
    d = m.close.index
    saturday = d[30] + pd.Timedelta(days=(5 - d[30].dayofweek) % 7 or 7)
    history = tx(deposit(d[0], 500_000),
                 trade(d[1], "BUY", "VOO", 10, m.close.loc[d[1], "VOO"] * 1.002, m.fx.loc[d[1]] + 0.25, fees=100),
                 trade(d[2], "BUY", "BND", 8, m.close.loc[d[2], "BND"], m.fx.loc[d[2]] + 0.25),
                 {"date": d[20], "action": "DIVIDEND", "asset": "VOO", "amount_jpy": 3_000, "tax_jpy": 300},
                 trade(d[25], "SELL", "VOO", 4, m.close.loc[d[25], "VOO"] * 0.999, m.fx.loc[d[25]] - 0.25, fees=50,
                       tax=200),
                 deposit(saturday, 100_000),                                     # counts on the next market day
                 trade(d[40], "BUY", "EWJ", 5, m.close.loc[d[40], "EWJ"], m.fx.loc[d[40]] + 0.25),
                 {"date": d[50], "action": "FEE", "amount_jpy": 110})
    targets = {d[0]: {"VOO": 0.6, "BND": 0.4}, d[35]: {"VOO": 0.4, "BND": 0.3, "EWJ": 0.3}}
    daily = at.daily_attribution(history, m, targets)
    assert at.check_identity(daily)
    total = at.totals(daily)
    state = rl.rebuild(history)
    final_value = state.cash_jpy + sum(q * m.close[a].iloc[-1] * m.fx.iloc[-1]
                                       for a, q in state.positions["quantity"].items())
    assert total["profit"] == pytest.approx(final_value - 600_000)
    assert abs(total["unexplained"]) < 1e-6
    assert total["fx"] != 0 and total["execution"] < 0 and total["dividend"] == 3_000
    assert total["fee_tax"] == pytest.approx(-(100 + 50 + 200 + 300 + 110))


def test_comparison_metrics_use_common_dates():
    idx = pd.bdate_range("2025-01-01", periods=300)
    a = pd.Series(0.001, index=idx)
    b = pd.Series(0.0005, index=idx[50:])
    table = ea.comparison({"a": a, "b": b}, {"a": 0.001})
    assert (table["start"] == idx[50].date()).all()
    assert table.loc["a", "total_return"] == pytest.approx(1.001 ** 250 - 1)


# --- settings, privacy, demo, dashboard ----------------------------------------------------------------

def test_personal_settings_override_defaults(tmp_path):
    local = tmp_path / "live.local.toml"
    local.write_text('[limits]\nmax_order_jpy = 12345\n[signal]\nstrategy = "ml"\n', encoding="utf-8")
    settings = load_settings(local_path=local)
    assert settings["limits"]["max_order_jpy"] == 12345 and settings["limits"]["max_orders_per_day"] == 3
    assert settings["signal"]["strategy"] == "ml" and settings["signal"]["mode"] == "buy_only"


def test_real_money_files_are_never_committed():
    paths = ["data/live/transactions.csv", "data/live/audit_logs/decisions.jsonl", "data/live/market/close.csv",
             "config/live.local.toml"]
    try:
        result = subprocess.run(["git", "check-ignore", *paths], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        pytest.skip("git is not available")
    assert sorted(result.stdout.split()) == sorted(paths)


def test_demo_runs_the_whole_workflow(tmp_path):
    m = demo.research_market()
    settings = demo.demo_settings(load_settings(local_path=None))
    store = demo.build(tmp_path / "demo", settings, m, start="2026-04-01", reject_every=3)
    history = rl.load_transactions(store.transactions)
    state = rl.rebuild(history)
    assert state.cash_jpy >= 0 and len(state.positions)
    decisions = audit.decision_table(store)
    assert {"APPROVED", "REJECTED"} <= set(decisions["decision"])
    assert at.check_identity(at.daily_attribution(history, m, audit.approved_targets(store)))
    with pytest.raises(ValueError, match="not empty"):
        demo.build(tmp_path / "demo", settings, m)


LIVE_PAGES = ["tab_live_portfolio", "tab_approval", "tab_execution_diff"]


def live_app() -> AppTest:
    at_ = AppTest.from_file("../app.py", default_timeout=900)
    at_.session_state["_live_mode"] = "デモ（合成データ）"   # never the real data/live in tests
    return at_


@pytest.mark.parametrize("page", LIVE_PAGES)
def test_live_pages_render_in_demo_mode(page):
    app = live_app().run()
    app.switch_page(f"src/dashboard/{page}.py").run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_approval_page_blocks_approval_during_emergency_stop():
    app = live_app().run()
    app.switch_page("src/dashboard/tab_approval.py").run()
    next(b for b in app.button if b.label == "緊急停止").click().run()
    assert any("緊急停止中" in e.value for e in app.error)
    approve = [b for b in app.button if b.label.startswith("承認")]
    assert approve and all(b.disabled for b in approve)
