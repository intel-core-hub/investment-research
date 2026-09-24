"""The live bridge workflow, shared by run_live.py, the dashboard and the demo.

    market data -> propose (pipeline, audit log, shadow trade)
    -> a person approves or rejects (the only way an order reaches the queue)
    -> the person places the order at the broker -> record the fill in the ledger
    -> recalculate holdings, cash and valuations (real and paper)

No function here sends an order anywhere.
"""
from __future__ import annotations

import pandas as pd

import audit
import fx_and_taxes as fxt
import paper_engine as paper
import real_ledger as rl
import risk_controller as rc
import signal_pipeline as sp
from market import MarketData
from settings import LiveStore


class BridgeError(ValueError):
    pass


def load_states(store: LiveStore):
    """(real transactions, real state, paper transactions, paper state)."""
    tx = rl.load_transactions(store.transactions)
    paper_tx = rl.load_transactions(store.paper_transactions)
    return tx, rl.rebuild(tx), paper_tx, rl.rebuild(paper_tx)


def propose(store: LiveStore, settings: dict, market: MarketData, today: pd.Timestamp | None = None):
    """Run the pipeline and record the proposal once. Returns (proposal, newly_recorded)."""
    today = pd.Timestamp(today or pd.Timestamp.now().normalize())
    costs = fxt.CostSettings.from_config(settings)
    paper.sync_deposits(store)
    paper.fill_pending(store, market, costs)
    tx, state, _, paper_state = load_states(store)
    proposal = sp.run(store, settings, market, state, today, known_order_ids=audit.known_order_ids(store, tx),
                      open_orders=audit.open_orders(store, tx), orders_today=audit.orders_approved_on(store, today),
                      paper_state=paper_state)
    recorded = audit.record_proposal(store, proposal)
    if recorded and proposal.status == "PASS":
        paper.record_signals(store, proposal)
    return proposal, recorded


def record_fill(store: LiveStore, order_id: str, date, quantity: float, price: float, fx_rate: float,
                account: str, fees_jpy: float = 0.0, tax_jpy: float = 0.0, broker: str = "", note: str = ""):
    """Record the broker's fill of an approved, still open order."""
    tx = rl.load_transactions(store.transactions)
    queue = audit.queue_status(store, tx)
    match = queue[queue["order_id"] == order_id]
    if match.empty:
        raise BridgeError(f"order {order_id} is not in the order queue (only approved orders can be filled)")
    if match["status"].iloc[0] == "FILLED":
        raise BridgeError(f"order {order_id} is already filled")
    fxt.check_account(account)
    return rl.append_transactions(store.transactions, [{
        "date": pd.Timestamp(date).strftime("%Y-%m-%d"), "action": match["side"].iloc[0].upper(),
        "asset": match["asset"].iloc[0], "quantity": quantity, "price": price, "fx_rate": fx_rate,
        "fees_jpy": fees_jpy, "tax_jpy": tax_jpy, "account": account, "broker": broker, "order_id": order_id,
        "note": note}])


def record_cash(store: LiveStore, action: str, date, amount_jpy: float, account: str, asset: str = "",
                tax_jpy: float = 0.0, broker: str = "", note: str = ""):
    """DEPOSIT, WITHDRAW, FEE, TAX, or DIVIDEND (amount = gross, tax = withheld)."""
    if action not in ("DEPOSIT", "WITHDRAW", "FEE", "TAX", "DIVIDEND"):
        raise BridgeError(f"use record_fill for {action}")
    fxt.check_account(account)
    return rl.append_transactions(store.transactions, [{
        "date": pd.Timestamp(date).strftime("%Y-%m-%d"), "action": action, "asset": asset, "amount_jpy": amount_jpy,
        "tax_jpy": tax_jpy, "account": account, "broker": broker, "note": note}])


def recalculate(store: LiveStore, market: MarketData) -> dict:
    """Rebuild both ledgers from their transactions and rewrite holdings, cash and valuations."""
    tx, state, paper_tx, paper_state = load_states(store)
    valuations = rl.valuation_history(state, market.close, market.fx)
    rl.save_derived(state, valuations, store.holdings, store.cash, store.valuations)
    return {"transactions": tx, "state": state, "valuations": valuations, "paper_transactions": paper_tx,
            "paper_state": paper_state,
            "paper_valuations": rl.valuation_history(paper_state, market.close, market.fx)}


def status(store: LiveStore, market: MarketData | None, today: pd.Timestamp) -> dict:
    """What the dashboard shows as system status."""
    tx = rl.load_transactions(store.transactions)
    decisions = audit.decision_table(store)
    queue = audit.queue_status(store, tx)
    errors = []
    if store.errors.exists():
        errors = pd.read_json(store.errors, lines=True).to_dict("records")
    return {
        "market_as_of": market.as_of.date() if market is not None else None,
        "market_age_business_days": rc.business_days_between(market.as_of, today) if market is not None else None,
        "fx_as_of": market.fx.dropna().index[-1].date() if market is not None else None,
        "emergency_stop": rc.emergency_stop_active(store),
        "pending_decisions": int((decisions["decision"] == "PENDING").sum()) if len(decisions) else 0,
        "last_proposal": decisions.iloc[-1].to_dict() if len(decisions) else None,
        "open_orders": int((queue["status"] == "OPEN").sum()),
        "errors": errors,
    }
