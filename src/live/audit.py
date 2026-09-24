"""9.6 / 9.9 Decision audit log and order queue.

audit_logs/decisions.jsonl is append-only, one JSON object per line:
  PROPOSED            the whole proposal: decision_id, timestamp, data_version,
                      strategy_version, git_commit, signal, target weights, orders, risk checks
  APPROVED / REJECTED a person's decision on that decision_id, with a note
Lines are never edited, so every past decision can be traced to the data, settings and
code that produced it. A decision is made once; a proposal whose risk check said STOP
cannot be approved; nothing can be approved while the emergency stop is set.

order_queue.csv lists approved orders for the person to place at the broker. An order
counts as filled once the ledger has a BUY or SELL row with its order_id.
"""
from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime

import pandas as pd

import risk_controller as rc
from settings import LiveStore

APPROVED, REJECTED, PROPOSED = "APPROVED", "REJECTED", "PROPOSED"
QUEUE_COLUMNS = ["order_id", "decision_id", "approved_at", "date", "asset", "side", "quantity", "est_price_usd",
                 "est_fx", "est_value_jpy"]
ORDER_FIELDS = [f.name for f in fields(rc.Order)]


class AuditError(ValueError):
    pass


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_events(store: LiveStore) -> list[dict]:
    if not store.decisions.exists():
        return []
    with open(store.decisions, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_event(store: LiveStore, event: dict) -> None:
    store.ensure()
    with open(store.decisions, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def proposals(store: LiveStore) -> dict[str, dict]:
    return {e["decision_id"]: e for e in read_events(store) if e["event"] == PROPOSED}


def final_decisions(store: LiveStore) -> dict[str, dict]:
    return {e["decision_id"]: e for e in read_events(store) if e["event"] in (APPROVED, REJECTED)}


def record_proposal(store: LiveStore, proposal) -> bool:
    """Write a proposal once; False if the same decision_id was already proposed (nothing written)."""
    if proposal.decision_id in proposals(store):
        return False
    append_event(store, {"event": PROPOSED, "timestamp": now_iso(), **proposal.to_record()})
    rc.log_stop(store, {"decision_id": proposal.decision_id, "as_of": proposal.as_of,
                        "strategy": proposal.strategy}, proposal.risk)
    return True


def orders_of(record: dict, key: str = "orders") -> list[rc.Order]:
    return [rc.Order(**{k: o[k] for k in ORDER_FIELDS}) for o in record.get(key, [])]


def load_queue(store: LiveStore) -> pd.DataFrame:
    if not store.order_queue.exists():
        return pd.DataFrame(columns=QUEUE_COLUMNS)
    return pd.read_csv(store.order_queue, dtype={"order_id": str, "decision_id": str})


def decide(store: LiveStore, decision_id: str, approve: bool, note: str = "") -> dict:
    """A person's decision. Approval re-checks the emergency stop and duplicate orders."""
    record = proposals(store).get(decision_id)
    if record is None:
        raise AuditError(f"no proposal {decision_id}")
    if decision_id in final_decisions(store):
        raise AuditError(f"{decision_id} was already decided: {final_decisions(store)[decision_id]['event']}")
    orders = orders_of(record)
    if approve:
        if record["risk"]["status"] != "PASS":
            raise AuditError(f"{decision_id}: the risk check said STOP, so it cannot be approved")
        if rc.emergency_stop_active(store):
            raise AuditError("the emergency stop is set; nothing can be approved")
        queued = set(load_queue(store)["order_id"])
        repeated = [o.order_id for o in orders if o.order_id in queued]
        if repeated:
            raise AuditError(f"orders already in the queue: {repeated}")
    event = {"event": APPROVED if approve else REJECTED, "decision_id": decision_id, "timestamp": now_iso(),
             "note": note}
    append_event(store, event)
    if approve and orders:
        rows = pd.DataFrame([{"order_id": o.order_id, "decision_id": decision_id, "approved_at": event["timestamp"],
                              "date": o.date, "asset": o.asset, "side": o.side, "quantity": o.quantity,
                              "est_price_usd": o.price, "est_fx": o.fx_rate, "est_value_jpy": o.value_jpy}
                             for o in orders], columns=QUEUE_COLUMNS)
        store.ensure()
        pd.concat([load_queue(store), rows], ignore_index=True).to_csv(store.order_queue, index=False)
    return event


BLOCKED, PENDING = "BLOCKED", "PENDING"


def decision_table(store: LiveStore) -> pd.DataFrame:
    """One row per proposal with its final decision: PENDING until a person decides,
    BLOCKED if the risk check stopped it (nothing to decide)."""
    decided = final_decisions(store)
    rows = []
    for decision_id, p in proposals(store).items():
        d = decided.get(decision_id, {})
        undecided = PENDING if p["risk"]["status"] == "PASS" else BLOCKED
        rows.append({"decision_id": decision_id, "proposed_at": p["timestamp"], "as_of": p["as_of"],
                     "strategy": p["strategy"], "risk": p["risk"]["status"], "decision": d.get("event", undecided),
                     "decided_at": d.get("timestamp", ""), "note": d.get("note", ""), "orders": len(p["orders"]),
                     "target_weights": json.dumps(p["target_weights"]), "data_version": p["data_version"],
                     "strategy_version": p["strategy_version"], "git_commit": p["git_commit"]})
    return pd.DataFrame(rows)


def queue_status(store: LiveStore, transactions: pd.DataFrame) -> pd.DataFrame:
    queue = load_queue(store)
    filled = set(transactions.loc[transactions["action"].isin(["BUY", "SELL"]), "order_id"])
    queue["status"] = ["FILLED" if o in filled else "OPEN" for o in queue["order_id"]]
    return queue


def open_orders(store: LiveStore, transactions: pd.DataFrame) -> list[rc.Order]:
    queue = queue_status(store, transactions)
    return [rc.Order(order_id=r.order_id, decision_id=r.decision_id, date=r.date, asset=r.asset, side=r.side,
                     quantity=float(r.quantity), price=float(r.est_price_usd), fx_rate=float(r.est_fx))
            for r in queue[queue.status == "OPEN"].itertuples()]


def known_order_ids(store: LiveStore, transactions: pd.DataFrame) -> set[str]:
    return set(load_queue(store)["order_id"]) | set(transactions["order_id"][transactions["order_id"] != ""])


def orders_approved_on(store: LiveStore, day: pd.Timestamp) -> int:
    queue = load_queue(store)
    return int((pd.to_datetime(queue["approved_at"]).dt.normalize() == pd.Timestamp(day).normalize()).sum())


def approved_targets(store: LiveStore) -> dict[pd.Timestamp, dict]:
    """Target weights of every approved decision, by the date of the data it used."""
    decided = final_decisions(store)
    return {pd.Timestamp(p["as_of"]): p["target_weights"] for i, p in proposals(store).items()
            if decided.get(i, {}).get("event") == APPROVED}
