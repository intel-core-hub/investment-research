"""9.3 Paper / shadow trading: what would happen if every signal were followed.

Every proposal that passes the risk check also goes to a paper account, without
approval (shadow trading). Paper orders fill at the close of the first trading day
after the signal (the Phase 5 backtest rule), with the configured slippage, FX
spread and commission. The paper account follows the same ledger rules
(paper_transactions.csv) and mirrors the real deposits and withdrawals, so two
gaps can be measured separately:
  backtest -> paper   timing, costs, whole shares, real data arriving day by day
  paper -> real       human decisions (rejections, delays) and real fills

paper_trading.csv, one row per asset per signal:
  order_id, decision_id, signal_date, asset, side (buy / sell / hold), quantity,
  target_weight, reference_price (close on the signal day), fill_date, fill_price,
  fx_rate, fees_jpy, status (PENDING / FILLED / NO_ORDER / REJECTED), detail
"""
from __future__ import annotations

import pandas as pd

import fx_and_taxes as fxt
import real_ledger as rl
from market import MarketData
from settings import LiveStore

COLUMNS = ["order_id", "decision_id", "signal_date", "asset", "side", "quantity", "target_weight", "reference_price",
           "fill_date", "fill_price", "fx_rate", "fees_jpy", "status", "detail"]
TEXT = ["order_id", "decision_id", "signal_date", "asset", "side", "fill_date", "status", "detail"]
MIRROR_PREFIX = "mirror of "


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    """Text columns as objects, so dates and notes can be written into columns that start empty."""
    frame = frame.reindex(columns=COLUMNS)
    return frame.astype({c: object for c in TEXT}).astype({c: float for c in COLUMNS if c not in TEXT})


def load(store: LiveStore) -> pd.DataFrame:
    if not store.paper_trading.exists():
        return _typed(pd.DataFrame(columns=COLUMNS))
    return _typed(pd.read_csv(store.paper_trading, dtype={c: str for c in TEXT}, keep_default_na=False,
                              na_values={c: [""] for c in COLUMNS if c not in TEXT}))


def _save(store: LiveStore, frame: pd.DataFrame) -> None:
    store.ensure()
    frame[COLUMNS].to_csv(store.paper_trading, index=False)


def record_signals(store: LiveStore, proposal) -> int:
    """Log the target of every asset for this signal; orders start PENDING."""
    log = load(store)
    if proposal.decision_id in set(log["decision_id"]):
        return 0
    by_asset = {o.asset: o for o in proposal.paper_orders}
    rows = []
    for asset in sorted(set(proposal.target_weights) | set(by_asset)):
        o = by_asset.get(asset)
        rows.append({"order_id": o.order_id if o else "", "decision_id": proposal.decision_id,
                     "signal_date": proposal.as_of, "asset": asset, "side": o.side if o else "hold",
                     "quantity": o.quantity if o else 0.0, "target_weight": proposal.target_weights.get(asset, 0.0),
                     "reference_price": o.price if o else None, "status": "PENDING" if o else "NO_ORDER",
                     "detail": ""})
    _save(store, pd.concat([log, _typed(pd.DataFrame(rows, columns=COLUMNS))], ignore_index=True))
    return len(rows)


def sync_deposits(store: LiveStore) -> int:
    """Copy real deposits and withdrawals the paper account does not have yet."""
    real = rl.load_transactions(store.transactions)
    paper = rl.load_transactions(store.paper_transactions)
    copied = set(paper["note"].str.removeprefix(MIRROR_PREFIX))
    rows = [{"date": r.date.strftime("%Y-%m-%d"), "action": r.action, "amount_jpy": r.amount_jpy, "account": "paper",
             "note": MIRROR_PREFIX + r.transaction_id}
            for r in real[real["action"].isin(["DEPOSIT", "WITHDRAW"])].itertuples()
            if r.transaction_id not in copied]
    if rows:
        rl.append_transactions(store.paper_transactions, rows)
    return len(rows)


def fill_pending(store: LiveStore, market: MarketData, costs: fxt.CostSettings) -> int:
    """Fill PENDING orders at the close of the next trading day after the signal, if it exists yet."""
    log = load(store)
    pending = log[log["status"] == "PENDING"]
    filled = 0
    # sells before buys on the same day, as in the Phase 5 engine
    for i in pending.sort_values(["signal_date", "side"], ascending=[True, False]).index:
        row = log.loc[i]
        later = market.close.index[market.close.index > pd.Timestamp(row.signal_date)]
        if len(later) == 0:
            continue
        day = later[0]
        fx_mid = float(market.fx_on(pd.DatetimeIndex([day])).iloc[0])
        trade = fxt.estimate_trade(row.side, float(row.quantity), float(market.close.loc[day, row.asset]), fx_mid,
                                   costs)
        try:
            rl.append_transactions(store.paper_transactions, [{
                "date": day.strftime("%Y-%m-%d"), "action": row.side.upper(), "asset": row.asset,
                "quantity": row.quantity, "price": trade["fill_price"], "fx_rate": trade["fx_rate"],
                "fees_jpy": trade["commission_jpy"], "account": "paper", "order_id": row.order_id}])
        except rl.LedgerError as e:
            log.loc[i, ["status", "detail"]] = ["REJECTED", str(e).splitlines()[-1]]
            continue
        log.loc[i, ["fill_date", "fill_price", "fx_rate", "fees_jpy", "status"]] = \
            [day.strftime("%Y-%m-%d"), trade["fill_price"], trade["fx_rate"], trade["commission_jpy"], "FILLED"]
        filled += 1
    _save(store, log)
    return filled
