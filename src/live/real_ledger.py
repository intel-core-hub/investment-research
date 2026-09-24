"""9.2 Real money ledger: the transaction history is the single source of truth.

transactions.csv, one row per event in the order it happened:
  transaction_id  unique id (T00001, ...), assigned when a row is appended
  date            YYYY-MM-DD, as shown by the broker
  action          DEPOSIT, WITHDRAW, BUY, SELL, DIVIDEND, FEE, TAX
  asset           ticker (BUY, SELL, DIVIDEND); every asset is a USD-listed ETF
  quantity        shares (BUY, SELL)
  price           execution price in USD (BUY, SELL)
  fx_rate         JPY per USD actually applied by the broker (BUY, SELL, DIVIDEND)
  fees_jpy        commission and other trade costs in JPY (BUY, SELL)
  tax_jpy         tax withheld (SELL in a taxable account, DIVIDEND)
  amount_jpy      cash amount (DEPOSIT, WITHDRAW, FEE, TAX; the gross dividend for DIVIDEND)
  account         nisa / tokutei / paper
  broker, order_id, note   free text; order_id links a fill to an approved order

Cash is held in JPY (yen settlement: the broker converts at fx_rate). Cash changes by
  DEPOSIT +amount   WITHDRAW -amount   FEE -amount   TAX -amount   DIVIDEND +(amount - tax)
  BUY  -(quantity x price x fx_rate + fees + tax)
  SELL +(quantity x price x fx_rate - fees - tax)
Cost basis is the moving average in JPY including fees (as Japanese brokers compute it).
A realized gain is the sale proceeds after fees minus the average cost of the shares sold.

Holdings, cash, cost basis, realized and unrealized P&L are never edited by hand:
`rebuild` recalculates them from the transactions, and refuses a history that is
inconsistent (selling shares not held, cash going negative, duplicated rows).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ACTIONS = ("DEPOSIT", "WITHDRAW", "BUY", "SELL", "DIVIDEND", "FEE", "TAX")
TRADES = ("BUY", "SELL")
CASH_ONLY = ("DEPOSIT", "WITHDRAW", "FEE", "TAX")
COLUMNS = ["transaction_id", "date", "action", "asset", "quantity", "price", "fx_rate", "fees_jpy", "tax_jpy",
           "amount_jpy", "account", "broker", "order_id", "note"]
NUMERIC = ["quantity", "price", "fx_rate", "fees_jpy", "tax_jpy", "amount_jpy"]
TEXT = ["transaction_id", "action", "asset", "account", "broker", "order_id", "note"]
TOLERANCE = 1e-9


class LedgerError(ValueError):
    pass


def empty() -> pd.DataFrame:
    return normalize(pd.DataFrame(columns=COLUMNS))


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Every column present with its type; blank numbers become 0 (fx_rate 1), blank text ''."""
    out = frame.copy()
    for col in COLUMNS:
        if col not in out:
            out[col] = np.nan
    out = out[COLUMNS]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[["quantity", "price", "fees_jpy", "tax_jpy", "amount_jpy"]] = \
        out[["quantity", "price", "fees_jpy", "tax_jpy", "amount_jpy"]].fillna(0.0)
    out["fx_rate"] = out["fx_rate"].fillna(1.0)
    for col in TEXT:
        out[col] = out[col].fillna("").astype(str).str.strip()
    out["action"] = out["action"].str.upper()
    out["asset"] = out["asset"].str.upper()
    return out.reset_index(drop=True)


def validate_rows(frame: pd.DataFrame) -> list[str]:
    """Problems visible in single rows or duplicates, before any balance is computed."""
    problems = []
    for i, r in frame.iterrows():
        where = f"row {i + 1} ({r.transaction_id or 'new'}, {r.action or '?'})"
        if pd.isna(r.date):
            problems.append(f"{where}: date is missing or not a date")
        if r.action not in ACTIONS:
            problems.append(f"{where}: unknown action; use one of {ACTIONS}")
            continue
        if min(r.fees_jpy, r.tax_jpy, r.amount_jpy) < 0:
            problems.append(f"{where}: fees, tax and amount must not be negative")
        if r.action in TRADES:
            if not r.asset:
                problems.append(f"{where}: asset is required")
            if not r.quantity > 0 or not r.price > 0 or not r.fx_rate > 0:
                problems.append(f"{where}: quantity, price and fx_rate must be positive")
        elif r.action == "DIVIDEND":
            if not r.asset:
                problems.append(f"{where}: asset is required")
            if r.tax_jpy > r.amount_jpy + TOLERANCE:
                problems.append(f"{where}: tax is larger than the dividend")
        elif not r.amount_jpy > 0:
            problems.append(f"{where}: amount_jpy must be positive")
    ids = frame["transaction_id"][frame["transaction_id"] != ""]
    for tid in ids[ids.duplicated()].unique():
        problems.append(f"transaction_id {tid} is used more than once")
    # a legitimate repeat (two equal deposits on one day) can be told apart with a note
    key = ["date", "action", "asset", "quantity", "price", "amount_jpy", "order_id", "note"]
    for i in frame.index[frame.duplicated(subset=key)]:
        problems.append(f"row {i + 1} repeats an earlier row exactly (entered twice?)")
    return problems


def cash_change(r) -> float:
    if r.action == "DEPOSIT":
        return r.amount_jpy
    if r.action in ("WITHDRAW", "FEE", "TAX"):
        return -r.amount_jpy
    if r.action == "DIVIDEND":
        return r.amount_jpy - r.tax_jpy
    value = r.quantity * r.price * r.fx_rate
    return -(value + r.fees_jpy + r.tax_jpy) if r.action == "BUY" else value - r.fees_jpy - r.tax_jpy


@dataclass(frozen=True)
class LedgerState:
    positions: pd.DataFrame       # index asset: quantity, cost_basis_jpy, average_cost_jpy (open positions only)
    cash_jpy: float
    realized: pd.DataFrame        # one row per sale: date, asset, account, quantity, proceeds_jpy, cost_jpy, gain_jpy
    totals: dict                  # deposits, withdrawals, dividends (gross), fees, taxes
    history: pd.DataFrame         # per date after its transactions: cash_jpy, <asset> quantity, cost:<asset>
    flows: pd.Series              # per date: deposits - withdrawals

    @property
    def realized_pnl_jpy(self) -> float:
        return float(self.realized["gain_jpy"].sum()) if len(self.realized) else 0.0


def rebuild(transactions: pd.DataFrame, allow_negative_cash: bool = False) -> LedgerState:
    """Recalculate holdings, cash, cost basis and realized P&L from the transactions."""
    frame = normalize(transactions)
    problems = validate_rows(frame)
    if problems:
        raise LedgerError("invalid transactions:\n  - " + "\n  - ".join(problems))
    frame = frame.sort_values("date", kind="stable")  # same-day rows keep the order they were entered

    qty: dict[str, float] = {}
    cost: dict[str, float] = {}
    cash = 0.0
    totals = dict.fromkeys(["deposits", "withdrawals", "dividends", "fees", "taxes"], 0.0)
    realized, snapshots, flows = [], {}, {}
    for r in frame.itertuples():
        where = f"{r.date.date()} {r.action} {r.asset} ({r.transaction_id or 'new'})".replace("  ", " ")
        if r.action == "BUY":
            qty[r.asset] = qty.get(r.asset, 0.0) + r.quantity
            cost[r.asset] = cost.get(r.asset, 0.0) + r.quantity * r.price * r.fx_rate + r.fees_jpy
        elif r.action == "SELL":
            held = qty.get(r.asset, 0.0)
            if r.quantity > held + TOLERANCE:
                raise LedgerError(f"{where}: sells {r.quantity:g} shares but only {held:g} are held")
            average = cost[r.asset] / held
            proceeds = r.quantity * r.price * r.fx_rate - r.fees_jpy
            realized.append({"date": r.date, "asset": r.asset, "account": r.account, "quantity": r.quantity,
                             "proceeds_jpy": proceeds, "cost_jpy": average * r.quantity,
                             "gain_jpy": proceeds - average * r.quantity})
            qty[r.asset] = held - r.quantity
            cost[r.asset] -= average * r.quantity
            if qty[r.asset] <= TOLERANCE:
                qty[r.asset], cost[r.asset] = 0.0, 0.0
        cash += cash_change(r)
        if cash < -TOLERANCE and not allow_negative_cash:
            raise LedgerError(f"{where}: cash would become {cash:,.0f} JPY (negative)")

        totals["deposits"] += r.amount_jpy if r.action == "DEPOSIT" else 0.0
        totals["withdrawals"] += r.amount_jpy if r.action == "WITHDRAW" else 0.0
        totals["dividends"] += r.amount_jpy if r.action == "DIVIDEND" else 0.0
        totals["fees"] += r.fees_jpy + (r.amount_jpy if r.action == "FEE" else 0.0)
        totals["taxes"] += r.tax_jpy + (r.amount_jpy if r.action == "TAX" else 0.0)
        flow = r.amount_jpy if r.action == "DEPOSIT" else (-r.amount_jpy if r.action == "WITHDRAW" else 0.0)
        flows[r.date] = flows.get(r.date, 0.0) + flow
        snapshots[r.date] = {"cash_jpy": cash, **{a: q for a, q in qty.items()},
                             **{f"cost:{a}": c for a, c in cost.items()}}

    held = [a for a, q in qty.items() if q > TOLERANCE]
    positions = pd.DataFrame({"quantity": [qty[a] for a in held], "cost_basis_jpy": [cost[a] for a in held]},
                             index=pd.Index(held, name="asset"))
    positions["average_cost_jpy"] = positions["cost_basis_jpy"] / positions["quantity"]
    history = pd.DataFrame.from_dict(snapshots, orient="index").sort_index().fillna(0.0)
    history.index.name = "date"
    return LedgerState(positions=positions, cash_jpy=cash,
                       realized=pd.DataFrame(realized, columns=["date", "asset", "account", "quantity",
                                                                 "proceeds_jpy", "cost_jpy", "gain_jpy"]),
                       totals=totals, history=history, flows=pd.Series(flows, dtype=float).sort_index())


# --- valuation ------------------------------------------------------------------------------

def holdings_table(state: LedgerState, prices: pd.Series, fx_rate: float) -> pd.DataFrame:
    """Open positions at the latest prices (USD) and FX rate: value, unrealized P&L and weight of total."""
    table = state.positions.copy()
    missing = [a for a in table.index if a not in prices or not prices[a] > 0]
    if missing:
        raise LedgerError(f"no valid price for {missing}")
    table["price"] = [float(prices[a]) for a in table.index]
    table["fx_rate"] = fx_rate
    table["value_jpy"] = table["quantity"] * table["price"] * fx_rate
    table["unrealized_pnl_jpy"] = table["value_jpy"] - table["cost_basis_jpy"]
    table["unrealized_return"] = table["unrealized_pnl_jpy"] / table["cost_basis_jpy"]
    total = table["value_jpy"].sum() + state.cash_jpy
    table["weight"] = table["value_jpy"] / total if total > 0 else np.nan
    return table


def summary(state: LedgerState, prices: pd.Series, fx_rate: float) -> dict:
    table = holdings_table(state, prices, fx_rate)
    invested = float(table["value_jpy"].sum())
    total = invested + state.cash_jpy
    net_deposits = state.totals["deposits"] - state.totals["withdrawals"]
    return {"total_value_jpy": total, "invested_value_jpy": invested, "cash_jpy": state.cash_jpy,
            "cost_basis_jpy": float(table["cost_basis_jpy"].sum()),
            "unrealized_pnl_jpy": float(table["unrealized_pnl_jpy"].sum()),
            "realized_pnl_jpy": state.realized_pnl_jpy, "net_deposits_jpy": net_deposits,
            "total_pnl_jpy": total - net_deposits, **{f"total_{k}_jpy": v for k, v in state.totals.items()}}


def valuation_history(state: LedgerState, prices: pd.DataFrame, fx: pd.Series) -> pd.DataFrame:
    """Daily value on market dates from the first transaction, with time-weighted returns.

    Holdings and cash are those after each date's transactions; a transaction on a
    non-trading day counts on the next market date. The daily time-weighted return is
    (V_t - flow_t) / V_{t-1} - 1, so deposits are not counted as gains.
    """
    if state.history.empty:
        return pd.DataFrame(columns=["cash_jpy", "invested_jpy", "total_jpy", "flow_jpy", "net_deposits_jpy",
                                     "daily_return", "growth"])
    first = state.history.index[0]
    dates = prices.index[prices.index >= first]
    fx_on = fx.sort_index().reindex(dates, method="ffill")
    held = state.history.reindex(state.history.index.union(dates)).ffill().loc[dates]
    assets = [c for c in state.history.columns if c != "cash_jpy" and not c.startswith("cost:")]
    invested = sum((held[a] * prices.loc[dates, a] * fx_on for a in assets), pd.Series(0.0, index=dates))
    net = state.flows.cumsum().reindex(state.flows.index.union(dates)).ffill().loc[dates].fillna(0.0)
    out = pd.DataFrame({"cash_jpy": held["cash_jpy"], "invested_jpy": invested}, index=dates)
    out["total_jpy"] = out["cash_jpy"] + out["invested_jpy"]
    out["net_deposits_jpy"] = net
    # deposits made before the first market date are part of that day's flow
    out["flow_jpy"] = net.diff().fillna(net.iloc[0])
    prev = out["total_jpy"].shift(1)
    out["daily_return"] = ((out["total_jpy"] - out["flow_jpy"]) / prev - 1).where(prev > 0)
    out["growth"] = (1 + out["daily_return"].fillna(0)).cumprod()
    out.index.name = "date"
    return out


# --- files ----------------------------------------------------------------------------------

def load_transactions(path: Path) -> pd.DataFrame:
    return normalize(pd.read_csv(path, dtype=str)) if Path(path).exists() else empty()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write through a temporary file so a crash never leaves a half-written ledger."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.10g")
    os.replace(tmp, path)


def save_transactions(frame: pd.DataFrame, path: Path) -> None:
    out = normalize(frame)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    _write_csv(out, path)


def next_transaction_number(frame: pd.DataFrame) -> int:
    numbers = frame["transaction_id"].str.extract(r"^T(\d+)$")[0].dropna().astype(int)
    return int(numbers.max()) + 1 if len(numbers) else 1


def append_transactions(path: Path, rows: list[dict]) -> pd.DataFrame:
    """Add rows (ids assigned) only if the whole history stays consistent; returns the new history."""
    current = load_transactions(path)
    new = normalize(pd.DataFrame(rows))
    first = next_transaction_number(current)
    new["transaction_id"] = [f"T{first + i:05d}" for i in range(len(new))]
    combined = pd.concat([current, new], ignore_index=True)
    rebuild(combined)  # raises LedgerError and writes nothing if the result is inconsistent
    save_transactions(combined, path)
    return combined


def save_derived(state: LedgerState, valuations: pd.DataFrame, holdings_path: Path, cash_path: Path,
                 valuations_path: Path) -> None:
    """holdings.csv (date, asset, quantity, cost basis), cash.csv and valuations.csv, all derived."""
    assets = [c for c in state.history.columns if c != "cash_jpy" and not c.startswith("cost:")]
    rows = [{"date": d.date(), "asset": a, "quantity": h[a], "cost_basis_jpy": h.get(f"cost:{a}", 0.0)}
            for d, h in state.history.iterrows() for a in assets]
    _write_csv(pd.DataFrame(rows, columns=["date", "asset", "quantity", "cost_basis_jpy"]), holdings_path)
    _write_csv(state.history[["cash_jpy"]].reset_index().assign(date=lambda f: f["date"].dt.date), cash_path)
    _write_csv(valuations.reset_index().assign(date=lambda f: pd.to_datetime(f["date"]).dt.date), valuations_path)
