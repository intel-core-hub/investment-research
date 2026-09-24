"""9.5 Risk controller and fail-safe: every proposal passes here before anyone can approve it.

A single failed check means STOP: the proposal cannot be approved and the reasons
are written to audit_logs/errors.jsonl. Checks, in order:
  switch   the emergency stop is not set (a file in the live data folder)
  data     the latest prices of every needed asset and the FX rate exist, are > 0,
           and are no older than `max_data_age_days` business days
  orders   positive quantities and known assets; no duplicate order (within the
           proposal, in the order queue or in the ledger); orders per day and amount
           per order within limits; sells within holdings; enough cash for the buys
           (after sells, commission and FX spread); no asset above `max_weight` afterwards
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd

import fx_and_taxes as fxt
from real_ledger import LedgerState
from settings import LiveStore

TOLERANCE = 1e-9


@dataclass(frozen=True)
class RiskLimits:
    max_orders_per_day: int
    max_order_jpy: float
    max_weight: float
    max_data_age_days: int

    @classmethod
    def from_config(cls, settings: dict) -> "RiskLimits":
        return cls(**settings["limits"])


@dataclass(frozen=True)
class Order:
    order_id: str
    decision_id: str
    date: str                 # signal date (YYYY-MM-DD)
    asset: str
    side: str                 # "buy" or "sell"
    quantity: float
    price: float              # latest close in USD
    fx_rate: float            # latest USD/JPY mid

    @property
    def value_jpy(self) -> float:
        return self.quantity * self.price * self.fx_rate


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def __post_init__(self):
        # comparisons on numpy values give numpy.bool_, which the JSON audit log would store as a string
        object.__setattr__(self, "passed", bool(self.passed))


@dataclass(frozen=True)
class RiskReport:
    checks: tuple[Check, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "STOP"

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_records(self) -> list[dict]:
        return [asdict(c) for c in self.checks]


# --- emergency stop ------------------------------------------------------------------------------

def emergency_stop_active(store: LiveStore) -> bool:
    return store.emergency_stop.exists()


def set_emergency_stop(store: LiveStore, reason: str) -> None:
    store.ensure()
    store.emergency_stop.write_text(f"{datetime.now().isoformat(timespec='seconds')} {reason}\n", encoding="utf-8")


def clear_emergency_stop(store: LiveStore) -> None:
    store.emergency_stop.unlink(missing_ok=True)


def check_switch(store: LiveStore) -> Check:
    if emergency_stop_active(store):
        reason = store.emergency_stop.read_text(encoding="utf-8").strip()
        return Check("emergency_stop", False, f"緊急停止中：{reason}")
    return Check("emergency_stop", True, "緊急停止は解除されている")


# --- data ------------------------------------------------------------------------------------------

def business_days_between(earlier: pd.Timestamp, later: pd.Timestamp) -> int:
    return int(np.busday_count(pd.Timestamp(earlier).date(), pd.Timestamp(later).date()))


def check_data(prices: pd.DataFrame, fx: pd.Series, assets: list[str], today: pd.Timestamp,
               max_age_days: int) -> list[Check]:
    """The latest row must have a positive price for every asset; prices and FX must be recent."""
    if prices is None or prices.empty:
        return [Check("prices_available", False, "価格データがない")]
    checks = []
    last = prices.index[-1]
    latest = prices.iloc[-1]
    for asset in assets:
        value = latest.get(asset, np.nan)
        if pd.isna(value):
            checks.append(Check(f"price:{asset}", False, f"{asset} の最新（{last.date()}）の価格が欠損"))
        elif not value > 0:
            checks.append(Check(f"price:{asset}", False, f"{asset} の最新価格が 0 以下（{value}）"))
        else:
            checks.append(Check(f"price:{asset}", True, f"{asset} {value:,.2f} USD（{last.date()}）"))
    age = business_days_between(last, today)
    checks.append(Check("prices_fresh", age <= max_age_days,
                        f"最新の価格は {last.date()}（{age} 営業日前、上限 {max_age_days}）"))

    fx = fx.dropna() if fx is not None else pd.Series(dtype=float)
    if fx.empty:
        checks.append(Check("fx_available", False, "為替レートがない"))
    else:
        fx_age = business_days_between(fx.index[-1], today)
        checks.append(Check("fx_positive", fx.iloc[-1] > 0, f"USD/JPY {fx.iloc[-1]:.3f}"))
        checks.append(Check("fx_fresh", fx_age <= max_age_days,
                            f"最新の為替は {fx.index[-1].date()}（{fx_age} 営業日前、上限 {max_age_days}）"))
    return checks


# --- orders ----------------------------------------------------------------------------------------

def check_orders(orders: list[Order], state: LedgerState, latest_prices: pd.Series, fx_mid: float,
                 limits: RiskLimits, costs: fxt.CostSettings, known_order_ids: set[str],
                 orders_already_today: int) -> list[Check]:
    checks = []
    ids = [o.order_id for o in orders]
    repeated = sorted({i for i in ids if ids.count(i) > 1} | (set(ids) & known_order_ids))
    checks.append(Check("no_duplicate_order", not repeated,
                        f"同じ注文がすでにある：{repeated}" if repeated else "重複する注文はない"))

    bad = [o.order_id for o in orders if not o.quantity > 0 or o.side not in fxt.SIDES]
    checks.append(Check("valid_quantity", not bad, f"数量または売買の向きが不正：{bad}" if bad else "数量は正"))
    unknown = [o.asset for o in orders if o.asset not in latest_prices.index or not latest_prices[o.asset] > 0]
    checks.append(Check("known_asset", not unknown, f"価格のない資産：{unknown}" if unknown else "全資産に価格がある"))
    if bad or unknown:
        return checks

    count = orders_already_today + len(orders)
    checks.append(Check("max_orders_per_day", count <= limits.max_orders_per_day,
                        f"本日の注文 {count} 件（上限 {limits.max_orders_per_day}）"))

    trades = {o.order_id: fxt.estimate_trade(o.side, o.quantity, o.price, fx_mid, costs) for o in orders}
    too_large = [f"{o.asset} {abs(trades[o.order_id]['cash_jpy']):,.0f} 円" for o in orders
                 if abs(trades[o.order_id]["cash_jpy"]) > limits.max_order_jpy + TOLERANCE]
    checks.append(Check("max_order_amount", not too_large,
                        f"1 回の上限 {limits.max_order_jpy:,.0f} 円を超える：{too_large}" if too_large
                        else f"すべて 1 回の上限 {limits.max_order_jpy:,.0f} 円以内"))

    held = state.positions["quantity"]
    oversold = [o.asset for o in orders if o.side == "sell" and o.quantity > held.get(o.asset, 0.0) + TOLERANCE]
    checks.append(Check("sell_within_holdings", not oversold,
                        f"保有数を超える売り：{oversold}" if oversold else "売りは保有数以内"))

    cash_after = state.cash_jpy + sum(t["cash_jpy"] for t in trades.values())
    checks.append(Check("enough_cash", cash_after >= -TOLERANCE,
                        f"注文後の現金 {cash_after:,.0f} 円（現在 {state.cash_jpy:,.0f} 円）"))

    quantity = held.to_dict()
    for o in orders:
        quantity[o.asset] = quantity.get(o.asset, 0.0) + (o.quantity if o.side == "buy" else -o.quantity)
    values = {a: q * latest_prices[a] * fx_mid for a, q in quantity.items() if q > TOLERANCE}
    total = sum(values.values()) + max(cash_after, 0.0)
    heavy = {a: v / total for a, v in values.items() if total > 0 and v / total > limits.max_weight + TOLERANCE}
    checks.append(Check("max_weight", not heavy,
                        f"上限 {limits.max_weight:.0%} を超える比率：" + ", ".join(f"{a} {w:.1%}" for a, w in heavy.items())
                        if heavy else f"注文後のどの資産も比率の上限 {limits.max_weight:.0%} 以内"))
    return checks


def evaluate(store: LiveStore, data_checks: list[Check], order_checks: list[Check]) -> RiskReport:
    return RiskReport(tuple([check_switch(store), *data_checks, *order_checks]))


def log_stop(store: LiveStore, context: dict, report: RiskReport) -> None:
    """Append the failed checks to audit_logs/errors.jsonl (one JSON object per line)."""
    if report.passed:
        return
    store.ensure()
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **context,
              "failures": [asdict(c) for c in report.failures]}
    with open(store.errors, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
