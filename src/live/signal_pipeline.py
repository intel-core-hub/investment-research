"""9.4 Research -> signal pipeline.

    latest data -> data checks -> strategy or ML model -> target weights
    -> orders sized from the ledger -> risk controller -> Proposal

Nothing here places an order. A Proposal is written to the audit log and shown on
the dashboard, where a person approves or rejects it (9.6). Approved orders go to
the order queue; the person places them at the broker and records the fill in the
ledger.

Signals (computed on dividend-adjusted closes, as in the research):
- a Phase 5 strategy from config/backtest.toml: its target weights at the latest close,
  from the same `decide` code as the backtest, called as if a new period started today
- "ml": the Phase 7 pipeline refitted on every label already known; hold the target
  asset if the predicted probability of a rise is above the threshold, else the risk-free asset

Orders (sized with actual closes and the latest FX rate, including costs):
- buy_only: spend the available cash on the assets furthest below their target value
  (largest shortfall first); never sell
- rebalance: first sell what is above target, then buy as in buy_only
With whole shares only, a small amount of cash may not buy a single share; the
proposal then has no orders and the cash waits for the next contribution.
Approved orders that are not yet in the ledger count as done when sizing new ones.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import tomllib
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime

import pandas as pd

from settings import ROOT

sys.path[:0] = [str(ROOT / "src" / d) for d in ("backtest", "statistics", "ml")
                if str(ROOT / "src" / d) not in sys.path]

import fx_and_taxes as fxt  # noqa: E402
import risk_controller as rc  # noqa: E402
from market import MarketData  # noqa: E402
from provenance import git_state  # noqa: E402
from real_ledger import LedgerState  # noqa: E402
from settings import LiveStore  # noqa: E402
from strategies import build_strategy  # noqa: E402

MODES = ("buy_only", "rebalance")


class PipelineError(RuntimeError):
    pass


def research_config(name: str) -> dict:
    with open(ROOT / "config" / f"{name}.toml", "rb") as f:
        return tomllib.load(f)


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def data_version(market: MarketData) -> str:
    """Hash of the exact data the signal saw (closes, adjusted closes, volume, FX)."""
    h = hashlib.sha256()
    for frame in (market.close, market.adjusted, market.volume, market.fx):
        h.update(frame.to_csv().encode("utf-8"))
    return h.hexdigest()


# --- signal ---------------------------------------------------------------------------------------

def strategy_spec(settings: dict) -> dict:
    """The strategy definition the signal comes from (its hash is the strategy version)."""
    name = settings["signal"]["strategy"]
    if name == "ml":
        ml_cfg = research_config("ml")
        return {"name": "ml", "type": "ml", **settings["ml"], "target": ml_cfg["target"],
                "relative_to": ml_cfg["features"]["relative_to"], "select_k": ml_cfg["features"]["select_k"],
                "seed": ml_cfg["seed"], "regimes": research_config("statistics")["regimes"]}
    for spec in research_config("backtest")["strategies"]:
        if spec["name"] == name:
            return spec
    raise PipelineError(f"unknown strategy {name!r}: use a name from config/backtest.toml or 'ml'")


def signal_assets(spec: dict) -> list[str]:
    if spec["type"] == "ml":
        t = spec["target"]
        return list(dict.fromkeys([t["asset"], t["risk_free"], *spec["relative_to"]]))
    return build_strategy(spec).tickers


def target_from_strategy(adjusted: pd.DataFrame, spec: dict) -> tuple[dict, dict]:
    strategy = build_strategy(spec)
    target = strategy.decide(adjusted[strategy.tickers].dropna(), first_day=True)
    if target is None:
        raise PipelineError(f"{spec['name']}: not enough price history for a signal")
    return {a: float(w) for a, w in target.items() if w > 0}, {}


def target_from_ml(adjusted: pd.DataFrame, volume: pd.DataFrame, spec: dict) -> tuple[dict, dict]:
    import regimes as rg
    from features import build_features, complete_rows, direction_target, feature_sets, label_end_dates, \
        trainable_rows
    from models import make_model

    asset, risk_free, horizon = spec["target"]["asset"], spec["target"]["risk_free"], spec["target"]["horizon_days"]
    settings = rg.RegimeSettings(**{k: v for k, v in spec["regimes"].items() if k != "reference"})
    features = build_features(adjusted, volume[asset], asset, risk_free, spec["relative_to"], settings)
    cols = feature_sets(spec["relative_to"])[spec["feature_set"]]
    X = complete_rows(features, cols)
    if X.empty or X.index[-1] != adjusted.index[-1]:
        raise PipelineError("ML: features are not available for the latest date")
    y = direction_target(adjusted[asset], horizon).reindex(X.index)
    label_end = label_end_dates(adjusted.index, horizon).reindex(X.index)
    fit = trainable_rows(y, label_end, X.index[0], adjusted.index[-1])   # labels already known today
    model = make_model(spec["model"], spec["hyperparameters"], spec["seed"], spec["select_k"])
    model.fit(X.loc[fit], y.loc[fit].astype(int))
    probability = float(model.predict_proba(X.iloc[[-1]])[0, 1])
    target = {asset: 1.0} if probability > spec["threshold"] else {risk_free: 1.0}
    return target, {"probability_up": probability, "threshold": spec["threshold"], "fit_rows": len(fit),
                    "last_known_label": str(fit[-1].date())}


# --- orders ---------------------------------------------------------------------------------------

def with_open_orders(state: LedgerState, orders: list[rc.Order], fx_mid: float,
                     costs: fxt.CostSettings) -> LedgerState:
    """The state as if approved orders not yet in the ledger were filled at their estimates."""
    if not orders:
        return state
    positions = state.positions.copy()
    cash = state.cash_jpy
    for o in orders:
        cash += fxt.estimate_trade(o.side, o.quantity, o.price, fx_mid, costs)["cash_jpy"]
        held = positions["quantity"].get(o.asset, 0.0)
        new = held + (o.quantity if o.side == "buy" else -o.quantity)
        if o.asset in positions.index:
            positions.loc[o.asset, "quantity"] = new
        else:
            positions.loc[o.asset] = {"quantity": new, "cost_basis_jpy": 0.0, "average_cost_jpy": 0.0}
    return replace(state, positions=positions[positions["quantity"] > 1e-9], cash_jpy=cash)


def current_weights(state: LedgerState, close: pd.Series, fx_mid: float) -> dict:
    values = {a: q * close[a] * fx_mid for a, q in state.positions["quantity"].items()}
    total = sum(values.values()) + state.cash_jpy
    return {a: v / total for a, v in values.items()} if total > 0 else {}


def size_orders(target: dict, state: LedgerState, close: pd.Series, fx_mid: float, mode: str,
                fractional: bool, costs: fxt.CostSettings, decision_id: str, as_of: pd.Timestamp) -> list[rc.Order]:
    if mode not in MODES:
        raise PipelineError(f"unknown mode {mode!r}; use one of {MODES}")
    held = state.positions["quantity"].to_dict()
    values = {a: q * close[a] * fx_mid for a, q in held.items()}
    total = sum(values.values()) + state.cash_jpy
    cash = state.cash_jpy
    date = str(pd.Timestamp(as_of).date())

    def order(asset, side, quantity):
        return rc.Order(order_id=f"{decision_id}-{asset}-{side}", decision_id=decision_id, date=date, asset=asset,
                        side=side, quantity=float(quantity), price=float(close[asset]), fx_rate=float(fx_mid))

    def rounded(q):
        return q if fractional else math.floor(q + 1e-9)

    orders = []
    if mode == "rebalance":
        for asset in sorted(set(values) | set(target)):
            excess = values.get(asset, 0.0) - target.get(asset, 0.0) * total
            q = min(rounded(excess / (close[asset] * fx_mid)), held.get(asset, 0.0)) if excess > 0 else 0
            if q > 0:
                orders.append(order(asset, "sell", q))
                cash += fxt.estimate_trade("sell", q, close[asset], fx_mid, costs)["cash_jpy"]
                values[asset] -= q * close[asset] * fx_mid

    shortfall = {a: w * total - values.get(a, 0.0) for a, w in target.items()}
    for asset in sorted((a for a in shortfall if shortfall[a] > 0), key=lambda a: (-shortfall[a], a)):
        per_share = -fxt.estimate_trade("buy", 1, close[asset], fx_mid, costs)["cash_jpy"]
        q = rounded(min(shortfall[asset], cash) / per_share)
        while q > 0 and -fxt.estimate_trade("buy", q, close[asset], fx_mid, costs)["cash_jpy"] > cash:
            q = q - 1 if not fractional else q * 0.999   # minimum commission can push the cost over the budget
        if q > 0:
            orders.append(order(asset, "buy", q))
            cash += fxt.estimate_trade("buy", q, close[asset], fx_mid, costs)["cash_jpy"]
    return orders


# --- proposal -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Proposal:
    decision_id: str
    created_at: str
    as_of: str
    strategy: str
    strategy_version: str
    data_version: str
    git_commit: str
    git_dirty: bool
    signal: dict
    target_weights: dict
    current_weights: dict
    orders: tuple[rc.Order, ...]
    paper_orders: tuple[rc.Order, ...]
    risk: rc.RiskReport
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def status(self) -> str:
        return self.risk.status

    def to_record(self) -> dict:
        record = asdict(self)
        record["risk"] = {"status": self.status, "checks": self.risk.to_records()}
        return record


def run(store: LiveStore, settings: dict, market: MarketData, state: LedgerState, today: pd.Timestamp,
        known_order_ids: set[str], open_orders: list[rc.Order], orders_today: int,
        paper_state: LedgerState | None = None) -> Proposal:
    """One pass of the pipeline. Always returns a Proposal; a failed check gives status STOP."""
    spec = strategy_spec(settings)
    limits = rc.RiskLimits.from_config(settings)
    costs = fxt.CostSettings.from_config(settings)
    git = git_state(ROOT)
    as_of = market.as_of
    base = {"created_at": datetime.now().isoformat(timespec="seconds"), "as_of": str(as_of.date()),
            "strategy": spec["name"], "strategy_version": sha256_json(spec)[:16],
            "data_version": data_version(market)[:16], "git_commit": git["git_commit"],
            "git_dirty": git["git_dirty"]}

    needed = sorted(set(signal_assets(spec)) | set(state.positions.index))
    data_checks = rc.check_data(market.close, market.fx, needed, today, limits.max_data_age_days)

    def stopped(checks, signal, notes=()):
        decision_id = sha256_json({**base, "created_at": None, "stop": [c.name for c in checks if not c.passed]})[:12]
        return Proposal(decision_id=decision_id, **base, signal=signal, target_weights={}, current_weights={},
                        orders=(), paper_orders=(), risk=rc.evaluate(store, checks, []), notes=tuple(notes))

    if not all(c.passed for c in data_checks):
        return stopped(data_checks, {"error": "data checks failed"})
    try:
        if spec["type"] == "ml":
            target, info = target_from_ml(market.adjusted, market.volume, spec)
        else:
            target, info = target_from_strategy(market.adjusted, spec)
    except PipelineError as e:
        return stopped(data_checks + [rc.Check("signal", False, str(e))], {"error": str(e)})
    missing = [a for a in target if a not in market.close.columns]
    if missing:
        return stopped(data_checks + [rc.Check("signal", False, f"no price for {missing}")], {"error": "no price"})

    close = market.close.iloc[-1]
    fx_mid = float(market.fx.dropna().iloc[-1])
    effective = with_open_orders(state, open_orders, fx_mid, costs)
    # the same data, target and holdings give the same order ids, so a repeated order is detected
    order_key = sha256_json({"as_of": base["as_of"], "strategy_version": base["strategy_version"],
                             "target": target, "positions": effective.positions["quantity"].round(9).to_dict(),
                             "cash": round(effective.cash_jpy, 2)})[:12]
    mode, fractional = settings["signal"]["mode"], settings["signal"]["fractional_shares"]
    orders = size_orders(target, effective, close, fx_mid, mode, fractional, costs, order_key, as_of)
    paper = size_orders(target, paper_state, close, fx_mid, mode, fractional, costs, f"paper-{order_key}",
                        as_of) if paper_state is not None else []
    signal_check = rc.Check("signal", True, f"{spec['name']}: " + ", ".join(f"{a} {w:.0%}" for a, w in target.items()))
    order_checks = rc.check_orders(orders, effective, close, fx_mid, limits, costs, known_order_ids, orders_today)
    report = rc.evaluate(store, data_checks + [signal_check], order_checks)
    # a stopped proposal gets its own id, so lifting the cause (e.g. the emergency stop) allows a new proposal
    decision_id = order_key if report.passed else \
        f"{order_key}-{sha256_json([c.name for c in report.failures])[:4]}"
    notes = []
    if not orders:
        notes.append("注文なし：目標との差を埋めるのに必要な株数が 1 株未満か、使える現金がない。")
    return Proposal(decision_id=decision_id, **base, signal={"type": spec["type"], **info},
                    target_weights=target, current_weights=current_weights(effective, close, fx_mid),
                    orders=tuple(orders), paper_orders=tuple(paper), risk=report, notes=tuple(notes))
