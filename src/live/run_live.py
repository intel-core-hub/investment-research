"""Phase 9 command line. Approving, rejecting and recording fills are done on the dashboard.

    python src/live/run_live.py fetch            download prices and USD/JPY into data/live/market/
    python src/live/run_live.py propose          today's proposal (pipeline + risk check), recorded in the audit log
    python src/live/run_live.py recalc           rebuild holdings.csv, cash.csv, valuations.csv; fill paper orders
    python src/live/run_live.py status           data dates, pending decisions, open orders, errors
    python src/live/run_live.py stop "reason"    emergency stop: every proposal stops and nothing can be approved
    python src/live/run_live.py resume           lift the emergency stop

No command sends an order to a broker.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import bridge  # noqa: E402
import fx_and_taxes as fxt  # noqa: E402
import market as mk  # noqa: E402
import paper_engine as paper  # noqa: E402
import risk_controller as rc  # noqa: E402
import signal_pipeline as sp  # noqa: E402
from settings import default_store, load_settings  # noqa: E402


def need_market(store):
    data = mk.load(store)
    if data is None:
        sys.exit("no market data yet: run `python src/live/run_live.py fetch` first")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fetch", "propose", "recalc", "status", "stop", "resume"])
    parser.add_argument("reason", nargs="?", default="manual stop")
    args = parser.parse_args()

    settings = load_settings()
    store = default_store(settings).ensure()
    today = pd.Timestamp.now().normalize()

    if args.command == "fetch":
        tickers = [a["ticker"] for a in sp.research_config("assets")["assets"]]   # the research universe
        data = mk.fetch(tickers)
        mk.save(store, data)
        print(f"saved {', '.join(tickers)} to {data.as_of.date()} and USD/JPY to {data.fx.index[-1].date()}"
              f" in {store.market_prices.parent}")
    elif args.command == "propose":
        proposal, recorded = bridge.propose(store, settings, need_market(store), today)
        print(f"{proposal.decision_id} {proposal.status} ({'recorded' if recorded else 'already proposed'})")
        print(f"  target: {proposal.target_weights}")
        for o in proposal.orders:
            print(f"  {o.side} {o.quantity:g} {o.asset} (about {o.value_jpy:,.0f} JPY)")
        for c in proposal.risk.failures:
            print(f"  STOP {c.name}: {c.detail}")
        for note in proposal.notes:
            print(f"  {note}")
        print("approve or reject it on the dashboard (streamlit run app.py -> ⑧ Today's Decision)")
    elif args.command == "recalc":
        data = need_market(store)
        paper.sync_deposits(store)
        filled = paper.fill_pending(store, data, fxt.CostSettings.from_config(settings))
        result = bridge.recalculate(store, data)
        print(f"ledger: {len(result['transactions'])} transactions, cash {result['state'].cash_jpy:,.0f} JPY; "
              f"paper orders filled: {filled}")
    elif args.command == "status":
        info = bridge.status(store, mk.load(store), today)
        for key, value in info.items():
            print(f"{key}: {value if key != 'errors' else len(value)}")
    elif args.command == "stop":
        rc.set_emergency_stop(store, args.reason)
        print(f"emergency stop set: {store.emergency_stop}")
    elif args.command == "resume":
        rc.clear_emergency_stop(store)
        print("emergency stop lifted")


if __name__ == "__main__":
    main()
