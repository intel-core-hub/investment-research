"""Round-trip trade statistics for strategies that fully enter and exit positions.

A round trip in one ticker starts with a buy while no shares are held and ends
with the sell that brings the position back to zero. Its profit is everything
received from sells minus everything paid for buys, commissions included
(slippage is already in the fill prices). Positions still open at the end are
reported separately and are not counted as wins or losses.

Only strategy types that switch between whole positions are applicable. Buy &
hold never closes a position, and periodic rebalancing only trims and tops up
positions, so a win rate would not describe either of them.
"""
from __future__ import annotations

import pandas as pd

APPLICABLE_TYPES = {"moving_average", "momentum"}
NOT_APPLICABLE_REASON = {
    "buy_and_hold": "holds one position for the whole period; no trade is ever closed",
    "periodic_rebalance": "rebalancing adjusts positions without closing them; trades have no entry/exit",
}
POSITION_TOLERANCE = 1e-9


def round_trips(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(closed round trips, positions still open at the end)."""
    closed, still_open = [], []
    tx = transactions.sort_values("date", kind="stable")
    for ticker, group in tx.groupby("ticker", sort=False):
        held, cost, proceeds, opened = 0.0, 0.0, 0.0, None
        for row in group.itertuples():
            if row.side == "buy":
                if held <= POSITION_TOLERANCE:
                    held, cost, proceeds, opened = 0.0, 0.0, 0.0, row.date
                held += row.quantity
                cost += row.gross_value + row.commission
            else:
                held -= row.quantity
                proceeds += row.gross_value - row.commission
                if held <= POSITION_TOLERANCE and opened is not None:
                    closed.append({"ticker": ticker, "entry_date": opened, "exit_date": row.date,
                                   "holding_days": (row.date - opened).days, "cost": cost,
                                   "proceeds": proceeds, "profit": proceeds - cost,
                                   "return": proceeds / cost - 1})
                    held, opened = 0.0, None
        if opened is not None and held > POSITION_TOLERANCE:
            still_open.append({"ticker": ticker, "entry_date": opened, "shares": held, "cost": cost})
    columns = ["ticker", "entry_date", "exit_date", "holding_days", "cost", "proceeds", "profit", "return"]
    return pd.DataFrame(closed, columns=columns), pd.DataFrame(still_open)


def longest_streak(flags: list[bool], value: bool) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag == value else 0
        best = max(best, run)
    return best


def trade_stats(trips: pd.DataFrame) -> dict:
    """Win rate, average win/loss (returns), profit factor and streaks, in exit-date order.

    Profit factor = total profit of winners / |total loss of losers|; NaN if there
    are no losing trades (undefined rather than infinite).
    """
    t = trips.sort_values("exit_date", kind="stable")
    wins, losses = t[t["profit"] > 0], t[t["profit"] <= 0]
    loss_total = -losses["profit"].sum()
    flags = list(t["profit"] > 0)
    return {
        "closed_trades": len(t),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(t) if len(t) else float("nan"),
        "average_win_return": wins["return"].mean() if len(wins) else float("nan"),
        "average_loss_return": losses["return"].mean() if len(losses) else float("nan"),
        "average_win_profit": wins["profit"].mean() if len(wins) else float("nan"),
        "average_loss_profit": losses["profit"].mean() if len(losses) else float("nan"),
        "profit_factor": wins["profit"].sum() / loss_total if loss_total > 0 else float("nan"),
        "max_consecutive_wins": longest_streak(flags, True),
        "max_consecutive_losses": longest_streak(flags, False),
        "average_holding_days": t["holding_days"].mean() if len(t) else float("nan"),
        "total_profit": t["profit"].sum(),
    }


def strategy_trade_table(name: str, kind: str, transactions: pd.DataFrame, risk_off: str | None) -> list[dict]:
    """Rows for all round trips, and for round trips outside the risk-off asset if there is one."""
    if kind not in APPLICABLE_TYPES:
        return [{"strategy": name, "strategy_type": kind, "scope": "all", "applicable": False,
                 "reason": NOT_APPLICABLE_REASON.get(kind, "not a position-switching strategy")}]
    trips, still_open = round_trips(transactions)
    scopes = [("all", trips)]
    if risk_off:
        scopes.append((f"excluding {risk_off}", trips[trips["ticker"] != risk_off]))
    return [{"strategy": name, "strategy_type": kind, "scope": scope, "applicable": True, "reason": "",
             **trade_stats(t), "open_positions_at_end": len(still_open)} for scope, t in scopes]
