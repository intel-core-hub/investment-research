"""Calculations behind the dashboard (Phases 2-6), with no Streamlit dependency.

Every number comes from the existing modules, so the dashboard shows exactly the
definitions used in the reports:
  metrics (Phase 2)      cumulative, rolling and drawdown series
  portfolio (Phase 3-4)  lump-sum / DCA portfolios with rebalancing
  engine + strategies    Phase 5 backtests (next-day fills, costs)
  performance, inference, trades (Phase 6)  risk, Sharpe uncertainty, bootstrap, Holm
"""
from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src" / d) for d in ("analysis", "simulation", "portfolio", "backtest", "statistics")
                if str(ROOT / "src" / d) not in sys.path]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import inference as inf  # noqa: E402
import metrics as m  # noqa: E402
import performance as pf  # noqa: E402
import portfolio as pt  # noqa: E402
import trades as tr  # noqa: E402
from engine import BacktestResult, CostModel, run_backtest  # noqa: E402
from strategies import build_strategy  # noqa: E402

PRICES_PATH = ROOT / "data" / "processed" / "prices.csv"
CONFIG_DIR = ROOT / "config"
RISK_FREE = "BIL"


def load_prices(path: Path = PRICES_PATH) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.toml", "rb") as f:
        return tomllib.load(f)


def trading_days(prices: pd.DataFrame, start, end) -> pd.DataFrame:
    return prices.loc[pd.Timestamp(start):pd.Timestamp(end)]


def risk_free_returns(prices: pd.DataFrame) -> pd.Series:
    return pf.daily_returns(prices[RISK_FREE])


# --- ① Overview and ② Asset analysis ------------------------------------------------------

def asset_kpis(prices: pd.DataFrame, risk_free_prices: pd.Series) -> pd.DataFrame:
    """Phase 6 summary (cumulative return, CAGR, volatility, max DD, Sharpe, Sortino, Calmar) per asset."""
    rf = pf.daily_returns(risk_free_prices)
    table = pd.DataFrame({t: pf.summary(prices[t], rf) for t in prices}).T
    return table.drop(columns="final_equity")


def growth(prices: pd.DataFrame) -> pd.DataFrame:
    """Value of 1 invested on the first day."""
    return prices / prices.iloc[0]


def correlation(prices: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "daily":
        return m.correlation(m.daily_returns(prices))
    if frequency == "monthly":
        return m.correlation(pd.DataFrame({t: pf.monthly_returns(prices[t]) for t in prices}))
    raise ValueError(f"unknown frequency {frequency!r}")


# --- ③ Portfolio ---------------------------------------------------------------------------

def presets() -> dict[str, dict[str, float]]:
    return {p["name"]: p["weights"] for p in load_config("portfolios")["portfolios"]}


def weight_problems(percent: dict[str, float]) -> list[str]:
    """Why a set of percentage weights cannot be simulated (empty list = valid)."""
    total = sum(percent.values())
    problems = []
    if any(w < 0 for w in percent.values()):
        problems.append("比率にマイナスの値があります。")
    if total <= 0:
        problems.append("少なくとも1つの資産に比率を割り当ててください。")
    elif abs(total - 100) > 1e-6:
        problems.append(f"比率の合計が {total:g}% です。合計を 100% にしてください。")
    return problems


def normalize_percent(percent: dict[str, float]) -> dict[str, int]:
    """Scale to whole percentages summing to exactly 100 (largest-remainder rounding)."""
    total = sum(percent.values())
    if total <= 0:
        raise ValueError("weights sum to zero")
    exact = {t: w * 100 / total for t, w in percent.items()}
    floors = {t: int(np.floor(v)) for t, v in exact.items()}
    short = 100 - sum(floors.values())
    for t in sorted(exact, key=lambda t: exact[t] - floors[t], reverse=True)[:short]:
        floors[t] += 1
    return floors


def simulate(prices: pd.DataFrame, percent: dict[str, float], rebalance: str, method: str,
             amount: float) -> tuple[pt.PortfolioResult, dict]:
    """Phase 4 portfolio: `method` is "lump_sum" (amount on day 1) or "dca" (amount every month)."""
    weights = {t: w / 100 for t, w in percent.items() if w > 0}
    if method == "lump_sum":
        result = pt.simulate_portfolio(prices, weights, rebalance, initial_investment=amount)
    elif method == "dca":
        result = pt.simulate_portfolio(prices, weights, rebalance, monthly_contribution=amount)
    else:
        raise ValueError(f"unknown method {method!r}")
    return result, pt.summarize(result, prices[RISK_FREE])


def nav_drawdown(result: pt.PortfolioResult) -> pd.Series:
    """Drawdown of the time-weighted value (contributions do not hide losses)."""
    return m.drawdowns(pt.nav(result))


def diversification(prices: pd.DataFrame, percent: dict[str, float]) -> dict:
    """Volatility of the fixed-weight mix vs the weighted average of the assets' volatilities (Phase 4)."""
    weights = {t: w / 100 for t, w in percent.items() if w > 0}
    returns = m.daily_returns(prices[list(weights)])
    return pt.diversification(weights, returns.cov() * m.TRADING_DAYS)


# --- ④ Backtest ----------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestSettings:
    initial_cash: float
    commission_rate: float
    slippage_rate: float
    ma_window: int
    momentum_lookback: int
    momentum_top_n: int


def default_backtest_settings() -> BacktestSettings:
    cfg = load_config("backtest")
    by_type = {s["type"]: s for s in cfg["strategies"]}
    return BacktestSettings(
        initial_cash=float(cfg["initial_cash"]),
        commission_rate=cfg["costs"]["commission_rate"],
        slippage_rate=cfg["costs"]["slippage_rate"],
        ma_window=by_type["moving_average"]["window"],
        momentum_lookback=by_type["momentum"]["lookback_days"],
        momentum_top_n=by_type["momentum"]["top_n"],
    )


def strategy_specs(settings: BacktestSettings) -> dict[str, dict]:
    """The Phase 5 strategies from config/backtest.toml with the dashboard's parameters applied."""
    specs = {}
    for spec in load_config("backtest")["strategies"]:
        spec = dict(spec)
        if spec["type"] == "moving_average":
            spec["window"] = settings.ma_window
        elif spec["type"] == "momentum":
            spec.update(lookback_days=settings.momentum_lookback, top_n=settings.momentum_top_n)
        specs[spec["name"]] = spec
    return specs


def trading_start(prices: pd.DataFrame, start, warmup_days: int) -> pd.Timestamp:
    """Trading starts on the chosen date, but never before `warmup_days` of history exist."""
    if warmup_days >= len(prices):
        raise ValueError(f"need more than {warmup_days} trading days of data")
    position = min(int(prices.index.searchsorted(pd.Timestamp(start))), len(prices) - 1)
    return prices.index[max(position, warmup_days)]


def run_strategies(prices: pd.DataFrame, names: list[str], settings: BacktestSettings, start, end) \
        -> dict[str, BacktestResult]:
    """Phase 5 engine on every named strategy, all starting on the same trading day."""
    cfg = load_config("backtest")
    specs = strategy_specs(settings)
    history = prices.loc[:pd.Timestamp(end)]
    first = trading_start(history, start, cfg["warmup_days"])
    if first >= history.index[-1]:
        raise ValueError("the period ends before trading can start")
    costs = CostModel(commission_rate=settings.commission_rate, commission_min=cfg["costs"]["commission_min"],
                      slippage_rate=settings.slippage_rate)
    return {n: run_backtest(history, build_strategy(specs[n]), settings.initial_cash, costs, start=first)
            for n in names}


def backtest_table(results: dict[str, BacktestResult], settings: BacktestSettings,
                   risk_free_returns: pd.Series) -> pd.DataFrame:
    """Performance (Phase 6) plus trade statistics for each strategy."""
    specs = strategy_specs(settings)
    rows = []
    for name, result in results.items():
        spec = specs[name]
        tx = result.transactions
        trade = tr.strategy_trade_table(name, spec["type"], tx, spec.get("risk_off"))[0]
        rows.append({
            "strategy": name, "strategy_type": spec["type"], "start": result.equity.index[0].date(),
            "end": result.equity.index[-1].date(), **pf.summary(result.equity, risk_free_returns),
            "transactions": len(tx),
            "total_costs": float(tx["commission"].sum() + tx["slippage_cost"].sum()),
            # NaN where round trips do not apply (buy & hold, periodic rebalancing)
            "closed_trades": float(trade.get("closed_trades", np.nan)),
            "win_rate": float(trade.get("win_rate", np.nan)),
            "profit_factor": float(trade.get("profit_factor", np.nan)),
            "trade_stats_applicable": trade["applicable"],
        })
    return pd.DataFrame(rows).set_index("strategy")


def round_trips(result: BacktestResult) -> pd.DataFrame:
    return tr.round_trips(result.transactions)[0]


# --- ⑤ Statistics --------------------------------------------------------------------------

def aligned_returns(equities: pd.DataFrame) -> pd.DataFrame:
    """Daily returns on the dates every series shares, so all statistics use the same days."""
    common = equities.dropna()
    return pd.DataFrame({c: pf.daily_returns(common[c]) for c in common})


def sharpe_table(returns: pd.DataFrame, risk_free_returns: pd.Series, confidence: float) -> pd.DataFrame:
    rows = []
    for name in returns:
        stats = {(r["statistic"], r["method"]): r for r in inf.sharpe_inference(returns[name], risk_free_returns,
                                                                                  confidence)}
        lo, mertens = stats.get(("sharpe_ratio", "lo_2002_iid")), stats.get(("sharpe_ratio", "mertens_2002_non_normal"))
        psr = stats.get(("probabilistic_sharpe_ratio_vs_0", "bailey_lopez_de_prado_2012"))
        rows.append({
            "series": name,
            "sharpe_ratio": lo["estimate"] if lo else np.nan,
            "se_lo": lo["standard_error"] if lo else np.nan,
            "ci_lower_lo": lo["ci_lower"] if lo else np.nan,
            "ci_upper_lo": lo["ci_upper"] if lo else np.nan,
            "se_mertens": mertens["standard_error"] if mertens else np.nan,
            "ci_lower_mertens": mertens["ci_lower"] if mertens else np.nan,
            "ci_upper_mertens": mertens["ci_upper"] if mertens else np.nan,
            "psr": psr["estimate"] if psr else np.nan,
            "observations": len(returns[name].dropna()),
        })
    return pd.DataFrame(rows).set_index("series")


def mean_return_table(returns: pd.DataFrame, confidence: float) -> pd.DataFrame:
    rows = [{"series": name, **r} for name in returns for r in inf.mean_confidence_intervals(returns[name],
                                                                                              confidence)]
    return pd.DataFrame(rows)


def risk_table(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in returns:
        r = returns[name]
        dist = pf.distribution(r)
        row = {"series": name}
        for c in (0.95, 0.99):
            v = pf.historical_var_cvar(r, c)
            row[f"var_{int(c * 100)}"], row[f"cvar_{int(c * 100)}"] = v["var"], v["cvar"]
        row.update(skewness=dist["skewness"], excess_kurtosis=dist["excess_kurtosis"],
                   observations=dist["observations"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("series")


def bootstrap_table(returns: pd.DataFrame, risk_free_returns: pd.Series, benchmark: str, resamples: int,
                    seed: int, confidence: float, block_size: int) -> pd.DataFrame:
    """i.i.d. and block bootstrap intervals; p-values of differences Holm-adjusted per method and statistic."""
    rows = []
    for block in dict.fromkeys((1, block_size)):
        rows += inf.bootstrap(returns, risk_free_returns, benchmark, resamples, seed, confidence, block)
    table = pd.DataFrame(rows)
    table["p_value_holm"] = np.nan
    if "p_value" in table:
        for _, group in table[table["p_value"].notna()].groupby(["method", "statistic"]):
            table.loc[group.index, "p_value_holm"] = inf.holm_adjust(group["p_value"])
    return table
