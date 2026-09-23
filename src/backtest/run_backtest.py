"""Backtest every strategy in config/backtest.toml and compare it with the benchmark.

Each strategy runs twice on the same data and start date: with the configured
costs and without costs. Writes CSV tables, figures and run_info.json (input
hashes and versions for reproducibility) to reports/backtest/.
"""
import hashlib
import json
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# shared metrics (analysis) and crash-episode detection (simulation)
sys.path[:0] = [str(ROOT / "src" / "analysis"), str(ROOT / "src" / "simulation")]

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

import metrics as m  # noqa: E402
from engine import BacktestResult, CostModel, run_backtest  # noqa: E402
from scenarios import drawdown_episodes  # noqa: E402
from strategies import build_strategy  # noqa: E402

CONFIG_PATH = ROOT / "config" / "backtest.toml"
PRICES_PATH = ROOT / "data" / "processed" / "prices.csv"
OUT_DIR = ROOT / "reports" / "backtest"
FIG_DIR = OUT_DIR / "figures"
CRASH_THRESHOLD = 0.15

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ASSET_COLORS = {"VOO": "#2a78d6", "VTI": "#eb6834", "VT": "#1baf7a", "EWJ": "#eda100",
                "BND": "#e87ba4", "BIL": "#008300"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("red_gray_blue", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.edgecolor": BASELINE,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False,
    "lines.linewidth": 1.3,
})


def load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def build_all(config: dict, tickers) -> list:
    errors, strategies = [], []
    for spec in config.get("strategies", []):
        try:
            strategy = build_strategy(spec)
        except ValueError as e:
            errors.append(str(e))
            continue
        missing = set(strategy.tickers) - set(tickers)
        if missing:
            errors.append(f"{strategy.name}: {sorted(missing)} not in the price data")
        strategies.append(strategy)
    names = [s.name for s in strategies]
    if not strategies:
        errors.append("no strategies defined")
    for name in {n for n in names if names.count(n) > 1}:
        errors.append(f"strategy name {name!r} is used more than once")
    if config.get("benchmark") not in names:
        errors.append(f"benchmark {config.get('benchmark')!r} is not one of the strategies")
    if config.get("risk_free_ticker") not in tickers:
        errors.append(f"risk_free_ticker {config.get('risk_free_ticker')!r} is not in the price data")
    if errors:
        raise SystemExit("invalid backtest config:\n  - " + "\n  - ".join(errors))
    return strategies


def summarize(result: BacktestResult, risk_free: pd.Series) -> dict:
    equity = result.equity
    drawdown = m.drawdowns(equity)
    tx = result.transactions
    years = m.years(equity)
    return {
        "strategy": result.strategy,
        "start": equity.index[0].date(),
        "end": equity.index[-1].date(),
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(m.total_return(equity)),
        "cagr": float(m.cagr(equity)),
        "annual_volatility": float(m.annual_volatility(equity)),
        "max_drawdown": float(drawdown.min()),
        "max_drawdown_date": drawdown.idxmin().date(),
        "sharpe_ratio": float(m.sharpe_ratio(equity, risk_free.loc[equity.index])),
        "signals": len(result.signals),
        "trades": len(tx),
        "annual_turnover": float(tx["gross_value"].sum() / 2 / equity.mean() / years),
        "total_commission": float(tx["commission"].sum()),
        "total_slippage": float(tx["slippage_cost"].sum()),
    }


def summary_table(results: dict, config: dict, risk_free: pd.Series) -> pd.DataFrame:
    rows = [{**summarize(r, risk_free), "costs": scenario} for (name, scenario), r in results.items()]
    table = pd.DataFrame(rows)
    bench = table[table["strategy"] == config["benchmark"]].set_index("costs")["cagr"]
    table["excess_cagr_vs_benchmark"] = table["cagr"] - table["costs"].map(bench)
    gross = table[table["costs"] == "no costs"].set_index("strategy")["cagr"]
    table["cost_drag_cagr"] = table["strategy"].map(gross) - table["cagr"]
    first = ["strategy", "costs"]
    return table[first + [c for c in table.columns if c not in first]]


def yearly_returns(equities: pd.DataFrame) -> pd.DataFrame:
    yearly = m.period_returns(equities, "YE")
    yearly.index = yearly.index.year.rename("year")
    return yearly


def regime_returns(equities: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """Returns in each benchmark crash (peak to trough, trough to recovery) and in up/down months."""
    rows = []
    episodes = drawdown_episodes(equities[benchmark], CRASH_THRESHOLD)
    for ep in episodes.itertuples():
        end = ep.recovery if pd.notna(ep.recovery) else equities.index[-1]
        for phase, a, b in [("fall", ep.peak, ep.trough), ("recovery", ep.trough, end)]:
            window = equities.loc[a:b]
            rows.append({"regime": f"{ep.peak.date()} crash: {phase}", "start": a.date(), "end": b.date(),
                         **(window.iloc[-1] / window.iloc[0] - 1).to_dict()})
    monthly = m.period_returns(equities, "ME").iloc[1:]
    for label, mask in [("benchmark up months (average)", monthly[benchmark] > 0),
                        ("benchmark down months (average)", monthly[benchmark] <= 0)]:
        rows.append({"regime": label, "start": None, "end": f"{int(mask.sum())} months",
                     **monthly[mask].mean().to_dict()})
    return pd.DataFrame(rows)


def git_state() -> dict:
    """Commit checked out and whether tracked or untracked files differ from it (reports/ excluded).

    Call before writing any report, so this run's own outputs do not count as changes.
    """
    def git(*args) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout

    try:
        return {
            "git_commit": git("rev-parse", "HEAD").strip(),
            "git_dirty": bool(git("status", "--porcelain", "--", ".", ":(exclude)reports").strip()),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def run_info(config: dict, prices: pd.DataFrame, start, git: dict) -> dict:
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "prices_file": PRICES_PATH.relative_to(ROOT).as_posix(),
        "prices_sha256": sha256(PRICES_PATH),
        "config_file": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(CONFIG_PATH),
        **git,
        "data_start": str(prices.index[0].date()),
        "trading_start": str(start.date()),
        "end": str(prices.index[-1].date()),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "config": config,
    }


def save_figure(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=120)
    plt.close(fig)


def spread_labels(values: pd.Series, min_gap: float) -> pd.Series:
    positions = values.sort_values().copy()
    for i in range(1, len(positions)):
        positions.iloc[i] = max(positions.iloc[i], positions.iloc[i - 1] + min_gap)
    return positions


def plot_equity(equities: pd.DataFrame, colors: dict, initial_cash: float) -> None:
    growth = equities / initial_cash
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in growth.columns:
        ax.plot(growth.index, growth[name], color=colors[name], label=name)
    final = np.log10(growth.iloc[-1])
    span = np.log10(growth.max().max()) - np.log10(growth.min().min())
    label_y = spread_labels(final, 0.035 * span)
    for name in growth.columns:
        ax.annotate(f"{growth[name].iloc[-1]:.2f}", (growth.index[-1], 10 ** label_y[name]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                    color=INK_SECONDARY, annotation_clip=False)
    ax.set_yscale("log")
    ax.set_yticks([t for t in [1, 2, 3, 5, 10] if growth.min().min() <= t <= growth.max().max() * 1.1])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylabel("equity / initial cash")
    ax.set_title(f"Equity after costs, log scale ({equities.index[0].date()} to {equities.index[-1].date()})",
                 loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, "equity")


def plot_drawdowns(equities: pd.DataFrame, colors: dict) -> None:
    dd = m.drawdowns(equities)
    names = list(equities.columns)
    rows = int(np.ceil(len(names) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 2.4 * rows + 0.6), sharex=True, sharey=True, squeeze=False)
    for ax, name in zip(axes.flat, names):
        ax.fill_between(dd.index, dd[name], 0, color=colors[name], alpha=0.35, linewidth=0)
        ax.plot(dd.index, dd[name], color=colors[name], linewidth=1)
        ax.set_title(f"{name}  max {dd[name].min():.1%} ({dd[name].idxmin().date()})", loc="left", fontsize=10)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    fig.suptitle("Drawdown from previous peak (after costs)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "drawdowns")


def plot_allocation(results: dict, prices: pd.DataFrame) -> None:
    runs = [r for (name, scenario), r in results.items() if scenario == "with costs"]
    rows = int(np.ceil(len(runs) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 2.6 * rows + 0.8), sharex=True, sharey=True, squeeze=False)
    for ax, result in zip(axes.flat, runs):
        weights = result.weights(prices)
        held = [t for t in weights.columns if weights[t].max() > 1e-6]
        ax.stackplot(weights.index, [weights[t] for t in held],
                     colors=[ASSET_COLORS.get(t, INK_MUTED) for t in held], labels=held, linewidth=0)
        ax.set_title(result.strategy, loc="left", fontsize=10)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0, 1)
        ax.legend(loc="lower left", ncols=len(held), fontsize=8)
    for ax in axes.flat[len(runs):]:
        ax.set_visible(False)
    fig.suptitle("Holdings as share of equity", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "allocation")


def plot_yearly(yearly: pd.DataFrame) -> None:
    table = yearly.T
    vmax = np.ceil(table.abs().max().max() * 10) / 10
    fig, ax = plt.subplots(figsize=(13, 0.6 * len(table) + 1.6))
    ax.imshow(table.values, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 * vmax else INK)
    ax.set_title(f"Yearly returns after costs ({table.columns[0]} and {table.columns[-1]} are partial years)",
                 loc="left")
    fig.tight_layout()
    save_figure(fig, "yearly_returns")


def plot_costs(summary: pd.DataFrame, names: list) -> None:
    table = summary.pivot(index="strategy", columns="costs", values="cagr").loc[names]
    fig, ax = plt.subplots(figsize=(10, 0.8 * len(names) + 1.4))
    y = np.arange(len(names))[::-1]
    ax.barh(y + 0.18, table["no costs"], 0.34, color=BASELINE, label="no costs")
    ax.barh(y - 0.18, table["with costs"], 0.34, color="#2a78d6", label="with costs")
    with_costs = summary[summary["costs"] == "with costs"].set_index("strategy")
    for yi, name in zip(y, names):
        row = with_costs.loc[name]
        ax.text(table.loc[name].max() + 0.002, yi,
                f"-{row['cost_drag_cagr'] * 100:.2f} pt/yr  ({int(row['trades'])} trades)",
                va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(y, names)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("CAGR")
    ax.set_title("Effect of commission and slippage on CAGR", loc="left")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "costs")


def main() -> None:
    git = git_state()
    config = load_config()
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    strategies = build_all(config, prices.columns)
    if len(strategies) > len(SERIES_COLORS):
        raise SystemExit(f"{len(strategies)} strategies but only {len(SERIES_COLORS)} distinguishable colors")
    start = prices.index[config["warmup_days"]]
    costs = CostModel(**config["costs"])
    scenarios = {"with costs": costs, "no costs": CostModel()}

    results = {}
    for strategy in strategies:
        for label, cost_model in scenarios.items():
            results[(strategy.name, label)] = run_backtest(
                prices, strategy, config["initial_cash"], cost_model, start=start)

    names = [s.name for s in strategies]
    colors = dict(zip(names, SERIES_COLORS))
    summary = summary_table(results, config, prices[config["risk_free_ticker"]])
    equities = pd.DataFrame({n: results[(n, "with costs")].equity for n in names})
    transactions = pd.concat(
        [results[(n, "with costs")].transactions.assign(strategy=n) for n in names], ignore_index=True)
    signals = pd.concat(
        [results[(n, "with costs")].signals.assign(strategy=n) for n in names], ignore_index=True).fillna(0.0)
    yearly = yearly_returns(equities)
    regimes = regime_returns(equities, config["benchmark"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.round(6).to_csv(OUT_DIR / "backtest_summary.csv", index=False)
    equities.round(4).to_csv(OUT_DIR / "equity_curves.csv")
    transactions.to_csv(OUT_DIR / "transactions.csv", index=False, float_format="%.10g")
    signals.to_csv(OUT_DIR / "signals.csv", index=False, float_format="%.10g")
    yearly.round(6).to_csv(OUT_DIR / "yearly_returns.csv")
    regimes.round(6).to_csv(OUT_DIR / "regime_returns.csv", index=False)
    (OUT_DIR / "run_info.json").write_text(
        json.dumps(run_info(config, prices, start, git), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    plot_equity(equities, colors, config["initial_cash"])
    plot_drawdowns(equities, colors)
    plot_allocation(results, prices)
    plot_yearly(yearly)
    plot_costs(summary, names)

    shown = summary[summary["costs"] == "with costs"].set_index("strategy")[
        ["cagr", "annual_volatility", "max_drawdown", "sharpe_ratio", "trades", "cost_drag_cagr",
         "excess_cagr_vs_benchmark"]]
    print(f"trading {start.date()} to {prices.index[-1].date()}, with costs {config['costs']}\n")
    print(shown.to_string(formatters={
        "cagr": "{:.2%}".format, "annual_volatility": "{:.2%}".format, "max_drawdown": "{:.2%}".format,
        "sharpe_ratio": "{:.2f}".format, "cost_drag_cagr": "{:.3%}".format,
        "excess_cagr_vs_benchmark": "{:+.2%}".format,
    }))
    print(f"\n{len(transactions)} transactions; saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
