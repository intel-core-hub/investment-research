"""Phase 6: statistical evaluation of the Phase 5 backtests.

Re-runs every strategy in config/backtest.toml (with costs, same start date) and
writes the statistics below to reports/statistics/, plus figures and run_info.json.
"""
import json
import platform
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# backtest engine, strategies and provenance helpers from Phase 5
sys.path.insert(0, str(ROOT / "src" / "backtest"))

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

import inference as inf  # noqa: E402
import performance as pf  # noqa: E402
import regimes as rg  # noqa: E402
import trades as tr  # noqa: E402
from engine import CostModel, run_backtest  # noqa: E402
from provenance import content_sha256, git_state  # noqa: E402
from strategies import build_strategy  # noqa: E402

PRICES_PATH = ROOT / "data" / "processed" / "prices.csv"
BACKTEST_CONFIG = ROOT / "config" / "backtest.toml"
STATISTICS_CONFIG = ROOT / "config" / "statistics.toml"
OUT_DIR = ROOT / "reports" / "statistics"
FIG_DIR = OUT_DIR / "figures"

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
REGIME_COLORS = {"bull": "#1baf7a", "bear": "#e34948", "high_volatility": "#eb6834",
                 "low_volatility": "#86b6ef", "large_drawdown": "#4a3aa7"}
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


def load(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


# --- backtests -----------------------------------------------------------------------

def run_strategies(prices: pd.DataFrame, config: dict):
    specs = config["strategies"]
    strategies = [build_strategy(spec) for spec in specs]
    start = prices.index[config["warmup_days"]]
    costs = CostModel(**config["costs"])
    results = {s.name: run_backtest(prices, s, config["initial_cash"], costs, start=start) for s in strategies}
    kinds = {spec["name"]: spec["type"] for spec in specs}
    risk_off = {spec["name"]: spec.get("risk_off") for spec in specs}
    return results, kinds, risk_off, start


# --- tables --------------------------------------------------------------------------

def strategy_metrics(equities: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in equities.columns:
        e = pf.clean(equities[name])
        dd = pf.drawdown_stats(e)
        rows.append({
            "strategy": name, "start": e.index[0].date(), "end": e.index[-1].date(),
            "observations": len(e), "initial_equity": e.iloc[0], "final_equity": e.iloc[-1],
            "total_return": pf.total_return(e), "cagr": pf.cagr(e),
            "volatility_annualized": pf.volatility(pf.daily_returns(e)),
            "max_drawdown": dd["max_drawdown"],
            "max_drawdown_peak_date": _date(dd["peak_date"]),
            "max_drawdown_trough_date": _date(dd["trough_date"]),
            "recovery_date": _date(dd["recovery_date"]),
            "recovery_days": dd["recovery_days"], "underwater_days": dd["underwater_days"],
        })
    return pd.DataFrame(rows)


def _date(ts):
    return ts.date() if pd.notna(ts) else None


def risk_metrics(equities: pd.DataFrame, rf_prices: pd.Series) -> pd.DataFrame:
    rf = pf.daily_returns(rf_prices)
    rows = []
    for name in equities.columns:
        e = pf.clean(equities[name])
        r = pf.daily_returns(e)
        ex = pf.excess_returns(r, rf)
        rows.append({
            "strategy": name,
            "sharpe_ratio": pf.sharpe_ratio(r, rf),
            "sortino_ratio": pf.sortino_ratio(r, rf),
            "calmar_ratio": pf.calmar_ratio(e),
            "mean_excess_return_annualized": ex.mean() * pf.TRADING_DAYS,
            "excess_volatility_annualized": pf.volatility(ex),
            "downside_deviation_annualized": pf.downside_deviation(r, rf),
            "cagr": pf.cagr(e),
            "max_drawdown": pf.drawdown_stats(e)["max_drawdown"],
            "risk_free_cagr": pf.cagr(rf_prices.loc[e.index[0]:e.index[-1]]),
        })
    return pd.DataFrame(rows)


def by_frequency(equities: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "daily": pd.DataFrame({n: pf.daily_returns(equities[n]) for n in equities.columns}),
        "monthly": pd.DataFrame({n: pf.monthly_returns(equities[n]) for n in equities.columns}),
    }


def distribution_table(returns: dict) -> pd.DataFrame:
    return pd.DataFrame([{"strategy": n, "frequency": f, **pf.distribution(r[n])}
                         for f, r in returns.items() for n in r.columns])


def var_table(returns: dict, confidences: list) -> pd.DataFrame:
    return pd.DataFrame([{"strategy": n, "frequency": f, "confidence": c, **pf.historical_var_cvar(r[n], c)}
                         for f, r in returns.items() for n in r.columns for c in confidences])


def rolling_table(equities: pd.DataFrame, rf: pd.Series, windows: list, benchmark: str) -> pd.DataFrame:
    frames = []
    for w in windows:
        for name in equities.columns:
            table = pf.rolling_metrics(equities[name], rf, w, equities[benchmark])
            table.insert(0, "window_days", w)
            table.insert(0, "strategy", name)
            frames.append(table)
    out = pd.concat(frames)
    out.index.name = "date"
    return out.reset_index()


def statistical_tests(returns: pd.DataFrame, rf: pd.Series, benchmark: str, settings: dict) -> pd.DataFrame:
    confidence = settings["confidence_level"]
    boot = settings["bootstrap"]
    rows = []
    for name in returns.columns:
        for row in inf.mean_confidence_intervals(returns[name], confidence):
            rows.append({"strategy": name, **row})
        for row in inf.sharpe_inference(returns[name], rf, confidence):
            rows.append({"strategy": name, **row})
    for block in (1, boot["block_size"]):
        rows += inf.bootstrap(returns, rf, benchmark, boot["resamples"], boot["seed"], confidence, block)
    table = pd.DataFrame(rows)
    table["confidence"] = confidence
    table["p_value_holm"] = np.nan
    has_p = table["p_value"].notna()
    for _, group in table[has_p].groupby(["method", "statistic"]):
        table.loc[group.index, "p_value_holm"] = inf.holm_adjust(group["p_value"])
    order = ["strategy", "statistic", "method", "estimate", "standard_error", "ci_lower", "ci_upper",
             "confidence", "p_value", "p_value_holm", "observations", "resamples", "block_size", "seed", "lag"]
    return table[[c for c in order if c in table.columns]]


def regime_tables(equities, rf, reference_prices, settings: rg.RegimeSettings):
    labels = rg.classify(reference_prices, settings)
    return_days = pf.daily_returns(equities.iloc[:, 0]).index
    day_labels = rg.return_day_labels(labels, return_days)
    stats = pd.concat([rg.regime_stats(equities[n], rf, day_labels).assign(strategy=n)
                       for n in equities.columns], ignore_index=True)
    stats = stats[["strategy"] + [c for c in stats.columns if c != "strategy"]]
    return stats, rg.regime_periods(day_labels), labels.loc[equities.index[0]:]


def trade_tables(results: dict, kinds: dict, risk_off: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats, trips = [], []
    for name, result in results.items():
        stats += tr.strategy_trade_table(name, kinds[name], result.transactions, risk_off[name])
        if kinds[name] in tr.APPLICABLE_TYPES:
            closed, _ = tr.round_trips(result.transactions)
            trips.append(closed.assign(strategy=name))
    trips = pd.concat(trips, ignore_index=True) if trips else pd.DataFrame()
    if not trips.empty:
        trips = trips[["strategy"] + [c for c in trips.columns if c != "strategy"]]
    return pd.DataFrame(stats), trips


# --- figures -------------------------------------------------------------------------

def save_figure(fig, folder: str, name: str) -> None:
    path = FIG_DIR / folder
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"{name}.png", dpi=120)
    plt.close(fig)


def grid(n: int, height: float = 2.6):
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, height * rows + 0.8), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes


def plot_histograms(daily: pd.DataFrame, var: pd.DataFrame, colors: dict) -> None:
    lo, hi = np.nanpercentile(daily.to_numpy(), [0.2, 99.8])
    bins = np.linspace(lo, hi, 61)
    fig, axes = grid(daily.shape[1])
    for ax, name in zip(axes.flat, daily.columns):
        r = daily[name].dropna()
        ax.hist(r.clip(lo, hi), bins=bins, color=colors[name], alpha=0.8)
        for conf, style in [(0.95, (0, (4, 3))), (0.99, (0, (1, 2)))]:
            v = var[(var.strategy == name) & (var.frequency == "daily") & (var.confidence == conf)]["var"].iloc[0]
            ax.axvline(-v, color=INK_SECONDARY, linestyle=style, linewidth=1)
            ax.text(-v, ax.get_ylim()[1] * (0.9 if conf == 0.95 else 0.75), f" VaR{int(conf * 100)} {v:.1%}",
                    fontsize=8, color=INK_SECONDARY)
        ax.set_title(f"{name}  skew {r.skew():.2f}  excess kurtosis {r.kurt():.1f}", loc="left", fontsize=10)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    fig.suptitle("Distribution of daily returns (same bins; values beyond the edges are clipped into the edge bins)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "return_distribution", "daily_histograms")


def plot_percentiles(dist: pd.DataFrame, colors: dict, frequency: str) -> None:
    d = dist[dist.frequency == frequency].set_index("strategy")
    fig, ax = plt.subplots(figsize=(10, 0.8 * len(d) + 1.6))
    y = np.arange(len(d))[::-1]
    for yi, (name, row) in zip(y, d.iterrows()):
        c = colors[name]
        ax.plot([row.p01, row.p99], [yi, yi], color=c, linewidth=1)
        ax.plot([row.p05, row.p95], [yi, yi], color=c, linewidth=4, solid_capstyle="butt")
        ax.barh(yi, row.p75 - row.p25, left=row.p25, height=0.45, color=c, alpha=0.35)
        ax.plot(row.p50, yi, "o", color=c, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, d.index)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{frequency.capitalize()} return percentiles: thin line 1-99%, thick 5-95%, box 25-75%, dot median",
                 loc="left", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "return_distribution", f"percentiles_{frequency}")


def plot_rolling(rolling: pd.DataFrame, colors: dict, window: int) -> None:
    data = rolling[rolling.window_days == window]
    panels = [("rolling_return_annualized", "Return (annualized)", True),
              ("rolling_volatility_annualized", "Volatility (annualized)", True),
              ("rolling_sharpe", "Sharpe ratio", False),
              ("rolling_max_drawdown", "Max drawdown inside the window", True)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3 * len(panels)), sharex=True)
    for ax, (column, title, pct) in zip(axes, panels):
        for name, group in data.groupby("strategy", sort=False):
            ax.plot(group["date"], group[column], color=colors[name], label=name, linewidth=1.1)
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.set_title(title, loc="left", fontsize=11)
        if pct:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(loc="upper left", ncols=len(colors), fontsize=9)
    fig.suptitle(f"Rolling {window}-day statistics (each point uses the {window} trading days up to that date)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "rolling", f"rolling_{window}d")


def plot_drawdowns(equities: pd.DataFrame, colors: dict) -> None:
    fig, axes = grid(equities.shape[1])
    for ax, name in zip(axes.flat, equities.columns):
        dd = pf.drawdowns(equities[name])
        stats = pf.drawdown_stats(equities[name])
        ax.fill_between(dd.index, dd, 0, color=colors[name], alpha=0.35, linewidth=0)
        ax.plot(dd.index, dd, color=colors[name], linewidth=1)
        recovery = (f"recovered in {stats['recovery_days']:.0f} days" if pd.notna(stats["recovery_date"])
                    else "not recovered")
        ax.set_title(f"{name}  max {stats['max_drawdown']:.1%} on {_date(stats['trough_date'])}, {recovery}",
                     loc="left", fontsize=10)
        ax.axvspan(stats["peak_date"], stats["recovery_date"] if pd.notna(stats["recovery_date"])
                   else dd.index[-1], color=GRID, zorder=0)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.suptitle("Drawdown from previous peak (gray band: peak to recovery of the largest drawdown)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "drawdown", "drawdowns")


def plot_risk_return(metrics: pd.DataFrame, risk: pd.DataFrame, colors: dict) -> None:
    data = metrics.set_index("strategy").join(risk.set_index("strategy")[["sharpe_ratio", "sortino_ratio",
                                                                          "calmar_ratio"]])
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, row in data.iterrows():
        ax.scatter(row.volatility_annualized, row.cagr, s=90, color=colors[name], edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        ax.annotate(f"{name}\nSharpe {row.sharpe_ratio:.2f} / Sortino {row.sortino_ratio:.2f} / "
                    f"Calmar {row.calmar_ratio:.2f}", (row.volatility_annualized, row.cagr),
                    xytext=(8, -4), textcoords="offset points", fontsize=8.5, color=INK_SECONDARY, va="top")
    ax.set_xlim(0, data.volatility_annualized.max() * 1.35)
    ax.set_ylim(0, data.cagr.max() * 1.15)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("volatility (annualized)")
    ax.set_ylabel("CAGR")
    ax.set_title("Return vs risk after costs", loc="left")
    fig.tight_layout()
    save_figure(fig, "risk_return", "risk_return")


def plot_benchmark_comparison(comparison: pd.DataFrame, colors: dict, benchmark: str) -> None:
    metrics = list(dict.fromkeys(comparison.metric))
    percent = {"total_return", "cagr", "volatility_annualized", "max_drawdown"}
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, metric in zip(axes.flat, metrics):
        d = comparison[comparison.metric == metric]
        ax.bar(range(len(d)), d["difference"], color=[colors[s] for s in d.strategy], width=0.6)
        ax.axhline(0, color=INK_MUTED, linewidth=1)
        ax.set_xticks(range(len(d)), [s.replace(" ", "\n", 1) for s in d.strategy], fontsize=8)
        ax.set_title(metric, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
        if metric in percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.suptitle(f"Difference from {benchmark} (strategy minus benchmark; no judgement implied)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "risk_return", "benchmark_comparison")


def plot_correlation(corr: pd.DataFrame) -> None:
    kinds = [k for k in dict.fromkeys(corr.matrix) if k.startswith("correlation")]
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 5.6))
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        m = corr[corr.matrix == kind].set_index("series").drop(columns="matrix")
        ax.imshow(m.values, cmap=DIVERGING, vmin=-1, vmax=1)
        labels = [s.replace(" ", "\n", 1) for s in m.columns]
        ax.set_xticks(range(len(m)), labels, fontsize=8)
        ax.set_yticks(range(len(m)), labels, fontsize=8)
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(len(m)):
            for j in range(len(m)):
                v = m.iat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 else INK)
        ax.set_title(kind.replace("correlation_", "").replace("_", " "), loc="left", fontsize=11)
    fig.suptitle("Correlation of returns", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "correlation", "correlation")


def plot_regime_timeline(labels: pd.DataFrame, reference: pd.Series, name: str) -> None:
    fig, (ax, strips) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[3, 2])
    ax.plot(reference.index, reference, color=INK_SECONDARY, linewidth=1)
    ax.set_yscale("log")
    ax.set_yticks([t for t in [50, 100, 200, 300, 500, 700, 1000]
                   if reference.min() * 0.9 <= t <= reference.max() * 1.1])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_title(f"{name} price and regime labels (label at each close, past data only)", loc="left")
    for i, regime in enumerate(rg.REGIMES):
        on = labels[regime]
        for first, last in rg.spells(on):
            strips.axvspan(first, last, ymin=(len(rg.REGIMES) - 1 - i) / len(rg.REGIMES) + 0.02,
                           ymax=(len(rg.REGIMES) - i) / len(rg.REGIMES) - 0.02, color=REGIME_COLORS[regime])
    strips.set_yticks([(len(rg.REGIMES) - 0.5 - i) / len(rg.REGIMES) for i in range(len(rg.REGIMES))],
                      [r.replace("_", " ") for r in rg.REGIMES])
    strips.set_ylim(0, 1)
    strips.grid(False)
    fig.tight_layout()
    save_figure(fig, "regime", "regime_timeline")


def plot_regime_returns(stats: pd.DataFrame, colors: dict) -> None:
    names = list(dict.fromkeys(stats.strategy))
    panels = [("return_annualized", "Return (annualized) by regime"),
              ("volatility_annualized", "Volatility (annualized) by regime"),
              ("max_drawdown_within_spell", "Worst drawdown within a regime spell")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3.2 * len(panels)), sharex=True)
    x = np.arange(len(rg.REGIMES))
    width = 0.8 / len(names)
    for ax, (column, title) in zip(axes, panels):
        table = stats.pivot(index="regime", columns="strategy", values=column).loc[rg.REGIMES, names]
        for i, name in enumerate(names):
            ax.bar(x + (i - (len(names) - 1) / 2) * width, table[name], width * 0.92, color=colors[name], label=name)
        ax.axhline(0, color=INK_MUTED, linewidth=1)
        ax.set_title(title, loc="left", fontsize=11)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x", visible=False)
    days = stats.drop_duplicates("regime").set_index("regime")["trading_days"]
    axes[-1].set_xticks(x, [f"{r.replace('_', ' ')}\n({days[r]} days)" for r in rg.REGIMES])
    axes[0].legend(loc="upper right", ncols=len(names), fontsize=9)
    fig.tight_layout()
    save_figure(fig, "regime", "regime_returns")


def plot_bootstrap(tests: pd.DataFrame, colors: dict, benchmark: str) -> None:
    panels = [("sharpe_ratio", "Sharpe ratio", False),
              ("sharpe_ratio_difference_vs_benchmark", f"Sharpe ratio minus {benchmark}", False),
              ("cagr", "CAGR (geometric, from daily returns)", True),
              ("cagr_difference_vs_benchmark", f"CAGR minus {benchmark}", True)]
    methods = [("bootstrap_iid", "o"), ("bootstrap_block", "s")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax, (statistic, title, pct) in zip(axes.flat, panels):
        d = tests[(tests.statistic == statistic) & tests.method.isin([m for m, _ in methods])]
        names = list(dict.fromkeys(d.strategy))
        for i, name in enumerate(names):
            for k, (method, marker) in enumerate(methods):
                row = d[(d.strategy == name) & (d.method == method)].iloc[0]
                y = i + (k - 0.5) * 0.3
                ax.plot([row.ci_lower, row.ci_upper], [y, y], color=colors[name], linewidth=2)
                ax.plot(row.estimate, y, marker, color=colors[name], markersize=7, markeredgecolor=SURFACE)
        ax.axvline(0, color=INK_MUTED, linewidth=1)
        ax.set_yticks(range(len(names)), names, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11)
        if pct:
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    handles = [plt.Line2D([], [], marker=m, linestyle="-", color=INK_SECONDARY,
                          label="i.i.d. bootstrap" if meth == "bootstrap_iid" else "block bootstrap (21 days)")
               for meth, m in methods]
    fig.legend(handles=handles, loc="upper right", ncols=2, fontsize=9)
    fig.suptitle("95% bootstrap confidence intervals (point = estimate on the actual data)",
                 x=0.01, y=0.995, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "statistical", "bootstrap_intervals")


# --- main ----------------------------------------------------------------------------

def save(table: pd.DataFrame, name: str) -> None:
    table.to_csv(OUT_DIR / f"{name}.csv", index=False, float_format="%.10g")


def main() -> None:
    git = git_state(ROOT)
    bt_config, settings = load(BACKTEST_CONFIG), load(STATISTICS_CONFIG)
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    benchmark = bt_config["benchmark"]
    rf_prices = prices[bt_config["risk_free_ticker"]]
    rf = pf.daily_returns(rf_prices)

    results, kinds, risk_off, start = run_strategies(prices, bt_config)
    equities = pd.DataFrame({name: r.equity for name, r in results.items()})
    names = list(equities.columns)
    colors = dict(zip(names, SERIES_COLORS))

    returns = by_frequency(equities)
    reference = settings["regimes"]["reference"]
    market = prices.loc[equities.index, [reference, bt_config["risk_free_ticker"]]]
    all_daily = returns["daily"].join(pd.DataFrame({c: pf.daily_returns(market[c]) for c in market}))
    all_monthly = returns["monthly"].join(pd.DataFrame({c: pf.monthly_returns(market[c]) for c in market}))

    metrics = strategy_metrics(equities)
    risk = risk_metrics(equities, rf_prices)
    dist = distribution_table(returns)
    var = var_table(returns, settings["var_confidences"])
    rolling = rolling_table(equities, rf, settings["rolling_windows"], benchmark)
    comparison = pf.benchmark_comparison(equities, benchmark, rf)
    corr = pf.correlation_tables(all_daily, all_monthly, reference)
    regime_settings = rg.RegimeSettings(**{k: v for k, v in settings["regimes"].items() if k != "reference"})
    regime_stats, regime_periods, labels = regime_tables(equities, rf, prices[reference], regime_settings)
    tests = statistical_tests(returns["daily"], rf, benchmark, settings)
    trade_stats, trips = trade_tables(results, kinds, risk_off)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save(metrics, "strategy_metrics")
    save(risk, "risk_metrics")
    save(dist, "return_distribution")
    save(var, "var_cvar")
    save(rolling, "rolling_metrics")
    save(comparison, "benchmark_comparison")
    save(corr, "correlation")
    save(regime_stats, "regime_analysis")
    save(regime_periods, "regime_periods")
    save(tests, "statistical_tests")
    save(trade_stats, "trade_statistics")
    save(trips, "trades")

    plot_histograms(returns["daily"], var, colors)
    plot_percentiles(dist, colors, "daily")
    plot_percentiles(dist, colors, "monthly")
    for w in settings["rolling_windows"]:
        plot_rolling(rolling, colors, w)
    plot_drawdowns(equities, colors)
    plot_risk_return(metrics, risk, colors)
    plot_benchmark_comparison(comparison, colors, benchmark)
    plot_correlation(corr)
    plot_regime_timeline(labels, prices.loc[labels.index, reference], reference)
    plot_regime_returns(regime_stats, colors)
    plot_bootstrap(tests, colors, benchmark)

    info = {
        "prices_file": PRICES_PATH.relative_to(ROOT).as_posix(),
        "prices_sha256": content_sha256(PRICES_PATH),
        "backtest_config_file": BACKTEST_CONFIG.relative_to(ROOT).as_posix(),
        "backtest_config_sha256": content_sha256(BACKTEST_CONFIG),
        "statistics_config_file": STATISTICS_CONFIG.relative_to(ROOT).as_posix(),
        "statistics_config_sha256": content_sha256(STATISTICS_CONFIG),
        **git,
        "trading_start": str(start.date()),
        "end": str(equities.index[-1].date()),
        "daily_observations": int(returns["daily"].shape[0]),
        "monthly_observations": int(returns["monthly"].shape[0]),
        "bootstrap": settings["bootstrap"],
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    shown = metrics.set_index("strategy")[["cagr", "volatility_annualized", "max_drawdown"]].join(
        risk.set_index("strategy")[["sharpe_ratio", "sortino_ratio", "calmar_ratio"]])
    print(f"{start.date()} to {equities.index[-1].date()}, {returns['daily'].shape[0]} daily returns\n")
    print(shown.round(4).to_string())
    print(f"\nsaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
