"""Compare the portfolios in config/portfolios.toml under identical conditions.

Every portfolio runs over the same common period (data/processed/prices.csv),
with the same money, for every rebalance frequency, as a lump sum and as DCA.
Writes CSV tables and figures to reports/portfolio/.
"""
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# shared metric and drawdown definitions live in the analysis and simulation packages
sys.path[:0] = [str(ROOT / "src" / "analysis"), str(ROOT / "src" / "simulation")]

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

import metrics as m  # noqa: E402
from portfolio import (  # noqa: E402
    PortfolioConfigError,
    daily_returns,
    diversification,
    nav,
    simulate_portfolio,
    summarize,
    validate_config,
)

CONFIG_PATH = ROOT / "config" / "portfolios.toml"
PRICES_PATH = ROOT / "data" / "processed" / "prices.csv"
OUT_DIR = ROOT / "reports" / "portfolio"
FIG_DIR = OUT_DIR / "figures"

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
# same color per asset as in the Phase 2 comparison
ASSET_COLORS = {"VOO": "#2a78d6", "VTI": "#eb6834", "VT": "#1baf7a", "EWJ": "#eda100",
                "BND": "#e87ba4", "BIL": "#008300"}
# ordinal blue ramp: no rebalancing (lightest) to monthly (darkest)
FREQUENCY_COLORS = {"none": "#86b6ef", "annual": "#3987e5", "quarterly": "#256abf", "monthly": "#184f95"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

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


def format_weights(weights: pd.Series) -> str:
    return " / ".join(f"{t} {w:.0%}" for t, w in weights.items())


def run_all(prices: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    risk_free = prices[config["risk_free_ticker"]]
    modes = {
        "lump_sum": {"initial_investment": config["initial_investment"]},
        "dca": {"monthly_contribution": config["monthly_contribution"]},
    }
    rows, results = [], {}
    for p in config["portfolios"]:
        for freq in config["rebalance_frequencies"]:
            for mode, money in modes.items():
                result = simulate_portfolio(prices, p["weights"], rebalance=freq, **money)
                results[(p["name"], freq, mode)] = result
                rows.append({
                    "portfolio": p["name"],
                    "target_weights": format_weights(pd.Series(p["weights"])),
                    "rebalance": freq,
                    "mode": mode,
                    **summarize(result, risk_free),
                    "final_weights": format_weights(result.weights(prices).iloc[-1]),
                })
    return pd.DataFrame(rows), results


def diversification_table(prices: pd.DataFrame, config: dict, summary: pd.DataFrame) -> pd.DataFrame:
    cov = m.daily_returns(prices).cov() * m.TRADING_DAYS
    realized = summary[(summary["mode"] == "lump_sum") & (summary["rebalance"] == config["default_rebalance"])]
    realized = realized.set_index("portfolio")["annual_volatility"]
    rows = []
    for p in config["portfolios"]:
        rows.append({
            "portfolio": p["name"],
            "target_weights": format_weights(pd.Series(p["weights"])),
            **diversification(p["weights"], cov),
            f"realized_volatility_{config['default_rebalance']}_rebalance": realized[p["name"]],
        })
    return pd.DataFrame(rows)


def save_figure(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=120)
    plt.close(fig)


def spread_labels(values: pd.Series, min_gap: float) -> pd.Series:
    positions = values.sort_values().copy()
    for i in range(1, len(positions)):
        positions.iloc[i] = max(positions.iloc[i], positions.iloc[i - 1] + min_gap)
    return positions


def period_label(index: pd.DatetimeIndex) -> str:
    return f"{index[0].date()} to {index[-1].date()}"


def plot_growth(navs: pd.DataFrame, colors: dict, rebalance: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in navs.columns:
        ax.plot(navs.index, navs[name], color=colors[name], label=name)
    final = np.log10(navs.iloc[-1])
    span = np.log10(navs.max().max()) - np.log10(navs.min().min())
    label_y = spread_labels(final, 0.035 * span)
    for name in navs.columns:
        ax.annotate(f"{name} {navs[name].iloc[-1]:.2f}", (navs.index[-1], 10 ** label_y[name]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                    color=INK_SECONDARY, annotation_clip=False)
    ax.set_yscale("log")
    ax.set_yticks([t for t in [1, 2, 3, 5, 10] if navs.min().min() <= t <= navs.max().max() * 1.1])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylabel("value of 1 invested")
    ax.set_title(f"Growth of 1, {rebalance} rebalancing, log scale ({period_label(navs.index)})", loc="left")
    ax.legend(loc="upper left", ncols=3)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    save_figure(fig, "growth")


def plot_drawdowns(navs: pd.DataFrame, colors: dict, rebalance: str) -> None:
    dd = m.drawdowns(navs)
    names = list(navs.columns)
    rows = int(np.ceil(len(names) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 2.4 * rows + 0.6), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, names):
        ax.fill_between(dd.index, dd[name], 0, color=colors[name], alpha=0.35, linewidth=0)
        ax.plot(dd.index, dd[name], color=colors[name], linewidth=1)
        ax.set_title(f"{name}  max {dd[name].min():.1%} ({dd[name].idxmin().date()})", loc="left", fontsize=10)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    fig.suptitle(f"Drawdown from previous peak, {rebalance} rebalancing", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "drawdowns")


def plot_risk_return(summary: pd.DataFrame, prices: pd.DataFrame, colors: dict, config: dict) -> None:
    rebalance = config["default_rebalance"]
    rows = summary[(summary["mode"] == "lump_sum") & (summary["rebalance"] == rebalance)].set_index("portfolio")
    assets = pd.DataFrame({
        "annual_volatility": m.annual_volatility(prices),
        "cagr": m.cagr(prices),
    })
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.scatter(assets["annual_volatility"], assets["cagr"], s=50, facecolor=SURFACE,
               edgecolor=INK_MUTED, linewidth=1.2, zorder=2)
    for t, row in assets.iterrows():
        ax.text(row["annual_volatility"], row["cagr"] - 0.006, t, ha="center", va="top",
                fontsize=8, color=INK_MUTED)
    label_y = spread_labels(rows["cagr"], 0.045 * rows["cagr"].max())
    x_pad = 0.008 * assets["annual_volatility"].max()
    for name, row in rows.iterrows():
        ax.scatter(row["annual_volatility"], row["cagr"], s=80, color=colors[name],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(row["annual_volatility"] + x_pad, label_y[name],
                f"{name}  Sharpe {row['sharpe_ratio']:.2f}  MDD {row['max_drawdown']:.1%}",
                va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("annual volatility")
    ax.set_ylabel("CAGR")
    ax.set_title(f"Portfolios ({rebalance} rebalancing) vs single assets (gray), "
                 f"{period_label(prices.index)}", loc="left")
    fig.tight_layout()
    save_figure(fig, "risk_return")


def plot_rebalancing(summary: pd.DataFrame, config: dict) -> None:
    data = summary[summary["mode"] == "lump_sum"]
    names = [p["name"] for p in config["portfolios"]]
    freqs = config["rebalance_frequencies"]
    metrics_to_show = [("cagr", "CAGR"), ("annual_volatility", "Annual volatility"),
                       ("max_drawdown", "Max drawdown"), ("sharpe_ratio", "Sharpe ratio")]
    fig, axes = plt.subplots(len(metrics_to_show), 1, figsize=(12, 3 * len(metrics_to_show)), sharex=True)
    x = np.arange(len(names))
    width = 0.8 / len(freqs)
    for ax, (column, title) in zip(axes, metrics_to_show):
        table = data.pivot(index="portfolio", columns="rebalance", values=column).loc[names, freqs]
        for i, freq in enumerate(freqs):
            ax.bar(x + (i - (len(freqs) - 1) / 2) * width, table[freq], width * 0.92,
                   color=FREQUENCY_COLORS[freq], label=freq)
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.set_title(title, loc="left", fontsize=11)
        if column != "sharpe_ratio":
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x", visible=False)
    axes[-1].set_xticks(x, names)
    axes[0].legend(title="rebalancing", loc="upper right", ncols=len(freqs), fontsize=9)
    fig.suptitle("Effect of rebalancing frequency (lump sum)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "rebalancing")


def plot_diversification(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(table) + 1.4))
    y = np.arange(len(table))[::-1]
    ax.barh(y + 0.18, table["weighted_average_volatility"], 0.34, color=BASELINE,
            label="if all assets were perfectly correlated")
    ax.barh(y - 0.18, table["portfolio_volatility"], 0.34, color="#2a78d6",
            label="actual mix (constant weights)")
    for yi, (_, row) in zip(y, table.iterrows()):
        if row["volatility_reduction"] > 0.001:
            ax.text(row["weighted_average_volatility"] + 0.002, yi + 0.18,
                    f"-{row['volatility_reduction']:.0%} from diversification",
                    va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(y, table["portfolio"])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("annual volatility")
    ax.set_title("Diversification: volatility of the mix vs weighted average of asset volatilities", loc="left")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "diversification")


def plot_weight_drift(results: dict, prices: pd.DataFrame, config: dict) -> None:
    multi = [p for p in config["portfolios"] if len(p["weights"]) > 1]
    if not multi:
        return
    rows = int(np.ceil(len(multi) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 2.8 * rows + 0.8), sharex=True, sharey=True, squeeze=False)
    for ax, p in zip(axes.flat, multi):
        held = results[(p["name"], "none", "lump_sum")].weights(prices)
        for t, target in p["weights"].items():
            color = ASSET_COLORS.get(t, INK_MUTED)
            ax.plot(held.index, held[t], color=color, label=t)
            ax.axhline(target, color=color, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(f"{p['name']}: buy & hold weights (dashed = target)", loc="left", fontsize=10)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(loc="upper left", ncols=len(p["weights"]), fontsize=9)
    for ax in axes.flat[len(multi):]:
        ax.set_visible(False)
    axes.flat[0].set_ylim(0, 1)
    fig.suptitle("How weights drift without rebalancing", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "weight_drift")


def plot_dca(results: dict, config: dict, colors: dict) -> None:
    rebalance = config["default_rebalance"]
    fig, ax = plt.subplots(figsize=(12, 6))
    contributed = None
    for p in config["portfolios"]:
        result = results[(p["name"], rebalance, "dca")]
        ax.plot(result.value.index, result.value, color=colors[p["name"]], label=p["name"])
        contributed = result.contributions
    ax.plot(contributed.index, contributed, color=INK_MUTED, linestyle=(0, (4, 3)), label="total contributed")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M" if v >= 1e6 else f"{v:,.0f}"))
    ax.set_ylabel("portfolio value")
    ax.set_title(f"DCA of {config['monthly_contribution']:,} per month, {rebalance} rebalancing", loc="left")
    ax.legend(loc="upper left", ncols=4)
    fig.tight_layout()
    save_figure(fig, "dca_growth")


def main() -> None:
    config = load_config()
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    try:
        validate_config(config, prices.columns)
    except PortfolioConfigError as e:
        raise SystemExit(str(e))
    names = [p["name"] for p in config["portfolios"]]
    if len(names) > len(SERIES_COLORS):
        raise SystemExit(f"{len(names)} portfolios but only {len(SERIES_COLORS)} distinguishable colors")
    colors = dict(zip(names, SERIES_COLORS))
    rebalance = config["default_rebalance"]

    summary, results = run_all(prices, config)
    diversification_rows = diversification_table(prices, config, summary)

    lump = {n: results[(n, rebalance, "lump_sum")] for n in names}
    navs = pd.DataFrame({n: nav(r) for n, r in lump.items()})
    returns = pd.DataFrame({n: daily_returns(r) for n, r in lump.items()})
    dca_values = pd.DataFrame({n: results[(n, rebalance, "dca")].value for n in names})
    dca_values["total_contributed"] = results[(names[0], rebalance, "dca")].contributions

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.round(6).to_csv(OUT_DIR / "portfolio_summary.csv", index=False)
    diversification_rows.round(6).to_csv(OUT_DIR / "portfolio_diversification.csv", index=False)
    navs.round(6).to_csv(OUT_DIR / "portfolio_nav.csv")
    returns.round(8).to_csv(OUT_DIR / "portfolio_daily_returns.csv")
    dca_values.round(2).to_csv(OUT_DIR / "portfolio_dca_values.csv")

    plot_growth(navs, colors, rebalance)
    plot_drawdowns(navs, colors, rebalance)
    plot_risk_return(summary, prices, colors, config)
    plot_rebalancing(summary, config)
    plot_diversification(diversification_rows)
    plot_weight_drift(results, prices, config)
    plot_dca(results, config, colors)

    shown = summary[(summary["mode"] == "lump_sum") & (summary["rebalance"] == rebalance)].set_index("portfolio")
    shown = shown[["cagr", "annual_volatility", "max_drawdown", "sharpe_ratio", "final_weights"]]
    print(f"period: {period_label(prices.index)}, {rebalance} rebalancing, lump sum\n")
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(shown.to_string(formatters={
            "cagr": "{:.2%}".format, "annual_volatility": "{:.2%}".format,
            "max_drawdown": "{:.2%}".format, "sharpe_ratio": "{:.2f}".format,
        }))
    print(f"\n{len(summary)} runs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
