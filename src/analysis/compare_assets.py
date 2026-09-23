"""Compare every asset in config/assets.toml under identical conditions.

Input is data/processed/prices.csv (built by src/data/build_dataset.py), so all
assets share the same period and the same trading days. Outputs go to
reports/comparison/ as CSV tables and PNG figures.
"""

import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

import metrics as m

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "assets.toml"
PRICES_PATH = ROOT / "data" / "processed" / "prices.csv"
OUT_DIR = ROOT / "reports" / "comparison"
FIG_DIR = OUT_DIR / "figures"

# categorical slots in fixed order (validated for adjacent-pair CVD separation)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
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
    "axes.titlesize": 12,
    "axes.edgecolor": BASELINE,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.grid": True,
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


def load_prices() -> pd.DataFrame:
    return pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)


def summarize(prices: pd.DataFrame, assets: list[dict], window: int) -> pd.DataFrame:
    drawdown = m.drawdowns(prices)
    roll_ret = m.rolling_return(prices, window)
    roll_vol = m.rolling_volatility(prices, window)
    info = pd.DataFrame(assets).set_index("ticker")
    summary = pd.DataFrame({
        "name": info["name"],
        "category": info["category"],
        "start": prices.index[0].date(),
        "end": prices.index[-1].date(),
        "total_return": m.total_return(prices),
        "cagr": m.cagr(prices),
        "annual_volatility": m.annual_volatility(prices),
        "max_drawdown": drawdown.min(),
        "max_drawdown_date": drawdown.idxmin().dt.date,
        f"rolling_return_{window}d_min": roll_ret.min(),
        f"rolling_return_{window}d_median": roll_ret.median(),
        f"rolling_return_{window}d_max": roll_ret.max(),
        f"rolling_return_{window}d_share_positive": (roll_ret > 0).mean(),
        f"rolling_volatility_{window}d_min": roll_vol.min(),
        f"rolling_volatility_{window}d_median": roll_vol.median(),
        f"rolling_volatility_{window}d_max": roll_vol.max(),
    })
    summary.index.name = "ticker"
    return summary


def save_tables(prices: pd.DataFrame, summary: pd.DataFrame, window: int) -> dict:
    yearly = m.period_returns(prices, "YE")
    yearly.index = yearly.index.year.rename("year")
    monthly = m.period_returns(prices, "ME")
    monthly_by_label = monthly.copy()
    monthly_by_label.index = monthly.index.strftime("%Y-%m").rename("month")

    tables = {
        "summary_metrics": summary,
        "yearly_returns": yearly,
        "monthly_returns": monthly_by_label,
        f"rolling_return_{window}d": m.rolling_return(prices, window),
        f"rolling_volatility_{window}d": m.rolling_volatility(prices, window),
        "correlation_daily": m.correlation(m.daily_returns(prices)),
        "correlation_monthly": m.correlation(monthly),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.round(6).to_csv(OUT_DIR / f"{name}.csv")
    return tables


def save_figure(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=120)
    plt.close(fig)


def percent_axis(ax, axis: str = "y") -> None:
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(PercentFormatter(1.0))


def period_label(prices: pd.DataFrame) -> str:
    return f"{prices.index[0].date()} to {prices.index[-1].date()}"


def spread_labels(values: pd.Series, min_gap: float) -> pd.Series:
    """Push sorted label positions apart so neighbours are at least min_gap apart."""
    positions = values.sort_values().copy()
    for i in range(1, len(positions)):
        positions.iloc[i] = max(positions.iloc[i], positions.iloc[i - 1] + min_gap)
    return positions


def plot_growth(prices: pd.DataFrame, colors: dict) -> None:
    growth = prices / prices.iloc[0]
    fig, ax = plt.subplots(figsize=(12, 6))
    for t in prices.columns:
        ax.plot(growth.index, growth[t], color=colors[t], label=t)

    final = np.log10(growth.iloc[-1])
    span = np.log10(growth.max().max()) - np.log10(growth.min().min())
    label_y = spread_labels(final, min_gap=0.035 * span)
    for t in prices.columns:
        ax.annotate(
            f"{t} {growth[t].iloc[-1]:.2f}", (growth.index[-1], 10 ** label_y[t]),
            xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
            color=INK_SECONDARY, annotation_clip=False,
        )
    ax.set_yscale("log")
    ax.set_yticks([t for t in [0.5, 1, 2, 3, 5, 10, 20] if growth.min().min() <= t <= growth.max().max() * 1.1])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_title(f"Growth of 1 USD, log scale ({period_label(prices)})", loc="left")
    ax.set_ylabel("value of 1 USD invested")
    ax.legend(loc="upper left", ncols=len(prices.columns))
    fig.tight_layout()
    save_figure(fig, "cumulative_growth")


def plot_drawdowns(prices: pd.DataFrame, colors: dict) -> None:
    dd = m.drawdowns(prices)
    tickers = list(prices.columns)
    rows = int(np.ceil(len(tickers) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 2.4 * rows), sharex=True, sharey=True)
    for ax, t in zip(axes.flat, tickers):
        ax.fill_between(dd.index, dd[t], 0, color=colors[t], alpha=0.35, linewidth=0)
        ax.plot(dd.index, dd[t], color=colors[t], linewidth=1)
        ax.set_title(f"{t}  max {dd[t].min():.1%} ({dd[t].idxmin().date()})", loc="left", fontsize=10)
        percent_axis(ax)
    for ax in axes.flat[len(tickers):]:
        ax.set_visible(False)
    fig.suptitle(f"Drawdown from previous peak ({period_label(prices)})", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "drawdowns")


def plot_rolling(prices: pd.DataFrame, colors: dict, window: int) -> None:
    roll_ret = m.rolling_return(prices, window)
    roll_vol = m.rolling_volatility(prices, window)
    fig, (ax_ret, ax_vol) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for t in prices.columns:
        ax_ret.plot(roll_ret.index, roll_ret[t], color=colors[t], label=t)
        ax_vol.plot(roll_vol.index, roll_vol[t], color=colors[t], label=t)
    ax_ret.axhline(0, color=BASELINE, linewidth=1)
    ax_ret.set_title(f"Rolling {window}-day return (annualized)", loc="left")
    ax_vol.set_title(f"Rolling {window}-day volatility (annualized)", loc="left")
    for ax in (ax_ret, ax_vol):
        percent_axis(ax)
    ax_ret.legend(loc="upper left", ncols=len(prices.columns))
    fig.tight_layout()
    save_figure(fig, f"rolling_{window}d")


def draw_heatmap(ax, table: pd.DataFrame, vmax: float, fmt) -> None:
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
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.6 * vmax else INK)


def plot_correlation(tables: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, key, label in [
        (axes[0], "correlation_daily", "daily returns"),
        (axes[1], "correlation_monthly", "monthly returns"),
    ]:
        draw_heatmap(ax, tables[key], 1.0, lambda v: f"{v:.2f}")
        ax.set_title(f"Correlation of {label}", loc="left")
    fig.tight_layout()
    save_figure(fig, "correlation")


def plot_risk_return(summary: pd.DataFrame, colors: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    label_y = spread_labels(summary["cagr"], min_gap=0.05 * summary["cagr"].max())
    x_pad = 0.01 * summary["annual_volatility"].max()
    for t, row in summary.iterrows():
        ax.scatter(row["annual_volatility"], row["cagr"], s=80, color=colors[t],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(row["annual_volatility"] + x_pad, label_y[t],
                f"{t}  MDD {row['max_drawdown']:.1%}", va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=min(0, summary["cagr"].min() * 1.2))
    percent_axis(ax, "x")
    percent_axis(ax)
    ax.set_xlabel("annual volatility")
    ax.set_ylabel("CAGR")
    ax.set_title(f"Return vs risk ({summary['start'].iloc[0]} to {summary['end'].iloc[0]})", loc="left")
    fig.tight_layout()
    save_figure(fig, "risk_return")


def plot_yearly(tables: dict) -> None:
    yearly = tables["yearly_returns"]
    vmax = np.ceil(yearly.abs().max().max() * 10) / 10
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(yearly) + 1.5))
    draw_heatmap(ax, yearly, vmax, lambda v: f"{v:.1%}")
    ax.set_title(
        f"Yearly returns ({yearly.index[0]} and {yearly.index[-1]} are partial years)", loc="left"
    )
    fig.tight_layout()
    save_figure(fig, "yearly_returns")


def main() -> None:
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assets = config["assets"]
    window = config["analysis"]["rolling_window_days"]

    prices = load_prices()
    tickers = [a["ticker"] for a in assets]
    if list(prices.columns) != tickers:
        raise SystemExit(
            f"{PRICES_PATH} has {list(prices.columns)} but config lists {tickers}; "
            "run src/data/build_dataset.py first"
        )
    if len(tickers) > len(SERIES_COLORS):
        raise SystemExit(
            f"{len(tickers)} assets but only {len(SERIES_COLORS)} distinguishable colors; "
            "split the comparison into groups"
        )
    colors = dict(zip(tickers, SERIES_COLORS))

    summary = summarize(prices, assets, window)
    tables = save_tables(prices, summary, window)

    plot_growth(prices, colors)
    plot_drawdowns(prices, colors)
    plot_rolling(prices, colors, window)
    plot_correlation(tables)
    plot_risk_return(summary, colors)
    plot_yearly(tables)

    shown = summary[["total_return", "cagr", "annual_volatility", "max_drawdown", "max_drawdown_date"]]
    print(f"period: {period_label(prices)} ({len(prices)} trading days)\n")
    with pd.option_context("display.width", 200, "display.float_format", "{:.2%}".format):
        print(shown)
    print(f"\ncorrelation (monthly returns):\n{tables['correlation_monthly'].round(2)}")
    print(f"\ntables saved to {OUT_DIR}")
    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
