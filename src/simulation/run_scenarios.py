"""Compare DCA and lump sum across start dates, horizons and crash timing.

Writes CSV tables and figures to reports/simulation/.
"""
from pathlib import Path
import tomllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from dca import load_prices
from scenarios import (
    crash_start_runs,
    drawdown_episodes,
    rolling_windows,
    start_year_runs,
    window_stats,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "simulation.toml"
OUT_DIR = ROOT / "reports" / "simulation"
FIG_DIR = OUT_DIR / "figures"

DCA_COLOR = "#2a78d6"
LUMP_COLOR = "#eb6834"
FALL_SHADE = "#d6d5ce"
RECOVERY_SHADE = "#eceae4"
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


def load_settings() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def save_figure(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=120)
    plt.close(fig)


def shade_episodes(ax, episodes: pd.DataFrame, end: pd.Timestamp) -> None:
    for ep in episodes.itertuples():
        ax.axvspan(ep.peak, ep.trough, color=FALL_SHADE, linewidth=0, zorder=0)
        ax.axvspan(ep.trough, ep.recovery if pd.notna(ep.recovery) else end,
                   color=RECOVERY_SHADE, linewidth=0, zorder=0)


def plot_start_dates(windows: pd.DataFrame, episodes: pd.DataFrame, years: int, end: pd.Timestamp) -> None:
    data = windows[windows["horizon_years"] == years]
    tickers = list(data["ticker"].unique())
    rows = int(np.ceil(len(tickers) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 2.6 * rows + 0.8), sharex=True)
    for ax, t in zip(axes.flat, tickers):
        d = data[data["ticker"] == t]
        starts = pd.to_datetime(d["start"])
        shade_episodes(ax, episodes[episodes["ticker"] == t], end)
        ax.plot(starts, d["dca_return"], color=DCA_COLOR)
        ax.plot(starts, d["lump_sum_return"], color=LUMP_COLOR)
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        win = (d["dca_vs_lump_sum"] < 0).mean()
        ax.set_title(f"{t}  lump sum ahead in {win:.0%} of {len(d)} starts", loc="left", fontsize=10)
    starts = pd.to_datetime(data["start"])
    axes.flat[0].set_xlim(starts.min(), starts.max())
    for ax in axes.flat[len(tickers):]:
        ax.set_visible(False)
    fig.legend(
        handles=[
            plt.Line2D([], [], color=DCA_COLOR, label="DCA (monthly)"),
            plt.Line2D([], [], color=LUMP_COLOR, label="Lump sum (day 1)"),
            Patch(color=FALL_SHADE, label="peak to trough (drawdown >= threshold)"),
            Patch(color=RECOVERY_SHADE, label="trough to recovery"),
        ],
        loc="upper right", ncols=4, fontsize=9,
    )
    fig.suptitle(f"Return on total contributed by start date, {years}-year horizon",
                 x=0.01, y=0.995, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save_figure(fig, f"start_date_{years}y")


def plot_win_rate(stats: pd.DataFrame) -> None:
    table = stats.pivot(index="ticker", columns="horizon_years", values="lump_sum_win_rate")
    counts = stats.pivot(index="ticker", columns="horizon_years", values="windows")
    table = table.loc[stats["ticker"].unique()]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(table) + 1.6))
    ax.imshow(table.values, cmap=DIVERGING, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [f"{y}y" for y in table.columns])
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, t in enumerate(table.index):
        for j, y in enumerate(table.columns):
            v = table.loc[t, y]
            ax.text(j, i, f"{v:.0%}\n(n={counts.loc[t, y]})", ha="center", va="center", fontsize=9,
                    color="white" if abs(v - 0.5) > 0.3 else INK)
    ax.set_title("Share of start dates where lump sum beat DCA\n(blue: lump sum wins more, red: DCA wins more)",
                 loc="left", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "lump_sum_win_rate")


def save(table: pd.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.round(6).to_csv(OUT_DIR / f"{name}.csv", index=False)


def main() -> None:
    settings = load_settings()
    amount = float(settings["monthly_contribution"])
    horizons = list(settings["horizons_years"])
    threshold = float(settings["crash_threshold"])
    prices = load_prices()

    windows, by_year, episodes, crash = [], [], [], []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        windows.append(rolling_windows(series, horizons, amount))
        by_year.append(start_year_runs(series, amount))
        ep = drawdown_episodes(series, threshold)
        episodes.append(ep)
        crash.append(crash_start_runs(series, ep, horizons, amount))

    windows = pd.concat(windows, ignore_index=True)
    stats = window_stats(windows)
    episodes = pd.concat(episodes, ignore_index=True)

    save(windows, "dca_windows")
    save(stats, "dca_window_stats")
    save(pd.concat(by_year, ignore_index=True), "dca_by_start_year")
    save(episodes.assign(**{c: episodes[c].dt.date for c in ["peak", "trough", "recovery"]}),
         "drawdown_episodes")
    save(pd.concat(crash, ignore_index=True), "dca_crash_starts")

    for years in horizons:
        plot_start_dates(windows, episodes, years, prices.index[-1])
    plot_win_rate(stats)

    shown = stats.set_index(["ticker", "horizon_years"])[
        ["windows", "lump_sum_win_rate", "dca_vs_lump_sum_median", "dca_loss_rate", "lump_sum_loss_rate"]
    ]
    with pd.option_context("display.width", 200, "display.float_format", "{:.1%}".format):
        print(shown)
    print(f"\n{len(windows)} windows, {len(episodes)} crash episodes; saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
