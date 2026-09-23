"""Compute performance metrics for VOO and plot price, cumulative return and drawdown."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import metrics as m

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "raw" / "voo.csv"
REPORTS_DIR = ROOT / "reports"
FIGURE_PATH = REPORTS_DIR / "figures" / "voo_analysis.png"
YEARLY_FIGURE_PATH = REPORTS_DIR / "figures" / "voo_yearly_returns.png"
YEARLY_CSV_PATH = REPORTS_DIR / "voo_yearly_returns.csv"
MONTHLY_CSV_PATH = REPORTS_DIR / "voo_monthly_returns.csv"


def load_close(path: Path = CSV_PATH) -> pd.Series:
    # yfinance writes two extra header rows (Ticker, Date) below the column names
    df = pd.read_csv(path, skiprows=[1, 2], index_col=0, parse_dates=True)
    return df["Close"].rename("VOO")


def compute_metrics(close: pd.Series) -> dict:
    drawdown = m.drawdowns(close)
    return {
        "cumulative_return": m.total_return(close),
        "annual_return": m.cagr(close),
        "annual_volatility": m.annual_volatility(close),
        "max_drawdown": drawdown.min(),
        "max_drawdown_date": drawdown.idxmin(),
        "cumulative_series": m.cumulative_returns(close),
        "drawdown_series": drawdown,
    }


def period_returns(close: pd.Series, freq: str) -> pd.Series:
    return m.period_returns(close, freq).rename("return")


def save_period_returns(yearly: pd.Series, monthly: pd.Series) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    yearly_out = yearly.round(6)
    yearly_out.index = yearly_out.index.year.rename("year")
    yearly_out.to_csv(YEARLY_CSV_PATH)
    monthly_out = monthly.round(6)
    monthly_out.index = monthly_out.index.strftime("%Y-%m").rename("month")
    monthly_out.to_csv(MONTHLY_CSV_PATH)


def plot_yearly(yearly: pd.Series, path: Path = YEARLY_FIGURE_PATH) -> None:
    years = yearly.index.year
    values = yearly * 100
    colors = ["tab:green" if v >= 0 else "tab:red" for v in values]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(years.astype(str), values, color=colors)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=2, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(
        f"VOO Yearly Returns ({years[0]} and {years[-1]} are partial years)"
    )
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot(close: pd.Series, metrics: dict, path: Path = FIGURE_PATH) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(close.index, close, color="tab:blue")
    axes[0].set_title("VOO Price (adjusted close)")
    axes[0].set_ylabel("USD")

    axes[1].plot(close.index, metrics["cumulative_series"] * 100, color="tab:green")
    axes[1].set_title("Cumulative Return")
    axes[1].set_ylabel("%")

    dd = metrics["drawdown_series"] * 100
    axes[2].fill_between(dd.index, dd, 0, color="tab:red", alpha=0.4)
    axes[2].set_title("Drawdown")
    axes[2].set_ylabel("%")

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    close = load_close()
    metrics = compute_metrics(close)

    print(f"period: {close.index[0].date()} to {close.index[-1].date()}")
    print(f"cumulative return: {metrics['cumulative_return']:.2%}")
    print(f"annual return (CAGR): {metrics['annual_return']:.2%}")
    print(f"annual volatility: {metrics['annual_volatility']:.2%}")
    print(
        f"max drawdown: {metrics['max_drawdown']:.2%} "
        f"(on {metrics['max_drawdown_date'].date()})"
    )

    plot(close, metrics)
    print(f"figure saved to {FIGURE_PATH}")

    yearly = period_returns(close, "YE")
    monthly = period_returns(close, "ME")
    save_period_returns(yearly, monthly)
    plot_yearly(yearly)
    print(f"yearly returns saved to {YEARLY_CSV_PATH}")
    print(f"monthly returns saved to {MONTHLY_CSV_PATH}")
    print(f"figure saved to {YEARLY_FIGURE_PATH}")


if __name__ == "__main__":
    main()
