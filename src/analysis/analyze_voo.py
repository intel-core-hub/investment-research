"""Compute performance metrics for VOO and plot price, cumulative return and drawdown."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "raw" / "voo.csv"
FIGURE_PATH = ROOT / "reports" / "figures" / "voo_analysis.png"
TRADING_DAYS = 252


def load_close(path: Path = CSV_PATH) -> pd.Series:
    # yfinance writes two extra header rows (Ticker, Date) below the column names
    df = pd.read_csv(path, skiprows=[1, 2], index_col=0, parse_dates=True)
    return df["Close"].rename("VOO")


def compute_metrics(close: pd.Series) -> dict:
    daily_returns = close.pct_change().dropna()
    cumulative = close / close.iloc[0] - 1
    years = (close.index[-1] - close.index[0]).days / 365.25
    drawdown = close / close.cummax() - 1
    return {
        "cumulative_return": cumulative.iloc[-1],
        "annual_return": (1 + cumulative.iloc[-1]) ** (1 / years) - 1,
        "annual_volatility": daily_returns.std() * np.sqrt(TRADING_DAYS),
        "max_drawdown": drawdown.min(),
        "max_drawdown_date": drawdown.idxmin(),
        "cumulative_series": cumulative,
        "drawdown_series": drawdown,
    }


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


if __name__ == "__main__":
    main()
