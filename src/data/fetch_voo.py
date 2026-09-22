"""Fetch VOO historical price data via yfinance and save it as CSV."""

from pathlib import Path

import yfinance as yf

TICKER = "VOO"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "voo.csv"


def fetch_voo(period: str = "max", interval: str = "1d") -> None:
    data = yf.download(TICKER, period=period, interval=interval)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH)
    print(f"Saved {len(data)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    fetch_voo()
