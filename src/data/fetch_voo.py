"""Fetch VOO daily price history from Yahoo Finance and save it as CSV.

Data spec (data/raw/voo.csv):
    source    Yahoo Finance via yfinance
    ticker    VOO (Vanguard S&P 500 ETF)
    period    full history (first trading day 2010-09-09 to latest completed day)
    interval  1 day, regular trading hours only
    prices    split- and dividend-adjusted (Close is total-return basis)
    columns   Close, High, Low, Open, Volume (USD, shares)
    layout    3 header rows: Price / Ticker / Date, then one row per trading day
    cleaning  rows with any missing Open/High/Low/Close are dropped;
              today's row is dropped when fetched before the 16:00 New York close
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

TICKER = "VOO"
PERIOD = "max"
INTERVAL = "1d"
AUTO_ADJUST = True
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
MARKET_TZ = "America/New_York"
MARKET_CLOSE_HOUR = 16
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "voo.csv"


def drop_unusable_rows(data: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    # Yahoo sometimes returns the latest day with volume but no prices
    unusable = data[PRICE_COLUMNS].isna().any(axis=1)

    if now.hour < MARKET_CLOSE_HOUR:
        today = now.tz_localize(None).normalize()
        unusable |= data.index >= today

    if unusable.any():
        dropped = ", ".join(str(d.date()) for d in data.index[unusable])
        print(f"Dropped {unusable.sum()} incomplete row(s): {dropped}")
    return data[~unusable]


def fetch_voo() -> None:
    data = yf.download(
        TICKER,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=AUTO_ADJUST,
        actions=False,
        prepost=False,
        multi_level_index=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"no data returned for {TICKER}; {OUTPUT_PATH} left unchanged")

    data = drop_unusable_rows(data, pd.Timestamp.now(tz=MARKET_TZ))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH)
    print(
        f"Saved {len(data)} rows ({data.index[0].date()} to {data.index[-1].date()}) "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    fetch_voo()
