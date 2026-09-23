"""Shared price-data access: asset config, Yahoo Finance fetch and raw CSV I/O.

Data spec (data/raw/<ticker>.csv, one file per asset):
    source    Yahoo Finance via yfinance
    period    full history (first trading day to latest completed day)
    interval  1 day, regular trading hours only
    prices    split- and dividend-adjusted (Close is total-return basis)
    columns   Close, High, Low, Open, Volume (listing currency, shares)
    layout    3 header rows: Price / Ticker / Date, then one row per trading day
    cleaning  rows with any missing Open/High/Low/Close are dropped;
              today's row is dropped when fetched before the 16:00 New York close
"""

import tomllib
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "assets.toml"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PERIOD = "max"
INTERVAL = "1d"
AUTO_ADJUST = True
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
MARKET_TZ = "America/New_York"
MARKET_CLOSE_HOUR = 16


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_assets(path: Path = CONFIG_PATH) -> list[dict]:
    return load_config(path)["assets"]


def raw_path(ticker: str) -> Path:
    return RAW_DIR / f"{ticker.lower()}.csv"


def drop_unusable_rows(data: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    # Yahoo sometimes returns the latest day with volume but no prices
    unusable = data[PRICE_COLUMNS].isna().any(axis=1)

    if now.hour < MARKET_CLOSE_HOUR:
        today = now.tz_localize(None).normalize()
        unusable |= data.index >= today

    if unusable.any():
        dropped = ", ".join(str(d.date()) for d in data.index[unusable])
        print(f"  dropped {unusable.sum()} incomplete row(s): {dropped}")
    return data[~unusable]


def fetch(ticker: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=AUTO_ADJUST,
        actions=False,
        prepost=False,
        multi_level_index=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"no data returned for {ticker}")
    return drop_unusable_rows(data, pd.Timestamp.now(tz=MARKET_TZ))


def fetch_and_save(ticker: str) -> Path:
    data = fetch(ticker)
    path = raw_path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path)
    print(
        f"{ticker}: saved {len(data)} rows "
        f"({data.index[0].date()} to {data.index[-1].date()}) to {path}"
    )
    return path


def load_raw(ticker: str) -> pd.DataFrame:
    # skip the Ticker and Date header rows that yfinance writes below the column names
    df = pd.read_csv(raw_path(ticker), skiprows=[1, 2], index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df
