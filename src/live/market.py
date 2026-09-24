"""Latest market data for the live bridge, kept apart from the fixed research data.

The research data (data/raw, data/processed) stays exactly as the Phase 1-8 reports
used it. The live bridge downloads its own copy into data/live/market/:
  close.csv     actual daily closes in USD, one column per ticker: used for order
                sizes and valuation (real holdings receive dividends as cash, which
                the ledger records, so adjusted prices would count them twice)
  adjusted.csv  dividend-adjusted closes: used for signals, as in the research
  volume.csv    daily volume (the ML features use the target's volume)
  usdjpy.csv    USD/JPY daily close (Yahoo Finance ticker JPY=X)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src" / "data") not in sys.path:
    sys.path.insert(0, str(ROOT / "src" / "data"))

from settings import LiveStore  # noqa: E402

FX_TICKER = "JPY=X"
FILES = {"close": "close.csv", "adjusted": "adjusted.csv", "volume": "volume.csv"}


@dataclass(frozen=True)
class MarketData:
    close: pd.DataFrame       # actual closes (USD)
    adjusted: pd.DataFrame    # dividend-adjusted closes (USD)
    volume: pd.DataFrame
    fx: pd.Series             # JPY per USD

    @property
    def as_of(self) -> pd.Timestamp:
        return self.close.index[-1]

    def fx_on(self, dates: pd.DatetimeIndex) -> pd.Series:
        """FX rate on each date (the latest known rate on dates without an FX quote)."""
        return self.fx.sort_index().reindex(self.fx.index.union(dates)).ffill().loc[dates]


def fetch(tickers: list[str]) -> MarketData:
    """Download every ticker and USD/JPY from Yahoo Finance (needs the internet)."""
    import yfinance as yf

    import prices as rawdata

    now = pd.Timestamp.now(tz=rawdata.MARKET_TZ)
    frames = {}
    for ticker in tickers:
        data = yf.download(ticker, period="max", interval="1d", auto_adjust=False, actions=False, prepost=False,
                           multi_level_index=False, progress=False)
        if data.empty:
            raise RuntimeError(f"no data returned for {ticker}")
        frames[ticker] = rawdata.drop_unusable_rows(data, now)
    fx = yf.download(FX_TICKER, period="max", interval="1d", auto_adjust=False, actions=False,
                     multi_level_index=False, progress=False)["Close"].dropna()
    close = pd.DataFrame({t: f["Close"] for t, f in frames.items()}).dropna()  # days every ticker traded
    adjusted = pd.DataFrame({t: f["Adj Close"] for t, f in frames.items()}).reindex(close.index)
    volume = pd.DataFrame({t: f["Volume"] for t, f in frames.items()}).reindex(close.index)
    return MarketData(close, adjusted, volume, fx.rename("usdjpy"))


def save(store: LiveStore, market: MarketData) -> None:
    store.ensure()
    folder = store.market_prices.parent
    for name, file in FILES.items():
        getattr(market, name).to_csv(folder / file, index_label="Date")
    market.fx.to_frame("usdjpy").to_csv(store.market_fx, index_label="Date")


def load(store: LiveStore) -> MarketData | None:
    """Saved live market data, or None if it has not been downloaded yet."""
    folder = store.market_prices.parent
    paths = {name: folder / file for name, file in FILES.items()}
    if not all(p.exists() for p in paths.values()) or not store.market_fx.exists():
        return None
    frames = {name: pd.read_csv(p, index_col=0, parse_dates=True) for name, p in paths.items()}
    fx = pd.read_csv(store.market_fx, index_col=0, parse_dates=True)["usdjpy"]
    return MarketData(frames["close"], frames["adjusted"], frames["volume"], fx)
