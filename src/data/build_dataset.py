"""Build the unified comparison dataset data/processed/prices.csv.

Takes the adjusted Close of every asset in config/assets.toml and aligns them
on a common period: from the latest first date to the earliest last date among
all assets. Only dates on which every asset has a price are kept, so every
metric is computed over exactly the same days.

Output layout: single header row "Date,<ticker>,<ticker>,...", one row per day.
"""

import pandas as pd

from prices import PROCESSED_DIR, load_assets, load_raw

OUTPUT_PATH = PROCESSED_DIR / "prices.csv"


def build() -> pd.DataFrame:
    tickers = [asset["ticker"] for asset in load_assets()]
    closes = pd.concat({t: load_raw(t)["Close"] for t in tickers}, axis=1, sort=True)

    firsts = closes.apply(pd.Series.first_valid_index)
    lasts = closes.apply(pd.Series.last_valid_index)
    start, end = firsts.max(), lasts.min()
    print("available history per asset:")
    for t in tickers:
        print(f"  {t:<5} {firsts[t].date()} to {lasts[t].date()}")
    print(f"common period: {start.date()} ({firsts.idxmax()} starts last) to {end.date()}")

    common = closes.loc[start:end]
    gaps = common.isna().sum()
    if gaps.any():
        print(f"dates dropped because some asset has no price: {gaps[gaps > 0].to_dict()}")
    return common.dropna()


def main() -> None:
    prices = build()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUTPUT_PATH)
    print(f"saved {len(prices)} rows x {prices.shape[1]} assets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
