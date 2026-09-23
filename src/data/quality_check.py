"""Run data quality checks on every raw price file listed in config/assets.toml.

Pass criteria (all must hold): no missing values, no duplicate dates,
no non-positive prices, no rows with High < Low.
Informational only: weekdays with no row (market holidays) and the largest
absolute daily move, which helps spot adjustment glitches.
"""

import pandas as pd

from prices import PRICE_COLUMNS, ROOT, load_assets, load_raw

REPORT_PATH = ROOT / "reports" / "comparison" / "data_quality.csv"


def check_quality(df: pd.DataFrame) -> dict:
    daily_move = df["Close"].pct_change().abs()
    result = {
        "rows": len(df),
        "start": df.index.min().date(),
        "end": df.index.max().date(),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_dates": int(df.index.duplicated().sum()),
        "non_positive_prices": int((df[PRICE_COLUMNS] <= 0).sum().sum()),
        "high_below_low": int((df["High"] < df["Low"]).sum()),
        "missing_weekdays": len(pd.bdate_range(df.index.min(), df.index.max()).difference(df.index)),
        "max_abs_daily_move": round(daily_move.max(), 4),
        "max_abs_daily_move_date": daily_move.idxmax().date(),
    }
    result["passed"] = (
        result["missing_values"] == 0
        and result["duplicate_dates"] == 0
        and result["non_positive_prices"] == 0
        and result["high_below_low"] == 0
    )
    return result


def main() -> None:
    results = pd.DataFrame(
        {asset["ticker"]: check_quality(load_raw(asset["ticker"])) for asset in load_assets()}
    ).T
    results.index.name = "ticker"

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_PATH)
    print(f"\nsaved to {REPORT_PATH}")

    failed = results.index[~results["passed"].astype(bool)].tolist()
    if failed:
        raise SystemExit(f"quality check failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
