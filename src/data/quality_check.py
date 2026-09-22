"""Load data/raw/voo.csv and run basic data quality checks."""

from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "voo.csv"


def load_voo(path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=[1, 2], index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def check_quality(df: pd.DataFrame) -> None:
    print(f"rows: {len(df)}")
    print(f"date range: {df.index.min().date()} to {df.index.max().date()}")

    missing = df.isna().sum()
    print("\nmissing values per column:")
    print(missing[missing > 0] if missing.any() else "  none")

    dup_dates = df.index.duplicated().sum()
    print(f"\nduplicate dates: {dup_dates}")

    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    non_positive = (df[price_cols] <= 0).sum()
    print("\nnon-positive prices per column:")
    print(non_positive[non_positive > 0] if non_positive.any() else "  none")

    bad_hl = (df["High"] < df["Low"]).sum() if {"High", "Low"}.issubset(df.columns) else 0
    print(f"\nrows where High < Low: {bad_hl}")

    business_days = pd.bdate_range(df.index.min(), df.index.max())
    missing_business_days = business_days.difference(df.index)
    print(f"\nmissing business days in range: {len(missing_business_days)}")


if __name__ == "__main__":
    voo = load_voo()
    check_quality(voo)
