"""Run Phase 3 DCA simulations for every configured asset."""
from pathlib import Path
import tomllib

import pandas as pd

from dca import load_prices, simulate_dca, simulate_lump_sum, summarize

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "simulation.toml"
OUT_DIR = ROOT / "reports" / "simulation"


def load_settings() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def main() -> None:
    amount = float(load_settings()["monthly_contribution"])
    prices = load_prices()

    summaries = []
    paths = []

    for ticker in prices.columns:
        series = prices[ticker].dropna()
        dca = simulate_dca(series, amount)
        lump = simulate_lump_sum(series, amount)
        summaries.append(summarize(dca, lump))
        paths.append(pd.DataFrame({
            f"{ticker}_contributed": dca.contributions,
            f"{ticker}_dca_value": dca.portfolio_value,
            f"{ticker}_lump_sum_value": lump,
        }))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUT_DIR / "dca_summary.csv", index=False)
    pd.concat(paths, axis=1).to_csv(OUT_DIR / "dca_paths.csv")

    print(f"saved {len(summaries)} asset summaries to {OUT_DIR}")


if __name__ == "__main__":
    main()
