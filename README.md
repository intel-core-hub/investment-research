# investment-research

market data collection and analysis for investment research.

## setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## usage

```
python src/data/fetch_voo.py
```

fetches VOO historical price data and saves it to `data/raw/voo.csv`.

```
python src/data/quality_check.py
python src/analysis/analyze_voo.py
```

`quality_check.py` checks the CSV for missing values, duplicates and invalid prices.
`analyze_voo.py` prints cumulative return, CAGR, annual volatility and max drawdown,
and saves a chart to `reports/figures/voo_analysis.png`.
