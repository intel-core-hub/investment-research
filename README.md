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
