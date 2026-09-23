"""Fetch VOO only into data/raw/voo.csv (see prices.py for the data spec)."""

from prices import fetch_and_save

if __name__ == "__main__":
    fetch_and_save("VOO")
