"""Fetch every asset listed in config/assets.toml into data/raw/<ticker>.csv."""

from prices import fetch_and_save, load_assets


def main() -> None:
    failed = []
    for asset in load_assets():
        try:
            fetch_and_save(asset["ticker"])
        except RuntimeError as e:
            print(f"{asset['ticker']}: FAILED ({e}); existing file left unchanged")
            failed.append(asset["ticker"])
    if failed:
        raise SystemExit(f"fetch failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
