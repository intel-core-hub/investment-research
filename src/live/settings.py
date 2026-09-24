"""Phase 9 settings and file locations.

Public defaults live in config/live.toml. Personal values go in
config/live.local.toml, which git ignores and which overrides the defaults key
by key. Real-money data lives under data/live/ (also git-ignored), so nothing
that identifies an account, an amount or a trade is ever committed.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "live.toml"
LOCAL_CONFIG_PATH = ROOT / "config" / "live.local.toml"


def merge(base: dict, override: dict) -> dict:
    """Recursive merge: tables are merged key by key, other values are replaced."""
    out = dict(base)
    for key, value in override.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_settings(path: Path = CONFIG_PATH, local_path: Path | None = LOCAL_CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        settings = tomllib.load(f)
    if local_path is not None and local_path.exists():
        with open(local_path, "rb") as f:
            settings = merge(settings, tomllib.load(f))
    return settings


@dataclass(frozen=True)
class LiveStore:
    """Every file of one live (or demo) account under `root`.

    transactions.csv is the only source of truth that a person edits; holdings,
    cash and valuations are recalculated from it and can always be regenerated.
    """
    root: Path

    @property
    def transactions(self) -> Path:
        return self.root / "transactions.csv"

    @property
    def paper_transactions(self) -> Path:
        return self.root / "paper_transactions.csv"

    @property
    def holdings(self) -> Path:
        return self.root / "holdings.csv"

    @property
    def cash(self) -> Path:
        return self.root / "cash.csv"

    @property
    def valuations(self) -> Path:
        return self.root / "valuations.csv"

    @property
    def paper_trading(self) -> Path:
        return self.root / "paper_trading.csv"

    @property
    def order_queue(self) -> Path:
        return self.root / "order_queue.csv"

    @property
    def audit_dir(self) -> Path:
        return self.root / "audit_logs"

    @property
    def decisions(self) -> Path:
        return self.audit_dir / "decisions.jsonl"

    @property
    def errors(self) -> Path:
        return self.audit_dir / "errors.jsonl"

    @property
    def emergency_stop(self) -> Path:
        return self.root / "EMERGENCY_STOP"

    @property
    def market_prices(self) -> Path:
        return self.root / "market" / "close.csv"

    @property
    def market_fx(self) -> Path:
        return self.root / "market" / "usdjpy.csv"

    def ensure(self) -> "LiveStore":
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "market").mkdir(parents=True, exist_ok=True)
        return self


def default_store(settings: dict) -> LiveStore:
    return LiveStore(ROOT / settings["data_dir"])
