"""9.1 Currency conversion, trading costs and Japanese taxes for USD-listed ETFs bought from a JPY account.

Conventions
- `fx_rate` is JPY per 1 USD. Assets quoted in JPY use 1.0.
- A broker converts at the mid rate plus a spread: mid + spread when buying USD
  (TTS), mid - spread when selling USD (TTB).
- ETF expense ratios are already deducted from ETF prices (and from the adjusted
  prices used in Phases 1-8). They are reported, never subtracted a second time.
- Taxes as implemented here (check your own situation; rules can change):
  taxable account ("tokutei"): 20.315% on realized gains measured in JPY, and on
      dividends after the 10% US withholding; gains and losses of the same year offset.
  NISA: no Japanese tax; the 10% US withholding on dividends still applies.
  Reclaiming US withholding through a tax return (foreign tax credit) is not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ACCOUNT_TYPES = ("tokutei", "nisa")
SIDES = ("buy", "sell")


@dataclass(frozen=True)
class CostSettings:
    commission_rate: float = 0.0
    commission_min_jpy: float = 0.0
    fx_spread_jpy: float = 0.0
    slippage_rate: float = 0.0

    def __post_init__(self):
        if min(self.commission_rate, self.commission_min_jpy, self.fx_spread_jpy, self.slippage_rate) < 0:
            raise ValueError("costs must not be negative")

    @classmethod
    def from_config(cls, settings: dict) -> "CostSettings":
        return cls(**settings["costs"])


@dataclass(frozen=True)
class TaxSettings:
    capital_gains_rate: float = 0.20315
    dividend_rate_jp: float = 0.20315
    us_withholding_rate: float = 0.10

    @classmethod
    def from_config(cls, settings: dict) -> "TaxSettings":
        return cls(**settings["taxes"])


def check_account(account: str) -> None:
    if account not in ACCOUNT_TYPES:
        raise ValueError(f"unknown account type {account!r}; use one of {ACCOUNT_TYPES}")


# --- FX -----------------------------------------------------------------------------------

def conversion_rate(mid: float, spread: float, side: str) -> float:
    """Rate the broker charges: buying the asset needs USD (mid + spread), selling returns USD (mid - spread)."""
    if not mid > 0:
        raise ValueError(f"FX mid rate must be positive, got {mid}")
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    return mid + spread if side == "buy" else mid - spread


def decompose_jpy_return(p0: float, p1: float, x0: float, x1: float) -> dict:
    """JPY return of a USD asset split into price, FX and cross terms (they add up to the total).

    (1 + total) = (1 + asset)(1 + fx)  ->  total = asset + fx + asset x fx
    """
    asset, fx = p1 / p0 - 1, x1 / x0 - 1
    return {"asset": asset, "fx": fx, "cross": asset * fx, "total": (p1 * x1) / (p0 * x0) - 1}


# --- trading costs ----------------------------------------------------------------------------

def commission_jpy(value_jpy: float, costs: CostSettings) -> float:
    return max(costs.commission_rate * value_jpy, costs.commission_min_jpy) if value_jpy > 0 else 0.0


def estimate_trade(side: str, quantity: float, close: float, fx_mid: float, costs: CostSettings,
                   slippage: bool = True) -> dict:
    """Cash effect of a trade and each cost measured against trading at the close and the mid rate.

    cash_jpy is negative for a buy (money paid) and positive for a sell (money received).
    """
    if quantity < 0 or close <= 0:
        raise ValueError("quantity must be >= 0 and price > 0")
    sign = 1 if side == "buy" else -1
    fill = close * (1 + sign * costs.slippage_rate) if slippage else close
    rate = conversion_rate(fx_mid, costs.fx_spread_jpy, side)
    gross = quantity * fill * rate
    fee = commission_jpy(gross, costs)
    return {
        "side": side, "quantity": quantity, "fill_price": fill, "fx_rate": rate, "gross_jpy": gross,
        "commission_jpy": fee,
        "slippage_cost_jpy": quantity * abs(fill - close) * rate,
        "fx_cost_jpy": quantity * fill * abs(rate - fx_mid),
        "cash_jpy": -(gross + fee) if side == "buy" else gross - fee,
    }


def expense_drag_jpy(value_jpy: float, expense_ratio: float, days: float) -> float:
    """Expense ratio cost over `days` (for reporting; already inside the ETF price)."""
    return value_jpy * expense_ratio * days / 365


# --- taxes ----------------------------------------------------------------------------------

def capital_gains_tax(gain_jpy: float, account: str, taxes: TaxSettings) -> float:
    """Tax on one realized gain (no offsetting; see annual_capital_gains_tax)."""
    check_account(account)
    return 0.0 if account == "nisa" or gain_jpy <= 0 else gain_jpy * taxes.capital_gains_rate


def annual_capital_gains_tax(realized: pd.DataFrame, taxes: TaxSettings) -> pd.DataFrame:
    """Tax per year and account after offsetting gains and losses of the same year.

    realized: columns date, account, gain_jpy. Losses are not carried to later years
    (carrying forward needs a tax return).
    """
    if realized.empty:
        return pd.DataFrame(columns=["year", "account", "net_gain_jpy", "tax_jpy"])
    for account in realized["account"].unique():
        check_account(account)
    frame = realized.assign(year=pd.to_datetime(realized["date"]).dt.year)
    table = frame.groupby(["year", "account"], as_index=False)["gain_jpy"].sum() \
        .rename(columns={"gain_jpy": "net_gain_jpy"})
    table["tax_jpy"] = [capital_gains_tax(g, a, taxes) for g, a in zip(table.net_gain_jpy, table.account)]
    return table


def dividend_tax(gross_usd: float, fx_rate: float, account: str, taxes: TaxSettings) -> dict:
    """US withholding, then Japanese tax on the remainder (taxable account only)."""
    check_account(account)
    gross = gross_usd * fx_rate
    us = gross * taxes.us_withholding_rate
    jp = (gross - us) * taxes.dividend_rate_jp if account == "tokutei" else 0.0
    return {"gross_jpy": gross, "us_withholding_jpy": us, "jp_tax_jpy": jp, "net_jpy": gross - us - jp}
