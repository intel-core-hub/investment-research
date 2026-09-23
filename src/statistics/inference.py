"""Uncertainty of return statistics: confidence intervals, Sharpe standard errors, bootstrap.

All intervals are two-sided at `confidence` (e.g. 0.95 -> 2.5% and 97.5% bounds).
Normal quantiles are used for the analytic intervals (n is in the thousands here,
so the t distribution is practically identical).
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_NORMAL = NormalDist()


def z_value(confidence: float) -> float:
    return _NORMAL.inv_cdf(0.5 + confidence / 2)


def newey_west_lag(n: int) -> int:
    """Rule of thumb lag length floor(4 * (n / 100) ^ (2 / 9))."""
    return int(math.floor(4 * (n / 100) ** (2 / 9)))


def newey_west_variance(x: np.ndarray, lag: int) -> float:
    """Long-run variance of x with Bartlett weights (Newey & West, 1987)."""
    d = x - x.mean()
    n = len(d)
    total = d @ d / n
    for k in range(1, lag + 1):
        total += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return float(total)


def mean_confidence_intervals(returns: pd.Series, confidence: float) -> list[dict]:
    """Annualized mean daily return with an i.i.d. interval and a Newey-West (HAC) interval."""
    x = returns.dropna().to_numpy()
    n = len(x)
    if n < 2:
        return []
    z = z_value(confidence)
    mean = x.mean() * TRADING_DAYS
    lag = newey_west_lag(n)
    rows = []
    for method, se, extra in [
        ("normal_iid", x.std(ddof=1) / math.sqrt(n), {}),
        ("newey_west", math.sqrt(max(newey_west_variance(x, lag), 0.0) / n), {"lag": lag}),
    ]:
        se_annual = se * TRADING_DAYS
        rows.append({"statistic": "mean_return_annualized", "method": method, "estimate": mean,
                     "standard_error": se_annual, "ci_lower": mean - z * se_annual,
                     "ci_upper": mean + z * se_annual, "observations": n, **extra})
    return rows


def sharpe_inference(returns: pd.Series, risk_free_returns: pd.Series, confidence: float) -> list[dict]:
    """Standard error of the Sharpe ratio (Lo 2002, i.i.d.; Mertens 2002 adds skew/kurtosis)
    and the probabilistic Sharpe ratio P(true Sharpe > 0) (Bailey & Lopez de Prado 2012).

    Computed on daily excess returns and annualized by sqrt(252).
    """
    ex = (returns - risk_free_returns.reindex(returns.index)).dropna()
    n = len(ex)
    if n < 4 or not ex.std(ddof=1) > 0:
        return []
    sr = ex.mean() / ex.std(ddof=1)
    skew, kurt = ex.skew(), ex.kurt() + 3
    z = z_value(confidence)
    scale = math.sqrt(TRADING_DAYS)
    rows = []
    for method, var in [
        ("lo_2002_iid", (1 + sr ** 2 / 2) / n),
        ("mertens_2002_non_normal", (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / n),
    ]:
        se = math.sqrt(var) * scale
        rows.append({"statistic": "sharpe_ratio", "method": method, "estimate": sr * scale,
                     "standard_error": se, "ci_lower": sr * scale - z * se, "ci_upper": sr * scale + z * se,
                     "observations": n})
    denominator = math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    rows.append({"statistic": "probabilistic_sharpe_ratio_vs_0", "method": "bailey_lopez_de_prado_2012",
                 "estimate": _NORMAL.cdf(sr * math.sqrt(n - 1) / denominator), "observations": n})
    return rows


def bootstrap_indices(n: int, resamples: int, rng: np.random.Generator, block_size: int = 1) -> np.ndarray:
    """Row indices for resampling. block_size 1 = i.i.d.; > 1 = circular moving blocks."""
    if block_size <= 1:
        return rng.integers(0, n, size=(resamples, n))
    blocks = math.ceil(n / block_size)
    starts = rng.integers(0, n, size=(resamples, blocks))
    idx = (starts[:, :, None] + np.arange(block_size)) % n
    return idx.reshape(resamples, -1)[:, :n]


def _statistics(r: np.ndarray, rf: np.ndarray) -> dict[str, np.ndarray]:
    """Per-resample annualized mean, CAGR (geometric) and Sharpe. r: (B, n, k), rf: (B, n)."""
    n = r.shape[1]
    ex = r - rf[:, :, None]
    sd = ex.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(sd > 0, ex.mean(axis=1) / sd * math.sqrt(TRADING_DAYS), np.nan)
    growth = np.exp(np.log1p(r).sum(axis=1) * TRADING_DAYS / n)
    return {"mean_return_annualized": r.mean(axis=1) * TRADING_DAYS, "cagr": growth - 1, "sharpe_ratio": sharpe}


def bootstrap(returns: pd.DataFrame, risk_free_returns: pd.Series, benchmark: str, resamples: int,
              seed: int, confidence: float, block_size: int = 1, chunk: int = 200) -> list[dict]:
    """Percentile bootstrap intervals for each strategy and for its difference from the benchmark.

    Rows (days) are resampled jointly for all strategies and the risk-free series,
    so differences keep the day-by-day pairing. CAGR here is the geometric mean
    growth of the resampled daily returns (252 days a year). The p-value of a
    difference is two-sided: 2 x min(P(diff <= 0), P(diff >= 0)) over resamples,
    each probability estimated as (count + 1) / (resamples + 1).
    """
    data = returns.join(risk_free_returns.rename("__rf__"), how="inner").dropna()
    names = list(returns.columns)
    r_all, rf_all = data[names].to_numpy(), data["__rf__"].to_numpy()
    n = len(data)
    rng = np.random.default_rng(seed)
    idx = bootstrap_indices(n, resamples, rng, block_size)

    samples = {key: [] for key in ("mean_return_annualized", "cagr", "sharpe_ratio")}
    for start in range(0, resamples, chunk):
        part = idx[start:start + chunk]
        for key, value in _statistics(r_all[part], rf_all[part]).items():
            samples[key].append(value)
    samples = {key: np.concatenate(v) for key, v in samples.items()}

    point = _statistics(r_all[None], rf_all[None])
    method = "bootstrap_iid" if block_size <= 1 else "bootstrap_block"
    lo, hi = (1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100
    common = {"method": method, "observations": n, "resamples": resamples, "seed": seed,
              "block_size": block_size}
    rows = []
    b = names.index(benchmark)
    for j, name in enumerate(names):
        for key, values in samples.items():
            v = values[:, j][~np.isnan(values[:, j])]
            rows.append({"strategy": name, "statistic": key, "estimate": float(point[key][0, j]),
                         "ci_lower": float(np.percentile(v, lo)), "ci_upper": float(np.percentile(v, hi)),
                         "standard_error": float(v.std(ddof=1)), **common})
            if j == b:
                continue
            diff = values[:, j] - values[:, b]
            diff = diff[~np.isnan(diff)]
            # (count + 1) / (resamples + 1): a p-value can never be exactly 0 from finite resamples
            p = 2 * min(((diff <= 0).sum() + 1) / (len(diff) + 1), ((diff >= 0).sum() + 1) / (len(diff) + 1))
            rows.append({"strategy": name, "statistic": f"{key}_difference_vs_benchmark",
                         "estimate": float(point[key][0, j] - point[key][0, b]),
                         "ci_lower": float(np.percentile(diff, lo)), "ci_upper": float(np.percentile(diff, hi)),
                         "standard_error": float(diff.std(ddof=1)), "p_value": float(min(p, 1.0)), **common})
    return rows


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = p_values.dropna().sort_values()
    m = len(p)
    adjusted = np.maximum.accumulate([(m - i) * v for i, v in enumerate(p)])
    return pd.Series(np.minimum(adjusted, 1.0), index=p.index).reindex(p_values.index)
