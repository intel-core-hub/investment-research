"""Prediction-quality metrics for direction forecasts (kept separate from investment results)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y: pd.Series, score: pd.Series, threshold: float, is_probability: bool) -> dict:
    """Metrics on rows with a known label. Undefined values are NaN.

    ROC-AUC needs both classes; Brier score and log loss are only reported when the
    score is a probability (not for rule-based 0/1 baselines).
    """
    mask = y.notna() & score.notna()
    y, score = y[mask].astype(int), score[mask]
    pred = (score > threshold).astype(int)
    n = len(y)
    out = {"observations": n, "positive_rate": y.mean() if n else np.nan,
           "predicted_positive_rate": pred.mean() if n else np.nan}
    if n == 0:
        return out
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = y.nunique() == 2
    out.update({
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=np.nan),
        "recall": recall_score(y, pred, zero_division=np.nan),
        "f1": f1_score(y, pred, zero_division=np.nan),
        "roc_auc": roc_auc_score(y, score) if both and score.nunique() > 1 else np.nan,
        "brier_score": brier_score_loss(y, score.clip(0, 1)) if is_probability else np.nan,
        "log_loss": log_loss(y, score.clip(1e-6, 1 - 1e-6), labels=[0, 1]) if is_probability else np.nan,
        "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp),
    })
    return out


def calibration_table(y: pd.Series, probability: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability vs observed up rate in equal-width probability bins."""
    mask = y.notna() & probability.notna()
    y, p = y[mask], probability[mask]
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        in_bin = bucket == b
        rows.append({"bin_lower": edges[b], "bin_upper": edges[b + 1], "observations": int(in_bin.sum()),
                     "mean_predicted": p[in_bin].mean() if in_bin.any() else np.nan,
                     "observed_rate": y[in_bin].mean() if in_bin.any() else np.nan})
    return pd.DataFrame(rows)
