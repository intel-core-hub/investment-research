"""Phase 7 calculations for the dashboard, with no Streamlit dependency.

Reuses src/ml/run_ml.py, so the same leak-free rules apply as in the Phase 7 report:
labels purged at every boundary, scaling fitted on training rows only,
hyperparameters compared on validation, test predicted by a model fitted on
train + validation, walk-forward with the fixed default hyperparameters.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src" / "ml") not in sys.path:
    sys.path.insert(0, str(ROOT / "src" / "ml"))

import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

import run_ml as ml  # noqa: E402  (also puts src/backtest, src/statistics and src/data on sys.path)
from engine import CostModel  # noqa: E402
from evaluation import classification_metrics  # noqa: E402
from features import trainable_rows  # noqa: E402
from models import BASELINES, baseline_scores, param_grid  # noqa: E402

BUY_AND_HOLD = "Buy & Hold (always_up)"
RULE_BASELINES = {"momentum_252d": "momentum_252d", "moving_average_200d": "moving_average_200d"}


def load_setup() -> tuple[dict, ml.Data]:
    cfg = ml.load(ml.CONFIGS["ml"])
    return cfg, ml.Data(cfg, ml.load(ml.CONFIGS["statistics"]))


def model_label(model: str, fset: str) -> str:
    return f"{model} / {fset}"


def candidates(cfg: dict, model: str) -> list[dict]:
    """Hyperparameters that may be chosen: the Phase 7 grid, plus the walk-forward default."""
    grid = param_grid(cfg["models"][model]["grid"])
    default = cfg["models"][model]["default"]
    return grid + ([default] if default not in grid else [])


def validation_search(data: ml.Data, cfg: dict, model: str, fset: str) -> pd.DataFrame:
    """Validation ROC-AUC of every candidate, fitted on the (purged) training rows only."""
    rows = []
    y = data.y.loc[data.eval_rows["validation"]]
    for params in candidates(cfg, model):
        _, prob = ml.fit_predict(data, model, fset, params, cfg["seed"], cfg["features"]["select_k"],
                                 data.fit_rows["train"], data.eval_rows["validation"])
        m = classification_metrics(y, prob, cfg["trading"]["threshold"], True)
        rows.append({"hyperparameters": json.dumps(params, sort_keys=True), "validation_roc_auc": m["roc_auc"],
                     "in_grid": params in param_grid(cfg["models"][model]["grid"])})
    table = pd.DataFrame(rows)
    grid_rows = table[table.in_grid]
    table["selected"] = table.index == grid_rows["validation_roc_auc"].idxmax()
    return table


def predict(data: ml.Data, cfg: dict, model: str, fset: str, params: dict):
    """(pipeline fitted on train, probability for every row).

    Train and validation rows are scored by the model fitted on train; test rows
    by the model refitted on train + validation, as in Phase 7.
    """
    seed, k = cfg["seed"], cfg["features"]["select_k"]
    before_test = data.rows["train"].append(data.rows["validation"])
    pipe, prob = ml.fit_predict(data, model, fset, params, seed, k, data.fit_rows["train"], before_test)
    _, prob_test = ml.fit_predict(data, model, fset, params, seed, k, data.fit_rows["train+validation"],
                                  data.rows["test"])
    return pipe, pd.concat([prob, prob_test])


def baseline_score_table(data: ml.Data, cfg: dict) -> dict[str, pd.Series]:
    """Scores of the Phase 7 baselines for every row (per-segment training rates as in Phase 7)."""
    return {f"baseline:{name}": ml.baseline_predictions(data, name, cfg["seed"]) for name in BASELINES}


def segment_metrics(data: ml.Data, scores: dict[str, pd.Series], threshold: float) -> pd.DataFrame:
    """Classification metrics per segment. Baseline scores are not probabilities (except the
    historical mean), so their Brier score and log loss are NaN, as in Phase 7."""
    rows = []
    for name, score in scores.items():
        is_probability = not name.startswith("baseline:") or name == "baseline:historical_mean"
        for s in ml.SEGMENTS:
            rows_s = data.eval_rows[s]
            rows.append({"predictor": name, "segment": s,
                         **classification_metrics(data.y.loc[rows_s], score.loc[rows_s], threshold, is_probability)})
    return pd.DataFrame(rows)


def importance(data: ml.Data, cfg: dict, pipe, fset: str) -> pd.DataFrame:
    """Permutation importance on validation rows (drop in ROC-AUC when a feature is shuffled)."""
    cols = data.sets[fset]
    rows = data.eval_rows["validation"]
    perm = permutation_importance(pipe, data.X.loc[rows, cols], data.y.loc[rows].astype(int), scoring="roc_auc",
                                  n_repeats=10, random_state=cfg["seed"])
    return pd.DataFrame({"feature": cols, "importance_mean": perm.importances_mean,
                         "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)


def walk_forward(data: ml.Data, cfg: dict, model: str, fset: str) -> tuple[pd.Series, pd.DataFrame]:
    """Each year predicted by a model fitted on every label known by the previous year end.

    Uses the default hyperparameters: choosing them on the 2020-2022 validation
    period would leak into the earlier walk-forward years.
    """
    years = range(cfg["walk_forward"]["first_test_year"], data.last.year + 1)
    parts, rows = [], []
    for year in years:
        test = data.X.index[data.X.index.year == year]
        fit = trainable_rows(data.y, data.label_end, data.first, pd.Timestamp(year - 1, 12, 31))
        _, prob = ml.fit_predict(data, model, fset, cfg["models"][model]["default"], cfg["seed"],
                                 cfg["features"]["select_k"], fit, test)
        parts.append(prob)
        labelled = test[data.y.loc[test].notna().to_numpy()]
        m = classification_metrics(data.y.loc[labelled], prob.loc[labelled], cfg["trading"]["threshold"], True)
        rows.append({"year": year, "fit_end": fit[-1].date(), "fit_rows": len(fit), **m})
    return pd.concat(parts), pd.DataFrame(rows)


def rule_scores(data: ml.Data, cfg: dict, index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Buy & Hold and the two rule baselines on any rows (they do not depend on training data)."""
    scores = {BUY_AND_HOLD: baseline_scores("always_up", data.X.loc[index], 0.5, cfg["seed"])}
    for name, label in RULE_BASELINES.items():
        scores[label] = baseline_scores(name, data.X.loc[index], 0.5, cfg["seed"])
    return scores


def backtest(data: ml.Data, cfg: dict, scores: dict[str, pd.Series], start, end, initial_cash: float,
             commission_rate: float, slippage_rate: float, threshold: float):
    """Predictions -> monthly signals -> Phase 5 engine (next-day fills, costs) -> Phase 6 metrics."""
    commission_min = ml.load(ml.CONFIGS["backtest"])["costs"]["commission_min"]
    costs = CostModel(commission_rate=commission_rate, commission_min=commission_min, slippage_rate=slippage_rate)
    equities, rows = {}, []
    for name, score in scores.items():
        invest = score.loc[start:end] > threshold
        result = ml.backtest(data, name, invest, costs, initial_cash, cfg, start, end)
        equities[name] = result.equity
        rows.append({"predictor": name, **ml.investment_metrics(result, data.rf, data.risk_free)})
    return pd.DataFrame(equities), pd.DataFrame(rows).set_index("predictor")
