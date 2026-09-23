"""Model pipelines and baseline predictors.

Every model is a scikit-learn Pipeline: StandardScaler -> optional SelectKBest -> classifier.
Because scaling and feature selection live inside the pipeline, `fit` learns them
from the training rows only; validation and test rows are only transformed.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS = ("logistic_regression", "random_forest", "hist_gradient_boosting")


def make_model(name: str, params: dict, seed: int, select_k: int = 0) -> Pipeline:
    if name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, random_state=seed, **params)
    elif name == "random_forest":
        estimator = RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    elif name == "hist_gradient_boosting":
        # early stopping would carve a random (non-chronological) validation split out of the training rows
        estimator = HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **params)
    else:
        raise ValueError(f"unknown model {name!r}; use one of {MODELS}")
    steps = [("scale", StandardScaler())]
    if select_k:
        steps.append(("select", SelectKBest(f_classif, k=select_k)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def param_grid(grid: dict) -> list[dict]:
    """Every combination of the listed values, in a fixed order."""
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def complexity(pipeline: Pipeline, n_features: int) -> tuple[float, str]:
    """A rough size of the fitted model and what the number counts."""
    model = pipeline.named_steps["model"]
    if isinstance(model, LogisticRegression):
        return float(model.coef_.size + model.intercept_.size), "coefficients"
    if isinstance(model, RandomForestClassifier):
        return float(sum(t.tree_.node_count for t in model.estimators_)), "tree nodes"
    if isinstance(model, HistGradientBoostingClassifier):
        return float(model.n_iter_), "boosting iterations"
    return float(n_features), "features"


def selected_features(pipeline: Pipeline, features: list[str]) -> list[str]:
    select = pipeline.named_steps.get("select")
    return list(np.array(features)[select.get_support()]) if select is not None else list(features)


# --- baselines --------------------------------------------------------------------------

BASELINES = ("always_up", "historical_mean", "momentum_252d", "moving_average_200d", "random")


def baseline_scores(name: str, features: pd.DataFrame, train_positive_rate: float, seed: int) -> pd.Series:
    """Score for each row (higher = more likely up); the prediction is score > threshold.

    always_up: always 1 (the buy & hold decision)
    historical_mean: the share of up labels in the training rows, for every row
    momentum_252d: 1 if the trailing 252-day return is positive, else 0
    moving_average_200d: 1 if the close is above its 200-day average, else 0
    random: uniform random number + (training up share - 0.5), so it is above 0.5 for that
            share of rows while ranking rows at random (not a probability)
    """
    index = features.index
    if name == "always_up":
        return pd.Series(1.0, index=index)
    if name == "historical_mean":
        return pd.Series(train_positive_rate, index=index)
    if name == "momentum_252d":
        return (features["ret_252d"] > 0).astype(float)
    if name == "moving_average_200d":
        return (features["ma200_gap"] > 0).astype(float)
    if name == "random":
        u = np.random.default_rng(seed).uniform(size=len(index))
        return pd.Series(u + (train_positive_rate - 0.5), index=index)
    raise ValueError(f"unknown baseline {name!r}")
