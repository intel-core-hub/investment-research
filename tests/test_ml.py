import numpy as np
import pandas as pd
import pytest

import regimes as rg
from engine import CostModel, run_backtest
from evaluation import calibration_table, classification_metrics
from features import (
    build_features,
    complete_rows,
    direction_target,
    feature_sets,
    forward_return,
    label_end_dates,
    split_segments,
    trainable_rows,
)
from models import BASELINES, MODELS, baseline_scores, make_model, param_grid
from signals import PredictionStrategy

RELATIVE = ["VT", "EWJ", "BND", "BIL"]


def market(n: int = 700, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=n)
    prices = pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(mu, sd, n))
                           for t, mu, sd in [("VOO", 5e-4, 0.01), ("VT", 4e-4, 0.01), ("EWJ", 2e-4, 0.012),
                                             ("BND", 1e-4, 0.003), ("BIL", 1e-4, 1e-4)]}, index=index)
    volume = pd.Series(rng.integers(1_000_000, 5_000_000, n).astype(float), index=index)
    return prices, volume


def features(prices, volume):
    return build_features(prices, volume, "VOO", "BIL", RELATIVE, rg.RegimeSettings())


def dataset(n=700, seed=0, horizon=20):
    prices, volume = market(n, seed)
    X = complete_rows(features(prices, volume), feature_sets(RELATIVE)["extended"])
    y = direction_target(prices["VOO"], horizon).reindex(X.index)
    return prices, X, y


# --- data: features, target, splits ------------------------------------------------------

def test_features_do_not_use_future_data():
    prices, volume = market()
    full = features(prices, volume)
    cut = 500
    truncated = features(prices.iloc[:cut], volume.iloc[:cut])
    shocked_p, shocked_v = prices.copy(), volume.copy()
    shocked_p.iloc[cut:] *= 1.5
    shocked_v.iloc[cut:] *= 3
    shocked = features(shocked_p, shocked_v)

    pd.testing.assert_frame_equal(truncated, full.iloc[:cut])
    pd.testing.assert_frame_equal(shocked.iloc[:cut], full.iloc[:cut])


def test_feature_values():
    prices, volume = market()
    f = features(prices, volume)
    close = prices["VOO"]
    t = f.index[300]
    assert f.loc[t, "ret_20d"] == pytest.approx(close.loc[t] / close.shift(20).loc[t] - 1)
    assert f.loc[t, "ma200_gap"] == pytest.approx(close.loc[t] / close.loc[:t].iloc[-200:].mean() - 1)
    assert f.loc[t, "rel_ret_60d_vs_BND"] == pytest.approx(
        f.loc[t, "ret_60d"] - (prices["BND"].loc[t] / prices["BND"].shift(60).loc[t] - 1))
    assert f.loc[t, "drawdown"] == pytest.approx(close.loc[t] / close.loc[:t].max() - 1)


def test_direction_target_and_label_end():
    close = pd.Series([100, 101, 99, 102, 98], index=pd.bdate_range("2024-01-01", periods=5), dtype=float)
    y = direction_target(close, 2)
    assert list(y.iloc[:3]) == [0.0, 1.0, 0.0]  # 99 < 100, 102 > 101, 98 < 99
    assert y.iloc[3:].isna().all()
    assert forward_return(close, 2).iloc[1] == pytest.approx(102 / 101 - 1)
    ends = label_end_dates(close.index, 2)
    assert ends.iloc[0] == close.index[2] and pd.isna(ends.iloc[-1])


def test_training_rows_are_purged_before_the_boundary():
    index = pd.bdate_range("2024-01-01", periods=100)
    y = pd.Series(1.0, index=index)
    ends = label_end_dates(index, 20)
    rows = trainable_rows(y, ends, index[0], index[59])
    assert rows[-1] == index[39]  # the label of index[39] ends on index[59]
    assert len(rows) == 40
    assert (ends.loc[rows] <= index[59]).all()


def test_splits_are_chronological_disjoint_and_complete():
    index = pd.bdate_range("2018-01-01", "2024-12-31")
    parts = split_segments(index, "2020-12-31", "2022-12-31")
    assert parts["train"][-1] < parts["validation"][0]
    assert parts["validation"][-1] < parts["test"][0]
    assert sum(len(p) for p in parts.values()) == len(index)
    assert not set(parts["train"]) & set(parts["test"])
    with pytest.raises(ValueError):
        split_segments(index, "2022-12-31", "2020-12-31")


def test_complete_rows_drop_the_warm_up_without_imputation():
    prices, volume = market()
    f = features(prices, volume)
    cols = feature_sets(RELATIVE)["extended"]
    X = complete_rows(f, cols)
    assert not X.isna().any().any()
    assert X.index[0] == f.index[f[cols].notna().all(axis=1)][0]
    assert list(X.columns) == cols


# --- ML --------------------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS)
def test_models_fit_and_predict_probabilities(model):
    _, X, y = dataset()
    rows = y.dropna().index
    params = {"random_forest": {"n_estimators": 20}, "hist_gradient_boosting": {"max_iter": 20}}.get(model, {})
    pipe = make_model(model, params, seed=0).fit(X.loc[rows], y.loc[rows].astype(int))
    prob = pipe.predict_proba(X.loc[rows])[:, 1]
    assert prob.shape == (len(rows),) and ((prob >= 0) & (prob <= 1)).all()
    with pytest.raises(ValueError):
        pipe.predict_proba(X.loc[rows].iloc[:, :-1])  # feature count must match


def test_scaler_and_feature_selection_are_fitted_on_training_rows_only():
    _, X, y = dataset()
    # regime flags can be constant in a short synthetic sample, which SelectKBest cannot score
    X = X[[c for c in X.columns if not c.startswith("regime_")]]
    rows = y.dropna().index
    train, test = rows[:200], rows[200:]
    pipe = make_model("logistic_regression", {}, seed=0, select_k=5).fit(X.loc[train], y.loc[train].astype(int))
    assert pipe.named_steps["scale"].mean_ == pytest.approx(X.loc[train].mean().to_numpy())

    altered = X.copy()
    altered.loc[test] *= 10
    again = make_model("logistic_regression", {}, seed=0, select_k=5).fit(altered.loc[train], y.loc[train].astype(int))
    assert list(pipe.named_steps["select"].get_support()) == list(again.named_steps["select"].get_support())
    assert pipe.named_steps["model"].coef_ == pytest.approx(again.named_steps["model"].coef_)


def test_seed_makes_models_reproducible():
    _, X, y = dataset()
    rows = y.dropna().index

    def fit(seed):
        p = make_model("random_forest", {"n_estimators": 30, "max_features": 0.5}, seed).fit(X.loc[rows],
                                                                                           y.loc[rows].astype(int))
        return p.predict_proba(X.loc[rows])[:, 1]

    assert np.array_equal(fit(1), fit(1))
    assert not np.array_equal(fit(1), fit(2))


def test_param_grid_is_every_combination_in_a_fixed_order():
    assert param_grid({"b": [1, 2], "a": [3]}) == [{"a": 3, "b": 1}, {"a": 3, "b": 2}]


def test_baseline_definitions():
    X = pd.DataFrame({"ret_252d": [0.1, -0.1, 0.0], "ma200_gap": [-0.01, 0.02, 0.0]},
                     index=pd.bdate_range("2024-01-01", periods=3))
    assert list(baseline_scores("always_up", X, 0.6, 0)) == [1, 1, 1]
    assert list(baseline_scores("historical_mean", X, 0.6, 0)) == [0.6, 0.6, 0.6]
    assert list(baseline_scores("momentum_252d", X, 0.6, 0)) == [1, 0, 0]
    assert list(baseline_scores("moving_average_200d", X, 0.6, 0)) == [0, 1, 0]
    big = pd.DataFrame(index=pd.bdate_range("2000-01-01", periods=20000))
    random = baseline_scores("random", big, 0.7, seed=3)
    assert (random > 0.5).mean() == pytest.approx(0.7, abs=0.02)
    assert random.equals(baseline_scores("random", big, 0.7, seed=3))
    assert set(BASELINES) >= {"always_up", "historical_mean", "momentum_252d", "moving_average_200d", "random"}


def test_classification_metrics():
    y = pd.Series([1, 1, 0, 0, 1], dtype=float)
    score = pd.Series([0.9, 0.4, 0.6, 0.2, np.nan])
    m = classification_metrics(y, score, 0.5, is_probability=True)
    assert m["observations"] == 4
    assert (m["true_positive"], m["false_negative"], m["false_positive"], m["true_negative"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["roc_auc"] == pytest.approx(0.75)
    rule = classification_metrics(y, (score > 0.5).astype(float), 0.5, is_probability=False)
    assert np.isnan(rule["brier_score"])
    one_class = classification_metrics(pd.Series([1.0, 1.0]), pd.Series([0.3, 0.8]), 0.5, True)
    assert np.isnan(one_class["roc_auc"])


def test_calibration_table():
    y = pd.Series([0, 1, 1, 1], dtype=float)
    p = pd.Series([0.05, 0.15, 0.95, 0.99])
    table = calibration_table(y, p, bins=10)
    assert table.observations.sum() == 4
    assert table.loc[9, "observed_rate"] == 1.0 and table.loc[9, "observations"] == 2


# --- backtest connection ---------------------------------------------------------------------

def test_prediction_signal_fills_on_the_next_day_with_costs():
    prices, _ = market(120)
    invest = pd.Series(True, index=prices.index)
    costs = CostModel(commission_rate=0.001, slippage_rate=0.001)
    result = run_backtest(prices, PredictionStrategy("p", invest, "VOO", "BIL"), 1000, costs, start=prices.index[5])
    first = result.transactions.iloc[0]

    assert first["signal_date"] == prices.index[5]
    assert first["date"] == prices.index[6]
    assert first["commission"] > 0 and first["slippage_cost"] > 0


def test_signals_are_only_read_on_monthly_decision_days():
    prices, _ = market(120)
    invest = pd.Series(np.arange(120) % 2 == 0, index=prices.index)
    result = run_backtest(prices, PredictionStrategy("p", invest, "VOO", "BIL"), 1000)
    decided = pd.to_datetime(result.signals["date"])
    assert decided[0] == prices.index[0]
    assert all(d.month != p.month for d, p in zip(decided[1:], [prices.index[prices.index.get_loc(x) - 1]
                                                               for x in decided[1:]]))


def test_backtest_runs_on_out_of_sample_predictions_only():
    prices, _ = market(300)
    test_start = prices.index[200]
    oos = pd.Series(True, index=prices.index[200:])
    result = run_backtest(prices, PredictionStrategy("p", oos, "VOO", "BIL"), 1000, start=test_start)
    assert result.equity.index[0] == test_start
    with pytest.raises(KeyError, match="no prediction"):
        run_backtest(prices, PredictionStrategy("p", oos, "VOO", "BIL"), 1000, start=prices.index[150])


def test_changing_later_predictions_does_not_change_earlier_results():
    prices, _ = market(300)
    invest = pd.Series(np.arange(300) % 3 == 0, index=prices.index)
    changed = invest.copy()
    changed.iloc[200:] = ~changed.iloc[200:]
    a = run_backtest(prices, PredictionStrategy("p", invest, "VOO", "BIL"), 1000)
    b = run_backtest(prices, PredictionStrategy("p", changed, "VOO", "BIL"), 1000)
    # the first decision that can differ is on or after index 200; its fill is on the following day
    pd.testing.assert_series_equal(a.equity.iloc[:201], b.equity.iloc[:201])
