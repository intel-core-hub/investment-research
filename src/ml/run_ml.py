"""Phase 7: does machine learning improve on simple rules? Leak-free, reproducible, baseline-compared.

Pipeline:
  features (past data only) + direction target (next `horizon` days)
  -> chronological train / validation / test split, labels purged at each boundary
  -> hyperparameters chosen on validation; test predicted once by a model fitted on train + validation
  -> predictions -> monthly signals -> Phase 5 engine (next-day fills, costs) -> Phase 6 metrics
  -> expanding-window walk-forward with fixed default hyperparameters
Everything is written to reports/ml/.
"""
import hashlib
import json
import platform
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Phase 5 engine/provenance, Phase 6 statistics, and raw data access
sys.path[:0] = [str(ROOT / "src" / d) for d in ("backtest", "statistics", "data")]

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import inference as inf  # noqa: E402
import performance as pf  # noqa: E402
import prices as rawdata  # noqa: E402
import regimes as rg  # noqa: E402
from engine import CostModel, run_backtest  # noqa: E402
from evaluation import calibration_table, classification_metrics  # noqa: E402
from features import (  # noqa: E402
    build_features,
    complete_rows,
    direction_target,
    feature_sets,
    label_end_dates,
    split_segments,
    trainable_rows,
)
from models import BASELINES, MODELS, baseline_scores, complexity, make_model, param_grid  # noqa: E402
from provenance import content_sha256, git_state  # noqa: E402
from signals import PredictionStrategy  # noqa: E402

PRICES_PATH = ROOT / "data" / "processed" / "prices.csv"
CONFIGS = {name: ROOT / "config" / f"{name}.toml" for name in ("ml", "backtest", "statistics")}
OUT = ROOT / "reports" / "ml"
BENCHMARK = "baseline:always_up"
SEGMENTS = ("train", "validation", "test")

COLORS = {"ml:logistic_regression": "#2a78d6", "ml:random_forest": "#eb6834",
          "ml:hist_gradient_boosting": "#1baf7a", "baseline:always_up": "#eda100",
          "baseline:momentum_252d": "#e87ba4", "baseline:moving_average_200d": "#008300"}
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE_INK, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEGMENT_COLORS = {"train": "#86b6ef", "validation": "#3987e5", "test": "#184f95"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK, "axes.edgecolor": BASELINE_INK,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY, "legend.frameon": False, "lines.linewidth": 1.3,
})


def load(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


# --- data -----------------------------------------------------------------------------------

class Data:
    def __init__(self, cfg: dict, stats_cfg: dict):
        target = cfg["target"]
        self.asset, self.risk_free, self.horizon = target["asset"], target["risk_free"], target["horizon_days"]
        self.prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
        volume = rawdata.load_raw(self.asset)["Volume"]
        regime_settings = rg.RegimeSettings(**{k: v for k, v in stats_cfg["regimes"].items() if k != "reference"})
        relative = cfg["features"]["relative_to"]
        self.sets = feature_sets(relative)
        all_features = build_features(self.prices, volume, self.asset, self.risk_free, relative, regime_settings)
        self.X = complete_rows(all_features, self.sets["extended"])
        close = self.prices[self.asset]
        self.y = direction_target(close, self.horizon).reindex(self.X.index)
        self.label_end = label_end_dates(self.prices.index, self.horizon).reindex(self.X.index)
        self.rf = pf.daily_returns(self.prices[self.risk_free])

        first, last = self.X.index[0], self.X.index[-1]
        train_end, val_end = pd.Timestamp(cfg["splits"]["train_end"]), pd.Timestamp(cfg["splits"]["validation_end"])
        self.rows = split_segments(self.X.index, train_end, val_end)
        self.fit_rows = {"train": trainable_rows(self.y, self.label_end, first, train_end),
                         "train+validation": trainable_rows(self.y, self.label_end, first, val_end)}
        val_start = self.rows["validation"][0]
        self.eval_rows = {"train": self.fit_rows["train"],
                          "validation": trainable_rows(self.y, self.label_end, val_start, val_end),
                          "test": self.rows["test"][self.y.loc[self.rows["test"]].notna().to_numpy()]}
        self.period = {s: (self.rows[s][0].date(), self.rows[s][-1].date()) for s in SEGMENTS}
        self.first, self.last = first, last

    def split_table(self) -> pd.DataFrame:
        """Rows per segment; evaluated rows exclude labels that end after the segment (or the data)."""
        rows = []
        for s in SEGMENTS:
            fitted_on = "train+validation" if s == "test" else "train"
            rows.append({
                "segment": s, "start": self.period[s][0], "end": self.period[s][1], "rows": len(self.rows[s]),
                "evaluated_rows": len(self.eval_rows[s]),
                "rows_without_usable_label": len(self.rows[s]) - len(self.eval_rows[s]),
                "positive_rate": self.y.loc[self.eval_rows[s]].mean(),
                "model_fitted_on": fitted_on, "fit_rows": len(self.fit_rows[fitted_on]),
            })
        return pd.DataFrame(rows)


# --- predictions ----------------------------------------------------------------------------

def fit_predict(data: Data, model: str, fset: str, params: dict, seed: int, select_k: int,
                fit_rows: pd.Index, predict_rows: pd.Index):
    cols = data.sets[fset]
    pipe = make_model(model, params, seed, min(select_k, len(cols)) if select_k else 0)
    pipe.fit(data.X.loc[fit_rows, cols], data.y.loc[fit_rows].astype(int))
    prob = pd.Series(pipe.predict_proba(data.X.loc[predict_rows, cols])[:, 1], index=predict_rows)
    return pipe, prob


def search(data: Data, cfg: dict) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for fset in data.sets:
            for params in param_grid(cfg["models"][model]["grid"]):
                _, prob = fit_predict(data, model, fset, params, cfg["seed"], cfg["features"]["select_k"],
                                      data.fit_rows["train"], data.eval_rows["validation"])
                y = data.y.loc[data.eval_rows["validation"]]
                rows.append({"model": model, "feature_set": fset, "hyperparameters": json.dumps(params, sort_keys=True),
                             "validation_roc_auc": roc_auc_score(y, prob),
                             **{f"validation_{k}": v for k, v in classification_metrics(
                                 y, prob, cfg["trading"]["threshold"], True).items()
                                if k in ("accuracy", "log_loss", "brier_score")}})
    table = pd.DataFrame(rows)
    best = table.groupby(["model", "feature_set"], sort=False)["validation_roc_auc"].idxmax()
    table["selected"] = table.index.isin(best)
    return table


def baseline_predictions(data: Data, name: str, seed: int) -> pd.Series:
    """Scores for every segment; historical mean uses the rows the matching model would be fitted on."""
    parts = []
    for s in SEGMENTS:
        fit = data.fit_rows["train+validation" if s == "test" else "train"]
        rate = data.y.loc[fit].mean()
        parts.append(baseline_scores(name, data.X.loc[data.rows[s]], rate, seed + SEGMENTS.index(s)))
    return pd.concat(parts)


# --- backtests and investment metrics ---------------------------------------------------------

def backtest(data: Data, name: str, invest: pd.Series, costs: CostModel, cash: float, cfg: dict, start, end):
    strategy = PredictionStrategy(name, invest, data.asset, data.risk_free, cfg["trading"]["frequency"])
    return run_backtest(data.prices.loc[:end], strategy, cash, costs, start=start)


def investment_metrics(result, rf: pd.Series, risk_off: str) -> dict:
    e = result.equity
    r = pf.daily_returns(e)
    tx = result.transactions
    held = result.signals.drop(columns="date").idxmax(axis=1)
    out = {"start": e.index[0].date(), "end": e.index[-1].date(), **pf.summary(e, rf),
           "share_of_decisions_invested": float((held != risk_off).mean())}
    for c in (0.95, 0.99):
        v = pf.historical_var_cvar(r, c)
        out[f"var_{int(c * 100)}"], out[f"cvar_{int(c * 100)}"] = v["var"], v["cvar"]
    out.update({"trades": len(tx), "total_costs": float(tx["commission"].sum() + tx["slippage_cost"].sum())})
    return out


def run_backtests(data, predictions: dict, segments: dict, costs, cash, cfg):
    """predictions: name -> score series; segments: segment -> (start, end). Returns metrics and equities."""
    rows, equities = [], {}
    for seg, (start, end) in segments.items():
        eq = {}
        for name, score in predictions.items():
            invest = score.loc[start:end] > cfg["trading"]["threshold"]
            result = backtest(data, name, invest, costs, cash, cfg, start, end)
            eq[name] = result.equity
            rows.append({"segment": seg, "predictor": name, **investment_metrics(result, data.rf, data.risk_free)})
        equities[seg] = pd.DataFrame(eq)
    table = pd.DataFrame(rows)
    bench = table[table.predictor == BENCHMARK].set_index("segment")
    for col in ("cagr", "sharpe_ratio", "max_drawdown", "total_return"):
        table[f"{col}_minus_buy_and_hold"] = table[col] - table["segment"].map(bench[col])
    return table, equities


def statistical_tests(equities: pd.DataFrame, rf: pd.Series, boot: dict, confidence: float, segment: str):
    returns = pd.DataFrame({n: pf.daily_returns(equities[n]) for n in equities})
    rows = []
    for name in returns:
        rows += [{"predictor": name, **r} for r in inf.sharpe_inference(returns[name], rf, confidence)]
    for block in (1, boot["block_size"]):
        rows += [{**r, "predictor": r.pop("strategy")} for r in
                 inf.bootstrap(returns, rf, BENCHMARK, boot["resamples"], boot["seed"], confidence, block)]
    table = pd.DataFrame(rows)
    table.insert(0, "segment", segment)
    table["p_value_holm"] = np.nan
    has_p = table["p_value"].notna() if "p_value" in table else pd.Series(False, index=table.index)
    for _, group in table[has_p].groupby(["method", "statistic"]):
        table.loc[group.index, "p_value_holm"] = inf.holm_adjust(group["p_value"])
        table.loc[group.index, "comparisons_in_family"] = len(group)
    return table


# --- walk-forward --------------------------------------------------------------------------

def walk_forward(data: Data, cfg: dict):
    years = range(cfg["walk_forward"]["first_test_year"], data.last.year + 1)
    preds, rows = {}, []
    for model in MODELS:
        for fset in data.sets:
            name = f"ml:{model}/{fset}"
            parts = []
            for year in years:
                test = data.X.index[data.X.index.year == year]
                fit = trainable_rows(data.y, data.label_end, data.first, pd.Timestamp(year - 1, 12, 31))
                _, prob = fit_predict(data, model, fset, cfg["models"][model]["default"], cfg["seed"],
                                      cfg["features"]["select_k"], fit, test)
                parts.append(prob)
                labelled = test[data.y.loc[test].notna().to_numpy()]
                m = classification_metrics(data.y.loc[labelled], prob.loc[labelled], cfg["trading"]["threshold"], True)
                rows.append({"predictor": name, "year": year, "fit_rows": len(fit), "fit_end": fit[-1].date(), **m})
            preds[name] = pd.concat(parts)
    for b in BASELINES:
        parts = []
        for year in years:
            test = data.X.index[data.X.index.year == year]
            fit = trainable_rows(data.y, data.label_end, data.first, pd.Timestamp(year - 1, 12, 31))
            score = baseline_scores(b, data.X.loc[test], data.y.loc[fit].mean(), cfg["seed"] + year)
            parts.append(score)
            labelled = test[data.y.loc[test].notna().to_numpy()]
            m = classification_metrics(data.y.loc[labelled], score.loc[labelled], cfg["trading"]["threshold"],
                                       b == "historical_mean")
            rows.append({"predictor": f"baseline:{b}", "year": year, "fit_rows": len(fit), "fit_end": fit[-1].date(), **m})
        preds[f"baseline:{b}"] = pd.concat(parts)
    return preds, pd.DataFrame(rows)


# --- figures -------------------------------------------------------------------------------

def save_fig(fig, name: str) -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=120)
    plt.close(fig)


def short(name: str) -> str:
    return name.replace("ml:", "").replace("baseline:", "").replace("_", " ")


def plot_splits(data: Data, wf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for i, s in enumerate(SEGMENTS):
        a, b = data.rows[s][0], data.rows[s][-1]
        ax.barh(3, (b - a).days, left=a, color=SEGMENT_COLORS[s], height=0.6)
        ax.text(a + (b - a) / 2, 3, f"{s}\n{len(data.eval_rows[s])} labelled rows", ha="center", va="center",
                fontsize=8.5, color="white" if s != "train" else INK)
    years = sorted(wf.year.unique())
    for j, year in enumerate(years[::-1]):
        fit_end = pd.Timestamp(wf[wf.year == year].fit_end.iloc[0])
        y = 2.2 - j * (2.0 / len(years))
        ax.plot([data.first, fit_end], [y, y], color=SEGMENT_COLORS["train"], linewidth=3)
        ax.plot([pd.Timestamp(year, 1, 1), min(pd.Timestamp(year, 12, 31), data.last)], [y, y],
                color=SEGMENT_COLORS["test"], linewidth=3)
    ax.set_yticks([3, 1.2], ["final split", "walk-forward\n(train | test year)"])
    ax.grid(axis="y", visible=False)
    ax.set_title("Chronological splits (labels purged at every boundary)", loc="left")
    fig.tight_layout()
    save_fig(fig, "splits")


def plot_auc(metrics: pd.DataFrame) -> None:
    ml = metrics[metrics.predictor.str.startswith("ml:")]
    names = list(dict.fromkeys(ml.predictor))
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    for k, s in enumerate(SEGMENTS):
        vals = ml[ml.segment == s].set_index("predictor").loc[names, "roc_auc"]
        ax.bar(x + (k - 1) * 0.27, vals, 0.25, color=SEGMENT_COLORS[s], label=s)
    ax.axhline(0.5, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(-0.45, 0.505, "0.5 = no skill", ha="left", va="bottom", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xticks(x, [short(n).replace("/", "\n") for n in names], fontsize=8.5)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=3)
    ax.set_title("ROC-AUC by segment: the gap from train to validation/test shows overfitting", loc="left")
    fig.tight_layout()
    save_fig(fig, "roc_auc_by_segment")


def plot_walk_forward_auc(wf: pd.DataFrame, fset: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for model in MODELS:
        d = wf[wf.predictor == f"ml:{model}/{fset}"]
        ax.plot(d.year, d.roc_auc, marker="o", color=COLORS[f"ml:{model}"], label=model.replace("_", " "))
    ax.axhline(0.5, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Walk-forward ROC-AUC by test year ({fset} features, fixed default hyperparameters)", loc="left")
    ax.legend(loc="upper left", ncols=3)
    fig.tight_layout()
    save_fig(fig, f"walk_forward_roc_auc_{fset}")


def plot_equity(equities: pd.DataFrame, names: list, title: str, file: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for n in names:
        e = equities[n] / equities[n].iloc[0]
        ax.plot(e.index, e, color=COLORS[n.split("/")[0]], label=short(n))
        ax.annotate(f"{e.iloc[-1]:.2f}", (e.index[-1], e.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK_SECONDARY, va="center", annotation_clip=False)
    ax.set_ylabel("equity / initial cash")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", ncols=2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.97, 1))
    save_fig(fig, file)


def plot_calibration(cal: pd.DataFrame, fset: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1, label="perfect calibration")
    for model in MODELS:
        d = cal[(cal.predictor == f"ml:{model}/{fset}") & (cal.segment == "test") & (cal.observations > 0)]
        ax.plot(d.mean_predicted, d.observed_rate, marker="o", color=COLORS[f"ml:{model}"],
                label=model.replace("_", " "))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability of up")
    ax.set_ylabel("observed share of up")
    ax.set_title(f"Calibration on the test period ({fset})", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save_fig(fig, f"calibration_test_{fset}")


def plot_importance(perm: pd.DataFrame, fset: str) -> None:
    d = perm[perm.feature_set == fset]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 0.32 * d.feature.nunique() + 1.8), sharey=True)
    order = d.groupby("feature")["importance_mean"].mean().sort_values().index
    for ax, model in zip(axes, MODELS):
        m = d[d.model == model].set_index("feature").loc[order]
        ax.barh(range(len(m)), m.importance_mean, xerr=m.importance_std, color=COLORS[f"ml:{model}"],
                error_kw={"elinewidth": 0.8, "ecolor": INK_MUTED})
        ax.axvline(0, color=INK_MUTED, linewidth=1)
        ax.set_title(model.replace("_", " "), loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), order, fontsize=8.5)
    fig.suptitle(f"Permutation importance on validation (drop in ROC-AUC when a feature is shuffled; {fset})",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save_fig(fig, f"permutation_importance_{fset}")


def plot_sharpe_differences(tests: pd.DataFrame, segment: str) -> None:
    d = tests[(tests.segment == segment) & (tests.method == "bootstrap_block")
              & (tests.statistic == "sharpe_ratio_difference_vs_benchmark")]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(d) + 1.6))
    y = np.arange(len(d))[::-1]
    for yi, row in zip(y, d.itertuples()):
        color = COLORS.get(row.predictor.split("/")[0], INK_MUTED)
        ax.plot([row.ci_lower, row.ci_upper], [yi, yi], color=color, linewidth=2)
        ax.plot(row.estimate, yi, "o", color=color, markersize=7, markeredgecolor=SURFACE)
        ax.text(max(row.ci_upper, 0) + 0.02, yi, f"p(Holm) {row.p_value_holm:.2f}", va="center", fontsize=8,
                color=INK_SECONDARY)
    ax.axvline(0, color=INK_MUTED, linewidth=1)
    ax.set_yticks(y, [short(n) for n in d.predictor], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Sharpe ratio minus buy & hold, {segment} period (95% block bootstrap interval)", loc="left",
                 fontsize=11)
    fig.tight_layout()
    save_fig(fig, f"sharpe_difference_{segment}")


# --- main ----------------------------------------------------------------------------------

def experiment_id(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()[:12]


def main() -> None:
    git = git_state(ROOT)
    cfg, bt, st = load(CONFIGS["ml"]), load(CONFIGS["backtest"]), load(CONFIGS["statistics"])
    seed, threshold = cfg["seed"], cfg["trading"]["threshold"]
    data = Data(cfg, st)
    costs, cash = CostModel(**bt["costs"]), bt["initial_cash"]
    boot, confidence = st["bootstrap"], st["confidence_level"]
    for d in ("predictions", "metrics", "feature_importance", "figures"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    # 1. hyperparameter search on validation
    grid = search(data, cfg)
    chosen = grid[grid.selected]

    # 2. final models: fit on train (predict train + validation), refit on train + validation (predict test once)
    scores, pred_frames, metric_rows, cal_rows, perm_rows, coef_rows, overfit_rows = {}, [], [], [], [], [], []
    for row in chosen.itertuples():
        name, params, cols = f"ml:{row.model}/{row.feature_set}", json.loads(row.hyperparameters), data.sets[row.feature_set]
        select_k = cfg["features"]["select_k"]
        pipe_train, prob_early = fit_predict(data, row.model, row.feature_set, params, seed, select_k,
                                             data.fit_rows["train"],
                                             data.rows["train"].append(data.rows["validation"]))
        pipe_final, prob_test = fit_predict(data, row.model, row.feature_set, params, seed, select_k,
                                            data.fit_rows["train+validation"], data.rows["test"])
        scores[name] = pd.concat([prob_early, prob_test])
        size, unit = complexity(pipe_final, len(cols))
        overfit_rows.append({"predictor": name, "model": row.model, "feature_set": row.feature_set,
                             "features": len(cols), "hyperparameters": row.hyperparameters,
                             "hyperparameter_candidates": len(param_grid(cfg["models"][row.model]["grid"])),
                             "complexity": size, "complexity_unit": unit})
        val_rows = data.eval_rows["validation"]
        perm = permutation_importance(pipe_train, data.X.loc[val_rows, cols], data.y.loc[val_rows].astype(int),
                                      scoring="roc_auc", n_repeats=10, random_state=seed)
        perm_rows += [{"model": row.model, "feature_set": row.feature_set, "feature": f,
                       "importance_mean": perm.importances_mean[i], "importance_std": perm.importances_std[i]}
                      for i, f in enumerate(cols)]
        est = pipe_train.named_steps["model"]
        if hasattr(est, "coef_"):
            coef_rows += [{"model": row.model, "feature_set": row.feature_set, "feature": f, "kind": "coefficient",
                           "value": v} for f, v in zip(cols, est.coef_[0])]
        elif hasattr(est, "feature_importances_"):
            coef_rows += [{"model": row.model, "feature_set": row.feature_set, "feature": f,
                           "kind": "impurity_importance", "value": v} for f, v in zip(cols, est.feature_importances_)]

    for b in BASELINES:
        scores[f"baseline:{b}"] = baseline_predictions(data, b, seed)

    for name, score in scores.items():
        is_prob = name.startswith("ml:") or name == "baseline:historical_mean"
        segment_of = pd.Series(index=score.index, dtype=object)
        for s in SEGMENTS:
            segment_of.loc[data.rows[s]] = s
            m = classification_metrics(data.y.loc[data.eval_rows[s]], score.loc[data.eval_rows[s]], threshold, is_prob)
            metric_rows.append({"predictor": name, "segment": s, **m})
            if is_prob and s != "train":
                cal_rows += [{"predictor": name, "segment": s, **r}
                             for r in calibration_table(data.y.loc[data.eval_rows[s]], score.loc[data.eval_rows[s]])
                             .to_dict("records")]
        pd.DataFrame({"date": score.index.date, "segment": segment_of.to_numpy(), "y_true": data.y.loc[score.index],
                      "label_end": data.label_end.loc[score.index].dt.date, "score": score.to_numpy(),
                      "invest": (score > threshold).astype(int).to_numpy()}) \
            .to_csv(OUT / "predictions" / f"{name.replace(':', '_').replace('/', '_')}.csv", index=False,
                    float_format="%.10g")
    metrics = pd.DataFrame(metric_rows)

    # 3. investment results with the Phase 5 engine and Phase 6 metrics
    segments = {s: (data.rows[s][0], data.rows[s][-1]) for s in SEGMENTS}
    invest_table, equities = run_backtests(data, scores, segments, costs, cash, cfg)
    tests = pd.concat([statistical_tests(equities[s], data.rf, boot, confidence, s) for s in ("validation", "test")],
                      ignore_index=True)

    # 4. walk-forward
    wf_scores, wf_metrics = walk_forward(data, cfg)
    wf_start = wf_scores[BENCHMARK].index[0]
    wf_invest, wf_equities = run_backtests(data, wf_scores, {"walk_forward": (wf_start, data.last)}, costs, cash, cfg)
    tests = pd.concat([tests, statistical_tests(wf_equities["walk_forward"], data.rf, boot, confidence,
                                                "walk_forward")], ignore_index=True)
    invest_table = pd.concat([invest_table, wf_invest], ignore_index=True)
    year_end = wf_equities["walk_forward"].resample("YE").last()
    wf_yearly = year_end / year_end.shift(1) - 1
    wf_yearly.iloc[0] = year_end.iloc[0] / cash - 1
    wf_yearly.index = wf_yearly.index.year.rename("year")

    # 5. overfitting and stability
    seed_rows = []
    for model in ("random_forest", "hist_gradient_boosting"):
        for fset in data.sets:
            for s in cfg["stability"]["seeds"]:
                _, prob = fit_predict(data, model, fset, cfg["models"][model]["default"], s,
                                      cfg["features"]["select_k"], data.fit_rows["train"], data.eval_rows["validation"])
                seed_rows.append({"model": model, "feature_set": fset, "seed": s,
                                  "validation_roc_auc": roc_auc_score(data.y.loc[data.eval_rows["validation"]], prob)})
    seeds = pd.DataFrame(seed_rows)
    overfit = pd.DataFrame(overfit_rows)
    auc = metrics.pivot(index="predictor", columns="segment", values="roc_auc")
    acc = metrics.pivot(index="predictor", columns="segment", values="accuracy")
    for s in SEGMENTS:
        overfit[f"{s}_roc_auc"] = overfit.predictor.map(auc[s])
        overfit[f"{s}_accuracy"] = overfit.predictor.map(acc[s])
    overfit["train_minus_validation_roc_auc"] = overfit.train_roc_auc - overfit.validation_roc_auc
    overfit["validation_minus_test_roc_auc"] = overfit.validation_roc_auc - overfit.test_roc_auc
    wf_ml = wf_metrics.groupby("predictor")["roc_auc"]
    overfit["walk_forward_roc_auc_mean"] = overfit.predictor.map(wf_ml.mean())
    overfit["walk_forward_roc_auc_std"] = overfit.predictor.map(wf_ml.std())
    overfit["walk_forward_years_above_0_5"] = overfit.predictor.map(wf_metrics.assign(
        above=wf_metrics.roc_auc > 0.5).groupby("predictor")["above"].mean())
    seed_stats = seeds.groupby(["model", "feature_set"])["validation_roc_auc"].agg(["mean", "std", "min", "max"])
    for stat in ("mean", "std", "min", "max"):
        overfit[f"seed_validation_roc_auc_{stat}"] = [seed_stats[stat].get((m, f), np.nan)
                                                     for m, f in zip(overfit.model, overfit.feature_set)]

    # 6. experiment log
    hashes = {"prices_sha256": content_sha256(PRICES_PATH),
              "raw_target_sha256": content_sha256(rawdata.raw_path(data.asset)),
              **{f"{k}_config_sha256": content_sha256(p) for k, p in CONFIGS.items()}}
    common = {"target": f"{data.asset} {data.horizon}-day return > 0", "prediction_horizon_days": data.horizon,
              "train_period": f"{data.period['train'][0]} to {data.period['train'][1]}",
              "validation_period": f"{data.period['validation'][0]} to {data.period['validation'][1]}",
              "test_period": f"{data.period['test'][0]} to {data.period['test'][1]}", "seed": seed,
              "threshold": threshold, "n_train": len(data.fit_rows["train"]),
              "n_validation": len(data.eval_rows["validation"]), "n_test": len(data.eval_rows["test"]),
              **git, **hashes}
    exp_rows = []
    test_inv = invest_table[invest_table.segment == "test"].set_index("predictor")
    wf_inv = invest_table[invest_table.segment == "walk_forward"].set_index("predictor")
    for name in scores:
        ml = name.startswith("ml:")
        model, fset = (name[3:].split("/") if ml else (name.split(":")[1], "rule"))
        params = chosen[(chosen.model == model) & (chosen.feature_set == fset)].hyperparameters.iloc[0] if ml else "{}"
        record = {"kind": "ml" if ml else "baseline", "model": model, "feature_set": fset,
                  "features": " ".join(data.sets[fset]) if ml else "", "hyperparameters": params, **common}
        m = metrics[metrics.predictor == name].set_index("segment")
        exp_rows.append({"experiment_id": experiment_id({k: v for k, v in record.items() if k not in hashes}),
                         "predictor": name, **record,
                         "validation_roc_auc": m.loc["validation", "roc_auc"], "test_roc_auc": m.loc["test", "roc_auc"],
                         "test_accuracy": m.loc["test", "accuracy"], "test_cagr": test_inv.loc[name, "cagr"],
                         "test_sharpe_ratio": test_inv.loc[name, "sharpe_ratio"],
                         "test_max_drawdown": test_inv.loc[name, "max_drawdown"],
                         "test_cagr_minus_buy_and_hold": test_inv.loc[name, "cagr_minus_buy_and_hold"],
                         "walk_forward_cagr": wf_inv.loc[name, "cagr"] if name in wf_inv.index else np.nan})
    experiments = pd.DataFrame(exp_rows)
    first = ["experiment_id", "predictor", "kind", "model", "feature_set"]
    experiments = experiments[first + [c for c in experiments.columns if c not in first]]

    # 7. write
    def save(table, path):
        table.to_csv(OUT / path, index=False, float_format="%.10g")

    save(experiments, "experiments.csv")
    save(data.split_table(), "metrics/splits.csv")
    save(grid, "metrics/hyperparameter_search.csv")
    save(metrics, "metrics/prediction_metrics.csv")
    save(pd.DataFrame(cal_rows), "metrics/calibration.csv")
    save(invest_table, "metrics/investment_metrics.csv")
    save(tests, "metrics/statistical_tests.csv")
    save(wf_metrics, "metrics/walk_forward_prediction_metrics.csv")
    wf_yearly.to_csv(OUT / "metrics" / "walk_forward_yearly_returns.csv", float_format="%.10g")
    save(overfit, "metrics/overfitting.csv")
    save(seeds, "metrics/seed_stability.csv")
    save(pd.DataFrame(perm_rows), "feature_importance/permutation_importance_validation.csv")
    save(pd.DataFrame(coef_rows), "feature_importance/model_importance.csv")
    for seg, eq in {**equities, **wf_equities}.items():
        eq.to_csv(OUT / "metrics" / f"equity_{seg}.csv", float_format="%.6f")

    plot_splits(data, wf_metrics)
    plot_auc(metrics)
    for fset in data.sets:
        plot_walk_forward_auc(wf_metrics, fset)
        plot_calibration(pd.DataFrame(cal_rows), fset)
        plot_importance(pd.DataFrame(perm_rows), fset)
    shown = [f"ml:{m}/extended" for m in MODELS] + [BENCHMARK, "baseline:momentum_252d",
                                                   "baseline:moving_average_200d"]
    plot_equity(equities["test"], shown, f"Out-of-sample test period after costs ({data.period['test'][0]} to "
                f"{data.period['test'][1]}, extended features)", "equity_test")
    plot_equity(wf_equities["walk_forward"], shown, f"Walk-forward out-of-sample after costs ({wf_start.date()} to "
                f"{data.last.date()}, extended features, default hyperparameters)", "equity_walk_forward")
    for seg in ("test", "walk_forward"):
        plot_sharpe_differences(tests, seg)

    info = {**hashes, **git, "python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__, "seed": seed, "bootstrap": boot,
            "features_available_from": str(data.first.date()), "last_date": str(data.last.date()),
            "hyperparameter_candidates": len(grid), "selected_models": len(chosen)}
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    view = invest_table[invest_table.segment == "test"].set_index("predictor")[
        ["cagr", "sharpe_ratio", "max_drawdown", "share_of_decisions_invested"]].join(
        metrics[metrics.segment == "test"].set_index("predictor")[["roc_auc", "accuracy"]])
    print(view.round(3).to_string())
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
